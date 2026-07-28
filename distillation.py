import contextlib
import copy
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from device_utils import make_grad_scaler, normalize_device, use_amp_for_device
from numeric_integrity import NumericIntegrityError, require_finite_tensor

# -----------------------------------------------------------------------------
# IMPORTANT: Data normalization consistency
#
# In data_utils.py, CIFAR inputs are normalized with mean=0.5/std=0.5 per channel.
# The pretrained AutoEncoder (autoencoder_pretrained.py) uses a Sigmoid decoder,
# so its natural output is in [0, 1]. If we feed AE outputs directly into the
# classifiers (trained/evaluated on normalized inputs), the distribution shift
# can cause accuracy to stay flat across rounds.
#
# We therefore:
#   1) Denormalize real images back to [0,1] before passing into AE.encoder.
#   2) Normalize AE.decoder outputs back to the model's expected space.
# -----------------------------------------------------------------------------

_CIFAR_MEAN = (0.5, 0.5, 0.5)
_CIFAR_STD = (0.5, 0.5, 0.5)
_LOGIT_CLAMP = 30.0
_INPUT_CLAMP = 1.0
_DEFAULT_DISTILL_LR_MULT = 0.2
_DEFAULT_MIN_DISTILL_LR = 1e-4
_DEFAULT_DISTILL_GRAD_CLIP = 5.0

_AE_CACHE = {}
NO_TEACHER = object()


def _amp_context(device, enabled=True):
    dev = normalize_device(device)
    if bool(enabled) and use_amp_for_device(dev):
        return torch.autocast(device_type=dev.type, dtype=torch.float16)
    return contextlib.nullcontext()


def _should_run_numeric_check(step: int, args) -> bool:
    if bool(getattr(args, "strict_numeric_checks", False)):
        return True
    interval = int(max(0, getattr(args, "numeric_check_interval", 0)))
    return interval > 0 and int(step) % interval == 0


def _shareable_device_key(device) -> str:
    return normalize_device(device).type



def _normalize_for_model(x: torch.Tensor) -> torch.Tensor:
    """(x - mean) / std, supports NCHW."""
    mean = x.new_tensor(_CIFAR_MEAN).view(1, -1, 1, 1)
    std = x.new_tensor(_CIFAR_STD).view(1, -1, 1, 1)
    return (x - mean) / std



def _denormalize_for_ae(x: torch.Tensor) -> torch.Tensor:
    """x * std + mean, supports NCHW."""
    mean = x.new_tensor(_CIFAR_MEAN).view(1, -1, 1, 1)
    std = x.new_tensor(_CIFAR_STD).view(1, -1, 1, 1)
    return x * std + mean


try:
    from utils import KL_Loss as _ExtKL
    _HAS_EXT_KL = True
except Exception:
    _HAS_EXT_KL = False


class _KLDivWithT(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, student_logits, teacher_logits):
        T = self.temperature
        return F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2)



def _make_kl(temperature):
    if _HAS_EXT_KL:
        return _ExtKL(temperature)
    return _KLDivWithT(temperature)


try:
    from autoencoder_pretrained import create_autoencoder as _ext_create_autoencoder
    _HAS_EXT_AE = True
except Exception:
    _HAS_EXT_AE = False


class _Identity(nn.Module):
    def forward(self, x):
        return x


class _IdentityAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Identity()
        self.decoder = _Identity()



def create_autoencoder(device="cpu"):
    device = normalize_device(device)
    if _HAS_EXT_AE:
        return _ext_create_autoencoder(device=device)
    return _IdentityAE().to(device)

def get_shared_autoencoder(device="cpu"):
    key = _shareable_device_key(device)
    if key not in _AE_CACHE:
        model = create_autoencoder(device=key)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _AE_CACHE[key] = model
    return _AE_CACHE[key]



def _forward_logits(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out



def _sanitize_logits(x: torch.Tensor, clamp: float = _LOGIT_CLAMP) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp).clamp_(-clamp, clamp)



def _sanitize_model_input(x: torch.Tensor, clamp: float = _INPUT_CLAMP) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp).clamp_(-clamp, clamp)



def _model_is_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    for _, buf in model.named_buffers():
        if not torch.isfinite(buf).all():
            return False
    return True



def _freeze_model_state(model: nn.Module) -> dict:
    frozen = {}
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            frozen[k] = v.detach().cpu().clone()
        else:
            frozen[k] = copy.deepcopy(v)
    return frozen



def _restore_model_state(model: nn.Module, frozen_state: dict, device) -> None:
    model.load_state_dict(frozen_state, strict=True)
    model.to(device)



def _gradients_are_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and (not torch.isfinite(p.grad).all()):
            return False
    return True



def _resolve_distill_lr(args) -> float:
    if hasattr(args, "distill_lr") and getattr(args, "distill_lr") is not None:
        return float(getattr(args, "distill_lr"))
    base_lr = float(getattr(args, "lr", 0.01))
    return max(base_lr * _DEFAULT_DISTILL_LR_MULT, _DEFAULT_MIN_DISTILL_LR)



def _resolve_grad_clip(args) -> float:
    value = getattr(args, "distill_grad_clip_norm", _DEFAULT_DISTILL_GRAD_CLIP)
    if value is None:
        return 0.0
    return float(value)


_global_index = 0


class Node:
    def __init__(self, args, model, father=None):
        global _global_index
        self.args = args
        self.model = model.to(args.device)
        self.dataset = None
        self.father = father
        self.children = []
        self.index = _global_index
        _global_index += 1
        self.ae_device = normalize_device(getattr(args, "ae_device", "cpu")).type
        self.autoencoder = get_shared_autoencoder(device=self.ae_device)
        self.noises = []
        self.labels = []

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.father is None



def create_child_for_upper_level(args, upper_level, children_number, models):
    result = []
    model_iter = iter(models)
    for ele in upper_level:
        sub_nodes = []
        for _ in range(children_number):
            model = next(model_iter)
            sub_nodes.append(Node(args, model, ele))
        ele.children = sub_nodes
        result.extend(sub_nodes)
    return result


def build_server_client_tree(args, server_model, client_models: Sequence[nn.Module]):
    """Create the two-level FedAgg topology used by the reference system.

    The server is the single root/global student. Every client is a direct leaf
    and acts as a locally trained teacher during the existing bidirectional
    FedAgg distillation routine.
    """

    if not client_models:
        raise ValueError("At least one client model is required.")

    server = Node(args, server_model)
    clients = create_child_for_upper_level(
        args,
        [server],
        len(client_models),
        client_models,
    )
    server.role = "server"
    server.logical_id = 0
    for client_id, client in enumerate(clients):
        client.role = "client"
        client.logical_id = int(client_id)
    return server, clients


class Loss_Non_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=10):
        super().__init__()
        self.alpha = alpha
        self.kl_loss_crit = _make_kl(temperature)
        self.ce_loss_crit = nn.CrossEntropyLoss()

    def forward(self, output_batch, teacher_outputs, label):
        if getattr(self, "strict_numeric_checks", False):
            require_finite_tensor(output_batch, phase="distillation", metric="student_logits")
            require_finite_tensor(teacher_outputs, phase="distillation", metric="teacher_logits")
        output_batch = _sanitize_logits(output_batch)
        teacher_outputs = _sanitize_logits(teacher_outputs)
        loss_ce = self.ce_loss_crit(output_batch, label.long())
        loss_kl = self.kl_loss_crit(output_batch, teacher_outputs.detach())
        return loss_ce + self.alpha * loss_kl


class Loss_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=1, alpha2=1):
        super().__init__()
        self.non_leaf_loss_crit = Loss_Non_Leaf(temperature, alpha)
        self.ce_loss_crit = nn.CrossEntropyLoss()
        self.alpha2 = alpha2

    def forward(self, output_fake, teacher_outputs_fake, label_fake, output_true, label_true):
        output_true = _sanitize_logits(output_true)
        loss_leaf = self.non_leaf_loss_crit(output_fake, teacher_outputs_fake.detach(), label_fake.long())
        loss_ce = self.ce_loss_crit(output_true, label_true.long())
        return loss_leaf + self.alpha2 * loss_ce


@torch.no_grad()
def Init(node):
    if node.is_root():
        for child in node.children:
            Init(child)
    elif node.is_leaf():
        ae = node.autoencoder
        for img, label in node.dataset:
            if not torch.is_tensor(label):
                label = torch.tensor(label, dtype=torch.long)
            img = img.to(node.ae_device, non_blocking=True)
            img_ae = _denormalize_for_ae(img)
            img_ae = torch.nan_to_num(img_ae, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            noise = ae.encoder(img_ae)
            noise = torch.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
            node.noises.append(noise.detach().cpu())
            node.labels.append(label.detach().cpu())
        node.father.noises.extend(node.noises)
        node.father.labels.extend(node.labels)
    else:
        for child in node.children:
            Init(child)
        node.father.noises.extend(node.noises)
        node.father.labels.extend(node.labels)



def _resolve_teacher_model(node_origin, node_neigh, args):
    resolver = getattr(args, "teacher_resolver", None)
    if callable(resolver):
        teacher = resolver(node_origin, node_neigh, args)
        if teacher is NO_TEACHER:
            return None
        if teacher is not None:
            return teacher
    return node_neigh.model



def _call_before_parent_distill_hook(node, args):
    hook = getattr(args, "before_parent_distill_hook", None)
    if callable(hook):
        hook(node, args)



def BSBODP(node1, node2, args):
    BSBODP_dir(node1, node2, args)
    BSBODP_dir(node2, node1, args)



def BSBODP_dir(node_origin, node_neigh, args):
    noises = node_neigh.noises if len(node_neigh.noises) < len(node_origin.noises) else node_origin.noises
    labels = node_neigh.labels if len(node_neigh.labels) < len(node_origin.labels) else node_origin.labels
    T = getattr(args, "temperature", getattr(args, "T_agg", 1.0))
    crit_non_leaf = Loss_Non_Leaf(T)
    crit_leaf = Loss_Leaf(T)
    crit_non_leaf.strict_numeric_checks = bool(getattr(args, "strict_numeric_checks", False))
    crit_leaf.non_leaf_loss_crit.strict_numeric_checks = bool(getattr(args, "strict_numeric_checks", False))
    optimizer = torch.optim.SGD(
        node_origin.model.parameters(),
        lr=_resolve_distill_lr(args),
        momentum=0.9,
    )
    grad_clip_norm = _resolve_grad_clip(args)

    ae = node_neigh.autoencoder
    ae_device = node_neigh.ae_device
    device = normalize_device(args.device)
    amp_enabled = bool(getattr(args, "amp", False)) and use_amp_for_device(device)
    ae_amp_enabled = bool(getattr(args, "amp", False)) and use_amp_for_device(ae_device)
    scaler = make_grad_scaler(device, enabled=amp_enabled)

    teacher_model = _resolve_teacher_model(node_origin, node_neigh, args)
    if teacher_model is None:
        return
    teacher_model.eval()
    node_origin.model.train()

    try:
        for idx, (noise_cpu, label_cpu) in enumerate(zip(noises, labels), start=1):
            optimizer.zero_grad(set_to_none=True)

            label = label_cpu.to(device, non_blocking=True)
            noise = noise_cpu.to(ae_device, non_blocking=True)
            if bool(getattr(args, "strict_numeric_checks", False)):
                require_finite_tensor(noise, phase="distillation", metric="generated_input")
            noise = torch.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)

            with _amp_context(ae_device, enabled=ae_amp_enabled):
                fake_data = ae.decoder(noise)
            fake_data = torch.nan_to_num(fake_data, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            fake_data = _normalize_for_model(fake_data)
            fake_data = _sanitize_model_input(fake_data).to(device, non_blocking=True)

            with torch.no_grad():
                with _amp_context(device, enabled=amp_enabled):
                    nei_logits = _forward_logits(teacher_model, fake_data)
                if bool(getattr(args, "strict_numeric_checks", False)):
                    require_finite_tensor(nei_logits, phase="distillation", metric="teacher_logits")
                nei_logits = _sanitize_logits(nei_logits)

            with _amp_context(device, enabled=amp_enabled):
                logits_fake = _forward_logits(node_origin.model, fake_data)
                if bool(getattr(args, "strict_numeric_checks", False)):
                    require_finite_tensor(logits_fake, phase="distillation", metric="student_logits")
                logits_fake = _sanitize_logits(logits_fake)

                if node_origin.is_leaf():
                    img, label_true = node_origin.dataset[idx - 1]
                    if not torch.is_tensor(label_true):
                        label_true = torch.tensor(label_true, dtype=torch.long)
                    img = _sanitize_model_input(img.to(device, non_blocking=True))
                    label_true = label_true.to(device, non_blocking=True)
                    logits_true_raw = _forward_logits(node_origin.model, img)
                    if bool(getattr(args, "strict_numeric_checks", False)):
                        require_finite_tensor(logits_true_raw, phase="distillation", metric="student_logits_true")
                    logits_true = _sanitize_logits(logits_true_raw)
                    loss = crit_leaf(logits_fake, nei_logits, label, logits_true, label_true)
                else:
                    loss = crit_non_leaf(logits_fake, nei_logits, label)

            should_check = _should_run_numeric_check(idx, args)
            if should_check and (not torch.isfinite(loss).item()):
                if bool(getattr(args, "strict_numeric_checks", False)):
                    raise NumericIntegrityError("Non-finite distillation loss.", phase="distillation", metric="loss", value=float(loss.detach().cpu().item()))
                stats = getattr(args, "numeric_stats", None)
                if isinstance(stats, dict):
                    stats["numeric_failure_count"] = stats.get("numeric_failure_count", 0.0) + 1.0
                optimizer.zero_grad(set_to_none=True)
                continue

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            params = [p for p in node_origin.model.parameters() if p.grad is not None]
            if should_check and (not _gradients_are_finite(node_origin.model)):
                if bool(getattr(args, "strict_numeric_checks", False)):
                    raise NumericIntegrityError("Non-finite distillation gradient.", phase="distillation", metric="gradient", value="nonfinite")
                stats = getattr(args, "numeric_stats", None)
                if isinstance(stats, dict):
                    stats["numeric_failure_count"] = stats.get("numeric_failure_count", 0.0) + 1.0
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    _finalize_amp_skip(scaler, optimizer, strict_numeric_checks=getattr(args, "strict_numeric_checks", False))
                continue

            if grad_clip_norm > 0 and params:
                nn.utils.clip_grad_norm_(params, grad_clip_norm)
                if should_check and (not _gradients_are_finite(node_origin.model)):
                    if bool(getattr(args, "strict_numeric_checks", False)):
                        raise NumericIntegrityError("Non-finite distillation gradient after clipping.", phase="distillation", metric="gradient", value="nonfinite")
                    stats = getattr(args, "numeric_stats", None)
                    if isinstance(stats, dict):
                        stats["numeric_failure_count"] = stats.get("numeric_failure_count", 0.0) + 1.0
                    optimizer.zero_grad(set_to_none=True)
                    if amp_enabled:
                        _finalize_amp_skip(scaler, optimizer, strict_numeric_checks=getattr(args, "strict_numeric_checks", False))
                    continue

            # The old implementation cloned every parameter tensor to CPU before
            # every optimizer step.  That serialized CUDA execution.  Keep the
            # expensive per-step rollback only in strict debug mode; the simulator
            # still maintains a round-level pre-distillation rollback snapshot.
            state_backup = _freeze_model_state(node_origin.model) if should_check else None
            try:
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            except RuntimeError as exc:
                if bool(getattr(args, "strict_numeric_checks", False)):
                    raise NumericIntegrityError(
                        "Optimizer step failed during distillation.",
                        phase="distillation",
                        metric="optimizer_step",
                        value=type(exc).__name__,
                    ) from exc
                if state_backup is not None:
                    _restore_model_state(node_origin.model, state_backup, device)
                stats = getattr(args, "numeric_stats", None)
                if isinstance(stats, dict):
                    stats["round_rollback_count"] = stats.get("round_rollback_count", 0.0) + 1.0
                    stats["numeric_failure_count"] = stats.get("numeric_failure_count", 0.0) + 1.0
                optimizer.zero_grad(set_to_none=True)
                continue

            if should_check and (not _model_is_finite(node_origin.model)):
                if bool(getattr(args, "strict_numeric_checks", False)):
                    raise NumericIntegrityError("Non-finite distillation model parameter.", phase="distillation", metric="parameter", value="nonfinite")
                if state_backup is not None:
                    _restore_model_state(node_origin.model, state_backup, device)
                stats = getattr(args, "numeric_stats", None)
                if isinstance(stats, dict):
                    stats["round_rollback_count"] = stats.get("round_rollback_count", 0.0) + 1.0
                optimizer.zero_grad(set_to_none=True)
                continue
    finally:
        if teacher_model is not None and teacher_model is not node_neigh.model:
            del teacher_model



def train_FedAgg(node, args):
    if node.is_root():
        for child in node.children:
            train_FedAgg(child, args)
    elif node.is_leaf():
        _call_before_parent_distill_hook(node, args)
        BSBODP(node, node.father, args)
    else:
        for child in node.children:
            train_FedAgg(child, args)
        _call_before_parent_distill_hook(node, args)
        BSBODP(node, node.father, args)



def _finalize_amp_skip(scaler, optimizer, *, strict_numeric_checks=False):
    """Reset GradScaler state after an early-continue that happened post-unscale_."""
    try:
        scaler.step(optimizer)
    except (AssertionError, RuntimeError) as exc:
        if bool(strict_numeric_checks):
            raise NumericIntegrityError(
                "GradScaler failed while finalizing a numeric skip.",
                phase="distillation",
                metric="grad_scaler",
                value=type(exc).__name__,
            ) from exc
    scaler.update()


def distill_student_from_teacher(student, teacher, dataloader,
                                 device='cpu', lr=0.01, temperature=2.0, alpha=0.5,
                                 distill_lr=None, grad_clip_norm=_DEFAULT_DISTILL_GRAD_CLIP, amp=False,
                                 strict_numeric_checks=False, numeric_stats=None):
    device = normalize_device(device)
    amp_enabled = bool(amp) and use_amp_for_device(device)
    scaler = make_grad_scaler(device, enabled=amp_enabled)
    student.train()
    teacher.eval()
    actual_lr = float(distill_lr) if distill_lr is not None else max(float(lr) * _DEFAULT_DISTILL_LR_MULT, _DEFAULT_MIN_DISTILL_LR)
    optimizer = torch.optim.SGD(student.parameters(), lr=actual_lr, momentum=0.9)
    crit_non_leaf = Loss_Non_Leaf(temperature, alpha)
    for imgs, labels in dataloader:
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if bool(strict_numeric_checks):
            require_finite_tensor(imgs, phase="distillation", metric="input")
        imgs = _sanitize_model_input(imgs)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with _amp_context(device, enabled=amp_enabled):
                teacher_logits = _forward_logits(teacher, imgs)
            if bool(strict_numeric_checks):
                require_finite_tensor(teacher_logits, phase="distillation", metric="teacher_logits")
            teacher_logits = _sanitize_logits(teacher_logits)
            if not torch.isfinite(teacher_logits).all():
                if isinstance(numeric_stats, dict):
                    numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
                continue
        with _amp_context(device, enabled=amp_enabled):
            student_logits = _forward_logits(student, imgs)
        if bool(strict_numeric_checks):
            require_finite_tensor(student_logits, phase="distillation", metric="student_logits")
        student_logits = _sanitize_logits(student_logits)
        if not torch.isfinite(student_logits).all():
            if isinstance(numeric_stats, dict):
                numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
            continue
        loss = crit_non_leaf(student_logits, teacher_logits, labels)
        if not torch.isfinite(loss):
            if bool(strict_numeric_checks):
                raise NumericIntegrityError("Non-finite distillation loss.", phase="distillation", metric="loss", value=float(loss.detach().cpu().item()))
            if isinstance(numeric_stats, dict):
                numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
            continue
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if not _gradients_are_finite(student):
            if bool(strict_numeric_checks):
                raise NumericIntegrityError("Non-finite distillation gradient.", phase="distillation", metric="gradient", value="nonfinite")
            if isinstance(numeric_stats, dict):
                numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                _finalize_amp_skip(scaler, optimizer, strict_numeric_checks=strict_numeric_checks)
            continue
        params = [p for p in student.parameters() if p.grad is not None]
        if grad_clip_norm is not None and float(grad_clip_norm) > 0 and params:
            nn.utils.clip_grad_norm_(params, float(grad_clip_norm))
            if not _gradients_are_finite(student):
                if bool(strict_numeric_checks):
                    raise NumericIntegrityError("Non-finite distillation gradient after clipping.", phase="distillation", metric="gradient", value="nonfinite")
                if isinstance(numeric_stats, dict):
                    numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    _finalize_amp_skip(scaler, optimizer, strict_numeric_checks=strict_numeric_checks)
                continue
        state_backup = _freeze_model_state(student)
        try:
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        except RuntimeError as exc:
            if bool(strict_numeric_checks):
                raise NumericIntegrityError(
                    "Optimizer step failed during distillation.",
                    phase="distillation",
                    metric="optimizer_step",
                    value=type(exc).__name__,
                ) from exc
            _restore_model_state(student, state_backup, device)
            if isinstance(numeric_stats, dict):
                numeric_stats["round_rollback_count"] = numeric_stats.get("round_rollback_count", 0.0) + 1.0
                numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
            optimizer.zero_grad(set_to_none=True)
            continue
        if not _model_is_finite(student):
            if bool(strict_numeric_checks):
                raise NumericIntegrityError("Non-finite distillation model parameter.", phase="distillation", metric="parameter", value="nonfinite")
            _restore_model_state(student, state_backup, device)
            if isinstance(numeric_stats, dict):
                numeric_stats["round_rollback_count"] = numeric_stats.get("round_rollback_count", 0.0) + 1.0
            optimizer.zero_grad(set_to_none=True)
            continue
