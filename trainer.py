import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from device_utils import make_grad_scaler, normalize_device, use_amp_for_device
from numeric_integrity import NumericIntegrityError, require_finite_tensor


def _amp_context(device, enabled=True):
    dev = normalize_device(device)
    if bool(enabled) and use_amp_for_device(dev):
        return torch.autocast(device_type=dev.type, dtype=torch.float16)
    return contextlib.nullcontext()


DEFAULT_MAX_CONSECUTIVE_AMP_OVERFLOWS = 8


def _finalize_amp_skip(
    scaler,
    optimizer,
    *,
    strict_numeric_checks=False,
    numeric_context=None,
):
    """Skip an overflowed AMP step and advance GradScaler's backoff state."""

    scale_before = float(scaler.get_scale())
    try:
        # ``unscale_`` has already populated GradScaler's found-inf state.
        # ``step`` therefore skips optimizer.step() and ``update`` lowers the
        # dynamic loss scale.
        scaler.step(optimizer)
    except (AssertionError, RuntimeError) as exc:
        if bool(strict_numeric_checks):
            raise NumericIntegrityError(
                "GradScaler failed while finalizing a numeric skip.",
                phase="training",
                metric="grad_scaler",
                value=type(exc).__name__,
                context=numeric_context,
            ) from exc
    scaler.update()
    return scale_before, float(scaler.get_scale())


def _should_run_numeric_check(step: int, strict_numeric_checks: bool, numeric_check_interval: int) -> bool:
    """Return whether expensive CUDA-synchronizing finite checks should run.

    The historical implementation checked every output tensor, loss tensor and
    gradient tensor on every mini-batch.  Those Python-side boolean checks force
    CUDA synchronization and substantially reduce GPU utilization for the small
    CNNs used by this prototype.  In normal submission runs, round-level rollback
    in ``simulate.py`` remains enabled.  Use ``strict_numeric_checks=True`` for
    debugging or set ``numeric_check_interval`` to a positive interval.
    """
    if bool(strict_numeric_checks):
        return True
    interval = int(max(0, numeric_check_interval))
    return interval > 0 and int(step) % interval == 0


def _first_nonfinite_gradient(named_params):
    """Return the first non-finite gradient's name and value for diagnostics."""

    for name, param in named_params:
        if param.grad is None:
            continue
        flat = param.grad.detach().reshape(-1)
        bad = ~torch.isfinite(flat)
        if bool(bad.any().item()):
            value = flat[bad][0].detach().cpu().item()
            return str(name), value
    return None


def _batch_numeric_context(numeric_context, step: int):
    context = dict(numeric_context or {})
    task_key = str(context.get("key", "training"))
    context["key"] = f"{task_key}/batch:{int(step)}"
    return context


def _increment_stat(numeric_stats, key: str, amount: int = 1) -> None:
    if isinstance(numeric_stats, dict):
        numeric_stats[key] = int(numeric_stats.get(key, 0)) + int(amount)


def _record_amp_overflow(
    numeric_stats,
    *,
    scale_before: float,
    scale_after: float,
    consecutive_overflows: int,
) -> None:
    _increment_stat(numeric_stats, "amp_overflow_count")
    _increment_stat(numeric_stats, "optimizer_step_skipped_count")
    if isinstance(numeric_stats, dict):
        numeric_stats["amp_loss_scale_before"] = float(scale_before)
        numeric_stats["amp_loss_scale_after"] = float(scale_after)
        numeric_stats["max_consecutive_amp_overflows"] = max(
            int(numeric_stats.get("max_consecutive_amp_overflows", 0)),
            int(consecutive_overflows),
        )


def _raise_if_amp_overflow_streak_exceeded(
    *,
    consecutive_overflows: int,
    max_consecutive_amp_overflows: int,
    gradient_failure,
    context,
) -> None:
    if int(consecutive_overflows) <= int(max_consecutive_amp_overflows):
        return
    gradient_name, gradient_value = gradient_failure or (
        "unknown",
        "nonfinite",
    )
    raise NumericIntegrityError(
        "AMP gradient overflow did not recover within the configured streak limit.",
        phase="training",
        metric=f"amp_gradient_overflow:{gradient_name}",
        value=gradient_value,
        context=context,
    )


def local_train(
    model,
    dataloader,
    device='cpu',
    lr=0.01,
    epochs=1,
    grad_clip_norm=5.0,
    amp=False,
    strict_numeric_checks=False,
    numeric_check_interval=0,
    numeric_stats=None,
    optimizer=None,
    batch_transform=None,
    round_number: int = 0,
    numeric_context=None,
    scaler=None,
    max_consecutive_amp_overflows=DEFAULT_MAX_CONSECUTIVE_AMP_OVERFLOWS,
):
    """Train one local client model.

    ``strict_numeric_checks=False`` avoids forced CUDA synchronization on every
    mini-batch.  It does not remove the round-level post-distillation rollback in
    the simulator.  For numerical debugging, pass ``strict_numeric_checks=True``.
    """
    device = normalize_device(device)
    amp_enabled = bool(amp) and use_amp_for_device(device)
    if int(max_consecutive_amp_overflows) < 1:
        raise ValueError("max_consecutive_amp_overflows must be positive.")
    scaler = (
        make_grad_scaler(device, enabled=amp_enabled)
        if scaler is None
        else scaler
    )
    model.train()
    if bool(strict_numeric_checks):
        initial_context = dict(numeric_context or {})
        initial_context["key"] = (
            f"{initial_context.get('key', 'training')}/before_training"
        )
        for name, param in model.named_parameters():
            require_finite_tensor(
                param,
                phase="training",
                metric=f"parameter_before:{name}",
                context=initial_context,
            )
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        for group in optimizer.param_groups:
            group["lr"] = float(lr)
    loss_fn = nn.CrossEntropyLoss()
    step = 0
    consecutive_amp_overflows = 0

    for _ in range(epochs):
        for imgs, labels in dataloader:
            step += 1
            context = _batch_numeric_context(numeric_context, step)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if batch_transform is not None:
                imgs, labels = batch_transform(
                    imgs,
                    labels,
                    round_number=int(round_number),
                    batch_index=int(step),
                )
                if imgs.ndim != 4 or labels.ndim != 1:
                    raise ValueError(
                        "batch_transform must return [N,C,H,W] images and [N] labels."
                    )
                if int(imgs.shape[0]) != int(labels.shape[0]):
                    raise ValueError(
                        "batch_transform changed image/label batch cardinality."
                    )
            if bool(strict_numeric_checks):
                require_finite_tensor(
                    imgs,
                    phase="training",
                    metric="inputs",
                    context=context,
                )
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, enabled=amp_enabled):
                output = model(imgs)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if bool(strict_numeric_checks):
                    require_finite_tensor(
                        output,
                        phase="training",
                        metric="logits",
                        context=context,
                    )
                loss = loss_fn(output, labels)

            should_check = _should_run_numeric_check(step, strict_numeric_checks, numeric_check_interval)
            if should_check and (not torch.isfinite(loss).item()):
                if bool(strict_numeric_checks):
                    raise NumericIntegrityError(
                        "Non-finite local training loss.",
                        phase="training",
                        metric="loss",
                        value=float(loss.detach().cpu().item()),
                        context=context,
                    )
                if isinstance(numeric_stats, dict):
                    numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
                optimizer.zero_grad(set_to_none=True)
                continue

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            named_params = [
                (name, param)
                for name, param in model.named_parameters()
                if param.grad is not None
            ]
            params = [param for _, param in named_params]
            gradient_failure = (
                _first_nonfinite_gradient(named_params)
                if should_check
                else None
            )
            if gradient_failure is not None:
                gradient_name, gradient_value = gradient_failure
                if amp_enabled:
                    scale_before, scale_after = _finalize_amp_skip(
                        scaler,
                        optimizer,
                        strict_numeric_checks=strict_numeric_checks,
                        numeric_context=context,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_amp_overflows += 1
                    _record_amp_overflow(
                        numeric_stats,
                        scale_before=scale_before,
                        scale_after=scale_after,
                        consecutive_overflows=consecutive_amp_overflows,
                    )
                    _raise_if_amp_overflow_streak_exceeded(
                        consecutive_overflows=consecutive_amp_overflows,
                        max_consecutive_amp_overflows=(
                            max_consecutive_amp_overflows
                        ),
                        gradient_failure=gradient_failure,
                        context=context,
                    )
                    continue
                if bool(strict_numeric_checks):
                    raise NumericIntegrityError(
                        "Non-finite local training gradient.",
                        phase="training",
                        metric=f"gradient:{gradient_name}",
                        value=gradient_value,
                        context=context,
                    )
                if isinstance(numeric_stats, dict):
                    numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    _finalize_amp_skip(scaler, optimizer, strict_numeric_checks=strict_numeric_checks)
                continue

            if grad_clip_norm is not None and float(grad_clip_norm) > 0 and params:
                nn.utils.clip_grad_norm_(params, float(grad_clip_norm))
                gradient_failure = (
                    _first_nonfinite_gradient(named_params)
                    if should_check
                    else None
                )
                if gradient_failure is not None:
                    gradient_name, gradient_value = gradient_failure
                    if amp_enabled:
                        # The pre-clip gradients were finite, so GradScaler
                        # did not classify this as a recoverable overflow.
                        raise NumericIntegrityError(
                            "Gradient clipping produced a non-finite AMP gradient.",
                            phase="training",
                            metric=f"gradient_after_clip:{gradient_name}",
                            value=gradient_value,
                            context=context,
                        )
                    if bool(strict_numeric_checks):
                        raise NumericIntegrityError(
                            "Non-finite local training gradient after clipping.",
                            phase="training",
                            metric=f"gradient_after_clip:{gradient_name}",
                            value=gradient_value,
                            context=context,
                        )
                    if isinstance(numeric_stats, dict):
                        numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
                    optimizer.zero_grad(set_to_none=True)
                    if amp_enabled:
                        _finalize_amp_skip(scaler, optimizer, strict_numeric_checks=strict_numeric_checks)
                    continue

            if amp_enabled:
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                if scale_after < scale_before:
                    consecutive_amp_overflows += 1
                    _record_amp_overflow(
                        numeric_stats,
                        scale_before=scale_before,
                        scale_after=scale_after,
                        consecutive_overflows=consecutive_amp_overflows,
                    )
                    _raise_if_amp_overflow_streak_exceeded(
                        consecutive_overflows=consecutive_amp_overflows,
                        max_consecutive_amp_overflows=(
                            max_consecutive_amp_overflows
                        ),
                        gradient_failure=gradient_failure,
                        context=context,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
            else:
                optimizer.step()
            consecutive_amp_overflows = 0
            _increment_stat(numeric_stats, "optimizer_step_count")
            if bool(strict_numeric_checks):
                for name, param in model.named_parameters():
                    require_finite_tensor(
                        param,
                        phase="training",
                        metric=f"parameter:{name}",
                        context=context,
                    )


@torch.no_grad()
def evaluate(model, dataloader, device='cpu', amp=False):
    device = normalize_device(device)
    amp_enabled = bool(amp) and use_amp_for_device(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _amp_context(device, enabled=amp_enabled):
                outputs = model(imgs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def predict_logits(model, dataloader, device='cpu', amp=False):
    device = normalize_device(device)
    amp_enabled = bool(amp) and use_amp_for_device(device)
    model.eval()
    outputs_all = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            if not batch:
                raise ValueError("Proxy batch cannot be empty.")
            imgs = batch[0]
        else:
            imgs = batch
        imgs = imgs.to(device, non_blocking=True)
        with _amp_context(device, enabled=amp_enabled):
            outputs = model(imgs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)
        outputs_all.append(outputs.detach().cpu())
    if not outputs_all:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(outputs_all, dim=0)


def distill_with_logits(
    model,
    dataloader,
    target_logits,
    device='cpu',
    lr=1e-3,
    epochs=1,
    temperature=2.0,
    amp=False,
    grad_clip_norm=5.0,
    strict_numeric_checks=False,
    numeric_check_interval=0,
    numeric_stats=None,
    targets_are_probabilities=False,
    clean_ce_weight=0.0,
    numeric_context=None,
    scaler=None,
    max_consecutive_amp_overflows=DEFAULT_MAX_CONSECUTIVE_AMP_OVERFLOWS,
):
    device = normalize_device(device)
    amp_enabled = bool(amp) and use_amp_for_device(device)
    if int(max_consecutive_amp_overflows) < 1:
        raise ValueError("max_consecutive_amp_overflows must be positive.")
    scaler = (
        make_grad_scaler(device, enabled=amp_enabled)
        if scaler is None
        else scaler
    )
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr))
    T = float(temperature)
    step = 0
    consecutive_amp_overflows = 0
    for _ in range(int(max(1, epochs))):
        cursor = 0
        for batch_data in dataloader:
            labels = None
            if isinstance(batch_data, (tuple, list)):
                if not batch_data:
                    raise ValueError("Distillation batch cannot be empty.")
                imgs = batch_data[0]
                if len(batch_data) >= 2:
                    labels = batch_data[1]
            else:
                imgs = batch_data
            step += 1
            context = _batch_numeric_context(numeric_context, step)
            batch = imgs.size(0)
            target = target_logits[cursor: cursor + batch]
            if target.numel() == 0:
                break
            cursor += batch
            imgs = imgs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if bool(strict_numeric_checks):
                require_finite_tensor(
                    imgs,
                    phase="training",
                    metric="distillation_inputs",
                    context=context,
                )
                require_finite_tensor(
                    target,
                    phase="training",
                    metric="teacher_logits_raw",
                    context=context,
                )
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, enabled=amp_enabled):
                outputs = model(imgs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                if bool(strict_numeric_checks):
                    require_finite_tensor(
                        outputs,
                        phase="training",
                        metric="student_logits",
                        context=context,
                    )
                    require_finite_tensor(
                        target,
                        phase="training",
                        metric="teacher_logits",
                        context=context,
                    )
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)
                if targets_are_probabilities:
                    target_probabilities = torch.nan_to_num(
                        target,
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp_min(0.0)
                    target_probabilities = target_probabilities / (
                        target_probabilities.sum(
                            dim=1,
                            keepdim=True,
                        ).clamp_min(1e-8)
                    )
                else:
                    target = torch.nan_to_num(
                        target,
                        nan=0.0,
                        posinf=30.0,
                        neginf=-30.0,
                    ).clamp_(-30.0, 30.0)
                    target_probabilities = F.softmax(target / T, dim=1)
                loss = F.kl_div(
                    F.log_softmax(outputs / T, dim=1),
                    target_probabilities,
                    reduction='batchmean',
                ) * (T ** 2)
                if float(clean_ce_weight) > 0.0:
                    if labels is None:
                        raise ValueError(
                            "clean CE anchoring requires labels in the proxy loader."
                        )
                    labels = labels.to(device, non_blocking=True).long()
                    if int(labels.shape[0]) != int(outputs.shape[0]):
                        raise ValueError(
                            "Proxy labels and logits must share the batch cursor."
                        )
                    loss = loss + float(clean_ce_weight) * F.cross_entropy(
                        outputs,
                        labels,
                    )

            should_check = _should_run_numeric_check(step, strict_numeric_checks, numeric_check_interval)
            if should_check and (not torch.isfinite(loss).item()):
                if bool(strict_numeric_checks):
                    raise NumericIntegrityError(
                        "Non-finite distillation loss.",
                        phase="training",
                        metric="distillation_loss",
                        value=float(loss.detach().cpu().item()),
                        context=context,
                    )
                if isinstance(numeric_stats, dict):
                    numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
                optimizer.zero_grad(set_to_none=True)
                continue

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            named_params = [
                (name, param)
                for name, param in model.named_parameters()
                if param.grad is not None
            ]
            params = [param for _, param in named_params]
            gradient_failure = (
                _first_nonfinite_gradient(named_params)
                if should_check
                else None
            )
            if gradient_failure is not None:
                gradient_name, gradient_value = gradient_failure
                if amp_enabled:
                    scale_before, scale_after = _finalize_amp_skip(
                        scaler,
                        optimizer,
                        strict_numeric_checks=strict_numeric_checks,
                        numeric_context=context,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_amp_overflows += 1
                    _record_amp_overflow(
                        numeric_stats,
                        scale_before=scale_before,
                        scale_after=scale_after,
                        consecutive_overflows=consecutive_amp_overflows,
                    )
                    _raise_if_amp_overflow_streak_exceeded(
                        consecutive_overflows=consecutive_amp_overflows,
                        max_consecutive_amp_overflows=(
                            max_consecutive_amp_overflows
                        ),
                        gradient_failure=gradient_failure,
                        context=context,
                    )
                    continue
                if bool(strict_numeric_checks):
                    raise NumericIntegrityError(
                        "Non-finite distillation gradient.",
                        phase="training",
                        metric=f"distillation_gradient:{gradient_name}",
                        value=gradient_value,
                        context=context,
                    )
                if isinstance(numeric_stats, dict):
                    numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    _finalize_amp_skip(scaler, optimizer, strict_numeric_checks=strict_numeric_checks)
                continue

            if grad_clip_norm is not None and float(grad_clip_norm) > 0 and params:
                nn.utils.clip_grad_norm_(params, float(grad_clip_norm))
                gradient_failure = (
                    _first_nonfinite_gradient(named_params)
                    if should_check
                    else None
                )
                if gradient_failure is not None:
                    gradient_name, gradient_value = gradient_failure
                    if amp_enabled:
                        raise NumericIntegrityError(
                            "Gradient clipping produced a non-finite AMP distillation gradient.",
                            phase="training",
                            metric=(
                                "distillation_gradient_after_clip:"
                                f"{gradient_name}"
                            ),
                            value=gradient_value,
                            context=context,
                        )
                    if bool(strict_numeric_checks):
                        raise NumericIntegrityError(
                            "Non-finite distillation gradient after clipping.",
                            phase="training",
                            metric=f"distillation_gradient_after_clip:{gradient_name}",
                            value=gradient_value,
                            context=context,
                        )
                    if isinstance(numeric_stats, dict):
                        numeric_stats["numeric_failure_count"] = numeric_stats.get("numeric_failure_count", 0.0) + 1.0
                    optimizer.zero_grad(set_to_none=True)
                    if amp_enabled:
                        _finalize_amp_skip(scaler, optimizer, strict_numeric_checks=strict_numeric_checks)
                    continue

            if amp_enabled:
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                if scale_after < scale_before:
                    consecutive_amp_overflows += 1
                    _record_amp_overflow(
                        numeric_stats,
                        scale_before=scale_before,
                        scale_after=scale_after,
                        consecutive_overflows=consecutive_amp_overflows,
                    )
                    _raise_if_amp_overflow_streak_exceeded(
                        consecutive_overflows=consecutive_amp_overflows,
                        max_consecutive_amp_overflows=(
                            max_consecutive_amp_overflows
                        ),
                        gradient_failure=gradient_failure,
                        context=context,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
            else:
                optimizer.step()
            consecutive_amp_overflows = 0
            _increment_stat(numeric_stats, "optimizer_step_count")
            if bool(strict_numeric_checks):
                for name, param in model.named_parameters():
                    require_finite_tensor(
                        param,
                        phase="training",
                        metric=f"distillation_parameter:{name}",
                        context=context,
                    )
