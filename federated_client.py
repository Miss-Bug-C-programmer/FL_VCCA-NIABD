from __future__ import annotations

import hashlib
import time

import torch
import torch.nn as nn

from attacks import AttackPlan
from attacks.evaluation import apply_evaluation_trigger
from logits_transport import ClientLogitsPacket, ServerLogitsPacket
from trainer import distill_with_logits, local_train, predict_logits


class FederatedClient:
    """Client runtime that owns its model, private loader and local updates."""

    def __init__(
        self,
        *,
        client_id: int,
        model: nn.Module,
        train_loader,
        device,
        amp: bool = False,
        strict_numeric_checks: bool = False,
    ) -> None:
        self.client_id = int(client_id)
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.amp = bool(amp)
        self.strict_numeric_checks = bool(strict_numeric_checks)
        self.model_round = 0

    def train_local(
        self,
        *,
        epochs: int,
        learning_rate: float,
        batch_transform=None,
        round_number: int = 0,
    ) -> None:
        local_train(
            self.model,
            self.train_loader,
            device=self.device,
            lr=float(learning_rate),
            epochs=int(epochs),
            amp=self.amp,
            strict_numeric_checks=self.strict_numeric_checks,
            batch_transform=batch_transform,
            round_number=int(round_number),
        )
        self.model_round += 1

    def upload_proxy_logits(
        self,
        proxy_loader,
        *,
        query_id: str,
    ) -> ClientLogitsPacket:
        """Run the real local model and serialize its proxy logits for upload."""

        logits = predict_logits(
            self.model,
            proxy_loader,
            device=self.device,
            amp=self.amp,
        )
        return ClientLogitsPacket.from_logits(
            client_id=self.client_id,
            model_round=self.model_round,
            generated_at_s=float(time.monotonic()),
            query_id=query_id,
            logits=logits,
            source_round=self.model_round,
            # ``model_round`` is the version after this local update; the
            # update was based on the server/student version immediately
            # before it.  Keeping these distinct makes lineage auditable.
            base_server_round=max(0, self.model_round - 1),
            local_model_version=self.model_round,
        )

    def distill_from_server(
        self,
        proxy_loader,
        packet: ServerLogitsPacket,
        *,
        learning_rate: float,
        temperature: float,
    ) -> None:
        target_logits = packet.decode_logits()
        distill_with_logits(
            self.model,
            proxy_loader,
            target_logits,
            device=self.device,
            lr=float(learning_rate),
            epochs=1,
            temperature=float(temperature),
            amp=self.amp,
            strict_numeric_checks=self.strict_numeric_checks,
        )

    @torch.no_grad()
    def compute_backdoor_diagnostics(
        self,
        proxy_loader,
        *,
        plan: AttackPlan,
        dataset_name: str,
        experiment_seed: int,
        source_round: int,
    ) -> dict[str, object]:
        """Return experiment-only scalars; never construct a transport packet."""

        seed_material = (
            f"{dataset_name}|{int(experiment_seed)}|"
            f"{plan.config.attack_type}|{int(source_round)}|{self.client_id}"
        ).encode("utf-8")
        diagnostic_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            byteorder="big",
            signed=False,
        ) % (2**63 - 1)
        device_obj = torch.device(self.device)
        cuda_devices = (
            [device_obj.index if device_obj.index is not None else 0]
            if device_obj.type == "cuda"
            else []
        )
        was_training = bool(self.model.training)
        target_probability_clean = 0.0
        target_probability_triggered = 0.0
        l1_sum = 0.0
        l2_sum = 0.0
        flip_count = 0
        sample_count = 0
        dba_part = (
            plan.dba_part(self.client_id)
            if plan.config.attack_type == "dba"
            and plan.is_malicious(self.client_id)
            else None
        )
        try:
            self.model.eval()
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(diagnostic_seed)
                if device_obj.type == "cuda":
                    torch.cuda.manual_seed_all(diagnostic_seed)
                for batch in proxy_loader:
                    images = batch[0] if isinstance(batch, (tuple, list)) else batch
                    images = images.to(device_obj, non_blocking=True)
                    triggered = apply_evaluation_trigger(
                        images,
                        plan=plan,
                        round_number=int(source_round),
                        dba_part=dba_part,
                    )
                    clean_logits = self.model(images)
                    triggered_logits = self.model(triggered)
                    if isinstance(clean_logits, (tuple, list)):
                        clean_logits = clean_logits[0]
                    if isinstance(triggered_logits, (tuple, list)):
                        triggered_logits = triggered_logits[0]
                    clean_logits = torch.nan_to_num(
                        clean_logits,
                        nan=0.0,
                        posinf=30.0,
                        neginf=-30.0,
                    ).clamp(-30.0, 30.0)
                    triggered_logits = torch.nan_to_num(
                        triggered_logits,
                        nan=0.0,
                        posinf=30.0,
                        neginf=-30.0,
                    ).clamp(-30.0, 30.0)
                    clean_probabilities = torch.softmax(clean_logits, dim=1)
                    triggered_probabilities = torch.softmax(
                        triggered_logits,
                        dim=1,
                    )
                    difference = triggered_logits - clean_logits
                    batch_size = int(clean_logits.shape[0])
                    target = int(plan.config.target_label)
                    target_probability_clean += float(
                        clean_probabilities[:, target].sum().item()
                    )
                    target_probability_triggered += float(
                        triggered_probabilities[:, target].sum().item()
                    )
                    l1_sum += float(
                        difference.abs().mean(dim=1).sum().item()
                    )
                    l2_sum += float(
                        difference.square().sum(dim=1).sqrt().sum().item()
                    )
                    flip_count += int(
                        (
                            clean_logits.argmax(dim=1)
                            != triggered_logits.argmax(dim=1)
                        ).sum().item()
                    )
                    sample_count += batch_size
        finally:
            self.model.train(was_training)
        denominator = float(max(sample_count, 1))
        return {
            "diagnostic_scope": "experiment-only oracle diagnostic",
            "diagnostic_usage": "not a deployable defense signal",
            "diagnostic_reporter_trust": "not assumed",
            "diagnostic_seed": int(diagnostic_seed),
            "diagnostic_proxy_samples": int(sample_count),
            "clean_proxy_target_probability": (
                target_probability_clean / denominator
            ),
            "triggered_proxy_target_probability": (
                target_probability_triggered / denominator
            ),
            "clean_trigger_logit_l1_deviation": l1_sum / denominator,
            "clean_trigger_logit_l2_deviation": l2_sum / denominator,
            "clean_trigger_prediction_flip_rate": (
                float(flip_count) / denominator
            ),
        }
