from __future__ import annotations

import time

import torch.nn as nn

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
            base_server_round=self.model_round,
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
