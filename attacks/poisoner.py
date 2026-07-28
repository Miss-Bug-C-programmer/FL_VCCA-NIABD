from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from attacks.attack_plan import AttackPlan
from attacks.trigger import (
    apply_badnets,
    apply_blend,
    apply_dba,
    apply_dynamic,
)


@dataclass
class PoisonStats:
    eligible: int = 0
    poisoned: int = 0
    batches_seen: int = 0
    poisoned_batches: int = 0

    def add(self, *, eligible: int, poisoned: int) -> None:
        self.eligible += int(eligible)
        self.poisoned += int(poisoned)
        self.batches_seen += 1
        if int(poisoned) > 0:
            self.poisoned_batches += 1

    def to_dict(self) -> dict:
        return asdict(self)


class BackdoorBatchPoisoner:
    """Deterministic label-targeted image poisoning for a single client."""

    def __init__(self, *, plan: AttackPlan, client_id: int) -> None:
        self.plan = plan
        self.client_id = int(client_id)
        self._round_number = 0
        self._round_stats = PoisonStats()

    @property
    def round_stats(self) -> PoisonStats:
        return PoisonStats(**self._round_stats.to_dict())

    @property
    def last_stats(self) -> PoisonStats:
        """Backward-compatible snapshot of the current round statistics."""

        return self.round_stats

    def start_round(self, round_number: int) -> None:
        self._round_number = int(round_number)
        self._round_stats = PoisonStats()

    def _sample_indices(
        self,
        eligible_indices: torch.Tensor,
        *,
        round_number: int,
        batch_index: int,
        n_poison: int,
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(self.plan.seed) * 1_000_003
            + self.client_id * 10_007
            + int(round_number) * 101
            + int(batch_index)
        )
        permutation = torch.randperm(
            int(eligible_indices.numel()),
            generator=generator,
        )[: int(n_poison)]
        return eligible_indices[
            permutation.to(eligible_indices.device)
        ]

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        round_number: int,
        batch_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        round_number = int(round_number)
        if self._round_number != round_number:
            self.start_round(round_number)
        if not self.plan.active_for(self.client_id, round_number):
            self._round_stats.add(eligible=0, poisoned=0)
            return images, labels

        config = self.plan.config
        labels_out = labels.clone()
        eligible_indices = torch.nonzero(
            labels_out.long() != int(config.target_label),
            as_tuple=False,
        ).view(-1)
        eligible = int(eligible_indices.numel())
        if eligible == 0 or float(config.poison_ratio) <= 0.0:
            self._round_stats.add(eligible=eligible, poisoned=0)
            return images, labels_out

        n_poison = min(
            eligible,
            max(1, int(round(eligible * float(config.poison_ratio)))),
        )
        chosen = self._sample_indices(
            eligible_indices,
            round_number=round_number,
            batch_index=int(batch_index),
            n_poison=n_poison,
        )
        output = images.clone()
        selected = output[chosen]
        size = int(config.trigger_size)

        if config.attack_type == "badnets":
            selected = apply_badnets(
                selected,
                size=size,
                value=float(config.trigger_value),
            )
        elif config.attack_type == "dba":
            selected = apply_dba(
                selected,
                size=size,
                part=self.plan.dba_part(self.client_id),
                value=float(config.trigger_value),
            )
        elif config.attack_type == "blend":
            selected = apply_blend(
                selected,
                alpha=float(config.blend_alpha),
            )
        elif config.attack_type == "dynamic":
            selected = apply_dynamic(
                selected,
                size=size,
                round_number=round_number,
                attack_start_round=int(config.attack_start_round),
                period=int(config.dynamic_period),
            )
        else:
            raise ValueError(
                f"Unsupported active attack type={config.attack_type!r}."
            )

        output[chosen] = selected
        labels_out[chosen] = int(config.target_label)
        self._round_stats.add(eligible=eligible, poisoned=n_poison)
        return output, labels_out
