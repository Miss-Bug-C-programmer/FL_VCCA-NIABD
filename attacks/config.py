from __future__ import annotations

from dataclasses import asdict, dataclass


SUPPORTED_ATTACKS = ("none", "badnets", "dba", "blend", "dynamic")


@dataclass(frozen=True)
class AttackConfig:
    """Configuration for real client-side backdoor data poisoning.

    The attack layer is an experiment-only input transformation.  It does not
    modify VCAA, NIABD, logits transport, aggregation, or server access rules.
    """

    attack_type: str = "none"
    target_label: int = 0
    malicious_fraction: float = 0.20
    poison_ratio: float = 0.20
    attack_start_round: int = 15
    attack_end_round: int = 10**9
    poison_interval: int = 1
    trigger_size: int = 4
    trigger_value: float = 1.0
    blend_alpha: float = 0.20
    dynamic_period: int = 10
    dba_parts: int = 4

    def __post_init__(self) -> None:
        attack = str(self.attack_type).lower()
        object.__setattr__(self, "attack_type", attack)
        if attack not in SUPPORTED_ATTACKS:
            raise ValueError(
                f"Unsupported attack_type={attack!r}; expected one of "
                f"{', '.join(SUPPORTED_ATTACKS)}."
            )
        if int(self.target_label) < 0:
            raise ValueError("target_label must be non-negative.")
        if not 0.0 <= float(self.malicious_fraction) <= 1.0:
            raise ValueError("malicious_fraction must be in [0, 1].")
        if not 0.0 <= float(self.poison_ratio) <= 1.0:
            raise ValueError("poison_ratio must be in [0, 1].")
        if int(self.attack_start_round) < 1:
            raise ValueError("attack_start_round must be at least 1.")
        if int(self.attack_end_round) < int(self.attack_start_round):
            raise ValueError(
                "attack_end_round must not be below attack_start_round."
            )
        if int(self.poison_interval) < 1:
            raise ValueError("poison_interval must be at least 1.")
        if int(self.trigger_size) < 1:
            raise ValueError("trigger_size must be positive.")
        if not -1.0 <= float(self.trigger_value) <= 1.0:
            raise ValueError(
                "trigger_value must lie in [-1, 1] for the normalized images."
            )
        if not 0.0 < float(self.blend_alpha) <= 1.0:
            raise ValueError("blend_alpha must be in (0, 1].")
        if int(self.dynamic_period) < 1:
            raise ValueError("dynamic_period must be at least 1.")
        if int(self.dba_parts) != 4:
            raise ValueError(
                "The formal DBA construction uses exactly four local triggers."
            )

    def active(self, round_number: int) -> bool:
        """Return whether the attack schedule is active in a source round."""

        round_number = int(round_number)
        if self.attack_type == "none":
            return False
        if round_number < int(self.attack_start_round):
            return False
        if round_number > int(self.attack_end_round):
            return False
        return (
            (round_number - int(self.attack_start_round))
            % int(self.poison_interval)
            == 0
        )

    def to_dict(self) -> dict:
        return asdict(self)
