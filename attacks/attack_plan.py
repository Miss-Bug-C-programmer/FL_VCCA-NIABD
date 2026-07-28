from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Optional

from attacks.config import AttackConfig


@dataclass(frozen=True)
class AttackPlan:
    """Strategy-independent attack ground truth for one FL run.

    The plan is deterministic for (seed, num_clients, config).  It is supplied
    only to experiment orchestration and client-local poisoners.  VCAA/NIABD do
    not receive malicious identities or attack labels.
    """

    seed: int
    num_clients: int
    malicious_client_ids: tuple[int, ...]
    dba_trigger_assignments: tuple[tuple[int, int], ...]
    config: AttackConfig

    @classmethod
    def build(
        cls,
        *,
        seed: int,
        num_clients: int,
        config: AttackConfig,
    ) -> "AttackPlan":
        if int(num_clients) <= 0:
            raise ValueError("num_clients must be positive.")
        if config.attack_type == "none" or float(config.malicious_fraction) == 0.0:
            malicious: tuple[int, ...] = ()
        else:
            count = max(
                1,
                int(round(int(num_clients) * float(config.malicious_fraction))),
            )
            count = min(int(num_clients), count)
            client_ids = list(range(int(num_clients)))
            rng = random.Random(int(seed) + 19073)
            rng.shuffle(client_ids)
            malicious = tuple(sorted(client_ids[:count]))
        assignments = tuple(
            (client_id, position % int(config.dba_parts))
            for position, client_id in enumerate(malicious)
        )
        return cls(
            seed=int(seed),
            num_clients=int(num_clients),
            malicious_client_ids=malicious,
            dba_trigger_assignments=assignments,
            config=config,
        )

    @property
    def malicious_set(self) -> frozenset[int]:
        return frozenset(int(x) for x in self.malicious_client_ids)

    def is_malicious(self, client_id: int) -> bool:
        return int(client_id) in self.malicious_set

    def dba_part(self, client_id: int) -> int:
        mapping = dict(self.dba_trigger_assignments)
        client_id = int(client_id)
        if client_id not in mapping:
            raise KeyError(
                f"Client {client_id} has no DBA local-trigger assignment."
            )
        return int(mapping[client_id])

    def active_for(self, client_id: int, round_number: int) -> bool:
        return self.is_malicious(client_id) and self.config.active(round_number)

    def to_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "num_clients": int(self.num_clients),
            "malicious_client_ids": [
                int(x) for x in self.malicious_client_ids
            ],
            "dba_trigger_assignments": [
                [int(a), int(b)] for a, b in self.dba_trigger_assignments
            ],
            "config": self.config.to_dict(),
        }

    @property
    def identity(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AttackPlan":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            seed=int(payload["seed"]),
            num_clients=int(payload["num_clients"]),
            malicious_client_ids=tuple(
                int(x) for x in payload["malicious_client_ids"]
            ),
            dba_trigger_assignments=tuple(
                (int(a), int(b))
                for a, b in payload["dba_trigger_assignments"]
            ),
            config=AttackConfig(**payload["config"]),
        )

    @classmethod
    def resolve(
        cls,
        *,
        seed: int,
        num_clients: int,
        config: AttackConfig,
        plan_path: Optional[str] = None,
    ) -> "AttackPlan":
        if plan_path:
            plan = cls.load(plan_path)
            if int(plan.seed) != int(seed):
                raise ValueError("Loaded attack plan seed does not match run seed.")
            if int(plan.num_clients) != int(num_clients):
                raise ValueError(
                    "Loaded attack plan client count does not match the run."
                )
            if plan.config != config:
                raise ValueError(
                    "Loaded attack plan configuration does not match CLI config."
                )
            return plan
        return cls.build(seed=seed, num_clients=num_clients, config=config)
