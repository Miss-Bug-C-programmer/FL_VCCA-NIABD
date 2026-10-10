from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Optional, Sequence

from attacks.config import AttackConfig
from dataset_metadata import canonical_dataset_name, dataset_normalization


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
    client_sample_counts: tuple[int, ...] = ()
    dataset_name: str = "cifar10"

    def __post_init__(self) -> None:
        dataset_name = canonical_dataset_name(self.dataset_name)
        normalization = dataset_normalization(dataset_name)
        object.__setattr__(self, "dataset_name", dataset_name)
        if (
            normalization.raw_space_triggers
            and self.config.attack_type != "none"
            and not 0.0 <= float(self.config.trigger_value) <= 1.0
        ):
            raise ValueError(
                "CINIC-10 trigger_value is a raw pixel intensity and must "
                "lie in [0, 1]."
            )

    @staticmethod
    def _data_balanced_selection(
        *,
        seed: int,
        count: int,
        malicious_fraction: float,
        client_sample_counts: Sequence[int],
    ) -> tuple[int, ...]:
        """Select a reproducible random cohort with representative data mass.

        Client IDs are still randomized from the experiment seed. A local swap
        search then minimizes the difference between malicious client data mass
        and the configured malicious fraction. No labels, model outputs, test
        data, defense state, or expected metrics are consumed.
        """

        counts = tuple(int(value) for value in client_sample_counts)
        if any(value <= 0 for value in counts):
            raise ValueError("Every client sample count must be positive.")
        order = list(range(len(counts)))
        random.Random(int(seed) + 19073).shuffle(order)
        selected = set(order[: int(count)])
        target_mass = float(sum(counts)) * float(malicious_fraction)

        def error(ids) -> float:
            return abs(float(sum(counts[index] for index in ids)) - target_mass)

        rank = {client_id: position for position, client_id in enumerate(order)}
        while True:
            current_error = error(selected)
            candidates = []
            for outgoing in selected:
                for incoming in set(order) - selected:
                    proposal = (selected - {outgoing}) | {incoming}
                    proposal_error = error(proposal)
                    if proposal_error + 1e-12 < current_error:
                        candidates.append(
                            (
                                proposal_error,
                                rank[incoming],
                                rank[outgoing],
                                incoming,
                                outgoing,
                            )
                        )
            if not candidates:
                break
            _, _, _, incoming, outgoing = min(candidates)
            selected.remove(outgoing)
            selected.add(incoming)
        return tuple(sorted(selected))

    @classmethod
    def build(
        cls,
        *,
        seed: int,
        num_clients: int,
        config: AttackConfig,
        client_sample_counts: Optional[Sequence[int]] = None,
        dataset_name: str = "cifar10",
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
            if str(config.malicious_selection) == "data-balanced":
                if client_sample_counts is None:
                    raise ValueError(
                        "data-balanced malicious selection requires client sample counts."
                    )
                if len(client_sample_counts) != int(num_clients):
                    raise ValueError(
                        "client_sample_counts must contain one value per client."
                    )
                malicious = cls._data_balanced_selection(
                    seed=int(seed),
                    count=int(count),
                    malicious_fraction=float(config.malicious_fraction),
                    client_sample_counts=client_sample_counts,
                )
            else:
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
            client_sample_counts=(
                tuple(int(value) for value in client_sample_counts)
                if client_sample_counts is not None
                else ()
            ),
            dataset_name=canonical_dataset_name(dataset_name),
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
        payload = {
            "seed": int(self.seed),
            "num_clients": int(self.num_clients),
            "malicious_client_ids": [
                int(x) for x in self.malicious_client_ids
            ],
            "dba_trigger_assignments": [
                [int(a), int(b)] for a, b in self.dba_trigger_assignments
            ],
            "config": self.config.to_dict(),
            "client_sample_counts": [
                int(value) for value in self.client_sample_counts
            ],
            "malicious_sample_fraction": (
                float(
                    sum(
                        self.client_sample_counts[client_id]
                        for client_id in self.malicious_client_ids
                    )
                )
                / float(sum(self.client_sample_counts))
                if self.client_sample_counts
                else None
            ),
        }
        # Preserve existing CIFAR-10 plan JSON and identity. Other datasets
        # must carry their identity so trigger normalization cannot silently
        # fall back to CIFAR semantics when a plan is reused.
        if str(self.dataset_name).lower() != "cifar10":
            payload["dataset_name"] = str(self.dataset_name).lower()
        return payload

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
            client_sample_counts=tuple(
                int(value)
                for value in payload.get("client_sample_counts", ())
            ),
            dataset_name=canonical_dataset_name(
                payload.get("dataset_name", "cifar10")
            ),
        )

    @classmethod
    def resolve(
        cls,
        *,
        seed: int,
        num_clients: int,
        config: AttackConfig,
        plan_path: Optional[str] = None,
        client_sample_counts: Optional[Sequence[int]] = None,
        dataset_name: str = "cifar10",
    ) -> "AttackPlan":
        if plan_path:
            plan = cls.load(plan_path)
            if int(plan.seed) != int(seed):
                raise ValueError("Loaded attack plan seed does not match run seed.")
            if int(plan.num_clients) != int(num_clients):
                raise ValueError(
                    "Loaded attack plan client count does not match the run."
                )
            if plan.dataset_name != canonical_dataset_name(dataset_name):
                raise ValueError(
                    "Loaded attack plan dataset does not match the run."
                )
            if plan.config != config:
                raise ValueError(
                    "Loaded attack plan configuration does not match CLI config."
                )
            if (
                plan.client_sample_counts
                and client_sample_counts is not None
                and tuple(int(value) for value in client_sample_counts)
                != plan.client_sample_counts
            ):
                raise ValueError(
                    "Loaded attack plan client sample counts do not match the run."
                )
            return plan
        return cls.build(
            seed=seed,
            num_clients=num_clients,
            config=config,
            client_sample_counts=client_sample_counts,
            dataset_name=dataset_name,
        )
