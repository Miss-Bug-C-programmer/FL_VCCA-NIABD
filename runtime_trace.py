from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import random
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class RuntimeFaultEvent:
    """External system conditions for one client and one source round."""

    client_id: int
    source_round: int
    selected: bool
    available: bool
    compute_slowdown_factor: float
    upload_delay_s: float
    dropped_attempts: Tuple[int, ...]
    ack_delay_s_by_attempt: Tuple[Tuple[int, float], ...]

    def ack_delay(self, attempt_index: int) -> float:
        return float(dict(self.ack_delay_s_by_attempt).get(
            int(attempt_index),
            0.0,
        ))


@dataclass(frozen=True)
class RuntimeTrace:
    seed: int
    num_clients: int
    rounds: int
    warmup_rounds: int
    profile_name: str
    events: Tuple[RuntimeFaultEvent, ...]

    def __post_init__(self) -> None:
        expected = {
            (client_id, source_round)
            for source_round in range(1, int(self.rounds) + 1)
            for client_id in range(int(self.num_clients))
        }
        actual = {
            (int(event.client_id), int(event.source_round))
            for event in self.events
        }
        if len(actual) != len(self.events) or actual != expected:
            raise ValueError(
                "Runtime trace must define every client/round exactly once."
            )

    def event(
        self,
        client_id: int,
        source_round: int,
    ) -> RuntimeFaultEvent:
        key = (int(client_id), int(source_round))
        for event in self.events:
            if (event.client_id, event.source_round) == key:
                return event
        raise KeyError(f"Runtime trace has no event for {key}.")

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "seed": int(self.seed),
            "num_clients": int(self.num_clients),
            "rounds": int(self.rounds),
            "warmup_rounds": int(self.warmup_rounds),
            "profile_name": str(self.profile_name),
            "events": [
                {
                    **asdict(event),
                    "dropped_attempts": list(event.dropped_attempts),
                    "ack_delay_s_by_attempt": [
                        [int(attempt), float(delay)]
                        for attempt, delay
                        in event.ack_delay_s_by_attempt
                    ],
                }
                for event in self.events
            ],
        }

    def save(
        self,
        path: str,
        *,
        metadata: Dict[str, object] | None = None,
    ) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            values = self.to_dict()
            if metadata is not None:
                values["run_metadata"] = dict(metadata)
            json.dump(
                values,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

    @classmethod
    def load(cls, path: str) -> "RuntimeTrace":
        with open(path, "r", encoding="utf-8") as handle:
            values = json.load(handle)
        if int(values.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported runtime trace schema.")
        events = []
        for item in values["events"]:
            events.append(RuntimeFaultEvent(
                client_id=int(item["client_id"]),
                source_round=int(item["source_round"]),
                selected=bool(item["selected"]),
                available=bool(item["available"]),
                compute_slowdown_factor=float(
                    item["compute_slowdown_factor"]
                ),
                upload_delay_s=float(item["upload_delay_s"]),
                dropped_attempts=tuple(
                    int(attempt)
                    for attempt in item["dropped_attempts"]
                ),
                ack_delay_s_by_attempt=tuple(
                    (int(attempt), float(delay))
                    for attempt, delay
                    in item["ack_delay_s_by_attempt"]
                ),
            ))
        return cls(
            seed=int(values["seed"]),
            num_clients=int(values["num_clients"]),
            rounds=int(values["rounds"]),
            warmup_rounds=int(values["warmup_rounds"]),
            profile_name=str(values["profile_name"]),
            events=tuple(events),
        )


def load_runtime_profile(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict):
        raise ValueError("Runtime profile must be a JSON object.")
    forbidden = {
        "stale_ratio",
        "forced_version_lag",
        "fake_age",
        "fake_round",
        "fake_stale",
    }
    present = forbidden.intersection(profile)
    if present:
        raise ValueError(
            f"Runtime profile contains forbidden fields: {sorted(present)}"
        )
    return profile


def _keyed_random(seed: int, *parts: object) -> random.Random:
    material = "|".join([str(int(seed)), *map(str, parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _probability(profile: Dict[str, object], key: str, default: float) -> float:
    value = float(profile.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1].")
    return value


def generate_runtime_trace(
    *,
    profile: Dict[str, object],
    seed: int,
    num_clients: int,
    rounds: int,
    warmup_rounds: int,
    participation_rate: float,
) -> RuntimeTrace:
    """Generate fault decisions independently of VCAA/NIABD strategy."""

    if int(num_clients) <= 0 or int(rounds) <= 0:
        raise ValueError("Trace dimensions must be positive.")
    if not 0.0 < float(participation_rate) <= 1.0:
        raise ValueError("participation_rate must be in (0, 1].")
    slow_fraction = _probability(
        profile,
        "slow_client_fraction",
        0.0,
    )
    availability = _probability(
        profile,
        "availability_probability",
        1.0,
    )
    drop_probability = _probability(
        profile,
        "upload_attempt_drop_probability",
        0.0,
    )
    ack_probability = _probability(
        profile,
        "ack_delay_probability",
        0.0,
    )
    normal_factor = float(
        profile.get("normal_compute_slowdown_factor", 1.0)
    )
    slow_factor = float(
        profile.get("slow_compute_slowdown_factor", normal_factor)
    )
    normal_upload = float(profile.get("normal_upload_delay_s", 0.0))
    slow_upload = float(profile.get("slow_upload_delay_s", normal_upload))
    ack_delay_s = float(profile.get("ack_delay_s", 0.0))
    if min(normal_factor, slow_factor) < 1.0:
        raise ValueError("compute slowdown factors must be at least 1.")
    if min(normal_upload, slow_upload, ack_delay_s) < 0.0:
        raise ValueError("Runtime delays must be non-negative.")

    slow_count = int(math.ceil(int(num_clients) * slow_fraction))
    slow_clients = set(range(max(0, int(num_clients) - slow_count), int(num_clients)))
    explicit = profile.get("events", {})
    if not isinstance(explicit, dict):
        raise ValueError("Runtime profile events must be an object.")
    events: List[RuntimeFaultEvent] = []
    for source_round in range(1, int(rounds) + 1):
        warmup = source_round <= int(warmup_rounds)
        for client_id in range(int(num_clients)):
            selection_rng = _keyed_random(
                seed,
                client_id,
                source_round,
                "selection",
            )
            selected = (
                True
                if warmup
                else selection_rng.random() < float(participation_rate)
            )
            availability_rng = _keyed_random(
                seed,
                client_id,
                source_round,
                "availability",
            )
            available_value = (
                True
                if warmup
                else availability_rng.random() < availability
            )
            is_slow = client_id in slow_clients
            factor = (
                1.0
                if warmup
                else (slow_factor if is_slow else normal_factor)
            )
            upload_delay = (
                0.0
                if warmup
                else (slow_upload if is_slow else normal_upload)
            )
            drop_rng = _keyed_random(
                seed,
                client_id,
                source_round,
                1,
                "attempt_drop",
            )
            dropped = (
                ()
                if warmup or drop_rng.random() >= drop_probability
                else (1,)
            )
            ack_rng = _keyed_random(
                seed,
                client_id,
                source_round,
                1,
                "ack_delay",
            )
            ack_delays: Tuple[Tuple[int, float], ...] = (
                ()
                if warmup or ack_rng.random() >= ack_probability
                else ((1, ack_delay_s),)
            )

            override = explicit.get(f"{client_id}:{source_round}", {})
            if not isinstance(override, dict):
                raise ValueError(
                    "Runtime event override must be an object."
                )
            selected = bool(override.get("selected", selected))
            available_value = bool(
                override.get("available", available_value)
            )
            factor = float(
                override.get("compute_slowdown_factor", factor)
            )
            upload_delay = float(
                override.get("upload_delay_s", upload_delay)
            )
            dropped = tuple(
                int(attempt)
                for attempt in override.get(
                    "dropped_attempts",
                    dropped,
                )
            )
            raw_ack = override.get(
                "ack_delay_s_by_attempt",
                dict(ack_delays),
            )
            if isinstance(raw_ack, dict):
                ack_delays = tuple(sorted(
                    (int(attempt), float(delay))
                    for attempt, delay in raw_ack.items()
                ))
            else:
                ack_delays = tuple(
                    (int(pair[0]), float(pair[1]))
                    for pair in raw_ack
                )
            if factor < 1.0 or upload_delay < 0.0:
                raise ValueError("Runtime event contains invalid latency.")
            if any(attempt <= 0 for attempt in dropped):
                raise ValueError("Dropped attempt indices must be positive.")
            if any(delay < 0.0 for _, delay in ack_delays):
                raise ValueError("ACK delays must be non-negative.")
            events.append(RuntimeFaultEvent(
                client_id=client_id,
                source_round=source_round,
                selected=selected,
                available=available_value,
                compute_slowdown_factor=factor,
                upload_delay_s=upload_delay,
                dropped_attempts=dropped,
                ack_delay_s_by_attempt=ack_delays,
            ))
    return RuntimeTrace(
        seed=int(seed),
        num_clients=int(num_clients),
        rounds=int(rounds),
        warmup_rounds=int(warmup_rounds),
        profile_name=str(profile.get("name", "runtime-profile")),
        events=tuple(events),
    )


def trace_fault_sequence(
    trace: RuntimeTrace,
) -> Iterable[Tuple[object, ...]]:
    """Return strategy-independent decisions for replay assertions."""

    for event in trace.events:
        yield (
            event.client_id,
            event.source_round,
            event.selected,
            event.available,
            event.compute_slowdown_factor,
            event.upload_delay_s,
            event.dropped_attempts,
            event.ack_delay_s_by_attempt,
        )
