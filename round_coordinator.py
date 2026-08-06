from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple
import uuid

from logits_transport import ClientLogitsPacket, ServerLogitsPacket
from runtime_trace import RuntimeTrace


CLIENT_IDLE = "IDLE"
CLIENT_BUSY = "BUSY"
CLIENT_OFFLINE = "OFFLINE"
CLIENT_FAILED = "FAILED"
CLIENT_STOPPED = "STOPPED"

TASK_DISPATCHED = "DISPATCHED"
TASK_IN_FLIGHT = "IN_FLIGHT"
TASK_KNOWLEDGE_RECEIVED = "KNOWLEDGE_RECEIVED"
TASK_RESERVED = "RESERVED"
TASK_CONSUMED = "CONSUMED"
TASK_FAILED = "FAILED"
TASK_EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class ClientTask:
    task_id: str
    client_id: int
    source_round: int
    base_server_round: int
    proxy_version: str
    dispatched_at_s: float
    local_epochs: int
    learning_rate: float
    distillation_temperature: float
    enable_client_distillation: bool
    compute_slowdown_factor: float
    upload_delay_s: float
    dropped_attempts: Tuple[int, ...]
    server_logits_packet: Optional[ServerLogitsPacket]

    def rpc_metadata(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "client_id": int(self.client_id),
            "source_round": int(self.source_round),
            "base_server_round": int(self.base_server_round),
            "proxy_version": self.proxy_version,
            "dispatched_at_s": float(self.dispatched_at_s),
            "local_epochs": int(self.local_epochs),
            "learning_rate": float(self.learning_rate),
            "distillation_temperature": float(
                self.distillation_temperature
            ),
            "enable_client_distillation": bool(
                self.enable_client_distillation
            ),
            "compute_slowdown_factor": float(
                self.compute_slowdown_factor
            ),
            "upload_delay_s": float(self.upload_delay_s),
            "dropped_attempts": list(self.dropped_attempts),
            "server_logits": (
                None
                if self.server_logits_packet is None
                else self.server_logits_packet.rpc_metadata()
            ),
        }


@dataclass
class TaskRegistryEntry:
    task: ClientTask
    status: str
    received_at_s: float = float("nan")
    consumed_at_s: float = float("nan")
    packet_id: str = ""
    failure: str = ""


@dataclass(frozen=True)
class DispatchSummary:
    selected_clients: Tuple[int, ...]
    dispatched_clients: Tuple[int, ...]
    busy_skipped_clients: Tuple[int, ...]
    offline_clients: Tuple[int, ...]


class SemiAsyncRoundCoordinator:
    """Server-owned task lineage, lifecycle and no-backlog dispatch."""

    def __init__(
        self,
        *,
        trace: RuntimeTrace,
        proxy_version: str,
        local_epochs: int,
        learning_rate: float,
        distillation_temperature: float,
        enable_client_distillation: bool,
    ) -> None:
        self.trace = trace
        self.proxy_version = str(proxy_version)
        self.local_epochs = int(local_epochs)
        self.learning_rate = float(learning_rate)
        self.distillation_temperature = float(distillation_temperature)
        self.enable_client_distillation = bool(
            enable_client_distillation
        )
        self.task_registry: Dict[str, TaskRegistryEntry] = {}
        self._pending_by_client: Dict[int, ClientTask] = {}
        self._active_task_by_client: Dict[int, str] = {}
        self._client_state = {
            client_id: CLIENT_IDLE
            for client_id in range(int(trace.num_clients))
        }
        self._client_pid: Dict[int, int] = {}
        self._seen_packet_ids: Dict[str, Tuple[str, str, float]] = {}
        self._rollback_by_client: Dict[int, str] = {}
        self._shutdown = False
        self._lock = threading.RLock()

    @property
    def client_pids(self) -> Dict[int, int]:
        with self._lock:
            return dict(self._client_pid)

    @property
    def client_states(self) -> Dict[int, str]:
        with self._lock:
            return dict(self._client_state)

    def snapshot_state(self) -> dict:
        """Capture task lineage for an atomic process-runtime checkpoint."""

        with self._lock:
            return {
                "proxy_version": self.proxy_version,
                "client_state": dict(self._client_state),
                "client_pid": dict(self._client_pid),
                "task_registry": {
                    task_id: {
                        "task": entry.task,
                        "status": entry.status,
                        "received_at_s": float(entry.received_at_s),
                        "consumed_at_s": float(entry.consumed_at_s),
                        "packet_id": entry.packet_id,
                        "failure": entry.failure,
                    }
                    for task_id, entry in self.task_registry.items()
                },
                "pending_by_client": dict(self._pending_by_client),
                "active_task_by_client": dict(self._active_task_by_client),
                "seen_packet_ids": dict(self._seen_packet_ids),
                "rollback_by_client": dict(self._rollback_by_client),
                "shutdown": bool(self._shutdown),
            }

    def restore_state(self, state: dict) -> None:
        with self._lock:
            if str(state.get("proxy_version")) != self.proxy_version:
                raise ValueError("Coordinator proxy_version mismatch.")
            for key in (
                "client_state",
                "client_pid",
                "task_registry",
                "pending_by_client",
                "active_task_by_client",
                "seen_packet_ids",
                "rollback_by_client",
            ):
                if key not in state:
                    raise ValueError(f"Coordinator snapshot missing {key}.")
            self._client_state = {
                int(key): str(value)
                for key, value in state["client_state"].items()
            }
            self._client_pid = {
                int(key): int(value)
                for key, value in state["client_pid"].items()
            }
            self.task_registry = {}
            for task_id, raw in state["task_registry"].items():
                if not isinstance(raw, dict) or not isinstance(
                    raw.get("task"), ClientTask
                ):
                    raise ValueError("Coordinator task snapshot is invalid.")
                self.task_registry[str(task_id)] = TaskRegistryEntry(
                    task=raw["task"],
                    status=str(raw["status"]),
                    received_at_s=float(raw["received_at_s"]),
                    consumed_at_s=float(raw["consumed_at_s"]),
                    packet_id=str(raw["packet_id"]),
                    failure=str(raw["failure"]),
                )
            self._pending_by_client = {
                int(key): value
                for key, value in state["pending_by_client"].items()
            }
            self._active_task_by_client = {
                int(key): str(value)
                for key, value in state["active_task_by_client"].items()
            }
            self._seen_packet_ids = dict(state["seen_packet_ids"])
            self._rollback_by_client = {
                int(key): str(value)
                for key, value in state["rollback_by_client"].items()
            }
            self._shutdown = bool(state.get("shutdown", False))

    def register_client(self, client_id: int, pid: int) -> None:
        with self._lock:
            client_id = int(client_id)
            if client_id not in self._client_state:
                raise ValueError(f"Unknown client_id={client_id}.")
            previous = self._client_pid.get(client_id)
            if previous is not None and previous != int(pid):
                raise ValueError(
                    f"Client {client_id} attempted PID replacement."
                )
            self._client_pid[client_id] = int(pid)

    def all_clients_registered(self) -> bool:
        with self._lock:
            return len(self._client_pid) == int(self.trace.num_clients)

    def dispatch_round(
        self,
        *,
        server_round: int,
        latest_server_packet: Optional[ServerLogitsPacket],
    ) -> DispatchSummary:
        selected: List[int] = []
        dispatched: List[int] = []
        busy: List[int] = []
        offline: List[int] = []
        with self._lock:
            for client_id in range(int(self.trace.num_clients)):
                event = self.trace.event(client_id, int(server_round))
                if not event.selected:
                    continue
                selected.append(client_id)
                state = self._client_state[client_id]
                if state == CLIENT_BUSY:
                    busy.append(client_id)
                    continue
                if state in {CLIENT_FAILED, CLIENT_STOPPED}:
                    busy.append(client_id)
                    continue
                if not event.available:
                    self._client_state[client_id] = CLIENT_OFFLINE
                    offline.append(client_id)
                    continue
                if state == CLIENT_OFFLINE:
                    self._client_state[client_id] = CLIENT_IDLE
                if client_id in self._pending_by_client:
                    raise RuntimeError(
                        f"Client {client_id} already has a pending task."
                    )
                base_round = (
                    int(latest_server_packet.model_round)
                    if (
                        self.enable_client_distillation
                        and latest_server_packet is not None
                    )
                    else int(server_round)
                )
                task_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"fedagg:{self.trace.seed}:{client_id}:"
                        f"{server_round}:{self.proxy_version}"
                    ),
                ).hex
                task = ClientTask(
                    task_id=task_id,
                    client_id=client_id,
                    source_round=int(server_round),
                    base_server_round=base_round,
                    proxy_version=self.proxy_version,
                    dispatched_at_s=time.monotonic(),
                    local_epochs=self.local_epochs,
                    learning_rate=self.learning_rate,
                    distillation_temperature=(
                        self.distillation_temperature
                    ),
                    enable_client_distillation=(
                        self.enable_client_distillation
                    ),
                    compute_slowdown_factor=float(
                        event.compute_slowdown_factor
                    ),
                    upload_delay_s=float(event.upload_delay_s),
                    dropped_attempts=tuple(event.dropped_attempts),
                    server_logits_packet=(
                        latest_server_packet
                        if self.enable_client_distillation
                        else None
                    ),
                )
                self.task_registry[task_id] = TaskRegistryEntry(
                    task=task,
                    status=TASK_DISPATCHED,
                )
                self._pending_by_client[client_id] = task
                self._active_task_by_client[client_id] = task_id
                self._client_state[client_id] = CLIENT_BUSY
                dispatched.append(client_id)
        return DispatchSummary(
            selected_clients=tuple(selected),
            dispatched_clients=tuple(dispatched),
            busy_skipped_clients=tuple(busy),
            offline_clients=tuple(offline),
        )

    def get_task(
        self,
        *,
        client_id: int,
        pid: int,
    ) -> Tuple[str, Optional[ClientTask]]:
        with self._lock:
            self.register_client(client_id, pid)
            if self._shutdown:
                self._client_state[int(client_id)] = CLIENT_STOPPED
                return "STOP", None
            rollback_task_id = self._rollback_by_client.get(int(client_id))
            if rollback_task_id is not None:
                return "ROLLBACK", self.task_registry[rollback_task_id].task
            task = self._pending_by_client.pop(int(client_id), None)
            if task is None:
                return "NO_TASK", None
            entry = self.task_registry[task.task_id]
            if entry.status != TASK_DISPATCHED:
                raise RuntimeError(
                    f"Task {task.task_id} is not dispatchable."
                )
            entry.status = TASK_IN_FLIGHT
            return "TASK", task

    def request_client_rollback(self, *, task_id: str, reason: str) -> None:
        """Schedule an explicit model restore for the task's owning client."""

        with self._lock:
            entry = self.task_registry[str(task_id)]
            if entry.status not in {TASK_KNOWLEDGE_RECEIVED, TASK_RESERVED}:
                raise RuntimeError(
                    f"Cannot roll back task in state {entry.status}."
                )
            client_id = int(entry.task.client_id)
            entry.failure = str(reason)
            self._rollback_by_client[client_id] = str(task_id)

    def acknowledge_client_rollback(
        self,
        *,
        client_id: int,
        pid: int,
        task_id: str,
    ) -> None:
        with self._lock:
            self.register_client(client_id, pid)
            expected = self._rollback_by_client.get(int(client_id))
            if expected != str(task_id):
                raise RuntimeError("Rollback acknowledgement does not match task.")
            entry = self.task_registry[str(task_id)]
            entry.status = TASK_FAILED
            entry.failure = entry.failure or "round_transaction_rollback"
            self._rollback_by_client.pop(int(client_id), None)
            self._active_task_by_client.pop(int(client_id), None)
            if self._client_state[int(client_id)] != CLIENT_FAILED:
                self._client_state[int(client_id)] = CLIENT_IDLE

    def validate_and_accept(
        self,
        packet: ClientLogitsPacket,
        *,
        received_at_s: float,
    ) -> str:
        with self._lock:
            duplicate_identity = self._seen_packet_ids.get(
                packet.packet_id
            )
            if duplicate_identity is not None:
                expected_identity = (
                    packet.task_id,
                    packet.payload_sha256,
                    float(packet.generated_at_s),
                )
                if duplicate_identity != expected_identity:
                    raise ValueError(
                        "Duplicate packet_id changed packet identity."
                    )
                return "duplicate"
            entry = self.task_registry.get(packet.task_id)
            if entry is None:
                raise ValueError("Packet references an unknown task_id.")
            task = entry.task
            checks = {
                "client_id": (
                    int(packet.client_id),
                    int(task.client_id),
                ),
                "source_round": (
                    int(packet.source_round),
                    int(task.source_round),
                ),
                "base_server_round": (
                    int(packet.base_server_round),
                    int(task.base_server_round),
                ),
                "proxy_version": (
                    str(packet.proxy_version),
                    str(task.proxy_version),
                ),
            }
            mismatches = [
                key for key, (actual, expected) in checks.items()
                if actual != expected
            ]
            if mismatches:
                raise ValueError(
                    "Packet lineage mismatch: "
                    + ", ".join(mismatches)
                )
            if entry.status not in {
                TASK_IN_FLIGHT,
                TASK_KNOWLEDGE_RECEIVED,
                TASK_RESERVED,
            }:
                raise ValueError(
                    f"Task status does not accept knowledge: {entry.status}."
                )
            self._seen_packet_ids[packet.packet_id] = (
                packet.task_id,
                packet.payload_sha256,
                float(packet.generated_at_s),
            )
            entry.status = TASK_KNOWLEDGE_RECEIVED
            entry.received_at_s = float(received_at_s)
            entry.packet_id = packet.packet_id
            return "accepted"

    def reserve_consumed(self, *, task_id: str) -> None:
        """Reserve a packet for a round; commit happens after all work succeeds."""

        with self._lock:
            entry = self.task_registry[str(task_id)]
            if entry.status != TASK_KNOWLEDGE_RECEIVED:
                raise RuntimeError(
                    f"Cannot reserve task in state {entry.status}."
                )
            entry.status = TASK_RESERVED

    def commit_consumed(
        self,
        *,
        task_id: str,
        consumed_at_s: float,
    ) -> None:
        with self._lock:
            entry = self.task_registry[str(task_id)]
            if entry.status != TASK_RESERVED:
                raise RuntimeError(
                    f"Cannot commit task in state {entry.status}."
                )
            entry.status = TASK_CONSUMED
            entry.consumed_at_s = float(consumed_at_s)
            client_id = int(entry.task.client_id)
            self._active_task_by_client.pop(client_id, None)
            if self._client_state[client_id] != CLIENT_FAILED:
                self._client_state[client_id] = CLIENT_IDLE

    def abort_consumed(self, *, task_id: str, reason: str = "") -> None:
        with self._lock:
            entry = self.task_registry[str(task_id)]
            if entry.status != TASK_RESERVED:
                raise RuntimeError(
                    f"Cannot abort task in state {entry.status}."
                )
            entry.status = TASK_KNOWLEDGE_RECEIVED
            entry.failure = str(reason)

    def mark_consumed(
        self,
        *,
        task_id: str,
        consumed_at_s: float,
    ) -> None:
        """Backward-compatible atomic helper; production uses reserve/commit."""

        self.reserve_consumed(task_id=task_id)
        self.commit_consumed(task_id=task_id, consumed_at_s=consumed_at_s)

    def mark_failed(self, task_id: str, reason: str) -> None:
        with self._lock:
            entry = self.task_registry.get(str(task_id))
            if entry is None:
                raise ValueError("Failure references an unknown task.")
            entry.status = TASK_FAILED
            entry.failure = str(reason)
            client_id = int(entry.task.client_id)
            self._client_state[client_id] = CLIENT_FAILED
            self._pending_by_client.pop(client_id, None)

    def expire_unfinished(self) -> None:
        with self._lock:
            for entry in self.task_registry.values():
                if entry.status in {
                    TASK_DISPATCHED,
                    TASK_IN_FLIGHT,
                    TASK_KNOWLEDGE_RECEIVED,
                    TASK_RESERVED,
                }:
                    entry.status = TASK_EXPIRED

    def request_shutdown(self) -> None:
        with self._lock:
            self._shutdown = True

    def ack_delay(self, packet: ClientLogitsPacket, attempt: int) -> float:
        event = self.trace.event(
            int(packet.client_id),
            int(packet.source_round),
        )
        return event.ack_delay(int(attempt))
