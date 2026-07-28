from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import multiprocessing
import os
import queue
import socket
import statistics
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from attacks import (
    AttackPlan,
    BackdoorBatchPoisoner,
    evaluate_backdoor_suite,
    split_defense_diagnostics,
)
from admission import (
    AdmissionDecision,
    TeacherAdmissionController,
    TeacherKnowledge,
    TeacherMetadata,
)
from data_utils import (
    FederatedDataPlan,
    build_client_dataloaders_from_plan,
    cleanup_dataloaders,
)
from defense import DefenseResult, KnowledgeDefenseController
from federated_runtime import (
    _decision_metrics,
    _defense_metrics,
    _freeze_state_dict,
    _model_is_finite,
    _restore_model,
    _validate_decision,
    evaluate_with_loss,
)
from federated_server import FederatedServer
from logits_transport import ClientLogitsPacket, ServerLogitsPacket
from model_factory import build_model
from round_coordinator import (
    CLIENT_BUSY,
    CLIENT_FAILED,
    SemiAsyncRoundCoordinator,
)
from rpc_transport import (
    DEFAULT_MAX_MESSAGE_BYTES,
    RpcFrame,
    RpcProtocolError,
    RpcResponse,
    RpcServer,
    rpc_call,
)
from runtime_trace import RuntimeTrace
from trainer import distill_with_logits, local_train, predict_logits


@dataclass(frozen=True)
class ProcessRuntimeConfig:
    participation_rate: float = 1.0
    quorum_fraction: float = 0.5
    warmup_rounds: int = 1
    soft_deadline_factor: float = 1.5
    hard_deadline_factor: float = 2.0
    soft_deadline_override_s: float = 0.0
    hard_deadline_override_s: float = 0.0
    rpc_timeout_s: float = 1.0
    max_retries: int = 3
    retry_backoff_s: float = 0.05
    poll_interval_s: float = 0.02
    registration_timeout_s: float = 60.0
    shutdown_timeout_s: float = 60.0
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    server_device: str = "cpu"
    client_device: str = "cpu"
    amp: bool = False
    strict_numeric_checks: bool = False
    client_num_workers: int = 0
    client_torch_threads: int = 1
    client_pin_memory: bool = False
    loader_mp_context: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 < float(self.participation_rate) <= 1.0:
            raise ValueError("participation_rate must be in (0, 1].")
        if not 0.0 < float(self.quorum_fraction) <= 1.0:
            raise ValueError("quorum_fraction must be in (0, 1].")
        if int(self.warmup_rounds) < 1:
            raise ValueError("warmup_rounds must be at least 1.")
        if float(self.soft_deadline_factor) <= 0.0:
            raise ValueError("soft_deadline_factor must be positive.")
        if (
            float(self.hard_deadline_factor)
            <= float(self.soft_deadline_factor)
        ):
            raise ValueError(
                "hard_deadline_factor must exceed soft_deadline_factor."
            )
        if float(self.soft_deadline_override_s) < 0.0:
            raise ValueError("soft deadline override cannot be negative.")
        if float(self.hard_deadline_override_s) < 0.0:
            raise ValueError("hard deadline override cannot be negative.")
        if (
            float(self.soft_deadline_override_s) > 0.0
        ) != (
            float(self.hard_deadline_override_s) > 0.0
        ):
            raise ValueError(
                "Absolute soft and hard deadlines must be set together."
            )
        if (
            float(self.soft_deadline_override_s) > 0.0
            and float(self.hard_deadline_override_s)
            <= float(self.soft_deadline_override_s)
        ):
            raise ValueError(
                "hard deadline override must exceed soft override."
            )
        if float(self.rpc_timeout_s) <= 0.0:
            raise ValueError("rpc_timeout_s must be positive.")
        if int(self.max_retries) < 0:
            raise ValueError("max_retries cannot be negative.")
        if float(self.retry_backoff_s) < 0.0:
            raise ValueError("retry_backoff_s cannot be negative.")
        if float(self.poll_interval_s) <= 0.0:
            raise ValueError("poll_interval_s must be positive.")
        if int(self.max_message_bytes) <= 0:
            raise ValueError("max_message_bytes must be positive.")
        if int(self.client_torch_threads) <= 0:
            raise ValueError("client_torch_threads must be positive.")


@dataclass(frozen=True)
class _MailboxItem:
    packet: ClientLogitsPacket
    received_at_s: float


class _ProcessRpcService:
    def __init__(
        self,
        *,
        coordinator: SemiAsyncRoundCoordinator,
        expected_proxy_samples: int,
        expected_num_classes: int,
    ) -> None:
        self.coordinator = coordinator
        self.expected_proxy_samples = int(expected_proxy_samples)
        self.expected_num_classes = int(expected_num_classes)
        self.mailbox: queue.Queue[_MailboxItem] = queue.Queue()
        self._event_state: Dict[str, Dict[str, object]] = {}
        self._transport_errors: List[str] = []
        self._current_server_round = 0
        self._lock = threading.RLock()

    def set_server_round(self, server_round: int) -> None:
        with self._lock:
            self._current_server_round = int(server_round)

    def handle_rpc(self, request: RpcFrame) -> RpcResponse:
        if request.message_type == "GET_TASK":
            return self._handle_get_task(request)
        if request.message_type == "UPLOAD_KNOWLEDGE":
            return self._handle_upload(request)
        if request.message_type == "TASK_FAILURE":
            return self._handle_task_failure(request)
        raise RpcProtocolError(
            f"Unsupported RPC method={request.message_type!r}."
        )

    def _handle_get_task(self, request: RpcFrame) -> RpcResponse:
        if request.payload:
            raise RpcProtocolError("GET_TASK cannot contain a payload.")
        client_id = int(request.metadata["client_id"])
        pid = int(request.metadata["pid"])
        status, task = self.coordinator.get_task(
            client_id=client_id,
            pid=pid,
        )
        if status != "TASK":
            return RpcResponse(status, {"client_id": client_id})
        if task is None:
            raise RuntimeError("TASK response is missing a task.")
        packet = task.server_logits_packet
        return RpcResponse(
            "TASK",
            task.rpc_metadata(),
            b"" if packet is None else packet.logits_payload,
        )

    def _handle_upload(self, request: RpcFrame) -> RpcResponse:
        raw_packet = request.metadata.get("packet")
        raw_attempt = request.metadata.get("attempt")
        if not isinstance(raw_packet, dict):
            raise RpcProtocolError(
                "UPLOAD_KNOWLEDGE packet metadata is missing."
            )
        if not isinstance(raw_attempt, dict):
            raise RpcProtocolError(
                "UPLOAD_KNOWLEDGE attempt metadata is missing."
            )
        packet = ClientLogitsPacket.from_rpc_parts(
            raw_packet,
            request.payload,
        )
        expected_shape = (
            self.expected_proxy_samples,
            self.expected_num_classes,
        )
        if tuple(packet.logits_shape) != expected_shape:
            raise ValueError(
                "Client logits shape does not match the active proxy."
            )
        packet.decode_logits()
        received_at_s = time.monotonic()
        status = self.coordinator.validate_and_accept(
            packet,
            received_at_s=received_at_s,
        )
        attempt_index = int(raw_attempt["attempt_index"])
        with self._lock:
            state = self._event_state.setdefault(
                packet.packet_id,
                {
                    "packet": packet,
                    "received_at_s": received_at_s,
                    "receive_server_round": int(
                        self._current_server_round
                    ),
                    "first_upload_attempt_at_s": float(
                        raw_attempt["first_upload_attempt_at_s"]
                    ),
                    "injected_upload_delay_s": float(
                        raw_attempt["injected_upload_delay_s"]
                    ),
                    "upload_attempts": attempt_index,
                    "upload_attempt_drop_count": int(
                        raw_attempt["upload_attempt_drop_count"]
                    ),
                    "rpc_timeout_count": int(
                        raw_attempt["rpc_timeout_count"]
                    ),
                    "retry_count": int(raw_attempt["retry_count"]),
                    "duplicate_receive_count": 0,
                    "request_wire_bytes": 0,
                    "response_wire_bytes": 0,
                    "rpc_elapsed_s": float(
                        raw_attempt.get("rpc_elapsed_s", 0.0)
                    ),
                    "transport_status": "accepted",
                },
            )
            state["upload_attempts"] = max(
                int(state["upload_attempts"]),
                attempt_index,
            )
            state["upload_attempt_drop_count"] = max(
                int(state["upload_attempt_drop_count"]),
                int(raw_attempt["upload_attempt_drop_count"]),
            )
            state["rpc_timeout_count"] = max(
                int(state["rpc_timeout_count"]),
                int(raw_attempt["rpc_timeout_count"]),
            )
            state["retry_count"] = max(
                int(state["retry_count"]),
                int(raw_attempt["retry_count"]),
            )
            state["request_wire_bytes"] = int(
                state["request_wire_bytes"]
            ) + int(request.wire_bytes)
            if status == "duplicate":
                state["duplicate_receive_count"] = int(
                    state["duplicate_receive_count"]
                ) + 1
                state["transport_status"] = "duplicate"
        if status == "accepted":
            self.mailbox.put(_MailboxItem(
                packet=packet,
                received_at_s=received_at_s,
            ))
        return RpcResponse(
            "ACK",
            {
                "packet_id": packet.packet_id,
                "status": status,
            },
            delay_s=self.coordinator.ack_delay(
                packet,
                attempt_index,
            ),
            packet_id=packet.packet_id,
        )

    def _handle_task_failure(self, request: RpcFrame) -> RpcResponse:
        if request.payload:
            raise RpcProtocolError("TASK_FAILURE cannot contain payload.")
        task_id = str(request.metadata["task_id"])
        reason = str(request.metadata["reason"])
        self.coordinator.mark_failed(task_id, reason)
        return RpcResponse(
            "ACK",
            {"task_id": task_id, "status": "failed-recorded"},
        )

    def record_response_wire(
        self,
        packet_id: str,
        response_wire_bytes: int,
    ) -> None:
        if not packet_id:
            return
        with self._lock:
            state = self._event_state.get(str(packet_id))
            if state is not None:
                state["response_wire_bytes"] = int(
                    state["response_wire_bytes"]
                ) + int(response_wire_bytes)

    def record_transport_error(self, message: str) -> None:
        with self._lock:
            self._transport_errors.append(str(message))

    def drain_mailbox(self) -> List[_MailboxItem]:
        items: List[_MailboxItem] = []
        while True:
            try:
                items.append(self.mailbox.get_nowait())
            except queue.Empty:
                return items

    def mailbox_size(self) -> int:
        return int(self.mailbox.qsize())

    def event_state(self, packet_id: str) -> Dict[str, object]:
        with self._lock:
            return dict(self._event_state[str(packet_id)])

    @property
    def all_event_states(self) -> List[Dict[str, object]]:
        with self._lock:
            return [
                dict(value) for value in self._event_state.values()
            ]

    @property
    def transport_errors(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._transport_errors)


def _restore_client_model(model, snapshot: Dict[str, object], device) -> None:
    model.load_state_dict(snapshot, strict=True)
    model.to(device)


def _client_task_from_response(
    response: RpcFrame,
) -> Tuple[Dict[str, object], Optional[ServerLogitsPacket]]:
    metadata = dict(response.metadata)
    server_metadata = metadata.get("server_logits")
    if server_metadata is None:
        if response.payload:
            raise RpcProtocolError(
                "Task has server payload without server metadata."
            )
        return metadata, None
    if not isinstance(server_metadata, dict):
        raise RpcProtocolError("server_logits metadata is invalid.")
    packet = ServerLogitsPacket.from_rpc_parts(
        server_metadata,
        response.payload,
    )
    return metadata, packet


def _send_task_failure(
    host: str,
    port: int,
    *,
    client_id: int,
    task_id: str,
    reason: str,
    timeout_s: float,
    max_message_bytes: int,
) -> None:
    rpc_call(
        host,
        port,
        message_type="TASK_FAILURE",
        metadata={
            "client_id": int(client_id),
            "task_id": str(task_id),
            "reason": str(reason),
        },
        timeout_s=float(timeout_s),
        max_message_bytes=int(max_message_bytes),
    )


def _client_process_main(
    *,
    client_id: int,
    host: str,
    port: int,
    data_plan: FederatedDataPlan,
    dataset_name: str,
    config: ProcessRuntimeConfig,
    seed: int,
    error_queue,
    attack_plan: Optional[AttackPlan] = None,
    attack_stats_queue=None,
) -> None:
    """Persistent spawned Client owning model, private data and retries."""

    torch.set_num_threads(int(config.client_torch_threads))
    torch.manual_seed(int(seed) + 10000 + int(client_id))
    if str(config.client_device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "client_device requests CUDA but CUDA is unavailable."
            )
        torch.cuda.set_device(torch.device(config.client_device))
    device = torch.device(config.client_device)
    dataloaders = None
    active_task_id = ""
    try:
        private_loader, proxy_input_loader = (
            build_client_dataloaders_from_plan(
                data_plan,
                client_id=int(client_id),
                num_workers=int(config.client_num_workers),
                pin_memory=bool(config.client_pin_memory),
                loader_mp_context=config.loader_mp_context,
            )
        )
        dataloaders = {
            "private": private_loader,
            "proxy_input": proxy_input_loader,
        }
        model = build_model(
            "resnet18",
            dataset_name=dataset_name,
            device=device,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        local_model_version = 0
        poisoner = (
            BackdoorBatchPoisoner(plan=attack_plan, client_id=int(client_id))
            if attack_plan is not None and attack_plan.is_malicious(client_id)
            else None
        )

        while True:
            response, _, _ = rpc_call(
                host,
                port,
                message_type="GET_TASK",
                metadata={
                    "client_id": int(client_id),
                    "pid": int(os.getpid()),
                },
                timeout_s=float(config.rpc_timeout_s),
                max_message_bytes=int(config.max_message_bytes),
            )
            if response.message_type == "STOP":
                break
            if response.message_type == "NO_TASK":
                time.sleep(float(config.poll_interval_s))
                continue
            if response.message_type != "TASK":
                raise RpcProtocolError(
                    f"Unexpected GET_TASK response={response.message_type}."
                )
            task, server_packet = _client_task_from_response(response)
            active_task_id = str(task["task_id"])
            snapshot = {
                key: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
                for key, value in model.state_dict().items()
            }
            compute_started_at_s = time.monotonic()
            try:
                if bool(task["enable_client_distillation"]):
                    if server_packet is None:
                        raise RuntimeError(
                            "Reverse distillation task lacks server logits."
                        )
                    if server_packet.proxy_version != str(
                        task["proxy_version"]
                    ):
                        raise ValueError(
                            "Server logits proxy_version mismatch."
                        )
                    distill_with_logits(
                        model,
                        proxy_input_loader,
                        server_packet.decode_logits(),
                        device=device,
                        lr=max(float(task["learning_rate"]) * 0.2, 1e-4),
                        epochs=1,
                        temperature=float(
                            task["distillation_temperature"]
                        ),
                        amp=bool(config.amp),
                        strict_numeric_checks=bool(
                            config.strict_numeric_checks
                        ),
                    )
                source_round = int(task["source_round"])
                if poisoner is not None:
                    poisoner.start_round(source_round)
                local_train(
                    model,
                    private_loader,
                    device=device,
                    lr=float(task["learning_rate"]),
                    epochs=int(task["local_epochs"]),
                    amp=bool(config.amp),
                    strict_numeric_checks=bool(
                        config.strict_numeric_checks
                    ),
                    optimizer=optimizer,
                    batch_transform=poisoner,
                    round_number=source_round,
                )
                if attack_stats_queue is not None:
                    stats = poisoner.round_stats if poisoner is not None else None
                    attack_stats_queue.put({
                        "task_id": active_task_id,
                        "client_id": int(client_id),
                        "source_round": source_round,
                        "is_malicious": bool(
                            attack_plan is not None
                            and attack_plan.is_malicious(client_id)
                        ),
                        "attack_active": bool(
                            attack_plan is not None
                            and attack_plan.active_for(client_id, source_round)
                        ),
                        "poisoned_samples": int(
                            stats.poisoned if stats is not None else 0
                        ),
                        "eligible_poison_samples": int(
                            stats.eligible if stats is not None else 0
                        ),
                        "poisoned_batches": int(
                            stats.poisoned_batches if stats is not None else 0
                        ),
                        "dba_trigger_part": (
                            int(attack_plan.dba_part(client_id))
                            if attack_plan is not None
                            and attack_plan.config.attack_type == "dba"
                            and attack_plan.is_malicious(client_id)
                            else -1
                        ),
                    })
                if not _model_is_finite(model):
                    raise RuntimeError(
                        "Client model became non-finite after local update."
                )
                local_model_version += 1
                inference_started = time.monotonic()
                logits = predict_logits(
                    model,
                    proxy_input_loader,
                    device=device,
                    amp=bool(config.amp),
                )
                generated_at_s = time.monotonic()
                proxy_inference_time_s = (
                    generated_at_s - inference_started
                )
                if not bool(torch.isfinite(logits).all().item()):
                    raise RuntimeError(
                        "Client proxy inference produced non-finite logits."
                    )
            except Exception:
                _restore_client_model(model, snapshot, device)
                raise

            actual_compute_time_s = (
                generated_at_s - compute_started_at_s
            )
            factor = float(task["compute_slowdown_factor"])
            injected_compute_delay_s = max(
                0.0,
                actual_compute_time_s * factor - actual_compute_time_s,
            )
            if injected_compute_delay_s > 0.0:
                time.sleep(injected_compute_delay_s)
            compute_finished_at_s = generated_at_s
            total_compute_phase_s = (
                time.monotonic() - compute_started_at_s
            )
            packet = ClientLogitsPacket.from_logits(
                client_id=int(client_id),
                model_round=int(task["source_round"]),
                generated_at_s=generated_at_s,
                query_id=str(task["proxy_version"]),
                logits=logits,
                task_id=active_task_id,
                source_round=int(task["source_round"]),
                base_server_round=int(task["base_server_round"]),
                local_model_version=local_model_version,
                proxy_version=str(task["proxy_version"]),
                dispatched_at_s=float(task["dispatched_at_s"]),
                compute_started_at_s=compute_started_at_s,
                compute_finished_at_s=compute_finished_at_s,
                actual_compute_time_s=actual_compute_time_s,
                injected_compute_delay_s=injected_compute_delay_s,
                total_compute_phase_s=total_compute_phase_s,
                proxy_inference_time_s=proxy_inference_time_s,
                client_pid=os.getpid(),
                local_train_count=1,
                predict_logits_calls=1,
            )
            upload_delay_s = float(task["upload_delay_s"])
            if upload_delay_s > 0.0:
                time.sleep(upload_delay_s)
            first_upload_attempt_at_s = time.monotonic()
            drop_count = 0
            timeout_count = 0
            rpc_elapsed_total = 0.0
            acknowledged = False
            total_attempts = 1 + int(config.max_retries)
            dropped_attempts = {
                int(attempt)
                for attempt in task["dropped_attempts"]
            }
            for attempt_index in range(1, total_attempts + 1):
                if attempt_index in dropped_attempts:
                    drop_count += 1
                    if attempt_index < total_attempts:
                        time.sleep(float(config.retry_backoff_s))
                    continue
                attempt_metadata = {
                    "attempt_index": attempt_index,
                    "first_upload_attempt_at_s": (
                        first_upload_attempt_at_s
                    ),
                    "injected_upload_delay_s": upload_delay_s,
                    "upload_attempt_drop_count": drop_count,
                    "rpc_timeout_count": timeout_count,
                    "retry_count": attempt_index - 1,
                    "rpc_elapsed_s": rpc_elapsed_total,
                }
                rpc_started = time.monotonic()
                try:
                    upload_response, _, rpc_elapsed = rpc_call(
                        host,
                        port,
                        message_type="UPLOAD_KNOWLEDGE",
                        metadata={
                            "packet": packet.rpc_metadata(),
                            "attempt": attempt_metadata,
                        },
                        payload=packet.logits_payload,
                        timeout_s=float(config.rpc_timeout_s),
                        max_message_bytes=int(
                            config.max_message_bytes
                        ),
                    )
                    rpc_elapsed_total += float(rpc_elapsed)
                except (socket.timeout, TimeoutError):
                    rpc_elapsed_total += time.monotonic() - rpc_started
                    timeout_count += 1
                    if attempt_index < total_attempts:
                        time.sleep(float(config.retry_backoff_s))
                    continue
                if upload_response.message_type != "ACK":
                    raise RpcProtocolError(
                        "UPLOAD_KNOWLEDGE did not return ACK."
                    )
                if (
                    str(upload_response.metadata["packet_id"])
                    != packet.packet_id
                ):
                    raise RpcProtocolError("ACK packet_id mismatch.")
                status = str(upload_response.metadata["status"])
                if status not in {"accepted", "duplicate"}:
                    raise RpcProtocolError(
                        f"Unexpected ACK status={status!r}."
                    )
                acknowledged = True
                break
            if not acknowledged:
                raise RuntimeError(
                    f"Client {client_id} exhausted upload retries for "
                    f"packet {packet.packet_id}."
                )
            active_task_id = ""
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if active_task_id:
            try:
                _send_task_failure(
                    host,
                    port,
                    client_id=int(client_id),
                    task_id=active_task_id,
                    reason=reason,
                    timeout_s=float(config.rpc_timeout_s),
                    max_message_bytes=int(config.max_message_bytes),
                )
            except (OSError, RpcProtocolError, TimeoutError) as report_exc:
                reason += (
                    f"; failure-report-error={type(report_exc).__name__}: "
                    f"{report_exc}"
                )
        error_queue.put({
            "client_id": int(client_id),
            "pid": int(os.getpid()),
            "error": reason,
        })
        raise
    finally:
        cleanup_dataloaders(dataloaders)


def _wait_for_clients(
    *,
    coordinator: SemiAsyncRoundCoordinator,
    processes: Sequence[multiprocessing.Process],
    error_queue,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while not coordinator.all_clients_registered():
        _raise_child_error(error_queue, processes)
        if time.monotonic() >= deadline:
            raise TimeoutError("Client process registration timed out.")
        time.sleep(0.02)


def _raise_child_error(error_queue, processes) -> None:
    try:
        failure = error_queue.get_nowait()
    except queue.Empty:
        failure = None
    if failure is not None:
        raise RuntimeError(
            "Client process failure: "
            f"client_id={failure['client_id']} pid={failure['pid']} "
            f"error={failure['error']}"
        )
    for process in processes:
        if process.exitcode not in {None, 0}:
            raise RuntimeError(
                f"Client process pid={process.pid} exited with "
                f"code={process.exitcode}."
            )


def _deadline_values(
    config: ProcessRuntimeConfig,
    reference_time_s: Optional[float],
) -> Tuple[float, float]:
    if float(config.soft_deadline_override_s) > 0.0:
        return (
            float(config.soft_deadline_override_s),
            float(config.hard_deadline_override_s),
        )
    reference = (
        max(float(reference_time_s), 1e-3)
        if reference_time_s is not None
        else max(1.0, float(config.rpc_timeout_s) * 4.0)
    )
    return (
        float(config.soft_deadline_factor) * reference,
        float(config.hard_deadline_factor) * reference,
    )


def _wait_collection_window(
    *,
    service: _ProcessRpcService,
    coordinator: SemiAsyncRoundCoordinator,
    processes: Sequence[multiprocessing.Process],
    error_queue,
    dispatched_count: int,
    quorum_required: int,
    soft_deadline_s: float,
    hard_deadline_s: float,
    warmup: bool,
) -> bool:
    started = time.monotonic()
    soft_at = started + float(soft_deadline_s)
    hard_at = started + float(hard_deadline_s)
    while True:
        _raise_child_error(error_queue, processes)
        available = service.mailbox_size()
        if warmup and available >= int(dispatched_count):
            return available >= int(quorum_required)
        now = time.monotonic()
        if now >= soft_at and available >= int(quorum_required):
            return True
        if now >= hard_at:
            return available >= int(quorum_required)
        time.sleep(0.01)


def _nan() -> float:
    return float("nan")


def _event_from_item(
    item: _MailboxItem,
    *,
    consumed_round: int,
    consumed_at_s: float,
    state: Dict[str, object],
) -> Dict[str, object]:
    packet = item.packet
    version_lag = int(consumed_round) - int(packet.source_round)
    base_version_lag = (
        int(consumed_round) - int(packet.base_server_round)
    )
    return {
        "client_id": int(packet.client_id),
        "client_pid": int(packet.client_pid),
        "task_id": packet.task_id,
        "packet_id": packet.packet_id,
        "payload_sha256": packet.payload_sha256,
        "inference_sha256": packet.inference_sha256,
        "source_round": int(packet.source_round),
        "base_server_round": int(packet.base_server_round),
        "receive_server_round": int(state["receive_server_round"]),
        "consumed_round": int(consumed_round),
        "local_model_version": int(packet.local_model_version),
        "dispatch_at_s": float(packet.dispatched_at_s),
        "dispatched_at_s": float(packet.dispatched_at_s),
        "compute_started_at_s": float(packet.compute_started_at_s),
        "compute_finished_at_s": float(packet.compute_finished_at_s),
        "generated_at_s": float(packet.generated_at_s),
        "first_upload_attempt_at_s": float(
            state["first_upload_attempt_at_s"]
        ),
        "received_at_s": float(item.received_at_s),
        "consumed_at_s": float(consumed_at_s),
        "actual_compute_time_s": float(
            packet.actual_compute_time_s
        ),
        "proxy_inference_time_s": float(
            packet.proxy_inference_time_s
        ),
        "injected_compute_delay_s": float(
            packet.injected_compute_delay_s
        ),
        "total_compute_phase_s": float(
            packet.total_compute_phase_s
        ),
        "injected_upload_delay_s": float(
            state["injected_upload_delay_s"]
        ),
        "knowledge_age_s": (
            float(consumed_at_s) - float(packet.generated_at_s)
        ),
        "transport_age_s": (
            float(item.received_at_s) - float(packet.generated_at_s)
        ),
        "version_lag": int(version_lag),
        "base_version_lag": int(base_version_lag),
        "upload_attempts": int(state["upload_attempts"]),
        "upload_attempt_drop_count": int(
            state["upload_attempt_drop_count"]
        ),
        "rpc_timeout_count": int(state["rpc_timeout_count"]),
        "retry_count": int(state["retry_count"]),
        "duplicate_receive_count": int(
            state["duplicate_receive_count"]
        ),
        "rpc_elapsed_s": max(
            0.0,
            float(item.received_at_s)
            - float(state["first_upload_attempt_at_s"]),
        ),
        "payload_bytes": int(packet.payload_bytes),
        "wire_bytes": (
            int(state["request_wire_bytes"])
            + int(state["response_wire_bytes"])
        ),
        "logits_dtype": packet.logits_dtype,
        "logits_shape": "x".join(map(str, packet.logits_shape)),
        "proxy_version": packet.proxy_version,
        "local_train_count": int(packet.local_train_count),
        "predict_logits_calls": int(packet.predict_logits_calls),
        "transport_status": str(state["transport_status"]),
        "rpc_accept_status": "accepted",
        "vcaa_version_score": _nan(),
        "vcaa_content_score": _nan(),
        "vcaa_final_score": _nan(),
        "vcaa_threshold": _nan(),
        "proxy_accuracy": _nan(),
        "mean_entropy": _nan(),
        "mean_kl": _nan(),
        "admitted": _nan(),
        "niabd_anomaly_fraction": _nan(),
        "niabd_mean_suppression": _nan(),
    }


def _unconsumed_event_from_state(
    state: Dict[str, object],
) -> Dict[str, object]:
    packet = state["packet"]
    if not isinstance(packet, ClientLogitsPacket):
        raise TypeError("Runtime event state has no ClientLogitsPacket.")
    received_at_s = float(state["received_at_s"])
    event = _event_from_item(
        _MailboxItem(packet=packet, received_at_s=received_at_s),
        consumed_round=int(packet.source_round),
        consumed_at_s=received_at_s,
        state=state,
    )
    event.update({
        "receive_server_round": int(state["receive_server_round"]),
        "consumed_round": _nan(),
        "consumed_at_s": _nan(),
        "knowledge_age_s": _nan(),
        "version_lag": _nan(),
        "base_version_lag": _nan(),
        "vcaa_version_score": _nan(),
        "vcaa_content_score": _nan(),
        "vcaa_final_score": _nan(),
        "vcaa_threshold": _nan(),
        "proxy_accuracy": _nan(),
        "mean_entropy": _nan(),
        "mean_kl": _nan(),
        "admitted": _nan(),
        "niabd_anomaly_fraction": _nan(),
        "niabd_mean_suppression": _nan(),
        "transport_status": "accepted-unconsumed",
    })
    return event


def _drain_attack_stats(attack_stats_queue, cache: Dict[str, Dict[str, object]]) -> None:
    if attack_stats_queue is None:
        return
    while True:
        try:
            record = attack_stats_queue.get_nowait()
        except queue.Empty:
            return
        if not isinstance(record, dict) or "task_id" not in record:
            raise RuntimeError("Client attack stats queue returned an invalid record.")
        cache[str(record["task_id"])] = dict(record)


def _append_metric(metrics: Dict[str, object], key: str, value) -> None:
    values = metrics.setdefault(key, [])
    if not isinstance(values, list):
        raise TypeError(f"Metric {key} is not appendable.")
    values.append(value)


def _initialize_metrics(
    *,
    num_clients: int,
    admission_controller,
    defense_controller,
    config: ProcessRuntimeConfig,
    attack_plan: Optional[AttackPlan] = None,
) -> Dict[str, object]:
    return {
        "topology": "server-client",
        "runtime": "process-semi-async",
        "knowledge_interface": "localhost-tcp-serialized-proxy-logits",
        "aggregation_rule": "mean-soft-probabilities",
        "server_role": "global-student",
        "client_role": "persistent-process-local-teacher",
        "num_clients": int(num_clients),
        "server_device": str(config.server_device),
        "client_device": str(config.client_device),
        "admission_method": (
            admission_controller.name
            if admission_controller is not None
            else "none"
        ),
        "vcaa_enabled": int(admission_controller is not None),
        "defense_method": (
            defense_controller.name
            if defense_controller is not None
            else "none"
        ),
        "niabd_enabled": int(defense_controller is not None),
        "runtime_events": [],
        "teacher_admission_records": [],
        "teacher_defense_records": [],
        "attack_type": (
            attack_plan.config.attack_type if attack_plan is not None else "none"
        ),
        "attack_plan_id": (
            attack_plan.identity if attack_plan is not None else ""
        ),
        "target_label": (
            int(attack_plan.config.target_label) if attack_plan is not None else -1
        ),
        "malicious_client_ids": (
            list(attack_plan.malicious_client_ids) if attack_plan is not None else []
        ),
        "malicious_fraction": (
            float(attack_plan.config.malicious_fraction) if attack_plan is not None else 0.0
        ),
        "poison_ratio": (
            float(attack_plan.config.poison_ratio) if attack_plan is not None else 0.0
        ),
        "attack_start_round": (
            int(attack_plan.config.attack_start_round) if attack_plan is not None else -1
        ),
        "backdoor_client_records": [],
    }


def run_fedagg_server_client_process_async(
    *,
    server_model,
    server_dataloaders: Dict[str, object],
    data_plan: FederatedDataPlan,
    trace: RuntimeTrace,
    config: ProcessRuntimeConfig,
    local_epochs: int,
    rounds: int,
    learning_rate: float,
    distill_temperature: float,
    admission_controller: Optional[TeacherAdmissionController] = None,
    defense_controller: Optional[KnowledgeDefenseController] = None,
    enable_client_distillation: bool = True,
    attack_plan: Optional[AttackPlan] = None,
) -> Dict[str, object]:
    """Run real persistent Client processes through localhost TCP RPC."""

    if int(rounds) != int(trace.rounds):
        raise ValueError("Runtime trace rounds do not match the run.")
    if data_plan.num_clients != int(trace.num_clients):
        raise ValueError("Runtime trace client count does not match data plan.")
    if attack_plan is not None and int(attack_plan.num_clients) != data_plan.num_clients:
        raise ValueError("Attack plan client count does not match data plan.")
    proxy_loader = server_dataloaders.get("proxy")
    test_loader = server_dataloaders.get("test")
    if proxy_loader is None or test_loader is None:
        raise ValueError("Server proxy and test loaders are required.")
    server = FederatedServer(
        model=server_model,
        proxy_loader=proxy_loader,
        device=torch.device(config.server_device),
        amp=bool(config.amp),
        strict_numeric_checks=bool(config.strict_numeric_checks),
    )
    initial_logits = server.student_proxy_logits()
    if initial_logits.ndim != 2:
        raise ValueError("Server proxy logits must be two-dimensional.")
    coordinator = SemiAsyncRoundCoordinator(
        trace=trace,
        proxy_version=data_plan.proxy_version,
        local_epochs=int(local_epochs),
        learning_rate=float(learning_rate),
        distillation_temperature=float(distill_temperature),
        enable_client_distillation=bool(enable_client_distillation),
    )
    service = _ProcessRpcService(
        coordinator=coordinator,
        expected_proxy_samples=int(initial_logits.shape[0]),
        expected_num_classes=int(initial_logits.shape[1]),
    )
    rpc_server = RpcServer(
        service,
        max_message_bytes=int(config.max_message_bytes),
    )
    rpc_server.start()
    host, port = rpc_server.address
    context = torch.multiprocessing.get_context("spawn")
    error_queue = context.Queue()
    attack_stats_queue = context.Queue()
    processes = [
        context.Process(
            target=_client_process_main,
            kwargs={
                "client_id": client_id,
                "host": host,
                "port": port,
                "data_plan": data_plan,
                "dataset_name": data_plan.dataset_name,
                "config": config,
                "seed": int(trace.seed),
                "error_queue": error_queue,
                "attack_plan": attack_plan,
                "attack_stats_queue": attack_stats_queue,
            },
            name=f"fedagg-client-{client_id}",
        )
        for client_id in range(data_plan.num_clients)
    ]
    metrics = _initialize_metrics(
        num_clients=data_plan.num_clients,
        admission_controller=admission_controller,
        defense_controller=defense_controller,
        config=config,
        attack_plan=attack_plan,
    )
    latest_server_packet = ServerLogitsPacket.from_logits(
        model_round=0,
        query_id=data_plan.proxy_version,
        proxy_version=data_plan.proxy_version,
        logits=initial_logits,
    )
    reference_times: List[float] = []
    attack_stats_cache: Dict[str, Dict[str, object]] = {}
    wall_start = time.monotonic()
    try:
        for process in processes:
            process.start()
        _wait_for_clients(
            coordinator=coordinator,
            processes=processes,
            error_queue=error_queue,
            timeout_s=float(config.registration_timeout_s),
        )
        metrics["parent_pid"] = os.getpid()
        metrics["client_pids"] = coordinator.client_pids
        server_attributes = tuple(sorted(vars(server)))
        metrics["server_attribute_names"] = server_attributes
        metrics["server_has_client_model_refs"] = any(
            "client" in name.lower() for name in server_attributes
        )

        for server_round in range(1, int(rounds) + 1):
            service.set_server_round(server_round)
            round_started = time.monotonic()
            dispatch = coordinator.dispatch_round(
                server_round=server_round,
                latest_server_packet=latest_server_packet,
            )
            dispatched_count = len(dispatch.dispatched_clients)
            quorum_required = (
                int(math.ceil(
                    dispatched_count * float(config.quorum_fraction)
                ))
                if dispatched_count
                else 0
            )
            reference = (
                statistics.median(reference_times)
                if reference_times
                else None
            )
            soft_deadline_s, hard_deadline_s = _deadline_values(
                config,
                reference,
            )
            if server_round <= int(config.warmup_rounds):
                hard_deadline_s = max(
                    hard_deadline_s,
                    float(config.registration_timeout_s),
                )
                soft_deadline_s = min(
                    soft_deadline_s,
                    hard_deadline_s * 0.75,
                )
            quorum_reached = _wait_collection_window(
                service=service,
                coordinator=coordinator,
                processes=processes,
                error_queue=error_queue,
                dispatched_count=dispatched_count,
                quorum_required=quorum_required,
                soft_deadline_s=soft_deadline_s,
                hard_deadline_s=hard_deadline_s,
                warmup=(
                    server_round <= int(config.warmup_rounds)
                ),
            )
            candidates = service.drain_mailbox()
            _drain_attack_stats(attack_stats_queue, attack_stats_cache)
            consumed_at_s = time.monotonic()
            knowledge_by_client: Dict[int, TeacherKnowledge] = {}
            round_events: List[Dict[str, object]] = []
            for item in candidates:
                packet = item.packet
                if int(packet.client_id) in knowledge_by_client:
                    raise RuntimeError(
                        "One aggregation batch contains multiple packets "
                        "from the same client."
                    )
                knowledge_by_client[int(packet.client_id)] = (
                    TeacherKnowledge(
                        metadata=TeacherMetadata(
                            client_id=int(packet.client_id),
                            model_round=int(packet.source_round),
                            generated_at_s=float(packet.generated_at_s),
                        ),
                        logits=packet.decode_logits(),
                    )
                )
                state = service.event_state(packet.packet_id)
                event = _event_from_item(
                    item,
                    consumed_round=server_round,
                    consumed_at_s=consumed_at_s,
                    state=state,
                )
                _drain_attack_stats(attack_stats_queue, attack_stats_cache)
                attack_record = attack_stats_cache.get(str(packet.task_id))
                event.update({
                    "is_malicious": bool(
                        attack_plan is not None
                        and attack_plan.is_malicious(packet.client_id)
                    ),
                    "attack_active": bool(
                        attack_plan is not None
                        and attack_plan.active_for(
                            packet.client_id, packet.source_round
                        )
                    ),
                    "poisoned_samples": (
                        int(attack_record["poisoned_samples"])
                        if attack_record is not None else _nan()
                    ),
                    "eligible_poison_samples": (
                        int(attack_record["eligible_poison_samples"])
                        if attack_record is not None else _nan()
                    ),
                    "poisoned_batches": (
                        int(attack_record["poisoned_batches"])
                        if attack_record is not None else _nan()
                    ),
                    "dba_trigger_part": (
                        int(attack_record["dba_trigger_part"])
                        if attack_record is not None else (
                            int(attack_plan.dba_part(packet.client_id))
                            if attack_plan is not None
                            and attack_plan.config.attack_type == "dba"
                            and attack_plan.is_malicious(packet.client_id)
                            else -1
                        )
                    ),
                    "attack_stats_missing": int(attack_record is None),
                })
                round_events.append(event)
                coordinator.mark_consumed(
                    task_id=packet.task_id,
                    consumed_at_s=consumed_at_s,
                )
                if server_round <= int(config.warmup_rounds):
                    reference_times.append(
                        float(packet.total_compute_phase_s)
                        + float(event["transport_age_s"])
                    )

            admission_started = time.monotonic()
            decision: Optional[AdmissionDecision] = None
            if knowledge_by_client:
                decision = server.apply_admission(
                    knowledge_by_client,
                    current_round=server_round,
                    controller=admission_controller,
                )
            candidate_ids = sorted(knowledge_by_client)
            if decision is None:
                admitted_ids = candidate_ids
            else:
                _validate_decision(decision, candidate_ids)
                admitted_ids = list(decision.admitted_client_ids)
            admission_time = time.monotonic() - admission_started
            admission_metrics = _decision_metrics(
                decision,
                num_clients=len(candidate_ids),
            )
            record_by_client = (
                {
                    int(record.client_id): record
                    for record in decision.records
                }
                if decision is not None
                else {}
            )
            for event in round_events:
                record = record_by_client.get(int(event["client_id"]))
                if record is None:
                    event["admitted"] = True
                else:
                    event["vcaa_version_score"] = float(
                        record.components["version_score"]
                    )
                    event["vcaa_content_score"] = float(
                        record.components["content_score"]
                    )
                    event["vcaa_final_score"] = float(record.score)
                    event["vcaa_threshold"] = float(decision.threshold)
                    event["proxy_accuracy"] = float(
                        record.components["proxy_accuracy"]
                    )
                    event["mean_entropy"] = float(
                        record.components["mean_entropy"]
                    )
                    event["mean_kl"] = float(
                        record.components["mean_kl"]
                    )
                    event["admitted"] = bool(record.admitted)

            defense_started = time.monotonic()
            defense_result: Optional[DefenseResult] = None
            if knowledge_by_client:
                defense_result = server.apply_defense(
                    knowledge_by_client,
                    admitted_client_ids=admitted_ids,
                    current_round=server_round,
                    controller=defense_controller,
                )
            if defense_result is not None:
                purified = {
                    int(item.metadata.client_id): item
                    for item in defense_result.purified_knowledge
                }
                if set(purified) != set(admitted_ids):
                    raise RuntimeError(
                        "Defense did not return every admitted teacher."
                    )
                knowledge_by_client.update(purified)
            defense_time = time.monotonic() - defense_started
            defense_metrics = _defense_metrics(
                defense_result,
                method=(
                    defense_controller.name
                    if defense_controller is not None
                    else "none"
                ),
            )
            defense_split = split_defense_diagnostics(
                defense_metrics["records"],
                malicious_client_ids=(
                    attack_plan.malicious_client_ids
                    if attack_plan is not None else ()
                ),
            )
            defense_by_client = (
                {
                    int(record.client_id): record
                    for record in defense_result.records
                }
                if defense_result is not None
                else {}
            )
            for event in round_events:
                record = defense_by_client.get(int(event["client_id"]))
                if record is not None:
                    event["niabd_anomaly_fraction"] = float(
                        record.anomaly_fraction
                    )
                    event["niabd_mean_suppression"] = float(
                        record.mean_suppression
                    )

            server_snapshot = _freeze_state_dict(
                server.model.state_dict()
            )
            aggregation_started = time.monotonic()
            aggregate = server.aggregate_admitted_probabilities(
                knowledge_by_client,
                admitted_ids,
                temperature=float(distill_temperature),
            )
            aggregation_time = time.monotonic() - aggregation_started
            distill_started = time.monotonic()
            server_updated = server.train_from_teacher_probabilities(
                aggregate,
                learning_rate=max(float(learning_rate) * 0.2, 1e-4),
                temperature=float(distill_temperature),
            )
            distill_time = time.monotonic() - distill_started
            rollback = 0
            if not _model_is_finite(server.model):
                _restore_model(
                    server.model,
                    server_snapshot,
                    torch.device(config.server_device),
                )
                rollback = 1
                server_updated = False
            latest_server_packet = ServerLogitsPacket.from_logits(
                model_round=server_round,
                query_id=data_plan.proxy_version,
                proxy_version=data_plan.proxy_version,
                logits=server.student_proxy_logits(),
            )
            accuracy, loss, nonfinite_eval = evaluate_with_loss(
                server.model,
                test_loader,
                device=torch.device(config.server_device),
                amp=bool(config.amp),
            )
            if attack_plan is not None:
                backdoor_eval = evaluate_backdoor_suite(
                    server.model,
                    test_loader,
                    device=torch.device(config.server_device),
                    plan=attack_plan,
                    round_number=server_round,
                    amp=bool(config.amp),
                )
            else:
                backdoor_eval = {
                    "basr_global": _nan(),
                    "basr_global_numerator": 0,
                    "basr_global_denominator": 0,
                    "basr_local_1": _nan(),
                    "basr_local_2": _nan(),
                    "basr_local_3": _nan(),
                    "basr_local_4": _nan(),
                }
            round_time = time.monotonic() - round_started
            version_lags = [
                int(event["version_lag"]) for event in round_events
            ]
            ages = [
                float(event["knowledge_age_s"]) for event in round_events
            ]
            admission_records = list(admission_metrics["records"])
            for record in admission_records:
                matching = next(
                    event for event in round_events
                    if int(event["client_id"]) == int(record["client_id"])
                )
                record.update({
                    "task_id": matching["task_id"],
                    "packet_id": matching["packet_id"],
                    "source_round": matching["source_round"],
                    "consumed_round": matching["consumed_round"],
                    "version_lag": matching["version_lag"],
                    "knowledge_age_s": matching["knowledge_age_s"],
                })
            defense_records = list(defense_metrics["records"])
            for record in defense_records:
                matching = next(
                    event for event in round_events
                    if int(event["client_id"]) == int(record["client_id"])
                )
                record.update({
                    "task_id": matching["task_id"],
                    "packet_id": matching["packet_id"],
                    "source_round": matching["source_round"],
                    "consumed_round": matching["consumed_round"],
                    "version_lag": matching["version_lag"],
                })
            metric_values = {
                "acc_list": float(accuracy),
                "loss_list": float(loss),
                "local_train_time_s": float(sum(
                    event["actual_compute_time_s"]
                    for event in round_events
                )),
                "upload_time_s": float(sum(
                    event["rpc_elapsed_s"]
                    for event in round_events
                )),
                "admission_time_s": float(admission_time),
                "defense_time_s": float(defense_time),
                "aggregation_time_s": float(aggregation_time),
                "distill_time_s": float(distill_time),
                "round_time_s": float(round_time),
                "wall_clock_time_s": float(
                    time.monotonic() - wall_start
                ),
                "clients_trained": len(round_events),
                "client_upload_bytes": int(sum(
                    event["payload_bytes"] for event in round_events
                )),
                "client_wire_bytes": int(sum(
                    event["wire_bytes"] for event in round_events
                )),
                "server_broadcast_bytes": int(
                    latest_server_packet.payload_bytes
                    * len(dispatch.dispatched_clients)
                ),
                "server_client_distillations": len(admitted_ids),
                "server_updates_from_clients": len(admitted_ids),
                "client_reverse_distillations": sum(
                    1 for event in round_events
                    if int(event["base_server_round"]) >= 0
                    and enable_client_distillation
                ),
                "server_update_applied": int(server_updated),
                "teachers_admitted": int(
                    admission_metrics["admitted"]
                ),
                "teachers_rejected": int(
                    admission_metrics["rejected"]
                ),
                "teacher_utilization": float(
                    admission_metrics["utilization"]
                ),
                "admission_threshold": float(
                    admission_metrics["threshold"]
                ),
                "admission_score_mean": float(
                    admission_metrics["score_mean"]
                ),
                "vcaa_version_score_mean": float(
                    admission_metrics["version_score_mean"]
                ),
                "vcaa_content_score_mean": float(
                    admission_metrics["content_score_mean"]
                ),
                "vcaa_proxy_accuracy_mean": float(
                    admission_metrics["proxy_accuracy_mean"]
                ),
                "vcaa_entropy_mean": float(
                    admission_metrics["entropy_mean"]
                ),
                "vcaa_kl_mean": float(
                    admission_metrics["kl_mean"]
                ),
                "teachers_purified": int(
                    defense_metrics["teachers_purified"]
                ),
                "niabd_warmup": float(defense_metrics["warmup"]),
                "niabd_anomaly_fraction": float(
                    defense_metrics["anomaly_fraction"]
                ),
                "niabd_mean_suppression": float(
                    defense_metrics["mean_suppression"]
                ),
                "niabd_threshold_mean": float(
                    defense_metrics["threshold_mean"]
                ),
                "niabd_threshold_min": float(
                    defense_metrics["threshold_min"]
                ),
                "niabd_threshold_max": float(
                    defense_metrics["threshold_max"]
                ),
                "niabd_prototype_updated": float(
                    defense_metrics["prototype_updated"]
                ),
                "niabd_prototype_observations": float(
                    defense_metrics["prototype_observations"]
                ),
                "niabd_memory_eligible_teachers": float(
                    defense_metrics["memory_eligible_teachers"]
                ),
                "nonfinite_eval_batches": int(nonfinite_eval),
                "nonfinite_distill_rollbacks": int(rollback),
                "numeric_failure_count": float(
                    nonfinite_eval + rollback
                ),
                "attack_active": (
                    int(attack_plan.config.active(server_round))
                    if attack_plan is not None else 0
                ),
                "poisoned_samples": int(sum(
                    int(event["poisoned_samples"])
                    for event in round_events
                    if not (
                        isinstance(event["poisoned_samples"], float)
                        and math.isnan(event["poisoned_samples"])
                    )
                )),
                "eligible_poison_samples": int(sum(
                    int(event["eligible_poison_samples"])
                    for event in round_events
                    if not (
                        isinstance(event["eligible_poison_samples"], float)
                        and math.isnan(event["eligible_poison_samples"])
                    )
                )),
                "attack_stats_missing_count": sum(
                    int(event["attack_stats_missing"])
                    for event in round_events
                ),
                "basr_global": float(backdoor_eval["basr_global"]),
                "basr_global_numerator": int(
                    backdoor_eval["basr_global_numerator"]
                ),
                "basr_global_denominator": int(
                    backdoor_eval["basr_global_denominator"]
                ),
                "basr_local_1": float(backdoor_eval["basr_local_1"]),
                "basr_local_2": float(backdoor_eval["basr_local_2"]),
                "basr_local_3": float(backdoor_eval["basr_local_3"]),
                "basr_local_4": float(backdoor_eval["basr_local_4"]),
                "malicious_mean_anomaly_fraction": float(
                    defense_split["malicious_mean_anomaly_fraction"]
                ),
                "benign_mean_anomaly_fraction": float(
                    defense_split["benign_mean_anomaly_fraction"]
                ),
                "malicious_mean_suppression": float(
                    defense_split["malicious_mean_suppression"]
                ),
                "benign_mean_suppression": float(
                    defense_split["benign_mean_suppression"]
                ),
                "malicious_memory_eligible_rate": float(
                    defense_split["malicious_memory_eligible_rate"]
                ),
                "benign_memory_eligible_rate": float(
                    defense_split["benign_memory_eligible_rate"]
                ),
                "selected_clients": len(dispatch.selected_clients),
                "dispatched_clients": len(
                    dispatch.dispatched_clients
                ),
                "busy_skipped_clients": len(
                    dispatch.busy_skipped_clients
                ),
                "offline_clients": len(dispatch.offline_clients),
                "packets_consumed": len(round_events),
                "fresh_packets": sum(
                    lag == 0 for lag in version_lags
                ),
                "mild_stale_packets": sum(
                    lag == 1 for lag in version_lags
                ),
                "moderate_stale_packets": sum(
                    lag in {2, 3} for lag in version_lags
                ),
                "severe_stale_packets": sum(
                    lag >= 4 for lag in version_lags
                ),
                "mean_version_lag": (
                    float(statistics.fmean(version_lags))
                    if version_lags else _nan()
                ),
                "max_version_lag": (
                    max(version_lags) if version_lags else _nan()
                ),
                "mean_knowledge_age_s": (
                    float(statistics.fmean(ages))
                    if ages else _nan()
                ),
                "max_knowledge_age_s": (
                    max(ages) if ages else _nan()
                ),
                "upload_attempt_drop_count": sum(
                    int(event["upload_attempt_drop_count"])
                    for event in round_events
                ),
                "rpc_timeout_count": sum(
                    int(event["rpc_timeout_count"])
                    for event in round_events
                ),
                "retry_count": sum(
                    int(event["retry_count"])
                    for event in round_events
                ),
                "quorum_required": int(quorum_required),
                "quorum_reached": int(quorum_reached),
                "soft_deadline_s": float(soft_deadline_s),
                "hard_deadline_s": float(hard_deadline_s),
            }
            for key, value in metric_values.items():
                _append_metric(metrics, key, value)
            metrics["teacher_admission_records"].append(
                admission_records
            )
            metrics["teacher_defense_records"].append(defense_records)
            metrics["backdoor_client_records"].append([
                {
                    "client_id": int(event["client_id"]),
                    "task_id": str(event["task_id"]),
                    "packet_id": str(event["packet_id"]),
                    "source_round": int(event["source_round"]),
                    "consumed_round": int(event["consumed_round"]),
                    "version_lag": int(event["version_lag"]),
                    "is_malicious": bool(event["is_malicious"]),
                    "attack_active": bool(event["attack_active"]),
                    "poisoned_samples": event["poisoned_samples"],
                    "eligible_poison_samples": event["eligible_poison_samples"],
                    "poisoned_batches": event["poisoned_batches"],
                    "dba_trigger_part": int(event["dba_trigger_part"]),
                    "attack_stats_missing": int(event["attack_stats_missing"]),
                }
                for event in round_events
            ])
            metrics["runtime_events"].extend(round_events)
            print(
                f"[Round {server_round}] runtime=process-semi-async "
                f"selected={len(dispatch.selected_clients)} "
                f"dispatched={dispatched_count} "
                f"packets={len(round_events)} "
                f"fresh={metric_values['fresh_packets']} "
                f"stale={len(round_events) - metric_values['fresh_packets']} "
                f"admitted={len(admitted_ids)} acc={accuracy:.4f} "
                f"attack={metrics['attack_type']} "
                f"poisoned={metric_values['poisoned_samples']} "
                f"basr={metric_values['basr_global']:.4f}"
            )

        coordinator.request_shutdown()
        shutdown_deadline = (
            time.monotonic() + float(config.shutdown_timeout_s)
        )
        for process in processes:
            remaining = max(0.0, shutdown_deadline - time.monotonic())
            process.join(timeout=remaining)
        alive = [process for process in processes if process.is_alive()]
        if alive:
            for process in alive:
                process.terminate()
            for process in alive:
                process.join(timeout=5.0)
            raise TimeoutError(
                "Client processes did not stop before shutdown timeout."
            )
        _raise_child_error(error_queue, processes)
        coordinator.expire_unfinished()
        _drain_attack_stats(attack_stats_queue, attack_stats_cache)
        consumed_packet_ids = {
            str(event["packet_id"])
            for event in metrics["runtime_events"]
        }
        for state in service.all_event_states:
            packet = state.get("packet")
            if (
                isinstance(packet, ClientLogitsPacket)
                and packet.packet_id not in consumed_packet_ids
            ):
                metrics["runtime_events"].append(
                    _unconsumed_event_from_state(state)
                )
        for event in metrics["runtime_events"]:
            if "is_malicious" not in event:
                task_id = str(event["task_id"])
                attack_record = attack_stats_cache.get(task_id)
                client_id = int(event["client_id"])
                source_round = int(event["source_round"])
                event.update({
                    "is_malicious": bool(
                        attack_plan is not None
                        and attack_plan.is_malicious(client_id)
                    ),
                    "attack_active": bool(
                        attack_plan is not None
                        and attack_plan.active_for(client_id, source_round)
                    ),
                    "poisoned_samples": (
                        int(attack_record["poisoned_samples"])
                        if attack_record is not None else _nan()
                    ),
                    "eligible_poison_samples": (
                        int(attack_record["eligible_poison_samples"])
                        if attack_record is not None else _nan()
                    ),
                    "poisoned_batches": (
                        int(attack_record["poisoned_batches"])
                        if attack_record is not None else _nan()
                    ),
                    "dba_trigger_part": (
                        int(attack_record["dba_trigger_part"])
                        if attack_record is not None else -1
                    ),
                    "attack_stats_missing": int(attack_record is None),
                })
            state = service.event_state(str(event["packet_id"]))
            event["upload_attempts"] = int(state["upload_attempts"])
            event["upload_attempt_drop_count"] = int(
                state["upload_attempt_drop_count"]
            )
            event["rpc_timeout_count"] = int(
                state["rpc_timeout_count"]
            )
            event["retry_count"] = int(state["retry_count"])
            event["duplicate_receive_count"] = int(
                state["duplicate_receive_count"]
            )
            event["wire_bytes"] = (
                int(state["request_wire_bytes"])
                + int(state["response_wire_bytes"])
            )
        metrics["all_clients_stopped"] = True
        metrics["transport_error_count"] = len(
            service.transport_errors
        )
        return metrics
    finally:
        coordinator.request_shutdown()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        rpc_server.close()
        error_queue.close()
        error_queue.join_thread()
        attack_stats_queue.close()
        attack_stats_queue.join_thread()
