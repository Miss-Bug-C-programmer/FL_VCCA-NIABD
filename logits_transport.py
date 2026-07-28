from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Dict, Tuple
import uuid

import torch


FLOAT32_DTYPE = "float32"


def _encode_float32_tensor(
    tensor: torch.Tensor,
) -> Tuple[Tuple[int, ...], bytes]:
    cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(cpu_tensor).all().item()):
        raise ValueError("Logits tensor contains non-finite values.")
    cpu_tensor = cpu_tensor.contiguous()
    return tuple(int(size) for size in cpu_tensor.shape), bytes(
        cpu_tensor.numpy().tobytes(order="C")
    )


def _decode_float32_tensor(
    shape: Tuple[int, ...],
    payload: bytes,
) -> torch.Tensor:
    expected_values = 1
    for size in shape:
        if int(size) < 0:
            raise ValueError("Logits packet shape cannot contain negatives.")
        expected_values *= int(size)
    expected_bytes = int(expected_values) * 4
    if len(payload) != expected_bytes:
        raise ValueError(
            "Logits packet payload size does not match its declared shape."
        )
    buffer = bytearray(payload)
    tensor = torch.frombuffer(
        buffer,
        dtype=torch.float32,
        count=expected_values,
    ).clone().reshape(shape)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("Decoded logits contain non-finite values.")
    return tensor


def _payload_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ClientLogitsPacket:
    """Immutable client knowledge generated once and reused for retries."""

    client_id: int
    model_round: int
    generated_at_s: float
    query_id: str
    logits_shape: Tuple[int, ...]
    logits_payload: bytes
    task_id: str = ""
    packet_id: str = ""
    source_round: int = -1
    base_server_round: int = -1
    local_model_version: int = 0
    proxy_version: str = ""
    logits_dtype: str = FLOAT32_DTYPE
    payload_sha256: str = ""
    dispatched_at_s: float = 0.0
    compute_started_at_s: float = 0.0
    compute_finished_at_s: float = 0.0
    actual_compute_time_s: float = 0.0
    injected_compute_delay_s: float = 0.0
    total_compute_phase_s: float = 0.0
    proxy_inference_time_s: float = 0.0
    client_pid: int = 0
    local_train_count: int = 0
    predict_logits_calls: int = 0
    inference_sha256: str = ""

    def __post_init__(self) -> None:
        if self.logits_dtype != FLOAT32_DTYPE:
            raise ValueError("Only float32 logits packets are supported.")
        if not math.isfinite(float(self.generated_at_s)):
            raise ValueError("generated_at_s must be finite.")
        if int(self.source_round) < 0:
            object.__setattr__(self, "source_round", int(self.model_round))
        if int(self.base_server_round) < 0:
            object.__setattr__(
                self,
                "base_server_round",
                int(self.source_round),
            )
        if not self.packet_id:
            object.__setattr__(self, "packet_id", uuid.uuid4().hex)
        expected_hash = _payload_hash(self.logits_payload)
        if self.payload_sha256 and self.payload_sha256 != expected_hash:
            raise ValueError("Client logits payload SHA-256 mismatch.")
        object.__setattr__(self, "payload_sha256", expected_hash)
        if not self.inference_sha256:
            object.__setattr__(self, "inference_sha256", expected_hash)
        if self.inference_sha256 != expected_hash:
            raise ValueError(
                "Inference tensor identity does not match packet payload."
            )
        if len(self.logits_shape) != 2:
            raise ValueError(
                "Client proxy logits must have shape [samples, classes]."
            )
        _decode_float32_tensor(self.logits_shape, self.logits_payload)

    @classmethod
    def from_logits(
        cls,
        *,
        client_id: int,
        model_round: int,
        generated_at_s: float,
        query_id: str,
        logits: torch.Tensor,
        task_id: str = "",
        packet_id: str = "",
        source_round: int | None = None,
        base_server_round: int | None = None,
        local_model_version: int = 0,
        proxy_version: str = "",
        dispatched_at_s: float = 0.0,
        compute_started_at_s: float = 0.0,
        compute_finished_at_s: float = 0.0,
        actual_compute_time_s: float = 0.0,
        injected_compute_delay_s: float = 0.0,
        total_compute_phase_s: float = 0.0,
        proxy_inference_time_s: float = 0.0,
        client_pid: int = 0,
        local_train_count: int = 0,
        predict_logits_calls: int = 0,
    ) -> "ClientLogitsPacket":
        shape, payload = _encode_float32_tensor(logits)
        digest = _payload_hash(payload)
        return cls(
            client_id=int(client_id),
            model_round=int(model_round),
            generated_at_s=float(generated_at_s),
            query_id=str(query_id),
            logits_shape=shape,
            logits_payload=payload,
            task_id=str(task_id),
            packet_id=str(packet_id),
            source_round=(
                int(model_round)
                if source_round is None
                else int(source_round)
            ),
            base_server_round=(
                int(model_round)
                if base_server_round is None
                else int(base_server_round)
            ),
            local_model_version=int(local_model_version),
            proxy_version=str(proxy_version),
            payload_sha256=digest,
            dispatched_at_s=float(dispatched_at_s),
            compute_started_at_s=float(compute_started_at_s),
            compute_finished_at_s=float(compute_finished_at_s),
            actual_compute_time_s=float(actual_compute_time_s),
            injected_compute_delay_s=float(injected_compute_delay_s),
            total_compute_phase_s=float(total_compute_phase_s),
            proxy_inference_time_s=float(proxy_inference_time_s),
            client_pid=int(client_pid),
            local_train_count=int(local_train_count),
            predict_logits_calls=int(predict_logits_calls),
            inference_sha256=digest,
        )

    @classmethod
    def from_rpc_parts(
        cls,
        metadata: Dict[str, object],
        payload: bytes,
    ) -> "ClientLogitsPacket":
        required = {
            "client_id",
            "model_round",
            "generated_at_s",
            "query_id",
            "logits_shape",
            "task_id",
            "packet_id",
            "source_round",
            "base_server_round",
            "local_model_version",
            "proxy_version",
            "logits_dtype",
            "payload_sha256",
        }
        missing = sorted(required.difference(metadata))
        if missing:
            raise ValueError(
                f"Client packet metadata missing fields: {missing}"
            )
        values = dict(metadata)
        values["logits_shape"] = tuple(
            int(size) for size in values["logits_shape"]
        )
        values["logits_payload"] = payload
        allowed = set(cls.__dataclass_fields__)
        return cls(**{
            key: value for key, value in values.items()
            if key in allowed
        })

    def rpc_metadata(self) -> Dict[str, object]:
        values = asdict(self)
        values.pop("logits_payload")
        values["logits_shape"] = list(self.logits_shape)
        return values

    def decode_logits(self) -> torch.Tensor:
        if _payload_hash(self.logits_payload) != self.payload_sha256:
            raise ValueError("Client logits payload SHA-256 mismatch.")
        return _decode_float32_tensor(
            self.logits_shape,
            self.logits_payload,
        )

    @property
    def payload_bytes(self) -> int:
        return len(self.logits_payload)


@dataclass(frozen=True)
class ServerLogitsPacket:
    """Binary server knowledge distributed through the task RPC."""

    model_round: int
    query_id: str
    logits_shape: Tuple[int, ...]
    logits_payload: bytes
    proxy_version: str = ""
    logits_dtype: str = FLOAT32_DTYPE
    payload_sha256: str = ""

    def __post_init__(self) -> None:
        if self.logits_dtype != FLOAT32_DTYPE:
            raise ValueError("Only float32 server logits are supported.")
        if len(self.logits_shape) != 2:
            raise ValueError(
                "Server proxy logits must have shape [samples, classes]."
            )
        expected_hash = _payload_hash(self.logits_payload)
        if self.payload_sha256 and self.payload_sha256 != expected_hash:
            raise ValueError("Server logits payload SHA-256 mismatch.")
        object.__setattr__(self, "payload_sha256", expected_hash)
        _decode_float32_tensor(self.logits_shape, self.logits_payload)

    @classmethod
    def from_logits(
        cls,
        *,
        model_round: int,
        query_id: str,
        logits: torch.Tensor,
        proxy_version: str = "",
    ) -> "ServerLogitsPacket":
        shape, payload = _encode_float32_tensor(logits)
        return cls(
            model_round=int(model_round),
            query_id=str(query_id),
            logits_shape=shape,
            logits_payload=payload,
            proxy_version=str(proxy_version),
            payload_sha256=_payload_hash(payload),
        )

    @classmethod
    def from_rpc_parts(
        cls,
        metadata: Dict[str, object],
        payload: bytes,
    ) -> "ServerLogitsPacket":
        return cls(
            model_round=int(metadata["model_round"]),
            query_id=str(metadata["query_id"]),
            logits_shape=tuple(
                int(size) for size in metadata["logits_shape"]
            ),
            logits_payload=payload,
            proxy_version=str(metadata.get("proxy_version", "")),
            logits_dtype=str(
                metadata.get("logits_dtype", FLOAT32_DTYPE)
            ),
            payload_sha256=str(metadata["payload_sha256"]),
        )

    def rpc_metadata(self) -> Dict[str, object]:
        return {
            "model_round": int(self.model_round),
            "query_id": str(self.query_id),
            "logits_shape": list(self.logits_shape),
            "proxy_version": str(self.proxy_version),
            "logits_dtype": str(self.logits_dtype),
            "payload_sha256": str(self.payload_sha256),
        }

    def decode_logits(self) -> torch.Tensor:
        if _payload_hash(self.logits_payload) != self.payload_sha256:
            raise ValueError("Server logits payload SHA-256 mismatch.")
        return _decode_float32_tensor(
            self.logits_shape,
            self.logits_payload,
        )

    @property
    def payload_bytes(self) -> int:
        return len(self.logits_payload)
