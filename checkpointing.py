from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

from result_schema import (
    AGGREGATION_ALGORITHM_VERSION,
    NIABD_ALGORITHM_VERSION,
    RESULT_SCHEMA_VERSION,
    VCAA_ALGORITHM_VERSION,
)


CHECKPOINT_VERSION = "checkpoint-v3"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not required.issubset(state):
        raise ValueError("Checkpoint RNG state is incomplete.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _state_dict(model) -> dict:
    return {
        str(key): value.detach().cpu().clone()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in model.state_dict().items()
    }


def _controller_state(controller, name: str) -> Optional[dict]:
    if controller is None:
        return None
    if not hasattr(controller, "snapshot_state"):
        raise TypeError(f"{name} controller does not expose snapshot_state().")
    return controller.snapshot_state()


def build_checkpoint_payload(
    *,
    current_round: int,
    expected_rounds: int,
    run_uid: str,
    config_sha256: str,
    runtime: str,
    server_model,
    client_models: Optional[Sequence[Any]] = None,
    optimizers: Optional[Sequence[Any]] = None,
    admission_controller=None,
    defense_controller=None,
    attack_plan_state: Optional[dict] = None,
    data_identity: Optional[dict] = None,
    proxy_identity: Optional[dict] = None,
    architecture_assignment: Optional[dict] = None,
    coordinator_state: Optional[dict] = None,
    manifest_identity: Optional[dict] = None,
    metrics_state: Optional[dict] = None,
) -> dict:
    if int(current_round) < 0 or int(current_round) > int(expected_rounds):
        raise ValueError("Checkpoint current_round is outside expected range.")
    client_state = None if client_models is None else [_state_dict(model) for model in client_models]
    optimizer_state = None if optimizers is None else [optimizer.state_dict() for optimizer in optimizers]
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "current_round": int(current_round),
        "expected_rounds": int(expected_rounds),
        "run_uid": str(run_uid),
        "config_sha256": str(config_sha256),
        "runtime": str(runtime),
        "server_model": _state_dict(server_model),
        "client_models": client_state,
        "optimizer_states": optimizer_state,
        "vcaa_state": _controller_state(admission_controller, "VCAA"),
        "niabd_state": _controller_state(defense_controller, "NIABD"),
        "attack_plan_state": attack_plan_state,
        "data_identity": data_identity,
        "proxy_identity": proxy_identity,
        "architecture_assignment": architecture_assignment,
        "coordinator_state": coordinator_state,
        "manifest_identity": manifest_identity,
        "metrics_state": metrics_state,
        "algorithm_versions": {
            "vcaa_algorithm_version": (
                VCAA_ALGORITHM_VERSION if admission_controller is not None else "none"
            ),
            "niabd_algorithm_version": (
                NIABD_ALGORITHM_VERSION if defense_controller is not None else "none"
            ),
            "aggregation_algorithm_version": AGGREGATION_ALGORITHM_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
        },
        "rng_state": capture_rng_state(),
    }


def save_checkpoint_atomic(payload: dict, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)
    return sha256_file(target)


def load_checkpoint(
    path: str | Path,
    *,
    expected_config_sha256: Optional[str] = None,
    expected_runtime: Optional[str] = None,
    expected_rounds: Optional[int] = None,
    expected_algorithm_versions: Optional[dict[str, str]] = None,
) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    # Checkpoints are produced by this trusted local runner and contain RNG
    # objects (including NumPy state) that the PyTorch 2.6 weights-only loader
    # intentionally rejects.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported or incomplete checkpoint version.")
    if expected_config_sha256 is not None and payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("Checkpoint config hash mismatch.")
    if expected_runtime is not None and payload.get("runtime") != str(expected_runtime):
        raise ValueError("Checkpoint runtime mismatch.")
    if expected_rounds is not None and int(payload.get("expected_rounds", -1)) != int(expected_rounds):
        raise ValueError("Checkpoint expected-round mismatch.")
    if expected_algorithm_versions is not None and payload.get("algorithm_versions") != expected_algorithm_versions:
        raise ValueError("Checkpoint algorithm/schema version mismatch.")
    required = {"server_model", "vcaa_state", "niabd_state", "rng_state", "proxy_identity", "data_identity"}
    if not required.issubset(payload):
        raise ValueError("Checkpoint is incomplete.")
    return payload


def restore_checkpoint(
    payload: dict,
    *,
    server_model,
    client_models: Optional[Sequence[Any]] = None,
    admission_controller=None,
    defense_controller=None,
    optimizers: Optional[Sequence[Any]] = None,
) -> None:
    server_model.load_state_dict(payload["server_model"], strict=True)
    saved_clients = payload.get("client_models")
    if saved_clients is not None:
        if client_models is None or len(client_models) != len(saved_clients):
            raise ValueError("Checkpoint client-model count mismatch.")
        for model, state in zip(client_models, saved_clients):
            model.load_state_dict(state, strict=True)
    if payload.get("optimizer_states") is not None:
        if optimizers is None or len(optimizers) != len(payload["optimizer_states"]):
            raise ValueError("Checkpoint optimizer-state count mismatch.")
        for optimizer, state in zip(optimizers, payload["optimizer_states"]):
            optimizer.load_state_dict(state)
    if admission_controller is not None and payload.get("vcaa_state") is not None:
        admission_controller.restore_state(payload["vcaa_state"])
    if defense_controller is not None and payload.get("niabd_state") is not None:
        defense_controller.restore_state(payload["niabd_state"])
    restore_rng_state(payload["rng_state"])


def checkpoint_is_complete(path: str | Path, *, expected_rounds: int) -> bool:
    try:
        payload = load_checkpoint(path, expected_rounds=expected_rounds)
    except (OSError, RuntimeError, ValueError, EOFError, TypeError):
        return False
    return int(payload.get("current_round", -1)) >= 0
