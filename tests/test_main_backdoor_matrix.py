from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pytest

from scripts.run_attack_viability_controls import (
    _load_json_or_inline as load_control_roots,
)
from scripts.run_main_backdoor_matrix import (
    _load_json_or_inline as load_matrix_roots,
    build_command,
)


def test_formal_main_matrix_is_exactly_240_jobs():
    config = json.loads(
        Path("configs/main_backdoor_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    jobs = list(product(
        config["datasets"],
        config["attacks"],
        config["methods"],
        config["seeds"],
    ))
    assert config["datasets"] == [
        "cifar10",
        "cinic10",
        "tiny-imagenet-200",
    ]
    assert config["attacks"] == ["badnets", "dba", "blend", "dynamic"]
    assert config["methods"] == [
        "baseline",
        "vcaa",
        "niabd",
        "vcaa-niabd",
    ]
    assert config["seeds"] == [0, 1, 2, 3, 4]
    assert len(jobs) == 240
    assert config["partition_scheme"] == "dirichlet"
    assert config["dirichlet_alpha"] == 0.5


@pytest.mark.parametrize(
    "loader",
    [load_control_roots, load_matrix_roots],
)
def test_dataset_roots_missing_file_has_actionable_error(loader, tmp_path):
    missing = tmp_path / "dataset_roots.container.json"
    with pytest.raises(FileNotFoundError, match="file not found"):
        loader(str(missing))


@pytest.mark.parametrize(
    "loader",
    [load_control_roots, load_matrix_roots],
)
def test_dataset_roots_accepts_inline_json_object(loader):
    assert loader('{"cifar10": "/data"}') == {"cifar10": "/data"}


def test_process_matrix_passes_shared_timeouts_and_deadlines():
    config = json.loads(
        Path("configs/main_backdoor_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    config["runtime"] = "process-semi-async"
    runtime_trace = Path("results/shared-trace.json")
    command = build_command(
        dataset_root="/data",
        dataset="cifar10",
        attack="badnets",
        method="vcaa-niabd",
        seed=0,
        config=config,
        outdir=Path("/results/job"),
        device="cuda",
        server_device="cuda",
        client_device="cpu",
        num_workers=0,
        runtime_trace=runtime_trace,
        runtime_registration_timeout_s=900.0,
        runtime_shutdown_timeout_s=900.0,
        soft_deadline_s=300.0,
        hard_deadline_s=600.0,
    )
    assert Path(command[command.index("--runtime-trace") + 1]) == (
        runtime_trace
    )
    assert command[command.index("--runtime-registration-timeout-s") + 1] == (
        "900.0"
    )
    assert command[command.index("--runtime-shutdown-timeout-s") + 1] == (
        "900.0"
    )
    assert command[command.index("--soft-deadline-s") + 1] == "300.0"
    assert command[command.index("--hard-deadline-s") + 1] == "600.0"
