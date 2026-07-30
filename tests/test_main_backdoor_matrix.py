from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pytest

from scripts.run_attack_viability_controls import (
    _load_json_or_inline as load_control_roots,
    build_control_command,
    build_control_jobs,
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


def test_attack_viability_control_modes_are_separate_from_formal_matrix():
    config = json.loads(
        Path("configs/main_backdoor_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    clean = build_control_jobs(config, "clean")
    triggered = build_control_jobs(config, "triggered-no-poison")
    both = build_control_jobs(config, "both")

    assert len(clean) == 15
    assert len(triggered) == 60
    assert len(both) == 75
    assert all(attack == "none" for _, attack, _ in clean)
    assert all(attack in config["attacks"] for _, attack, _ in triggered)


def test_triggered_no_poison_control_passes_formal_configuration():
    config = json.loads(
        Path("configs/main_backdoor_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    command = build_control_command(
        dataset_root="/data",
        dataset="cifar10",
        attack="badnets",
        seed=3,
        config=config,
        outdir=Path("/controls/triggered-no-poison/cifar10/badnets/seed_3"),
        device="cuda:0",
        num_workers=2,
    )

    assert command[command.index("--method") + 1] == "baseline"
    assert command[command.index("--attack") + 1] == "badnets"
    assert command[command.index("--poison-ratio") + 1] == "0.0"
    assert command[command.index("--target-label") + 1] == str(
        config["target_label"]
    )
    assert command[command.index("--malicious-fraction") + 1] == str(
        config["malicious_fraction"]
    )
    assert command[command.index("--attack-start-round") + 1] == str(
        config["attack_start_round"]
    )
    assert command[command.index("--attack-end-round") + 1] == str(
        config["attack_end_round"]
    )
    assert command[command.index("--proxy-dataset-size") + 1] == str(
        config["proxy_dataset_size"]
    )
    assert command[command.index("--runtime") + 1] == config["runtime"]
    assert command[command.index("--num-workers") + 1] == "2"
