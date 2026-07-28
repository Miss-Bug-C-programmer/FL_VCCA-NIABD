from __future__ import annotations

import json
from itertools import product
from pathlib import Path


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
