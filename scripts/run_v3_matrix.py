"""Enumerate or execute an auditable v3 experiment matrix.

The default is a dry-run so a formal 240-run plan can be inspected before any
training is started.  Existing ``experiment_runner.py`` remains the sole
training entry point.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_backdoor_experiment.json")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = list(config["datasets"])
    attacks = list(config["attacks"])
    methods = list(config["methods"])
    seeds = list(config["seeds"])
    jobs = list(itertools.product(datasets, attacks, methods, seeds))
    if args.limit > 0:
        jobs = jobs[: int(args.limit)]
    print(json.dumps({
        "config": str(config_path),
        "jobs": len(jobs),
        "full_factorization": {
            "datasets": len(datasets),
            "attacks": len(attacks),
            "methods": len(methods),
            "seeds": len(seeds),
        },
        "dry_run": not bool(args.execute),
    }, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    root = config_path.parent.parent
    runner = root / "experiment_runner.py"
    dataset_paths = config.get("dataset_paths")
    if not isinstance(dataset_paths, dict):
        raise SystemExit(
            "--execute requires a config.dataset_paths mapping; dry-run remains safe."
        )
    for dataset, attack, method, seed in jobs:
        command = [
            sys.executable,
            str(runner),
            "--dataset", str(config["dataset_paths"][dataset]),
            "--dataset-name", str(dataset),
            "--attack", str(attack),
            "--method", str(method),
            "--seeds", str(seed),
            "--run-class", "formal",
            "--runtime", str(config.get("runtime", "sync")),
            "--runtime-profile", str(
                config.get("runtime_profile", "configs/runtime_moderate.json")
            ),
            "--runtime-warmup-rounds", str(
                config.get("runtime_warmup_rounds", 1)
            ),
            "--quorum-fraction", str(config.get("quorum_fraction", 0.5)),
            "--vcaa-age-scale-mode", str(
                config.get("vcaa_age_scale_mode", "runtime-calibrated")
            ),
            "--vcaa-max-version-lag", str(
                config.get("vcaa_max_version_lag", 1)
            ),
            "--vcaa-version-lag-half-life-rounds", str(
                config.get("vcaa_version_lag_half_life_rounds", 1.0)
            ),
            "--vcaa-minimum-content-history-size", str(
                config.get("vcaa_minimum_content_history_size", 3)
            ),
        ]
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
