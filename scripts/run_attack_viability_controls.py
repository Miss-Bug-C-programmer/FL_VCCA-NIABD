from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json_or_inline(value: str) -> dict:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run 3 datasets x 5 seeds clean no-attack baseline controls. "
            "These controls are separate from the 240 requested attacked runs."
        )
    )
    parser.add_argument("--dataset-roots", required=True)
    parser.add_argument(
        "--config",
        default="configs/main_backdoor_experiment.json",
    )
    parser.add_argument(
        "--outdir",
        default="experiment_results_attack_viability_controls",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    roots = _load_json_or_inline(args.dataset_roots)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = list(product(config["datasets"], config["seeds"]))
    if len(jobs) != 15:
        raise ValueError(f"Expected 15 control runs, found {len(jobs)}.")
    for index, (dataset, seed) in enumerate(jobs, start=1):
        output = Path(args.outdir) / dataset / f"seed_{seed}"
        command = [
            sys.executable,
            "experiment_runner.py",
            "--dataset",
            str(roots[dataset]),
            "--dataset-name",
            dataset,
            "--method",
            "baseline",
            "--attack",
            "none",
            "--rounds",
            str(config["rounds"]),
            "--epochs",
            str(config["local_epochs"]),
            "--batch-size",
            str(config["batch_size"]),
            "--num-clients-list",
            str(config["num_clients"]),
            "--seeds",
            str(seed),
            "--partition-schemes",
            str(config["partition_scheme"]),
            "--dirichlet-alpha",
            str(config["dirichlet_alpha"]),
            "--proxy-ratio",
            str(config["proxy_ratio"]),
            "--val-ratio",
            str(config["val_ratio"]),
            "--device",
            args.device,
            "--outdir",
            str(output),
        ]
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
