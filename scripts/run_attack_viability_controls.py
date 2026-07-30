from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_MODES = ("clean", "triggered-no-poison", "both")


def _load_json_or_inline(value: str) -> dict:
    path = Path(value)
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
    elif str(value).lstrip().startswith("{"):
        loaded = json.loads(value)
    else:
        raise FileNotFoundError(
            "dataset-roots must be an existing JSON file or an inline "
            f"JSON object; file not found: {path}"
        )
    if not isinstance(loaded, dict):
        raise ValueError("dataset-roots must decode to a JSON object.")
    return loaded


def build_control_jobs(config: dict, control_mode: str) -> list[tuple[str, str, int]]:
    """Build controls outside the formal method dimension and 240-run matrix."""

    mode = str(control_mode).lower()
    if mode not in CONTROL_MODES:
        raise ValueError(f"Unsupported control mode: {control_mode!r}.")
    jobs: list[tuple[str, str, int]] = []
    if mode in {"clean", "both"}:
        jobs.extend(product(config["datasets"], ["none"], config["seeds"]))
    if mode in {"triggered-no-poison", "both"}:
        jobs.extend(product(
            config["datasets"],
            config["attacks"],
            config["seeds"],
        ))
    return [
        (str(dataset), str(attack), int(seed))
        for dataset, attack, seed in jobs
    ]


def _control_job_dir(
    root: Path,
    *,
    dataset: str,
    attack: str,
    seed: int,
) -> Path:
    if attack == "none":
        # Preserve the original clean-control directory layout.
        return root / dataset / f"seed_{seed}"
    return (
        root
        / "triggered-no-poison"
        / dataset
        / attack
        / f"seed_{seed}"
    )


def _summary_complete(
    job_dir: Path,
    *,
    dataset: str,
    seed: int,
    rounds: int,
    attack: str,
) -> bool:
    """Accept resume only for a parseable, matching, completed summary row."""

    path = job_dir / f"fedagg_run_summary_{dataset}.csv"
    if not path.is_file():
        return False
    required = {
        "dataset",
        "seed",
        "runtime",
        "strategy",
        "attack_type",
        "rounds",
        "final_accuracy",
        "final_loss",
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(set(reader.fieldnames or ())):
                return False
            rows = list(reader)
        if len(rows) != 1:
            return False
        row = rows[0]
        return (
            row["dataset"] == str(dataset)
            and int(row["seed"]) == int(seed)
            and int(row["rounds"]) == int(rounds)
            and row["strategy"] == "baseline"
            and row["attack_type"] == str(attack)
            and row["runtime"] != ""
            and row["final_accuracy"] != ""
            and row["final_loss"] != ""
        )
    except (OSError, TypeError, ValueError, csv.Error):
        return False


def build_control_command(
    *,
    dataset_root: str,
    dataset: str,
    attack: str,
    seed: int,
    config: dict,
    outdir: Path,
    device: str,
    num_workers: int,
) -> list[str]:
    """Pass formal settings while forcing a clean or no-poison control."""

    trigger_size = 8 if dataset == "tiny-imagenet-200" else 4
    return [
        sys.executable,
        "experiment_runner.py",
        "--dataset", str(dataset_root),
        "--dataset-name", dataset,
        "--method", "baseline",
        "--attack", attack,
        "--target-label", str(config["target_label"]),
        "--malicious-fraction", str(config["malicious_fraction"]),
        "--poison-ratio", "0.0",
        "--attack-start-round", str(config["attack_start_round"]),
        "--attack-end-round", str(config["attack_end_round"]),
        "--poison-interval", str(config["poison_interval"]),
        "--trigger-size", str(trigger_size),
        "--blend-alpha", str(config["blend_alpha"]),
        "--dynamic-period", str(config["dynamic_period"]),
        "--rounds", str(config["rounds"]),
        "--epochs", str(config["local_epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--num-clients-list", str(config["num_clients"]),
        "--seeds", str(seed),
        "--partition-schemes", str(config["partition_scheme"]),
        "--dirichlet-alpha", str(config["dirichlet_alpha"]),
        "--proxy-ratio", str(config["proxy_ratio"]),
        "--val-ratio", str(config["val_ratio"]),
        "--proxy-dataset-size", str(config["proxy_dataset_size"]),
        "--distill-temperature", str(config["distill_temperature"]),
        "--niabd-warmup-rounds", str(config["niabd_warmup_rounds"]),
        "--runtime", str(config["runtime"]),
        "--runtime-profile", str(config["runtime_profile"]),
        "--device", device,
        "--num-workers", str(num_workers),
        "--outdir", str(outdir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run clean and/or triggered-without-poison controls. Controls use "
            "separate directories and are not part of the formal 240 runs."
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
    parser.add_argument(
        "--control-mode",
        choices=CONTROL_MODES,
        default="clean",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    roots = _load_json_or_inline(args.dataset_roots)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = set(config["datasets"]) - set(roots)
    if missing:
        raise ValueError(f"dataset-roots is missing: {sorted(missing)}")
    jobs = build_control_jobs(config, args.control_mode)
    expected = {
        "clean": 15,
        "triggered-no-poison": 60,
        "both": 75,
    }[args.control_mode]
    if len(jobs) != expected:
        raise ValueError(
            f"Expected {expected} {args.control_mode} controls, found {len(jobs)}."
        )
    root_out = Path(args.outdir)
    for index, (dataset, attack, seed) in enumerate(jobs, start=1):
        output = _control_job_dir(
            root_out,
            dataset=dataset,
            attack=attack,
            seed=seed,
        )
        if args.resume and _summary_complete(
            output,
            dataset=dataset,
            seed=seed,
            rounds=int(config["rounds"]),
            attack=attack,
        ):
            print(f"[{index}/{len(jobs)}] skip complete {output}", flush=True)
            continue
        command = build_control_command(
            dataset_root=str(roots[dataset]),
            dataset=dataset,
            attack=attack,
            seed=seed,
            config=config,
            outdir=output,
            device=args.device,
            num_workers=int(args.num_workers),
        )
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
