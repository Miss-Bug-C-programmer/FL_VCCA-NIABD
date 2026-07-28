from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_trace import RuntimeTrace, generate_runtime_trace, load_runtime_profile


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


def _job_dir(root: Path, dataset: str, attack: str, method: str, seed: int) -> Path:
    return root / dataset / attack / method / f"seed_{seed}"


def _summary_exists(job_dir: Path, dataset: str) -> bool:
    return (job_dir / f"fedagg_run_summary_{dataset}.csv").is_file()


def _ensure_trace(
    *,
    root: Path,
    dataset: str,
    seed: int,
    config: dict,
) -> Path:
    trace_path = root / "_shared_runtime_traces" / dataset / f"seed_{seed}.json"
    if trace_path.exists():
        trace = RuntimeTrace.load(str(trace_path))
        expected = (
            int(seed),
            int(config["num_clients"]),
            int(config["rounds"]),
        )
        actual = (int(trace.seed), int(trace.num_clients), int(trace.rounds))
        if actual != expected:
            raise ValueError(
                f"Existing trace dimensions {actual} do not match {expected}: {trace_path}"
            )
        return trace_path
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profile = load_runtime_profile(str(config["runtime_profile"]))
    trace = generate_runtime_trace(
        profile=profile,
        seed=int(seed),
        num_clients=int(config["num_clients"]),
        rounds=int(config["rounds"]),
        warmup_rounds=1,
        participation_rate=1.0,
    )
    trace.save(
        str(trace_path),
        metadata={
            "purpose": "strategy-independent-main-backdoor-trace",
            "dataset": dataset,
            "seed": int(seed),
        },
    )
    return trace_path


def build_command(
    *,
    dataset_root: str,
    dataset: str,
    attack: str,
    method: str,
    seed: int,
    config: dict,
    outdir: Path,
    device: str,
    server_device: str,
    client_device: str,
    num_workers: int,
    runtime_trace: Path | None,
    runtime_registration_timeout_s: float,
    runtime_shutdown_timeout_s: float,
    soft_deadline_s: float,
    hard_deadline_s: float,
) -> list[str]:
    trigger_size = 8 if dataset == "tiny-imagenet-200" else 4
    command = [
        sys.executable,
        "experiment_runner.py",
        "--dataset",
        str(dataset_root),
        "--dataset-name",
        dataset,
        "--method",
        method,
        "--attack",
        attack,
        "--target-label",
        str(config["target_label"]),
        "--malicious-fraction",
        str(config["malicious_fraction"]),
        "--poison-ratio",
        str(config["poison_ratio"]),
        "--attack-start-round",
        str(config["attack_start_round"]),
        "--attack-end-round",
        str(config["attack_end_round"]),
        "--poison-interval",
        str(config["poison_interval"]),
        "--trigger-size",
        str(trigger_size),
        "--blend-alpha",
        str(config["blend_alpha"]),
        "--dynamic-period",
        str(config["dynamic_period"]),
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
        "--proxy-dataset-size",
        str(config["proxy_dataset_size"]),
        "--distill-temperature",
        str(config["distill_temperature"]),
        "--niabd-warmup-rounds",
        str(config["niabd_warmup_rounds"]),
        "--runtime",
        str(config["runtime"]),
        "--device",
        device,
        "--server-device",
        server_device,
        "--client-device",
        client_device,
        "--num-workers",
        str(num_workers),
        "--outdir",
        str(outdir),
    ]
    if str(config["runtime"]) == "process-semi-async":
        if runtime_trace is None:
            raise ValueError("Process runtime requires a shared runtime trace.")
        command.extend([
            "--runtime-trace",
            str(runtime_trace),
            "--runtime-registration-timeout-s",
            str(runtime_registration_timeout_s),
            "--runtime-shutdown-timeout-s",
            str(runtime_shutdown_timeout_s),
        ])
        if float(soft_deadline_s) > 0.0 or float(hard_deadline_s) > 0.0:
            if (
                float(soft_deadline_s) <= 0.0
                or float(hard_deadline_s) <= float(soft_deadline_s)
            ):
                raise ValueError(
                    "Process absolute deadlines require "
                    "0 < soft-deadline-s < hard-deadline-s."
                )
            command.extend([
                "--soft-deadline-s",
                str(soft_deadline_s),
                "--hard-deadline-s",
                str(hard_deadline_s),
            ])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the formal 3x4x4x5=240 backdoor-defense matrix."
    )
    parser.add_argument(
        "--dataset-roots",
        required=True,
        help=(
            "JSON file or inline object mapping cifar10, cinic10, and "
            "tiny-imagenet-200 to local dataset roots."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/main_backdoor_experiment.json",
    )
    parser.add_argument("--outdir", default="experiment_results_main_backdoor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-device", default="cuda")
    parser.add_argument("--client-device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--runtime-registration-timeout-s",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--runtime-shutdown-timeout-s",
        type=float,
        default=60.0,
    )
    parser.add_argument("--soft-deadline-s", type=float, default=0.0)
    parser.add_argument("--hard-deadline-s", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    roots = _load_json_or_inline(args.dataset_roots)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_datasets = set(config["datasets"])
    missing = required_datasets - set(roots)
    if missing:
        raise ValueError(f"dataset-roots is missing: {sorted(missing)}")

    jobs = list(product(
        config["datasets"],
        config["attacks"],
        config["methods"],
        config["seeds"],
    ))
    expected = 3 * 4 * 4 * 5
    if len(jobs) != expected:
        raise ValueError(
            f"Formal matrix must contain exactly {expected} jobs, found {len(jobs)}."
        )
    if int(args.limit) > 0:
        jobs = jobs[: int(args.limit)]

    root_out = Path(args.outdir)
    for index, (dataset, attack, method, seed) in enumerate(jobs, start=1):
        job_out = _job_dir(root_out, dataset, attack, method, int(seed))
        if args.resume and _summary_exists(job_out, dataset):
            print(f"[{index}/{len(jobs)}] skip complete {job_out}", flush=True)
            continue
        trace_path = None
        if str(config["runtime"]) == "process-semi-async" and not args.dry_run:
            trace_path = _ensure_trace(
                root=root_out,
                dataset=dataset,
                seed=int(seed),
                config=config,
            )
        elif str(config["runtime"]) == "process-semi-async":
            trace_path = root_out / "_shared_runtime_traces" / dataset / f"seed_{seed}.json"
        command = build_command(
            dataset_root=str(roots[dataset]),
            dataset=dataset,
            attack=attack,
            method=method,
            seed=int(seed),
            config=config,
            outdir=job_out,
            device=args.device,
            server_device=args.server_device,
            client_device=args.client_device,
            num_workers=int(args.num_workers),
            runtime_trace=trace_path,
            runtime_registration_timeout_s=float(
                args.runtime_registration_timeout_s
            ),
            runtime_shutdown_timeout_s=float(
                args.runtime_shutdown_timeout_s
            ),
            soft_deadline_s=float(args.soft_deadline_s),
            hard_deadline_s=float(args.hard_deadline_s),
        )
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            job_out.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
