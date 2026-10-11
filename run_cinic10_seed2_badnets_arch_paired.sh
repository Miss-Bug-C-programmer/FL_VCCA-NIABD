#!/usr/bin/env bash
set -euo pipefail

# Paired 50-round attack matrix for the current source tree.
# Defaults to the attacks not covered by the preceding BadNets experiment.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CODE_DIR="$(pwd)"
export DATASET_ROOT="$(pwd)/dataset"
#OUT_ROOT="${OUT_ROOT:-/home/root/lxy/06-app/FL/results_seed2_cifar10_mnist_other_attacks_20261011}"
export OUT_ROOT="$(pwd)/results_seed2_cifar10_mnist_other_attacks_20261011"
ATTACKS_TEXT="${ATTACKS:-dba blend dynamic}"

mkdir -p "${OUT_ROOT}"
cd "${CODE_DIR}"

python3 - <<'PY'
import torch
from model_factory import build_model

assert torch.cuda.is_available(), "CUDA unavailable"
for dataset_name, shape in (("cifar10", (2, 3, 32, 32)), ("mnist", (2, 1, 28, 28))):
    model = build_model("resnet18", dataset_name=dataset_name, device="cuda")
    output = model(torch.randn(*shape, device="cuda"))
    assert tuple(output.shape) == (2, 10)
    print(dataset_name, "forward", tuple(output.shape), torch.cuda.get_device_name(0))
PY

python3 - <<'PY'
from pathlib import Path
from torchvision.datasets import CIFAR10, MNIST
import os

root = Path(os.environ.get("DATASET_ROOT", "/home/root/lxy/06-app/FL/datasets"))
CIFAR10(root=str(root), train=True, download=False)
CIFAR10(root=str(root), train=False, download=False)
MNIST(root=str(root), train=True, download=False)
MNIST(root=str(root), train=False, download=False)
print("datasets ready:", root)
PY

run_complete() {
  local csv_path="$1"
  python3 - "${csv_path}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
with path.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
ok = (
    len(rows) == 50
    and [int(row["round"]) for row in rows] == list(range(1, 51))
    and all(row.get("transaction_status") == "committed" for row in rows)
    and sum(int(float(row.get("numeric_failure_count") or 0)) for row in rows) == 0
    and sum(int(float(row.get("nonfinite_distill_rollbacks") or 0)) for row in rows) == 0
)
raise SystemExit(0 if ok else 1)
PY
}

run_one() {
  local dataset_name="$1"
  local attack="$2"
  local method="$3"
  local plan_path="$4"
  local run_dir="${OUT_ROOT}/${dataset_name}/${attack}/${method}"
  local result_csv="${run_dir}/fedagg_experiment_results_${dataset_name}.csv"
  local log_path="${OUT_ROOT}/${dataset_name}_${attack}_${method}.log"

  if run_complete "${result_csv}"; then
    echo "[skip complete] ${dataset_name}/${attack}/${method}"
    return 0
  fi

  local plan_args=()
  if [[ -n "${plan_path}" ]]; then
    plan_args=(--attack-plan "${plan_path}")
  fi

  echo "[run] dataset=${dataset_name} attack=${attack} method=${method}"
  python3 -u experiment_runner.py \
    --dataset "${DATASET_ROOT}" \
    --dataset-name "${dataset_name}" \
    --rounds 50 \
    --epochs 1 \
    --batch-size 64 \
    --num-clients-list 20 \
    --seeds 2 \
    --partition-schemes dirichlet \
    --dirichlet-alpha 0.5 \
    --proxy-dataset-size 256 \
    --private-dataset-size 0 \
    --run-class formal \
    --runtime sync \
    --server-architecture resnet18 \
    --method "${method}" \
    --attack "${attack}" \
    --attack-condition attacked \
    --target-label 0 \
    --malicious-fraction 0.2 \
    --malicious-selection data-balanced \
    --poison-ratio 0.2 \
    --malicious-local-epoch-multiplier 2 \
    --attack-start-round 10 \
    --attack-end-round 30 \
    --poison-interval 1 \
    --trigger-size 4 \
    --trigger-value 1.0 \
    --blend-alpha 0.2 \
    --dynamic-period 10 \
    --enable-backdoor-diagnostics \
    --device cuda \
    --num-workers 0 \
    --client-torch-threads 1 \
    --auxiliary-num-workers 0 \
    --pin-memory \
    --strict-numeric-checks \
    --checkpoint-every-rounds 0 \
    --training-policy balanced \
    --server-distill-epochs 5 \
    --server-distill-lr 0.01 \
    --server-distill-momentum 0.9 \
    --client-learning-rate 0.02 \
    --client-momentum 0.9 \
    --client-weight-decay 0.0005 \
    --client-kd-weight 0.1 \
    --client-kd-warmup-updates 10 \
    --client-kd-ramp-updates 10 \
    --niabd-warmup-rounds 5 \
    --niabd-recovery-memory-lr 0.02 \
    "${plan_args[@]}" \
    --outdir "${run_dir}" \
    2>&1 | tee "${log_path}"

  run_complete "${result_csv}"
}

for dataset_name in cifar10 mnist; do
  for attack in ${ATTACKS_TEXT}; do
    defense_dir="${OUT_ROOT}/${dataset_name}/${attack}/vcaa-niabd"
    plan="${defense_dir}/attack_plans/attack_plan_${dataset_name}_seed_2_clients_20_dirichlet_${attack}.json"

    # Generate the plan once in the defense run, then reuse its exact bytes in
    # the baseline run so client selection and DBA assignments are paired.
    run_one "${dataset_name}" "${attack}" "vcaa-niabd" ""
    test -f "${plan}"
    run_one "${dataset_name}" "${attack}" "baseline" "${plan}"
  done
done

export MATRIX_OUT_ROOT="${OUT_ROOT}"
python3 - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["MATRIX_OUT_ROOT"])
rows = []
for dataset_name in ("cifar10", "mnist"):
    for attack in os.environ.get("ATTACKS", "dba blend dynamic").split():
        frames = {}
        for method in ("baseline", "vcaa-niabd"):
            path = root / dataset_name / attack / method / f"fedagg_experiment_results_{dataset_name}.csv"
            with path.open(newline="", encoding="utf-8-sig") as handle:
                frames[method] = list(csv.DictReader(handle))
        for phase, first, last in (("attack", 10, 30), ("recovery", 31, 50)):
            selected = {
                method: [row for row in frame if first <= int(row["round"]) <= last]
                for method, frame in frames.items()
            }
            def mean(method, key):
                values = [float(row[key]) for row in selected[method] if row.get(key) not in (None, "", "nan")]
                return sum(values) / len(values)
            rows.append({
                "dataset": dataset_name,
                "attack": attack,
                "phase": phase,
                "baseline_ta_pct": 100 * mean("baseline", "ta"),
                "defense_ta_pct": 100 * mean("vcaa-niabd", "ta"),
                "defense_minus_baseline_ta_pp": 100 * (mean("vcaa-niabd", "ta") - mean("baseline", "ta")),
                "baseline_aa_pct": 100 * mean("baseline", "aa"),
                "defense_aa_pct": 100 * mean("vcaa-niabd", "aa"),
                "baseline_minus_defense_aa_pp": 100 * (mean("baseline", "aa") - mean("vcaa-niabd", "aa")),
                "baseline_trigger_lift_pp": 100 * mean("baseline", "trigger_lift"),
                "defense_trigger_lift_pp": 100 * mean("vcaa-niabd", "trigger_lift"),
                "baseline_minus_defense_trigger_lift_pp": 100 * (mean("baseline", "trigger_lift") - mean("vcaa-niabd", "trigger_lift")),
            })

out = root / "paired_phase_summary.csv"
with out.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("[write]", out)
PY

