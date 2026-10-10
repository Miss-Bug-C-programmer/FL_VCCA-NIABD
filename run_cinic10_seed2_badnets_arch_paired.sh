#!/usr/bin/env bash
set -euo pipefail

#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
#export CODE_DIR="${CODE_DIR:-/home/root/lxy/06-app/FL/FL_CODE}"
#export DATASET_ROOT="${DATASET_ROOT:-${CODE_DIR}/dataset/CINIC-10}"

export CUDA_VISIBLE_DEVICES=0
export CODE_DIR="$(pwd)"
export DATASET_ROOT="$(pwd)/dataset"

export ATTACK="${ATTACK:-badnets}"
case "${ATTACK}" in
  badnets|dba|blend|dynamic) ;;
  *)
    echo "Unsupported ATTACK=${ATTACK}; use badnets, dba, blend, or dynamic." >&2
    exit 2
    ;;
esac
export OUT_ROOT="$(pwd)/results_seed2_arch_badnets_cinic10"
#export OUT_ROOT="${OUT_ROOT:-/home/root/lxy/06-app/FL/results_seed2_arch_${ATTACK}_cinic10}"
# 0 uses all CINIC training samples left after the proxy/validation split.
# Set PRIVATE_DATASET_SIZE=2000 only for an exact repeat of the earlier small
# CIFAR training budget; that budget averages about 100 samples per client.
export PRIVATE_DATASET_SIZE="${PRIVATE_DATASET_SIZE:-0}"

if [[ -e "${OUT_ROOT}" ]]; then
  echo "Refusing to mix results into existing path: ${OUT_ROOT}" >&2
  echo "Set OUT_ROOT to a new empty path and rerun." >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
cd "${CODE_DIR}"

# Fail before training if the dataset, official normalization, trigger mapping,
# CUDA runtime, or either requested model architecture is not usable.
python3 - <<'PY'
import os
import torch

from attacks.trigger import apply_badnets
from data_utils import _load_dataset_pair
from dataset_metadata import dataset_normalization
from model_factory import build_model, model_parameter_count

root = os.environ["DATASET_ROOT"]
train, test = _load_dataset_pair(root, "cinic10")
assert len(train) == 90_000, f"expected 90000 CINIC train images, got {len(train)}"
assert len(test) == 90_000, f"expected 90000 CINIC test images, got {len(test)}"
assert train.class_to_idx == test.class_to_idx
assert len(train.classes) == 10

normalization = dataset_normalization("cinic10")
assert normalization.transform_identity == "tensor-normalize-cinic10-official-v1"
sample, _ = train[0]
assert tuple(sample.shape) == (3, 32, 32)
triggered = apply_badnets(
    sample.unsqueeze(0),
    size=4,
    value=1.0,
    dataset_name="cinic10",
)
mean = torch.tensor(normalization.mean).view(1, 3, 1, 1)
std = torch.tensor(normalization.std).view(1, 3, 1, 1)
raw_trigger = triggered * std + mean
assert torch.allclose(raw_trigger[:, :, 27:31, 27:31], torch.ones(1, 3, 4, 4), atol=1e-6)

assert torch.cuda.is_available(), "CUDA unavailable"
for architecture in ("resnet18", "mobilenet_v2"):
    model = build_model(architecture, dataset_name="cinic10", device="cuda")
    inputs = torch.randn(2, 3, 32, 32, device="cuda")
    with torch.no_grad():
        outputs = model(inputs)
    assert tuple(outputs.shape) == (2, 10)
    assert torch.isfinite(outputs).all()
    print(
        architecture,
        "parameters=", model_parameter_count(model),
        "output=", tuple(outputs.shape),
        "device=", outputs.device,
    )
PY

COMMON_ARGS=(
  --dataset "${DATASET_ROOT}"
  --dataset-name cinic10
  --rounds 50
  --epochs 1
  --batch-size 64
  --num-clients-list 20
  --seeds 2
  --partition-schemes dirichlet
  --dirichlet-alpha 0.5
  --proxy-dataset-size 256
  --private-dataset-size "${PRIVATE_DATASET_SIZE}"
  --run-class formal
  --runtime sync
  --attack "${ATTACK}"
  --attack-condition attacked
  --target-label 0
  --malicious-fraction 0.2
  --malicious-selection data-balanced
  --poison-ratio 0.2
  --malicious-local-epoch-multiplier 2
  --attack-start-round 10
  --attack-end-round 30
  --poison-interval 1
  --trigger-size 4
  --trigger-value 1.0
  --enable-backdoor-diagnostics
  --device cuda
  --num-workers 0
  --client-torch-threads 1
  --auxiliary-num-workers 4
  --pin-memory
  --strict-numeric-checks
  --checkpoint-every-rounds 0
  --training-policy balanced
  --server-distill-lr 0.01
  --server-distill-momentum 0.9
  --server-distill-epochs 5
  --client-learning-rate 0.02
  --client-momentum 0.9
  --client-weight-decay 0.0005
  --niabd-warmup-rounds 5
  --niabd-recovery-memory-lr 0.02
)

RESNET_BASE="${OUT_ROOT}/resnet18_baseline"
RESNET_DEF="${OUT_ROOT}/resnet18_vcaa_niabd"
MOBILE_BASE="${OUT_ROOT}/mobilenet_v2_baseline"
MOBILE_DEF="${OUT_ROOT}/mobilenet_v2_vcaa_niabd"

# Generate the sole attack plan in the first run.
python3 -u experiment_runner.py \
  "${COMMON_ARGS[@]}" \
  --server-architecture resnet18 \
  --method baseline \
  --outdir "${RESNET_BASE}" \
  2>&1 | tee "${OUT_ROOT}/resnet18_baseline.log"

PLAN="${RESNET_BASE}/attack_plans/attack_plan_cinic10_seed_2_clients_20_dirichlet_${ATTACK}.json"
test -f "${PLAN}"
grep -q '"dataset_name": "cinic10"' "${PLAN}"
echo "Shared attack plan: ${PLAN}"

python3 -u experiment_runner.py \
  "${COMMON_ARGS[@]}" \
  --server-architecture resnet18 \
  --method vcaa-niabd \
  --attack-plan "${PLAN}" \
  --outdir "${RESNET_DEF}" \
  2>&1 | tee "${OUT_ROOT}/resnet18_vcaa_niabd.log"

python3 -u experiment_runner.py \
  "${COMMON_ARGS[@]}" \
  --server-architecture mobilenet_v2 \
  --method baseline \
  --attack-plan "${PLAN}" \
  --outdir "${MOBILE_BASE}" \
  2>&1 | tee "${OUT_ROOT}/mobilenet_v2_baseline.log"

python3 -u experiment_runner.py \
  "${COMMON_ARGS[@]}" \
  --server-architecture mobilenet_v2 \
  --method vcaa-niabd \
  --attack-plan "${PLAN}" \
  --outdir "${MOBILE_DEF}" \
  2>&1 | tee "${OUT_ROOT}/mobilenet_v2_vcaa_niabd.log"

# Validate pairing and print attack/recovery summaries directly from CSV.
python3 - <<'PY'
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["OUT_ROOT"])
runs = {
    "resnet18_baseline": root / "resnet18_baseline",
    "resnet18_vcaa_niabd": root / "resnet18_vcaa_niabd",
    "mobilenet_v2_baseline": root / "mobilenet_v2_baseline",
    "mobilenet_v2_vcaa_niabd": root / "mobilenet_v2_vcaa_niabd",
}
plan_ids = set()
summary = []
for name, directory in runs.items():
    path = directory / "fedagg_experiment_results_cinic10.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 50, f"{name}: expected 50 rows, got {len(frame)}"
    assert set(frame["dataset"]) == {"cinic10"}
    assert set(frame["seed"].astype(int)) == {2}
    assert frame["numeric_failure_count"].fillna(0).eq(0).all()
    assert frame["transaction_status"].eq("committed").all()
    plan_ids.update(frame["attack_plan_id"].dropna().astype(str).unique())
    for phase, lo, hi in (("attack", 10, 30), ("recovery", 31, 50)):
        part = frame[frame["round"].between(lo, hi)]
        summary.append({
            "run": name,
            "phase": phase,
            "rounds": len(part),
            "ta_mean_pct": 100.0 * part["ta"].mean(),
            "aa_mean_pct": 100.0 * part["aa"].mean(),
            "aa_max_pct": 100.0 * part["aa"].max(),
            "clean_target_mean_pct": 100.0 * part["clean_target_rate"].mean(),
            "trigger_lift_mean_pp": 100.0 * part["trigger_lift"].mean(),
        })

assert len(plan_ids) == 1, f"runs do not share one attack plan: {plan_ids}"
summary_frame = pd.DataFrame(summary)
summary_path = root / "cinic10_phase_summary.csv"
summary_frame.to_csv(summary_path, index=False)
print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("attack_plan_id=", next(iter(plan_ids)))
print("summary=", summary_path)
PY
