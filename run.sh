#!/usr/bin/env bash
set -euo pipefail

# Formal supplementary matrix for the VCAA v4 / NIABD recovery code.
# Override machine-specific values from the environment, for example:
#   DATASET_ROOT=/data/FedAgg/dataset DEVICE=cuda:0 ./run.sh

PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-./dataset}"
DATASET_NAME="${DATASET_NAME:-cifar10}"
DEVICE="${DEVICE:-cuda:0}"
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-./experiment_results_v3/cifar10_main_supplement}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_ARGS=(
  --dataset "${DATASET_ROOT}"
  --dataset-name "${DATASET_NAME}"
  --rounds 80
  --epochs 1
  --batch-size "${BATCH_SIZE}"
  --num-clients-list 20
  --partition-schemes dirichlet
  --dirichlet-alpha 0.5
  --proxy-ratio 0.1
  --val-ratio 0.1
  --distill-temperature 2.0

  --malicious-fraction 0.2
  --poison-ratio 0.2
  --target-label 0
  --attack-start-round 15
  --attack-end-round 35
  --poison-interval 1
  --trigger-size 4
  --blend-alpha 0.2
  --dynamic-period 10

  --server-architecture resnet18
  --aggregation-rule mean-soft-probabilities

  --device "${DEVICE}"
  --runtime sync
  --num-workers 4
  --auxiliary-num-workers 4
  --pin-memory
  --persistent-workers
  --amp
  --enable-backdoor-diagnostics
  --run-class formal
  --attack-condition attacked
  --checkpoint-every-rounds 0

  # VCAA v4 robust relative Stage-B calibration.
  --vcaa-content-scale-floor 0.05
  --vcaa-reliability-temperature 1.0
  --vcaa-reliability-z-cap 6.0
  --vcaa-minimum-content-cohort-size 3
)

ATTACKS=(badnets dba blend dynamic)
# Baseline results already exist in experiment_results_v3; this supplementary
# matrix intentionally runs only the three modified strategies.
METHODS=(vcaa niabd vcaa-niabd)
SEEDS=(0 1 2 3 4)

for attack in "${ATTACKS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_out="${OUT_ROOT}/${attack}/${method}/seed_${seed}"
      mkdir -p "${run_out}/checkpoints"

      echo "Running attack=${attack}, method=${method}, seed=${seed}"
      "${PYTHON}" experiment_runner.py \
        "${COMMON_ARGS[@]}" \
        --attack "${attack}" \
        --method "${method}" \
        --seeds "${seed}" \
        --outdir "${run_out}" \
        --checkpoint-dir "${run_out}/checkpoints"

      "${PYTHON}" scripts/validate_v3_results.py \
        --indir "${run_out}"
    done
  done
done

"${PYTHON}" scripts/collect_main_backdoor_results.py \
  --indir "${OUT_ROOT}" \
  --out "${OUT_ROOT}/${DATASET_NAME}_main_backdoor_summary.csv" \
  --expected-runs 60
