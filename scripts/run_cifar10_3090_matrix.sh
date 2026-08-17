#!/usr/bin/env bash
set -Eeuo pipefail

# CIFAR-10 formal matrix for one CUDA GPU.
#
# Default matrix: 4 attacks x 4 methods x 5 seeds = 80 runs.
# Default protocol: 80 rounds, 20 clients, one local epoch.
#
# The repaired freshness path uses process-semi-async by default.  Set
# RUNTIME=sync for a strict synchronous control.  The two runtime results
# must be stored under different OUT_ROOT values.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
CIFAR10="${CIFAR10:-${REPO_ROOT}/dataset}"
GPU="${GPU:-0}"

RUNTIME="${RUNTIME:-process-semi-async}"
if [[ -z "${RUNTIME_PROFILE:-}" ]]; then
    if [[ "${RUNTIME}" == "process-semi-async" ]]; then
        RUNTIME_PROFILE="configs/runtime_severe.json"
    else
        RUNTIME_PROFILE="configs/runtime_moderate.json"
    fi
fi

NUM_CLIENTS="${NUM_CLIENTS:-20}"
ROUNDS="${ROUNDS:-80}"
EPOCHS="${EPOCHS:-1}"
ATTACK_END_ROUND="${ATTACK_END_ROUND:-35}"

SERVER_DEVICE="${SERVER_DEVICE:-cuda:0}"
# A single 24 GB GPU cannot hold the Server plus 20 independent CUDA Client
# models/optimizers/activations.  Keep the Server on CUDA and run the
# persistent private-data Clients on CPU by default.  This preserves the
# Server--Client process/RPC/logits boundary and avoids multiplying one GPU's
# memory allocation by NUM_CLIENTS.  Users with a genuinely multi-device
# allocation may override CLIENT_DEVICE explicitly.
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AUXILIARY_NUM_WORKERS="${AUXILIARY_NUM_WORKERS:-0}"
CLIENT_TORCH_THREADS="${CLIENT_TORCH_THREADS:-1}"
PIN_MEMORY="${PIN_MEMORY:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
AMP="${AMP:-1}"
STRICT_NUMERIC_CHECKS="${STRICT_NUMERIC_CHECKS:-1}"
AMP_MAX_CONSECUTIVE_OVERFLOWS="${AMP_MAX_CONSECUTIVE_OVERFLOWS:-8}"

# With one visible GPU, process-semi-async keeps 20 persistent Client
# processes alive.  Their local training is CPU-backed by default, so the
# original CUDA-oriented batch=512 can exhaust host RAM when all clients
# train concurrently.  Keep 512 for sync or an explicitly selected value;
# use a conservative process-runtime default only when the caller did not
# provide BATCH_SIZE.
if [[ -z "${BATCH_SIZE+x}" ]]; then
    if [[ "${RUNTIME}" == "process-semi-async" && "${CLIENT_DEVICE}" == "cpu" ]]; then
        BATCH_SIZE="128"
    else
        BATCH_SIZE="512"
    fi
fi

PARTITION_SCHEME="${PARTITION_SCHEME:-dirichlet}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.5}"
PROXY_RATIO="${PROXY_RATIO:-0.1}"
VAL_RATIO="${VAL_RATIO:-0.1}"
PROXY_DATASET_SIZE="${PROXY_DATASET_SIZE:-0}"
PRIVATE_DATASET_SIZE="${PRIVATE_DATASET_SIZE:-0}"

RUNTIME_WARMUP_ROUNDS="${RUNTIME_WARMUP_ROUNDS:-2}"
PARTICIPATION_RATE="${PARTICIPATION_RATE:-1.0}"
QUORUM_FRACTION="${QUORUM_FRACTION:-0.5}"
RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-1.0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_S="${RETRY_BACKOFF_S:-0.05}"
REGISTRATION_TIMEOUT_S="${REGISTRATION_TIMEOUT_S:-180.0}"
SHUTDOWN_TIMEOUT_S="${SHUTDOWN_TIMEOUT_S:-180.0}"

# The oracle backdoor diagnostic is sync-only in the current runner.  It is
# deliberately excluded from the semi-async formal path so it cannot alter
# the traced scheduling path.
if [[ -z "${ENABLE_BACKDOOR_DIAGNOSTICS:-}" ]]; then
    if [[ "${RUNTIME}" == "sync" ]]; then
        ENABLE_BACKDOOR_DIAGNOSTICS="1"
    else
        ENABLE_BACKDOOR_DIAGNOSTICS="0"
    fi
fi

# RUN_LIMIT is for a CUDA pilot.  A partial matrix is never collected.
RUN_LIMIT="${RUN_LIMIT:-0}"
RESUME="${RESUME:-1}"
CHECKPOINT_EVERY_ROUNDS="${CHECKPOINT_EVERY_ROUNDS:-0}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiment_results_v3/cifar10_main_${RUNTIME}}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

if [[ "${RUNTIME}" != "sync" && "${RUNTIME}" != "process-semi-async" ]]; then
    echo "ERROR: RUNTIME must be sync or process-semi-async: ${RUNTIME}" >&2
    exit 2
fi
if [[ "${RUNTIME}" == "process-semi-async" && "${ENABLE_BACKDOOR_DIAGNOSTICS}" == "1" ]]; then
    echo "ERROR: backdoor diagnostics are sync-only in this runner." >&2
    exit 2
fi
if [[ "${RUNTIME}" == "process-semi-async" \
    && "${CLIENT_DEVICE}" == cuda* \
    && "${NUM_CLIENTS}" -gt 1 \
    && "${ALLOW_SHARED_CUDA_CLIENTS:-0}" != "1" ]]; then
    cat >&2 <<EOF
ERROR: process-semi-async is configured with ${NUM_CLIENTS} CUDA Client
processes on the shared device ${CLIENT_DEVICE}.  Each Client owns an
independent model, optimizer, and activation buffers; this commonly exhausts
a single 24 GB GPU before round 1.  Use the safe single-GPU setting:

  CLIENT_DEVICE=cpu ./scripts/run_cifar10_3090_matrix.sh

Set ALLOW_SHARED_CUDA_CLIENTS=1 only when the visible CUDA devices and memory
have been deliberately provisioned for this configuration.
EOF
    exit 2
fi
if [[ "${RUNTIME}" == "process-semi-async" \
    && "${CLIENT_DEVICE}" == "cpu" \
    && "${BATCH_SIZE}" -gt 256 ]]; then
    echo "WARNING: ${NUM_CLIENTS} concurrent CPU Clients with batch size ${BATCH_SIZE} may exhaust host RAM." >&2
    echo "         Prefer BATCH_SIZE=128 (or 64) unless host memory has been verified." >&2
fi
if (( RUN_LIMIT < 0 || RUN_LIMIT > 80 )); then
    echo "ERROR: RUN_LIMIT must be between 0 and 80." >&2
    exit 2
fi

echo "============================================================"
echo "FL_VCCA-NIABD CUDA pre-flight"
echo "============================================================"
echo "Repository           : ${REPO_ROOT}"
echo "Python               : ${PYTHON}"
echo "Dataset              : ${CIFAR10}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "Runtime              : ${RUNTIME}"
echo "Runtime profile      : ${RUNTIME_PROFILE}"
echo "Clients / rounds     : ${NUM_CLIENTS} / ${ROUNDS}"
echo "Attack window        : 15 / ${ATTACK_END_ROUND}"
echo "Batch / epochs       : ${BATCH_SIZE} / ${EPOCHS}"
echo "Server device        : ${SERVER_DEVICE}"
echo "Client device        : ${CLIENT_DEVICE}"
echo "AMP overflow limit   : ${AMP_MAX_CONSECUTIVE_OVERFLOWS} consecutive steps"
echo "Output               : ${OUT_ROOT}"
echo

[[ -d "${CIFAR10}" ]] || { echo "ERROR: missing dataset: ${CIFAR10}" >&2; exit 1; }
[[ -f "${RUNTIME_PROFILE}" ]] || { echo "ERROR: missing runtime profile: ${RUNTIME_PROFILE}" >&2; exit 1; }

"${PYTHON}" -c '
import sys
import torch

print("Python             :", sys.version.replace("\\n", " "))
print("PyTorch            :", torch.__version__)
print("CUDA available     :", torch.cuda.is_available())
print("torch.version.cuda :", torch.version.cuda)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: no CUDA device is available.")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"GPU {index}             : {props.name} ({props.total_memory / 1024**3:.2f} GB)")
name = torch.cuda.get_device_name(0)
if "3090" not in name:
    print("WARNING: visible GPU is not identified as RTX 3090:", name)
probe = torch.randn((8, 8), device="cuda", requires_grad=True)
probe.square().mean().backward()
torch.cuda.synchronize()
del probe
torch.cuda.empty_cache()
print("CUDA allocation/gradient probe: PASS")
'

"${PYTHON}" -m compileall -q .
echo "Python compile check: PASS"
echo "Repository HEAD: $(git rev-parse HEAD)"
git status --short

ATTACKS=(badnets dba blend dynamic)
METHODS=(baseline vcaa niabd vcaa-niabd)
SEEDS=(0 1 2 3 4)
EXPECTED_RUNS=$(( ${#ATTACKS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))
[[ "${EXPECTED_RUNS}" -eq 80 ]] || { echo "ERROR: matrix is not 80 runs." >&2; exit 2; }

# Explicit VCAA settings are recorded in each manifest and make the repaired
# semantics independent of parser defaults.
VCAA_ARGS=(
    --vcaa-version-weight 0.5
    --vcaa-time-decay-gamma 0.99
    --vcaa-time-unit-s 60.0
    --vcaa-max-version-lag 1
    --vcaa-version-lag-half-life-rounds 1.0
    --vcaa-age-scale-mode runtime-calibrated
    --vcaa-runtime-age-reference-multiplier 4.0
    --vcaa-runtime-age-half-life-floor-s 0.5
    --vcaa-runtime-age-half-life-ceiling-s 60.0
    --vcaa-runtime-max-age-multiplier 4.0
    --vcaa-content-scale-floor 0.05
    --vcaa-reliability-temperature 1.0
    --vcaa-reliability-z-cap 6.0
    --vcaa-minimum-content-cohort-size 3
    --vcaa-minimum-content-history-size 3
    --vcaa-accuracy-weight 0.5
    --vcaa-entropy-weight 0.25
    --vcaa-divergence-weight 0.25
    --vcaa-accuracy-scale 1.0
    --vcaa-divergence-scale 1.0
    --vcaa-window-rounds 5
    --vcaa-threshold-beta 1.0
    --vcaa-warmup-rounds 1
)

COMMON_ARGS=(
    --dataset "${CIFAR10}"
    --dataset-name cifar10
    --rounds "${ROUNDS}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-clients-list "${NUM_CLIENTS}"
    --partition-schemes "${PARTITION_SCHEME}"
    --dirichlet-alpha "${DIRICHLET_ALPHA}"
    --proxy-ratio "${PROXY_RATIO}"
    --val-ratio "${VAL_RATIO}"
    --proxy-dataset-size "${PROXY_DATASET_SIZE}"
    --private-dataset-size "${PRIVATE_DATASET_SIZE}"
    --distill-temperature 2.0
    --malicious-fraction 0.2
    --poison-ratio 0.2
    --target-label 0
    --attack-start-round 15
    --attack-end-round "${ATTACK_END_ROUND}"
    --poison-interval 1
    --trigger-size 4
    --blend-alpha 0.2
    --dynamic-period 10
    --server-architecture resnet18
    --aggregation-rule mean-soft-probabilities
    --device "${SERVER_DEVICE}"
    --server-device "${SERVER_DEVICE}"
    --client-device "${CLIENT_DEVICE}"
    --runtime "${RUNTIME}"
    --num-workers "${NUM_WORKERS}"
    --auxiliary-num-workers "${AUXILIARY_NUM_WORKERS}"
    --client-torch-threads "${CLIENT_TORCH_THREADS}"
    --amp-max-consecutive-overflows "${AMP_MAX_CONSECUTIVE_OVERFLOWS}"
    --run-class formal
    --attack-condition attacked
    --niabd-warmup-rounds 5
    "${VCAA_ARGS[@]}"
)

if [[ "${RUNTIME}" == "process-semi-async" ]]; then
    COMMON_ARGS+=(
        --runtime-profile "${RUNTIME_PROFILE}"
        --runtime-warmup-rounds "${RUNTIME_WARMUP_ROUNDS}"
        --participation-rate "${PARTICIPATION_RATE}"
        --quorum-fraction "${QUORUM_FRACTION}"
        --rpc-timeout-s "${RPC_TIMEOUT_S}"
        --max-retries "${MAX_RETRIES}"
        --retry-backoff-s "${RETRY_BACKOFF_S}"
        --runtime-registration-timeout-s "${REGISTRATION_TIMEOUT_S}"
        --runtime-shutdown-timeout-s "${SHUTDOWN_TIMEOUT_S}"
    )
fi
[[ "${PIN_MEMORY}" == "1" ]] && COMMON_ARGS+=(--pin-memory)
[[ "${PERSISTENT_WORKERS}" == "1" ]] && COMMON_ARGS+=(--persistent-workers)
[[ "${AMP}" == "1" ]] && COMMON_ARGS+=(--amp)
[[ "${STRICT_NUMERIC_CHECKS}" == "1" ]] && COMMON_ARGS+=(--strict-numeric-checks)
[[ "${ENABLE_BACKDOOR_DIAGNOSTICS}" == "1" ]] && COMMON_ARGS+=(--enable-backdoor-diagnostics)

TRACE_ROOT="${OUT_ROOT}/_shared_runtime_traces/cifar10"

ensure_trace() {
    local trace_path="$1"
    local seed="$2"
    mkdir -p "$(dirname -- "${trace_path}")"
    if [[ -f "${trace_path}" ]]; then
        "${PYTHON}" -c '
import sys
from runtime_trace import RuntimeTrace, load_runtime_profile
trace = RuntimeTrace.load(sys.argv[1])
profile = load_runtime_profile(sys.argv[2])
expected = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
actual = (int(trace.seed), int(trace.num_clients), int(trace.rounds))
if actual != expected:
    raise SystemExit(f"Existing trace dimensions {actual} != {expected}")
if str(trace.profile_name) != str(profile.get("name", "runtime-profile")):
    raise SystemExit(f"Existing trace profile {trace.profile_name!r} does not match the selected profile")
print(f"reuse runtime trace: {sys.argv[1]}")
' "${trace_path}" "${RUNTIME_PROFILE}" "${seed}" "${NUM_CLIENTS}" "${ROUNDS}"
        return
    fi
    "${PYTHON}" -c '
import sys
from runtime_trace import generate_runtime_trace, load_runtime_profile
trace_path, profile_path, seed, clients, rounds, warmup = sys.argv[1:]
profile = load_runtime_profile(profile_path)
trace = generate_runtime_trace(
    profile=profile,
    seed=int(seed),
    num_clients=int(clients),
    rounds=int(rounds),
    warmup_rounds=int(warmup),
    participation_rate=1.0,
)
trace.save(trace_path, metadata={"purpose": "cifar10-3090-formal-shared-runtime-trace"})
print(f"write runtime trace: {trace_path}")
' "${trace_path}" "${RUNTIME_PROFILE}" "${seed}" "${NUM_CLIENTS}" "${ROUNDS}" "${RUNTIME_WARMUP_ROUNDS}"
}

if [[ "${RUNTIME}" == "process-semi-async" ]]; then
    for seed in "${SEEDS[@]}"; do
        ensure_trace "${TRACE_ROOT}/seed_${seed}.json" "${seed}"
    done
fi

mkdir -p "${OUT_ROOT}"
RUN_INDEX=0
for attack in "${ATTACKS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN_INDEX=$((RUN_INDEX + 1))
            run_out="${OUT_ROOT}/cifar10/${attack}/${method}/seed_${seed}"
            summary="${run_out}/fedagg_run_summary_cifar10.csv"
            echo
            echo "============================================================"
            echo "RUN ${RUN_INDEX}/${EXPECTED_RUNS}: attack=${attack} method=${method} seed=${seed}"
            echo "runtime=${RUNTIME} output=${run_out}"
            echo "============================================================"

            if [[ "${RESUME}" == "1" && -f "${summary}" ]]; then
                if "${PYTHON}" scripts/validate_v3_results.py --indir "${run_out}"; then
                    echo "RUN ${RUN_INDEX}: existing validated result, skip"
                    if (( RUN_LIMIT > 0 && RUN_INDEX >= RUN_LIMIT )); then break 3; fi
                    continue
                fi
                echo "Existing result is incomplete/invalid; rerunning."
            fi

            mkdir -p "${run_out}/checkpoints"
            command=(
                "${PYTHON}" experiment_runner.py
                "${COMMON_ARGS[@]}"
                --attack "${attack}"
                --method "${method}"
                --seeds "${seed}"
                --outdir "${run_out}"
                --checkpoint-every-rounds "${CHECKPOINT_EVERY_ROUNDS}"
                --checkpoint-dir "${run_out}/checkpoints"
            )
            if [[ "${RUNTIME}" == "process-semi-async" ]]; then
                command+=(
                    --runtime-trace "${TRACE_ROOT}/seed_${seed}.json"
                    --runtime-trace-out "${run_out}/runtime_trace"
                )
            fi
            "${command[@]}"
            "${PYTHON}" scripts/validate_v3_results.py --indir "${run_out}"
            echo "RUN ${RUN_INDEX}: PASS"
            if (( RUN_LIMIT > 0 && RUN_INDEX >= RUN_LIMIT )); then break 3; fi
        done
    done
done

if (( RUN_LIMIT > 0 )); then
    echo "Partial matrix complete: ${RUN_INDEX}/${EXPECTED_RUNS}; collection skipped."
    exit 0
fi

"${PYTHON}" scripts/collect_main_backdoor_results.py \
    --indir "${OUT_ROOT}" \
    --out "${OUT_ROOT}/cifar10_main_backdoor_summary.csv" \
    --expected-runs "${EXPECTED_RUNS}"

echo "FORMAL MATRIX COMPLETE"
echo "Output root: ${OUT_ROOT}"
echo "Summary    : ${OUT_ROOT}/cifar10_main_backdoor_summary.csv"
