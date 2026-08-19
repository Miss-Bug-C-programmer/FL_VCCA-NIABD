#!/usr/bin/env bash
set -Eeuo pipefail

# CIFAR-10 formal matrix for one CUDA GPU.
#
# Mechanism ownership is explicit and controlled only by --method in this
# formal script:
#   baseline   -> VCAA OFF, NIABD OFF
#   vcaa       -> VCAA ON,  NIABD OFF
#   niabd      -> VCAA OFF, NIABD ON
#   vcaa-niabd -> VCAA ON,  NIABD ON
#
# Only methods with VCAA enabled may reject teacher knowledge at admission.
# NIABD is a post-admission purification mechanism and does not own admitted.
# Supplying VCAA_ARGS/NIABD_ARGS configures a mechanism but never enables it;
# the arrays are attached only to methods that actually enable that mechanism.
#
# Formal matrix: 4 attacks x 4 methods x 5 seeds = 80 runs.
# Default protocol: 80 rounds, 20 clients, one local epoch.
#
# IMPORTANT for process-semi-async:
# - The runtime launches 20 persistent client processes.
# - Each client owns its own model and uses CLIENT_DEVICE independently.
# - With one visible GPU, all 20 clients share cuda:0 plus the server model.
# - Therefore the formal default batch size remains 128 to preserve the old
#   experimental protocol. Use 256/512 only for a separately labeled new
#   protocol after a representative pilot.
# - Client DataLoader workers default to 0 because the 20 client processes
#   already provide parallelism; NUM_WORKERS=2 would create ~40 additional
#   DataLoader worker processes and is usually counterproductive for CIFAR-10.
#
# The repaired freshness path uses process-semi-async by default. Set
# RUNTIME=sync for the strict synchronous control. Runtime results are stored
# in distinct OUT_ROOT directories.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Support both layouts:
#   work/run.sh + work/FL_VCCA-NIABD/
# and
#   extracted-repo/run.sh + extracted-repo/experiment_runner.py
if [[ -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd)"
elif [[ -f "${SCRIPT_DIR}/FL_VCCA-NIABD/experiment_runner.py" ]]; then
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/FL_VCCA-NIABD" && pwd)"
elif [[ -f "${SCRIPT_DIR}/experiment_runner.py" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "ERROR: cannot locate FL_VCCA-NIABD. Set REPO_ROOT=/path/to/repo." >&2
    exit 1
fi
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

# Batch size is an experimental hyperparameter, not just a memory knob. Keep
# 128 for direct comparability with the old formal matrix. If you intentionally
# define a new protocol, override BATCH_SIZE=256 (or 512 after a pilot) and use
# a different RESULT_TAG/OUT_ROOT.
BATCH_SIZE="${BATCH_SIZE:-128}"
ATTACK_START_ROUND="${ATTACK_START_ROUND:-15}"
ATTACK_END_ROUND="${ATTACK_END_ROUND:-35}"

SERVER_DEVICE="${SERVER_DEVICE:-cuda:0}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cuda:0}"

# For process-semi-async, client_num_workers=0 is intentional: there are
# already NUM_CLIENTS independent client processes loading CIFAR-10.
NUM_WORKERS="${NUM_WORKERS:-0}"
AUXILIARY_NUM_WORKERS="${AUXILIARY_NUM_WORKERS:-2}"
CLIENT_TORCH_THREADS="${CLIENT_TORCH_THREADS:-1}"
LOADER_MP_CONTEXT="${LOADER_MP_CONTEXT:-none}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
AMP="${AMP:-1}"
AMP_MAX_CONSECUTIVE_OVERFLOWS="${AMP_MAX_CONSECUTIVE_OVERFLOWS:-8}"
STRICT_NUMERIC_CHECKS="${STRICT_NUMERIC_CHECKS:-1}"

PARTITION_SCHEME="${PARTITION_SCHEME:-dirichlet}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.5}"
PROXY_RATIO="${PROXY_RATIO:-0.1}"
VAL_RATIO="${VAL_RATIO:-0.1}"
PROXY_DATASET_SIZE="${PROXY_DATASET_SIZE:-0}"
PRIVATE_DATASET_SIZE="${PRIVATE_DATASET_SIZE:-0}"

RUNTIME_WARMUP_ROUNDS="${RUNTIME_WARMUP_ROUNDS:-2}"
PARTICIPATION_RATE="${PARTICIPATION_RATE:-1.0}"
QUORUM_FRACTION="${QUORUM_FRACTION:-0.5}"
SOFT_DEADLINE_FACTOR="${SOFT_DEADLINE_FACTOR:-1.5}"
HARD_DEADLINE_FACTOR="${HARD_DEADLINE_FACTOR:-2.0}"
RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-1.0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_S="${RETRY_BACKOFF_S:-0.05}"
REGISTRATION_TIMEOUT_S="${REGISTRATION_TIMEOUT_S:-180.0}"
SHUTDOWN_TIMEOUT_S="${SHUTDOWN_TIMEOUT_S:-180.0}"

# Oracle backdoor diagnostics remain sync-only. They are excluded from the
# semi-async formal path so they cannot perturb the traced scheduling path.
if [[ -z "${ENABLE_BACKDOOR_DIAGNOSTICS:-}" ]]; then
    if [[ "${RUNTIME}" == "sync" ]]; then
        ENABLE_BACKDOOR_DIAGNOSTICS="1"
    else
        ENABLE_BACKDOOR_DIAGNOSTICS="0"
    fi
fi

# RUN_LIMIT is only for a CUDA pilot. A partial matrix is never collected.
RUN_LIMIT="${RUN_LIMIT:-0}"
RESUME="${RESUME:-1}"

# Process-runtime checkpoint resume is intentionally unsupported by the
# current code because live client/coordinator state cannot yet be restored
# atomically. Keep this 0 for process-semi-async formal runs.
CHECKPOINT_EVERY_ROUNDS="${CHECKPOINT_EVERY_ROUNDS:-0}"

RESULT_TAG="${RESULT_TAG:-method_switch_admission_4090_48g}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiment_results_v3/cifar10_main_${RUNTIME}_${RESULT_TAG}}"

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
if (( NUM_CLIENTS <= 0 || ROUNDS <= 0 || EPOCHS <= 0 || BATCH_SIZE <= 0 )); then
    echo "ERROR: NUM_CLIENTS/ROUNDS/EPOCHS/BATCH_SIZE must be positive." >&2
    exit 2
fi
if (( ATTACK_START_ROUND < 1 || ATTACK_END_ROUND < ATTACK_START_ROUND || ATTACK_END_ROUND > ROUNDS )); then
    echo "ERROR: invalid attack window ${ATTACK_START_ROUND}-${ATTACK_END_ROUND} for ${ROUNDS} rounds." >&2
    exit 2
fi
if [[ "${RUNTIME}" == "process-semi-async" && "${CHECKPOINT_EVERY_ROUNDS}" != "0" ]]; then
    echo "ERROR: process-semi-async atomic resume is not implemented; set CHECKPOINT_EVERY_ROUNDS=0." >&2
    exit 2
fi

ATTACKS=(badnets dba blend dynamic)
METHODS=(baseline vcaa niabd vcaa-niabd)
SEEDS=(0 1 2 3 4)
EXPECTED_RUNS=$(( ${#ATTACKS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))
[[ "${EXPECTED_RUNS}" -eq 80 ]] || {
    echo "ERROR: formal matrix must contain exactly 80 runs, got ${EXPECTED_RUNS}." >&2
    exit 2
}
if (( RUN_LIMIT < 0 || RUN_LIMIT > EXPECTED_RUNS )); then
    echo "ERROR: RUN_LIMIT must be between 0 and ${EXPECTED_RUNS}." >&2
    exit 2
fi

[[ -f "experiment_runner.py" ]] || {
    echo "ERROR: missing experiment_runner.py under ${REPO_ROOT}" >&2
    exit 1
}
[[ -d "${CIFAR10}" ]] || {
    echo "ERROR: missing dataset: ${CIFAR10}" >&2
    exit 1
}
if [[ "${RUNTIME}" == "process-semi-async" ]]; then
    [[ -f "${RUNTIME_PROFILE}" ]] || {
        echo "ERROR: missing runtime profile: ${RUNTIME_PROFILE}" >&2
        exit 1
    }
fi

"${PYTHON}" - <<'PY'
from method_switches import resolve_method_switches

expected = {
    "baseline": (False, False),
    "vcaa": (True, False),
    "niabd": (False, True),
    "vcaa-niabd": (True, True),
}
for method, flags in expected.items():
    switches = resolve_method_switches(method)
    actual = (switches.enable_vcaa, switches.enable_niabd)
    if actual != flags:
        raise SystemExit(
            f"ERROR: method switch mapping {method} -> {actual}, expected {flags}"
        )
print("Method-switch preflight: PASS")
PY

echo "============================================================"
echo "FL_VCCA-NIABD CUDA pre-flight"
echo "============================================================"
echo "Repository             : ${REPO_ROOT}"
echo "Python                 : ${PYTHON}"
echo "Dataset                : ${CIFAR10}"
echo "CUDA_VISIBLE_DEVICES   : ${CUDA_VISIBLE_DEVICES}"
echo "Runtime                : ${RUNTIME}"
echo "Runtime profile        : ${RUNTIME_PROFILE}"
echo "Clients / rounds       : ${NUM_CLIENTS} / ${ROUNDS}"
echo "Attack window          : ${ATTACK_START_ROUND} / ${ATTACK_END_ROUND}"
echo "Batch / epochs         : ${BATCH_SIZE} / ${EPOCHS}"
echo "Client loader workers  : ${NUM_WORKERS} per client"
echo "Server loader workers  : ${AUXILIARY_NUM_WORKERS}"
echo "Server device          : ${SERVER_DEVICE}"
echo "Client device          : ${CLIENT_DEVICE}"
echo "AMP                    : ${AMP}"
echo "Output                 : ${OUT_ROOT}"
echo "Expected runs          : ${EXPECTED_RUNS}"
echo

"${PYTHON}" - <<'PY'
import sys
import torch

print("Python             :", sys.version.replace("\n", " "))
print("PyTorch            :", torch.__version__)
print("CUDA available     :", torch.cuda.is_available())
print("torch.version.cuda :", torch.version.cuda)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: no CUDA device is available.")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(
        f"GPU {index}             : {props.name} "
        f"({props.total_memory / 1024**3:.2f} GB, cc {props.major}.{props.minor})"
    )
props = torch.cuda.get_device_properties(0)
name = props.name
memory_gb = props.total_memory / 1024**3
if "4090" not in name:
    print("WARNING: visible GPU is not identified as RTX 4090:", name)
if memory_gb < 40.0:
    print(
        "WARNING: visible GPU has <40 GB memory; the configured BATCH_SIZE may need "
        "to be reduced for 20 concurrent CUDA clients."
    )
probe = torch.randn((1024, 1024), device="cuda", dtype=torch.float16, requires_grad=True)
probe.square().mean().backward()
torch.cuda.synchronize()
del probe
torch.cuda.empty_cache()
print("CUDA allocation/gradient probe: PASS")
PY

"${PYTHON}" -m compileall -q .
echo "Python compile check: PASS"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Repository HEAD       : $(git rev-parse HEAD)"
    echo "Repository dirty state:"
    git status --short
else
    echo "Repository HEAD       : unavailable (source ZIP without .git metadata)"
fi

# Explicit VCAA configuration. These values are configuration only; VCAA is
# enabled solely by --method vcaa / --method vcaa-niabd.
VCAA_ARGS=(
    --vcaa-version-weight 0.5
    --vcaa-time-decay-gamma 0.99
    --vcaa-time-unit-s 60.0
    --vcaa-max-version-lag 1
    --vcaa-version-lag-half-life-rounds 1.0
    --vcaa-max-knowledge-age-s 0.0
    --vcaa-age-half-life-s 0.0
    --vcaa-age-scale-mode runtime-calibrated
    --vcaa-runtime-age-reference-multiplier 4.0
    --vcaa-runtime-age-half-life-floor-s 0.5
    --vcaa-runtime-age-half-life-ceiling-s 60.0
    --vcaa-runtime-max-age-multiplier 4.0
    --vcaa-content-threshold-beta -1.0
    --vcaa-consensus-divergence-scale 0.0
    --vcaa-content-scale-floor 0.05
    --vcaa-reliability-temperature 1.0
    --vcaa-reliability-z-cap 6.0
    --vcaa-minimum-content-cohort-size 3
    --vcaa-minimum-content-history-size 3
    --vcaa-accuracy-weight 0.5
    --vcaa-entropy-weight 0.25
    --vcaa-divergence-weight 0.25
    --vcaa-accuracy-scale 1.0
    --vcaa-entropy-scale 0.0
    --vcaa-divergence-scale 1.0
    --vcaa-window-rounds 5
    --vcaa-threshold-beta 1.0
    --vcaa-warmup-rounds 1
)

# Explicit NIABD configuration. These values are configuration only; NIABD is
# enabled solely by --method niabd / --method vcaa-niabd.
NIABD_ARGS=(
    --niabd-initial-threshold 2.0
    --niabd-min-threshold 0.5
    --niabd-max-threshold 6.0
    --niabd-kappa 1.0
    --niabd-prototype-learning-rate 0.01
    --niabd-threshold-learning-rate 0.01
    --niabd-potentiation-balance 0.5
    --niabd-threshold-decay 0.01
    --niabd-benign-deviation-limit 4.0
    --niabd-warmup-rounds 5
    --niabd-min-standard-deviation 0.1
    --niabd-reference-source prototype
    --niabd-memory-quantile 0.95
    --niabd-maximum-memory-anomaly-fraction 0.10
    --niabd-teacher-score-beta 3.0
    --niabd-teacher-score-scale-floor 0.001
    --niabd-teacher-score-effective-floor 0.05
    --niabd-teacher-score-z-cap 12.0
    --niabd-consensus-purification-threshold 1.5
    --niabd-minimum-consensus-teachers 4
    --niabd-consensus-recovery-fraction 0.75
    --niabd-threshold-exposure-quantile 0.75
    --niabd-proxy-chunk-size 0
    --niabd-risk-ema-beta 0.30
    --niabd-risk-on 1.25
    --niabd-risk-off 0.60
    --niabd-onset-patience 2
    --niabd-recovery-patience 2
    --niabd-stable-patience 2
    --niabd-memory-clip-z 3.0
    --niabd-reference-clip-z 2.0
    --niabd-normal-memory-lr 0.0
    --niabd-suspicious-memory-lr 0.0
    --niabd-recovery-memory-lr 0.20
    --niabd-clean-ce-weight-normal 0.05
    --niabd-clean-ce-weight-suspicious 0.10
    --niabd-clean-ce-weight-recovery 0.20
    --niabd-threshold-upward-step-limit 0.05
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
    --attack-start-round "${ATTACK_START_ROUND}"
    --attack-end-round "${ATTACK_END_ROUND}"
    --poison-interval 1
    --trigger-size 4
    --blend-alpha 0.2
    --dynamic-period 10
    --server-architecture resnet18
    --aggregation-rule mean-soft-probabilities
    --clean-ce-weight 0.05
    --device "${SERVER_DEVICE}"
    --server-device "${SERVER_DEVICE}"
    --client-device "${CLIENT_DEVICE}"
    --runtime "${RUNTIME}"
    --num-workers "${NUM_WORKERS}"
    --auxiliary-num-workers "${AUXILIARY_NUM_WORKERS}"
    --client-torch-threads "${CLIENT_TORCH_THREADS}"
    --loader-mp-context "${LOADER_MP_CONTEXT}"
    --amp-max-consecutive-overflows "${AMP_MAX_CONSECUTIVE_OVERFLOWS}"
    --run-class formal
    --attack-condition attacked
)

if [[ "${RUNTIME}" == "process-semi-async" ]]; then
    COMMON_ARGS+=(
        --runtime-profile "${RUNTIME_PROFILE}"
        --runtime-warmup-rounds "${RUNTIME_WARMUP_ROUNDS}"
        --participation-rate "${PARTICIPATION_RATE}"
        --quorum-fraction "${QUORUM_FRACTION}"
        --soft-deadline-factor "${SOFT_DEADLINE_FACTOR}"
        --hard-deadline-factor "${HARD_DEADLINE_FACTOR}"
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
        "${PYTHON}" - "${trace_path}" "${RUNTIME_PROFILE}" "${seed}" \
            "${NUM_CLIENTS}" "${ROUNDS}" "${RUNTIME_WARMUP_ROUNDS}" \
            "${PARTICIPATION_RATE}" <<'PY'
import hashlib
import json
import sys
from runtime_trace import RuntimeTrace, load_runtime_profile

(
    trace_path,
    profile_path,
    seed,
    clients,
    rounds,
    warmup,
    participation_rate,
) = sys.argv[1:]

trace = RuntimeTrace.load(trace_path)
profile = load_runtime_profile(profile_path)
expected = (int(seed), int(clients), int(rounds), int(warmup))
actual = (
    int(trace.seed),
    int(trace.num_clients),
    int(trace.rounds),
    int(trace.warmup_rounds),
)
if actual != expected:
    raise SystemExit(f"Existing trace dimensions {actual} != {expected}")
if str(trace.profile_name) != str(profile.get("name", "runtime-profile")):
    raise SystemExit(
        f"Existing trace profile {trace.profile_name!r} does not match selected profile"
    )
with open(trace_path, "r", encoding="utf-8") as handle:
    raw = json.load(handle)
metadata = raw.get("run_metadata", {})
expected_profile_sha = hashlib.sha256(open(profile_path, "rb").read()).hexdigest()
if metadata.get("runtime_profile_sha256") != expected_profile_sha:
    raise SystemExit(
        "Existing runtime trace was generated from a different runtime-profile file. "
        "Use a new OUT_ROOT or delete the stale shared trace."
    )
actual_pr = float(metadata.get("participation_rate", -1.0))
if abs(actual_pr - float(participation_rate)) > 1e-12:
    raise SystemExit(
        f"Existing trace participation_rate={actual_pr} != {participation_rate}"
    )
print(f"reuse runtime trace: {trace_path}")
PY
        return
    fi

    "${PYTHON}" - "${trace_path}" "${RUNTIME_PROFILE}" "${seed}" \
        "${NUM_CLIENTS}" "${ROUNDS}" "${RUNTIME_WARMUP_ROUNDS}" \
        "${PARTICIPATION_RATE}" <<'PY'
import hashlib
import sys
from runtime_trace import generate_runtime_trace, load_runtime_profile

(
    trace_path,
    profile_path,
    seed,
    clients,
    rounds,
    warmup,
    participation_rate,
) = sys.argv[1:]

profile = load_runtime_profile(profile_path)
profile_sha = hashlib.sha256(open(profile_path, "rb").read()).hexdigest()
trace = generate_runtime_trace(
    profile=profile,
    seed=int(seed),
    num_clients=int(clients),
    rounds=int(rounds),
    warmup_rounds=int(warmup),
    participation_rate=float(participation_rate),
)
trace.save(
    trace_path,
    metadata={
        "purpose": "cifar10-4090-48g-formal-shared-runtime-trace",
        "runtime_profile_sha256": profile_sha,
        "participation_rate": float(participation_rate),
    },
)
print(f"write runtime trace: {trace_path}")
PY
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
                    if (( RUN_LIMIT > 0 && RUN_INDEX >= RUN_LIMIT )); then
                        break 3
                    fi
                    continue
                fi
                echo "Existing result is incomplete/invalid; rerunning from round 1."
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

            # Keep mechanism configuration and mechanism enablement aligned.
            # No --enable-vcaa/--enable-niabd flags are used here; --method is
            # the single source of truth for the formal matrix.
            case "${method}" in
                baseline)
                    ;;
                vcaa)
                    command+=("${VCAA_ARGS[@]}")
                    ;;
                niabd)
                    command+=("${NIABD_ARGS[@]}")
                    ;;
                vcaa-niabd)
                    command+=("${VCAA_ARGS[@]}" "${NIABD_ARGS[@]}")
                    ;;
                *)
                    echo "ERROR: unsupported method ${method}" >&2
                    exit 2
                    ;;
            esac

            if [[ "${RUNTIME}" == "process-semi-async" ]]; then
                command+=(
                    --runtime-trace "${TRACE_ROOT}/seed_${seed}.json"
                    --runtime-trace-out "${run_out}/runtime_trace"
                )
            fi

            "${command[@]}"
            "${PYTHON}" scripts/validate_v3_results.py --indir "${run_out}"
            echo "RUN ${RUN_INDEX}: PASS"

            if (( RUN_LIMIT > 0 && RUN_INDEX >= RUN_LIMIT )); then
                break 3
            fi
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
