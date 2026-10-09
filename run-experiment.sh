#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export CODE_DIR="$(pwd)"
export DATASET_ROOT="$(pwd)/dataset"
export OUT_ROOT="$(pwd)/results_seed2_arch_badnets_20261009"

mkdir -p "${OUT_ROOT}"
cd "${CODE_DIR}"

# 使用固定候选顺序和 seed=2 随机选择攻击；结果为 badnets。
ATTACK="$(
python3 - <<'PY'
import random
attacks = ["badnets", "blend", "dba", "dynamic"]
print(random.Random(2).choice(attacks))
PY
)"
export ATTACK
echo "Selected attack: ${ATTACK}"

# CUDA 与模型前向检查。
python3 - <<'PY'
import torch
from model_factory import build_model, model_parameter_count

assert torch.cuda.is_available(), "CUDA unavailable"

for architecture in ("resnet18", "mobilenet_v2"):
    model = build_model(
        architecture,
        dataset_name="cifar10",
        device="cuda",
    )
    inputs = torch.randn(2, 3, 32, 32, device="cuda")
    outputs = model(inputs)
    print(
        architecture,
        "parameters=", model_parameter_count(model),
        "output=", tuple(outputs.shape),
        "device=", outputs.device,
    )
PY

COMMON_ARGS=(
  --dataset "${DATASET_ROOT}"
  --dataset-name cifar10
  --rounds 50
  --epochs 1
  --batch-size 64
  --num-clients-list 20
  --seeds 2
  --partition-schemes dirichlet
  --dirichlet-alpha 0.5
  --proxy-dataset-size 256
  --private-dataset-size 2000
  --run-class smoke
  --runtime sync
  --attack "${ATTACK}"
  --attack-condition attacked
  --target-label 0
  --malicious-fraction 0.2
  --poison-ratio 0.2
  --attack-start-round 10
  --attack-end-round 30
  --poison-interval 1
  --trigger-size 4
  --trigger-value 1.0
  --enable-backdoor-diagnostics
  --device cuda
  --num-workers 0
  --client-torch-threads 1
  --auxiliary-num-workers 0
  --pin-memory
  --strict-numeric-checks
  --checkpoint-every-rounds 0
  --training-policy balanced
  --niabd-warmup-rounds 5
  --niabd-recovery-memory-lr 0.02
)

RESNET_DEF="${OUT_ROOT}/resnet18_vcaa_niabd"
RESNET_BASE="${OUT_ROOT}/resnet18_baseline"
MOBILE_DEF="${OUT_ROOT}/mobilenet_v2_vcaa_niabd"
MOBILE_BASE="${OUT_ROOT}/mobilenet_v2_baseline"

# 1. ResNet-18 + VCAA+NIABD，同时生成唯一攻击计划。
python3 -u experiment_runner.py \
  "${COMMON_ARGS[@]}" \
  --server-architecture resnet18 \
  --method vcaa-niabd \
  --outdir "${RESNET_DEF}" \
  2>&1 | tee "${OUT_ROOT}/resnet18_vcaa_niabd.log"

PLAN="${RESNET_DEF}/attack_plans/attack_plan_cifar10_seed_2_clients_20_dirichlet_${ATTACK}.json"
test -f "${PLAN}"
echo "Shared attack plan: ${PLAN}"

# 2. ResNet-18 baseline，复用完全相同的攻击计划。
python3 -u experiment_runner.py \
  "${COMMON_ARGS[@]}" \
  --server-architecture resnet18 \
  --method baseline \
  --attack-plan "${PLAN}" \
  --outdir "${RESNET_BASE}" \
  2>&1 | tee "${OUT_ROOT}/resnet18_baseline.log"

# 3. MobileNetV2 + VCAA+NIABD，复用同一个攻击计划。
python3 -u experiment_runner.py \
  "${COMMON_ARGS[@]}" \
  --server-architecture mobilenet_v2 \
  --method vcaa-niabd \
  --attack-plan "${PLAN}" \
  --outdir "${MOBILE_DEF}" \
  2>&1 | tee "${OUT_ROOT}/mobilenet_v2_vcaa_niabd.log"

# 4. MobileNetV2 baseline，复用同一个攻击计划。
python3 -u experiment_runner.py \
  "${COMMON_ARGS[@]}" \
  --server-architecture mobilenet_v2 \
  --method baseline \
  --attack-plan "${PLAN}" \
  --outdir "${MOBILE_BASE}" \
  2>&1 | tee "${OUT_ROOT}/mobilenet_v2_baseline.log"