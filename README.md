# FedAgg Server-Client with VCAA and NIABD

## NIABD v2 production semantics

The production implementation uses a proxy-conditioned class-response
prototype. For teacher logits `Z[k,p,c]`, the persistent memory is
`prototype_mean[p,c]` and `prototype_variance[p,c]`, both `[P,C]`; adaptive
`thresholds[c]` remains `[C]`. Proxy samples are never flattened together, so
the historical response for a fixed proxy query is preserved. `proxy_labels`
remains in the controller interface for compatibility, but NIABD does not read
its values; only Server-side VCAA may use the labels.

Eligibility is teacher-level and robust: anomaly fraction, the configured
`memory_quantile` deviation, mean excess, and current-consensus deviation are
robust-standardized and combined by their maximum. The legacy
`benign_deviation_limit` is a high-quantile history/consensus bound, not an
`amax(P,C)` bound. A single finite outlier therefore receives continuous
suppression without freezing every teacher. Memory and threshold updates use
the pre-purification state and only safe memory-eligible raw logits; freeze
rounds never absorb all teachers or potentiate thresholds.

New CLI controls are:

```text
--niabd-memory-quantile
--niabd-maximum-memory-anomaly-fraction
--niabd-teacher-score-beta
--niabd-teacher-score-scale-floor
--niabd-minimum-consensus-teachers
--niabd-consensus-recovery-fraction
--niabd-threshold-exposure-quantile
```

Results identify `niabd-v2-proxy-conditioned-robust-memory` and
`fedagg-results-v2`. Round and teacher-defense CSVs include the update reason,
robust teacher metrics, freeze streak, effective memory weight, eligible
observations, and memory-update rounds. Formal result collection is fail-closed
and keeps runtime, algorithm version, and result schema in its grouping key:

```powershell
python scripts\check_result_completeness.py `
  --indir experiment_results_main_backdoor `
  --config configs\main_backdoor_experiment.json
python scripts\collect_main_backdoor_results.py `
  --indir experiment_results_main_backdoor `
  --expected-runs 240
```

The complete CUDA/CoreX procedure is documented in
`GPU_VALIDATION_RUNBOOK.md`. CPU tests, synthetic logits tests, preservation
checks, and the 240-job dry-run are validation evidence only; formal CUDA
training remains unverified until executed on the target device.

当前分支采用服务器—客户端联邦蒸馏结构：

- 中心服务器维护 ResNet-18 全局学生模型。
- 多个客户端在本地私有数据上训练 ResNet-18 教师模型。
- 客户端使用真实本地模型对公共代理集执行推理，并上传序列化 logits。
- 服务器只接收 logits 字节载荷与公开版本元信息，不读取客户端模型。
- 服务器将教师 logits 转为温度软概率并聚合更新学生，可将学生 logits
  下发给客户端完成 FedAgg 风格的反向蒸馏。
- VCAA 作为独立的教师准入模块运行于服务器端，可按实验启用或关闭。
- NIABD 作为独立的 logits 净化模块运行于服务器端，也可单独开关。

## VCAA 与 NIABD 接入位置

每轮的真实知识交换顺序如下：

1. 每个客户端利用私有数据独立更新本地教师模型。
2. 客户端在公共代理集上运行真实模型，生成 logits 并编码为上传包。
3. 服务器验证查询编号、客户端编号、logits 形状和上传完整性。
4. 可选 VCAA 根据版本状态、准确率、熵和教师—学生 KL 散度执行准入。
5. 可选 NIABD 对已准入 logits 执行类别原型偏差检测和连续软净化。
6. 服务器对净化后的教师软概率求均值，在代理集上蒸馏学生模型。
7. 默认将更新后的学生 logits 序列化下发给客户端执行反向蒸馏。

可通过 `--disable-client-distillation` 关闭最后一步，用于单向联邦蒸馏
消融。无论是否启用 VCAA 或 NIABD，服务器都不会调用客户端模型。

NIABD 在服务器端维护逐类别 logit 均值、方差和自适应阈值。异常维度
不会导致整个教师被丢弃，而是按平滑兴奋—抑制权重拉回历史原型或学生
参考。仅满足良性偏差条件的教师能够更新原型记忆。

## 环境

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## 纯 FedAgg 基线

不提供 `--enable-vcaa` 和 `--enable-niabd` 即运行纯基线：

```powershell
python experiment_runner.py `
  --dataset D:\path\to\dataset `
  --dataset-name cifar10 `
  --rounds 5 `
  --epochs 1 `
  --batch-size 64 `
  --num-clients-list 6 `
  --seeds 0 `
  --partition-schemes iid `
  --outdir experiment_results_fedagg
```

## FedAgg + VCAA

```powershell
python experiment_runner.py `
  --dataset D:\path\to\dataset `
  --dataset-name cifar10 `
  --rounds 5 `
  --epochs 1 `
  --batch-size 64 `
  --num-clients-list 6 `
  --seeds 0 `
  --partition-schemes iid `
  --enable-vcaa `
  --vcaa-version-weight 0.5 `
  --vcaa-time-decay-gamma 0.99 `
  --vcaa-time-unit-s 60 `
  --vcaa-max-version-lag 1 `
  --vcaa-accuracy-weight 0.5 `
  --vcaa-entropy-weight 0.25 `
  --vcaa-divergence-weight 0.25 `
  --vcaa-window-rounds 5 `
  --vcaa-threshold-beta 1 `
  --vcaa-warmup-rounds 1 `
  --outdir experiment_results_vcaa
```

三个内容权重之和必须为 1。`--vcaa-entropy-scale 0` 表示自动使用
`log(类别数)` 作为熵归一化常数。

## FedAgg + NIABD

NIABD 可以不依赖 VCAA 单独运行：

```powershell
python experiment_runner.py `
  --dataset D:\path\to\dataset `
  --dataset-name cifar10 `
  --rounds 5 `
  --epochs 1 `
  --num-clients-list 6 `
  --enable-niabd `
  --niabd-initial-threshold 2.0 `
  --niabd-kappa 1.0 `
  --niabd-prototype-learning-rate 0.01 `
  --niabd-threshold-learning-rate 0.01 `
  --niabd-benign-deviation-limit 4.0 `
  --niabd-warmup-rounds 1 `
  --outdir experiment_results_niabd
```

## FedAgg + VCAA + NIABD

同时添加两个开关即可运行参考模板中的两阶段验证与安全净化：

```powershell
python experiment_runner.py `
  --dataset D:\path\to\dataset `
  --dataset-name cifar10 `
  --rounds 5 `
  --epochs 1 `
  --num-clients-list 6 `
  --enable-vcaa `
  --enable-niabd `
  --outdir experiment_results_vcaa_niabd
```

## 输出指标

逐轮结果除准确率、损失和耗时外，还包括：

- `teachers_admitted`、`teachers_rejected`
- `teacher_utilization`
- `client_upload_bytes`、`server_broadcast_bytes`
- `admission_threshold`、`admission_score_mean`
- `vcaa_version_score_mean`、`vcaa_content_score_mean`
- `vcaa_proxy_accuracy_mean`、`vcaa_entropy_mean`、`vcaa_kl_mean`
- `niabd_anomaly_fraction`、`niabd_mean_suppression`
- `niabd_threshold_mean`、`niabd_prototype_observations`
- `niabd_memory_eligible_teachers`

启用准入模块时，还会生成
`fedagg_teacher_admission_<dataset>.csv`，记录每轮每个客户端的总分、
准入结果及各评分分量，便于后续消融和方法对比。

启用 NIABD 时，还会生成
`fedagg_teacher_defense_<dataset>.csv`，记录每个教师的异常比例、
最大归一化偏差、平均抑制强度和原型更新资格。

汇总工具会按 VCAA 和 NIABD 两组开关及方法名称共同分组，避免四类
实验结果被错误合并。

## 验证

```powershell
python experiment_example.py
python -m pytest -q
```

## 进程级半异步运行时

`--runtime process-semi-async` 提供单主机、多进程的 Server–Client
Federated Distillation 运行时。每个 Client 是通过 `spawn` 创建并在整次
实验中持续存在的独立 OS process；Client 独立持有本地模型、私有数据
loader、优化器、随机状态以及 retry 状态。Server 不持有 Client model
引用，也不会调用 Client 的训练或推理函数。

Client 与 Server 的两个方向都使用 localhost TCP RPC：

- Client 通过 `GET_TASK` 获取由 Server 创建的 task lineage 和最新
  student logits；
- Client 完成真实反向蒸馏、本地训练和 proxy inference 后，通过
  `UPLOAD_KNOWLEDGE` 上传二进制 float32 logits；
- RPC 使用明确的 magic、协议版本、JSON header length 和 binary
  payload length，并验证消息上限、shape、dtype、有限数值和 SHA-256；
- ACK 超时或 application-level upload-attempt loss 后，Client 重试
  完全相同的 packet，不会重新训练或重新推理。

该运行时实现真实 process concurrency、tensor serialization、socket
transport、socket timeout/retry 和自然 late arrival。配置中的
`upload_attempt_drop` 是应用层可控的上传尝试丢弃，不是 kernel packet
loss、WAN packet loss 或真实 Internet 部署。

### 真实 freshness lineage

Server task registry 生成并验证：

- `task_id`
- `source_round`：Server 派发任务时的 round；
- `base_server_round`：任务携带的 student knowledge version，关闭反向
  蒸馏时等于任务 coordination round；
- `proxy_version`：由 dataset、固定 proxy indices/order 和 preprocessing
  identity 计算的 SHA-256；
- `local_model_version`：Client 成功完成本地更新的次数，仅用于审计。

Client 完成 proxy inference 后使用 `time.monotonic()` 记录
`generated_at_s`。Server 在完整 TCP packet 到达后记录 `received_at_s`，
在 coordinator 主线程将 packet 送入 VCAA 时记录 `consumed_at_s`。
因此：

```text
version_lag = consumed_round - source_round
knowledge_age_s = consumed_at_s - generated_at_s
```

late packet 不会被 RPC、task registry 或 mailbox 因版本较旧而拒绝；
它会进入现有 VCAA，由 version score 和 content score 联合决定准入，
随后按配置进入 NIABD、软概率聚合和 student distillation。

### Proxy label 隔离

主进程只进行一次确定性数据划分并生成 `FederatedDataPlan`。Client process
只根据自己的 private indices 构建本地 labeled loader；公共代理集在
Client 侧使用 `ProxyInputDataset`，只返回输入。VCAA 使用的 proxy labels
仅由 Server 的 labeled proxy loader 持有，不包含在 `ClientTask` 或 RPC
消息中。

### 运行示例

```powershell
python experiment_runner.py `
  --dataset dataset `
  --dataset-name cifar10 `
  --rounds 5 `
  --epochs 1 `
  --batch-size 64 `
  --num-clients-list 4 `
  --seeds 0 `
  --partition-schemes iid `
  --runtime process-semi-async `
  --runtime-profile configs/runtime_test_stale.json `
  --participation-rate 1.0 `
  --quorum-fraction 0.5 `
  --runtime-warmup-rounds 1 `
  --soft-deadline-factor 1.5 `
  --hard-deadline-factor 2.0 `
  --rpc-timeout-s 0.2 `
  --max-retries 4 `
  --retry-backoff-s 0.05 `
  --server-device cuda `
  --client-device cpu `
  --enable-vcaa `
  --enable-niabd `
  --outdir experiment_results_process
```

`--runtime sync` 是默认值，继续执行原有严格同步基线。进程运行时支持
baseline、VCAA-only、NIABD-only 和 VCAA+NIABD 四种策略组合。系统异构
条件来自预先生成或加载的 strategy-independent trace：

```text
configs/runtime_homogeneous.json
configs/runtime_moderate.json
configs/runtime_severe.json
configs/runtime_test_stale.json
```

可以使用 `--runtime-trace-out <path>` 保存 trace，并用
`--runtime-trace <file>` 在不同策略间重放相同的 selection、availability、
compute slowdown、upload delay、attempt drop 和 ACK delay。

### 半异步输出

除原有 round、summary、admission 和 defense CSV 外，进程运行时生成：

```text
fedagg_runtime_events_<dataset>.csv
```

每个真实 packet 一行，包含 task/packet ID、payload SHA-256、完整时间线、
source/base/receive/consume round、version lag、knowledge age、计算与通信
耗时、drop/timeout/retry/duplicate、payload/wire bytes、VCAA 分量和
NIABD 结果。未启用的算法指标或尚未被 coordinator 消费的 packet 使用
空值/NaN，不使用 `0` 冒充实际算法结果。

---

# 正式 NIABD 后门防御主实验扩展

本节是在上述 Server–Client Federated Distillation、VCAA、NIABD、TCP/RPC
以及 process-semi-async 运行时基础上的**增量扩展**。本扩展没有把系统改回
参数聚合 FL，也没有删除或绕过现有运行时。

源代码基线记录在 `BASELINE_MAIN_PRESERVATION.json`：来源为
`Miss-Bug-C-programmer/FL_VCCA-NIABD` 的 `main` 分支提交
`a9d9d58129ef460e85f9c70f424f8dcbd8494a70`。以下核心机制文件保持与该
提交**逐字节一致**：

```text
admission.py
defense.py
vcaa.py
niabd.py
federated_server.py
logits_transport.py
rpc_transport.py
round_coordinator.py
runtime_trace.py
models.py
numeric_integrity.py
```

可以本地执行：

```bash
python scripts/verify_preserved_main.py
```

该检查要求：原 main 快照中的全部文件仍然存在，并且上述 VCAA/NIABD、
Server 和协议关键文件的 SHA-256 与源提交一致。新增后门实验只在客户端本地
训练数据入口、数据集支持、实验编排、BASR 评估与结果记录层增加功能。

## 1. 正式主实验矩阵

主实验严格固定为：

```text
Datasets:
  CIFAR-10
  CINIC-10
  Tiny-ImageNet-200

Attacks:
  BadNets
  DBA
  Blend
  Dynamic

Methods:
  Baseline
  VCAA-only
  NIABD-only
  VCAA+NIABD

Seeds:
  0, 1, 2, 3, 4
```

因此主矩阵为：

```text
3 datasets × 4 attacks × 4 methods × 5 seeds = 240 runs
```

正式参数文件：

```text
configs/main_backdoor_experiment.json
```

当前默认正式设置为：

```text
clients                 20
malicious fraction      0.20
malicious clients       4
partition               class-wise Dirichlet
Dirichlet alpha         0.5
rounds                  50
local epochs            1
batch size              64
proxy ratio             0.10
validation ratio        0.10
distillation T          2.0
poison ratio            0.20
target label            0
NIABD warmup            5 rounds
attack start            round 15
attack interval         every round after attack start
main runtime            sync
```

主安全实验默认使用 `sync`，目的不是简化系统，而是首先隔离“攻击/防御”变量，
避免 staleness、timeout、drop 和 quorum 同时影响 BASR。原有
`process-semi-async` 没有删除，并且四类攻击也已接入该真实多进程 TCP/RPC
路径；完成安全主表后，可以将配置中的 `runtime` 改为
`process-semi-async`，继续做系统异构与攻击共同存在的第二阶段实验。

## 2. 数据集

### CIFAR-10

仍使用原来的 `torchvision.datasets.CIFAR10`。数据根目录应包含 torchvision
标准 CIFAR-10 文件。输入保持现有代码的归一化方式：

```text
ToTensor()
Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
```

### CINIC-10

新增 `ImageFolder` 支持，目录必须为：

```text
CINIC-10/
├── train/
│   ├── airplane/
│   ├── automobile/
│   └── ... 10 classes
├── valid/
└── test/
    ├── airplane/
    ├── automobile/
    └── ... 10 classes
```

联邦私有数据、proxy 和 validation 从 `train/` 进行互斥划分，正式测试使用
`test/`。加载器会检查 train/test 的类别映射一致，并检查训练集确实包含
10 个类别。

### Tiny-ImageNet-200

支持官方目录：

```text
tiny-imagenet-200/
├── train/
│   ├── n01443537/images/*.JPEG
│   └── ... 200 classes
└── val/
    ├── images/*.JPEG
    └── val_annotations.txt
```

训练集使用 `ImageFolder` 递归读取 `train/<wnid>/images`；validation 通过
`val_annotations.txt` 显式恢复标签，用作正式 clean test / triggered test。
代码要求训练集恰好包含 200 类。也兼容已经按类别重新整理后的 val 目录。

`model_factory.py` 新增：

```text
cinic10             -> 10 classes
tiny-imagenet-200   -> 200 classes
```

Server 和 Client 仍然全部使用原有 ResNet-18 构建路径，没有为攻击实验引入
测试专用模型。

## 3. 真正的 class-wise Dirichlet Non-IID

原 `quantity-skew` 中的 Dirichlet 只控制各客户端样本数量，并不等价于
常见 FL 论文中的 label-distribution Dirichlet Non-IID。

本扩展新增：

```bash
--partition-schemes dirichlet
--dirichlet-alpha 0.5
```

对于每个类别 `c`，独立采样：

```text
p_c ~ Dirichlet(alpha, ..., alpha)
```

再依据 `p_c` 把该类别的样本分给所有客户端。实现会检查分区没有丢失样本，
并继续沿用原代码的空客户端修复规则。

## 4. AttackPlan：策略无关的攻击真值

新增：

```text
attacks/config.py
attacks/attack_plan.py
```

每个 `(seed, num_clients, attack config)` 确定性生成一个 `AttackPlan`，包括：

```text
malicious_client_ids
DBA local-trigger assignment
target label
poison ratio
attack start/end round
poison interval
trigger parameters
```

正式 20-client、20% malicious 设置下，每个 run 恰好 4 个恶意客户端。
同一个 dataset / seed / attack 的 Baseline、VCAA-only、NIABD-only 和
VCAA+NIABD 使用相同的恶意客户端、相同攻击轮次和相同 trigger assignment。
每个运行会导出：

```text
<outdir>/attack_plans/attack_plan_*.json
```

也可以通过：

```bash
--attack-plan path/to/attack_plan.json
```

强制重放完全相同的 AttackPlan。配置或 seed/client 数不匹配时会直接报错，
不会静默重建。

**AttackPlan 中的恶意身份不传给 VCAA 或 NIABD。**

同步运行时中，它只用于选择对应客户端的本地 batch poisoner 和实验日志；
多进程运行时中，AttackPlan 在 process spawn 时作为客户端本地实验配置传入，
不会加入 `ClientLogitsPacket`、RPC payload、VCAA 输入或 NIABD 输入。

## 5. 四类真实客户端数据投毒

四类攻击都发生在：

```text
Client private labeled batch
        ↓
real image trigger / blend transform
        ↓
target-label replacement
        ↓
original local_train()
        ↓
real Client ResNet-18
        ↓
clean proxy inference
        ↓
serialized logits
        ↓
Server
```

没有使用：

```text
logits[:, target] += constant
伪造模型更新
parameter replacement
服务器直接读取客户端模型
```

`trainer.local_train()` 只新增了通用的可选 `batch_transform` 接口；不提供该
参数时执行路径与原代码一致。正式恶意客户端传入 `BackdoorBatchPoisoner`，
良性客户端仍使用原始训练行为。

### 5.1 BadNets

文件：

```text
attacks/trigger.py
attacks/poisoner.py
```

对被选中的非 target-class 本地样本，在图像右下角加入固定方形 patch，并将
标签改为 target label。

正式默认：

```text
CIFAR-10 / CINIC-10: 4×4 patch
Tiny-ImageNet-200:    8×8 patch
poison ratio:         20%
target label:         0
```

投毒只从 `label != target_label` 的样本中选择，原本属于 target class 的样本
不会用于后门 label replacement。

### 5.2 DBA

DBA 使用 4 个彼此分离的 local sub-trigger：

```text
local trigger 0     local trigger 1

local trigger 2     local trigger 3
```

正式 4 个 malicious clients 分别只看到一个 local trigger。
本地训练阶段不会给任一恶意客户端完整 global trigger。

评估阶段同时报告：

```text
BASR_global   # 四个 local trigger 的组合
BASR_local_1
BASR_local_2
BASR_local_3
BASR_local_4
```

这里没有照搬参数聚合 FL 中的 model-update scaling / model replacement；
当前系统的知识接口仍然是 proxy logits，所以 DBA 通过持续本地数据投毒形成
真实 backdoored teacher，再由 federated distillation 路径尝试传播至 student。

### 5.3 Blend

使用固定、确定性的 checker pattern：

```text
x_poison = (1-alpha) * x + alpha * pattern
```

正式默认：

```text
alpha = 0.2
```

pattern 和输入位于相同的 normalized `[-1, 1]` 空间，结果再次 clamp 到
`[-1,1]`。

### 5.4 Dynamic

Dynamic attack 不使用一个固定 trigger，而是根据 source round 周期性改变：

```text
trigger location
trigger intensity
trigger size scale
```

默认 `dynamic_period=10`。变化由 `(attack_start_round, round_number)` 完全
确定，因此同一 seed/round 的不同防御策略看到相同攻击状态。

这组实验主要用于检验 NIABD adaptive threshold 在长期、变化攻击下是否会
真正适应正常 drift，还是逐步把攻击“习惯化”。

## 6. 恶意样本选择的确定性

每个恶意客户端、每个 source round、每个 mini-batch 的 poisoned sample
indices 由下列信息共同决定：

```text
experiment seed
client_id
source_round
batch_index
```

因此四个 Methods 之间不会因为不同随机抽样而得到不同攻击强度。

每轮记录：

```text
eligible_poison_samples
poisoned_samples
poisoned_batches
```

process runtime 的这些值通过独立的 experiment-only multiprocessing queue
返回主进程，只用于记录，不写入 logits packet。如果该 side-channel 统计由于
异常没有到达，不会用 0 冒充，而是以 NaN / `attack_stats_missing` 标记。

## 7. NIABD 主链保持不变

后门扩展没有修改 `niabd.py`。

Server 仍然执行：

```text
receive serialized proxy logits
        ↓
optional VCAA
        ↓
optional NIABD
        ↓
mean soft probabilities
        ↓
student distillation
        ↓
optional reverse distillation
```

NIABD 仍然只使用上传 logits 和 student reference，继续维护原来的：

```text
per-output-dimension prototype mean
prototype variance
adaptive threshold
smooth excitation/inhibition weight
memory eligibility
```

后门实验中的 `is_malicious` 只在 NIABD 完成之后，与
`teacher_defense_records` 做离线 join，用于计算：

```text
malicious_mean_anomaly_fraction
benign_mean_anomaly_fraction
malicious_mean_suppression
benign_mean_suppression
malicious_memory_eligible_rate
benign_memory_eligible_rate
```

这些 ground-truth 字段不会参与 NIABD 的判断。

## 8. Triggered test 与 BASR

正式 test set 不参与任何本地训练、proxy inference、VCAA threshold、NIABD
prototype 或 hyper-parameter 更新。

BASR 仅在每轮 student 更新之后通过 test set 的只读 triggered view 计算。
设攻击目标为 `tau`，定义：

```text
BASR = count(f(T(x)) == tau, y != tau) / count(y != tau)
```

实现明确排除真实标签已经等于 target label 的测试样本，避免把正常 target-class
准确预测计入后门成功率。

对于 `attack=none`，BASR 字段为 NaN，而不是伪造 0。

## 9. Attack Viability Gate

不能只比较：

```text
Attack + NIABD
```

而不先证明攻击在无防御系统中能够真正传播。

正式证据链必须是：

```text
1. clean no-attack control
2. attacked Baseline
3. attacked NIABD-only
4. attacked VCAA-only
5. attacked VCAA+NIABD
```

只有 `attacked Baseline` 本身建立了显著后门 BASR，后续 BASR 下降才能归因于
防御。代码不硬编码诸如“BASR 必须 >80%”或“NIABD 必须 <3%”这样的通过阈值。
真实实验得到什么就记录什么。

主 240 组实验之外，提供 15 个 no-attack control：

```text
3 datasets × 5 seeds = 15 runs
```

运行：

```bash
python scripts/run_attack_viability_controls.py \
  --dataset-roots dataset_roots.json \
  --device cuda
```

随后：

```bash
python scripts/check_attack_viability.py
```

该脚本只输出 attacked Baseline 的真实 BASR、attack-window BASR、clean accuracy
变化和实际 poisoned sample 数，不伪造成功阈值。

## 10. 正式主矩阵运行

先准备：

```json
{
  "cifar10": "/data/cifar10",
  "cinic10": "/data/CINIC-10",
  "tiny-imagenet-200": "/data/tiny-imagenet-200"
}
```

保存为：

```text
dataset_roots.json
```

首先检查 240 个命令，不启动训练：

```bash
python scripts/run_main_backdoor_matrix.py \
  --dataset-roots dataset_roots.json \
  --dry-run
```

输出必须恰好 240 行。

正式执行：

```bash
python scripts/run_main_backdoor_matrix.py \
  --dataset-roots dataset_roots.json \
  --device cuda \
  --server-device cuda \
  --client-device cpu
```

支持断点续跑：

```bash
python scripts/run_main_backdoor_matrix.py \
  --dataset-roots dataset_roots.json \
  --resume \
  --device cuda \
  --server-device cuda \
  --client-device cpu
```

每个正式 job 使用独立目录：

```text
experiment_results_main_backdoor/
└── <dataset>/
    └── <attack>/
        └── <method>/
            └── seed_<seed>/
```

因此不会为了追加结果而覆盖其他策略，也不需要通过降低测试范围实现 resume。

## 11. 单个正式运行示例

例如 CIFAR-10 + DBA + NIABD-only：

```bash
python experiment_runner.py \
  --dataset /data/cifar10 \
  --dataset-name cifar10 \
  --method niabd \
  --attack dba \
  --target-label 0 \
  --malicious-fraction 0.2 \
  --poison-ratio 0.2 \
  --attack-start-round 15 \
  --attack-end-round 50 \
  --poison-interval 1 \
  --trigger-size 4 \
  --rounds 50 \
  --epochs 1 \
  --batch-size 64 \
  --num-clients-list 20 \
  --seeds 0 \
  --partition-schemes dirichlet \
  --dirichlet-alpha 0.5 \
  --proxy-ratio 0.1 \
  --val-ratio 0.1 \
  --niabd-warmup-rounds 5 \
  --device cuda \
  --outdir experiment_results_dba_niabd_seed0
```

`--method` 只是正式矩阵的便利别名。原来的：

```text
--enable-vcaa
--enable-niabd
```

继续完整支持；若 `--method` 与显式 flags 冲突，程序直接报错，避免实验标签与
真实算法开关不一致。

## 12. process-semi-async + attack

真实多进程路径同样支持四种攻击。

Client process 仍然：

```text
spawn
→ own local model
→ own private loader
→ own optimizer
→ optional reverse distillation
→ real poisoned/clean local training
→ clean proxy inference
→ ClientLogitsPacket
→ localhost TCP RPC
```

攻击不会绕过 packet serialization。

例如：

```bash
python experiment_runner.py \
  --dataset /data/cifar10 \
  --dataset-name cifar10 \
  --method vcaa-niabd \
  --attack badnets \
  --rounds 20 \
  --epochs 1 \
  --num-clients-list 20 \
  --seeds 0 \
  --partition-schemes dirichlet \
  --dirichlet-alpha 0.5 \
  --runtime process-semi-async \
  --runtime-profile configs/runtime_moderate.json \
  --server-device cuda \
  --client-device cpu \
  --outdir experiment_results_process_backdoor
```

在正式多策略 process 实验中，`scripts/run_main_backdoor_matrix.py` 会先生成
strategy-independent shared runtime trace，再把同一 dataset/seed trace 重放
给所有 attack/method 组合，因此 selection、availability、compute slowdown、
upload delay、attempt drop 和 ACK delay 不会因为防御策略变化。

真实数据上使用 CPU Client 时，本地 ResNet-18 一轮训练可能超过默认的 60 秒
首轮校准时间。增加 timeout 只会增加等待上限，不会加速 Client；尤其
`--client-torch-threads 1` 会让每个 Client 以单 CPU 线程训练完整私有分区。
若 warmup 已派发 Client、但 hard deadline 前一个 packet 都没有收到，运行时会
立即报出明确的 warmup `TimeoutError`，不会继续推进没有任何 Server update 的
空轮，也不会再额外等待一次完整 shutdown timeout。

完整数据 process 运行应根据容器可用 CPU 核数合理设置每个 Client 的线程数。
例如 4 个 Client、节点至少有 16 个可用 CPU 核时可先测试
`--client-torch-threads 4`。不得通过减少正式矩阵的 Client 数处理性能问题。
真实数据 smoke 可以使用单独标记的缩小配置验证程序路径，例如：

```bash
python experiment_runner.py \
  --dataset ./dataset \
  --dataset-name cifar10 \
  --method vcaa-niabd \
  --attack badnets \
  --target-label 0 \
  --malicious-fraction 0.25 \
  --poison-ratio 0.2 \
  --attack-start-round 1 \
  --attack-end-round 2 \
  --poison-interval 1 \
  --rounds 2 \
  --epochs 1 \
  --batch-size 64 \
  --num-clients-list 4 \
  --seeds 0 \
  --partition-schemes dirichlet \
  --dirichlet-alpha 0.5 \
  --proxy-ratio 0.1 \
  --val-ratio 0.89 \
  --proxy-dataset-size 64 \
  --runtime process-semi-async \
  --device cuda \
  --server-device cuda \
  --client-device cpu \
  --client-torch-threads 4 \
  --num-workers 0 \
  --runtime-registration-timeout-s 300 \
  --runtime-shutdown-timeout-s 300 \
  --outdir ./results/process_attack_smoke
```

这个命令仍使用真实 CIFAR 图像、真实 trigger、真实 ResNet-18、真实 TCP/RPC
和序列化 logits，但缩小数据划分只用于 smoke，不能替代正式配置或论文结果。
首轮成功返回 packet 后，后续相对 deadline 会由首轮真实 compute/transport
时间校准。若正式矩阵切换为 `process-semi-async`，矩阵入口支持统一传入：

```bash
python scripts/run_main_backdoor_matrix.py \
  --dataset-roots dataset_roots.json \
  --runtime-registration-timeout-s 900 \
  --runtime-shutdown-timeout-s 900
```

这些时间参数对矩阵中所有 Methods 完全相同；strategy-independent runtime
trace 的选择、可用性、slowdown、upload delay、drop 和 ACK delay 仍保持共享。

若 Client 也使用 CUDA，`--client-device cuda` 会规范化为当前容器可见的
`cuda:0`；也可以显式写 `--client-device cuda:0`。每个 Client 仍是独立进程并
独立持有模型，多个 Client process 只是共享同一可见加速卡，Server 仍只能通过
localhost TCP 接收序列化 logits。使用前必须确认 CoreX 支持多进程共享设备且
显存足以同时容纳 Server、所有 Client model、梯度和激活。若容器通过
`COREX_VISIBLE_DEVICES=0` 只暴露一张卡，则进程内唯一合法索引是 `cuda:0`，
不能使用宿主机物理编号代替容器内可见编号。

## 13. 新增输出

原有 CSV 全部保留：

```text
fedagg_experiment_results_<dataset>.csv
fedagg_run_summary_<dataset>.csv
fedagg_teacher_admission_<dataset>.csv
fedagg_teacher_defense_<dataset>.csv
fedagg_runtime_events_<dataset>.csv
```

新增：

```text
fedagg_backdoor_defense_<dataset>.csv
```

Round CSV 新增：

```text
attack_type
attack_plan_id
target_label
malicious_fraction
poison_ratio
attack_start_round
attack_active
poisoned_samples
eligible_poison_samples
attack_stats_missing_count
basr_global
basr_local_1 ... basr_local_4
malicious_mean_anomaly_fraction
benign_mean_anomaly_fraction
malicious_mean_suppression
benign_mean_suppression
malicious_memory_eligible_rate
benign_memory_eligible_rate
```

Backdoor-defense CSV 每轮/客户端一行，包括：

```text
client_id
is_malicious
attack_active
poisoned_samples
eligible_poison_samples
poisoned_batches
dba_trigger_part
admitted
admission_score
niabd_anomaly_fraction
niabd_mean_abs_deviation
niabd_max_abs_deviation
niabd_mean_suppression
niabd_memory_eligible
```

其中 `is_malicious` 是实验 ground truth，只在算法完成后用于日志关联。

Run summary 新增：

```text
final_basr_global
final_basr_local_1 ... final_basr_local_4
mean_attack_window_basr
total_poisoned_samples
total_attack_stats_missing
```

## 14. 汇总 5 seeds

240 组运行完成后：

```bash
python scripts/collect_main_backdoor_results.py
```

输出：

```text
experiment_results_main_backdoor/main_backdoor_mean_std.csv
```

按照：

```text
dataset × attack × strategy
```

报告：

```text
5-seed count
clean ACC mean ± std
final BASR mean ± std
attack-window BASR mean ± std
mean poisoned sample count
```

如果发现相同 dataset/attack/strategy/seed 重复结果，汇总脚本直接报错，不会
静默平均重复实验。

## 15. 本地完整验证

先安装原依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

然后执行：

```bash
python -m compileall -q .
python scripts/verify_preserved_main.py
python -m pytest -q
```

本代码包构造时使用上述命令验证，测试套件包含原 main 的全部测试，并增加：

```text
BadNets trigger / poisoning
DBA local/global trigger construction
Blend trigger
Dynamic round-varying trigger
AttackPlan determinism
BASR target-class exclusion
class-wise Dirichlet partition
CINIC-10 loader
Tiny-ImageNet official validation parser
sync serialized-logits attack path
process-semi-async TCP/RPC attack path
formal 240-job matrix cardinality
```

测试不是通过删除原测试、降低断言、缩短已有集成测试或增加 production 中的
“test-only”算法分支实现的。

正式数据上的 BASR/ACC 数值仍必须在目标机器上真实跑出。任何 smoke dataset
结果只用于验证程序路径，不得写成论文实验结果。

## 16. Codex 本地继续迭代

完整约束和执行步骤见：

```text
CODEX_LOCAL_ITERATION_PROMPT.md
```

Codex 必须在这个完整仓库中增量修改，禁止重建一个精简项目替换现有系统。
