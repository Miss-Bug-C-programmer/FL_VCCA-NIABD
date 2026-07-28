# FedAgg Server-Client with VCAA and NIABD

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
