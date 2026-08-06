# NIABD Core Fix Report

本报告记录本次 NIABD 生产修复的审计基线、数学定义、调用链、不变量和
验证结果。未在真实 CUDA/CoreX 设备上执行的项目均标记为“未验证”；CPU、
合成数据、静态检查和 dry-run 不用于推断正式 CUDA 实验成功。

## 1. 修改前审计基线

1. 修改前 git HEAD：`78d4a7b1afbb628fd62bdbef3f479e761c291b58`。
2. 修改前工作区状态：`?? results.zip`；该文件为已有未提交用户文件，本次不修改。
   `git diff --stat` 与 `git diff` 均为空。
3. 参考模板实际路径：
   `D:\research\my_papers\fedagg_rgate_icpp\papers\参考模板.docx`，文件存在并已读取文本和公式对象。
4. 参考模板引言同时包含分层 HFD 的后续草稿叙述和服务器直接协调客户端的 FD 叙述。
   本工程按引言确定的最高优先级 Server–Client FD 设计实现，不把后续 HFD 参数聚合草稿迁移进来。

## 2. 系统设计与调用图

当前生产链为：

```text
Client private labeled data
  -> FederatedClient local teacher training
  -> proxy inference
  -> ClientLogitsPacket serialized proxy logits
  -> FederatedServer receive_client_uploads
  -> optional VersionContentAwareAdmission (VCAA)
  -> optional NIABD logits purification
  -> mean soft probabilities
  -> server student distillation
  -> optional ServerLogitsPacket reverse distillation to Clients
```

同步入口为 `experiment_runner.py -> federated_runtime.py -> FederatedServer`；
process-semi-async 入口为 `experiment_runner.py -> process_runtime.py ->
RoundCoordinator/Client OS processes -> FederatedServer`。两条运行时均通过
`FederatedServer.apply_defense` 调用同一个 `KnowledgeDefenseController.purify`
接口，不维护第二套 NIABD 语义。

Server 只接收序列化 proxy logits 和协议元信息，不读取、保存或调用 Client
模型的参数、梯度、`forward`、`train` 或本地优化函数。`federated_server.py` 只从
Server proxy loader 收集 `proxy_labels`；标签不会进入 Client task、RPC 请求/响应、
packet metadata 或 binary payload。`ClientLogitsPacket`、`ServerLogitsPacket`、SHA-256、
shape/dtype/finite 校验、retry、timeout、duplicate、late packet、source/consume round、
version lag 和 knowledge age 保持在现有 transport/coordinator 链中。

不实现模板后续云—边—端参数聚合，是因为该内容描述新增边缘服务器、边缘教师、
FedAvg 和模型参数上传，与当前工程的 Server–Client FD 观测边界冲突；迁移它会把
prediction-level knowledge interface 改成 parameter-level FL，也会破坏异构 Client
模型和现有 process TCP/RPC 协议。因此术语固定为：模板云中心对应唯一 Server，
模板边缘教师对应 Client 本地教师，模板教师上传对应序列化 proxy logits，模板安全
蒸馏对应 Server 聚合净化 logits 并训练学生。

## 3. 参考模板公式审计

### 3.1 VCAA

模板第 4 节的公式与当前 `vcaa.py` 逐项一致：

| 模板项 | 当前代码映射 |
| --- | --- |
| `S_ver = gamma^Delta_tau * I(r >= R_min)` | `_version_scores` 的 wall-clock decay、median version floor 与 `max_version_lag` |
| `Q_acc` | `_content_statistics` 中 teacher probability top-1 与 Server proxy labels 的比较 |
| `H` | teacher probability entropy |
| `D_KL(p_student || p_teacher)` | student probability 为左侧、teacher probability 为右侧 |
| `S_con = alpha1*acc + alpha2*exp(-H/H0) + alpha3*exp(-KL/D0)` | `content_score` |
| `Psi = lambda*S_ver + (1-lambda)*S_con` | `score` |
| `theta_adm = mean(history) - beta*std(history)` | `_historical_threshold` |
| warmup/history window | `warmup_rounds` 与 bounded deque |

VCAA 继续只承担知识有效性和时效性准入，不承担后门安全保证，不访问攻击真值。
没有发现足以授权修改 `vcaa.py` 的实现偏差，因此本次不修改该文件。

### 3.2 当前 NIABD（修复前）

当前接口输入为 `teacher_knowledge`, `student_logits`, `proxy_labels`,
`current_round`，输出为 purified `TeacherKnowledge`、每教师记录和 metrics。

修复前维度和公式为：

```text
prototype_mean:     [C]
prototype_variance: [C]
thresholds:         [C]
flattened = Z.reshape(K*P, C)
mu = mean(flattened, dim=0)
v = var(flattened, dim=0)
delta = (Z - mu[None,None,:]) / (sqrt(v)[None,None,:] + epsilon)
weight = exp(-relu(abs(delta)-theta)^2 / (2*kappa^2))
purified = weight*Z + (1-weight)*reference
teacher_max_deviation = abs(delta).amax(dim=(1,2))
eligible = teacher_max_deviation < benign_deviation_limit
threshold exposure = mean(abs(delta), dim=(0,1))
```

这会混合不同 proxy 样本语义，并因单个有限极值拒绝整位教师；阈值 exposure 也会
被所有教师影响。报告中所称原型均为“固定代理查询条件下的类别响应原型”，不是
攻击样本原型、目标类原型或 Client 身份原型。

## 4. 已确认失败现象与根因

1. `amax(P,C)` 把一个正常教师的单个极值扩展成整位教师拒绝条件，解释了大 proxy
   集下 memory eligibility 归零、prototype observations 停在约 `500000` 的现象。
2. `reshape(-1, C)` 将所有教师和 proxy 样本压成一个 `[C]` 向量，抹除同一 proxy
   输入的历史响应语义，解释了 clean proxy 上的 NIABD ACC 早于攻击开始就退化。
3. threshold 使用所有教师、所有 proxy 元素的普通均值，允许异常教师参与阈值
   potentiation。
4. VCAA 当前是质量/时效准入，不是后门检测器；准入集合中的恶意比例不被假设为
   原始 20%，也不把攻击真值加入生产判定。
5. 中期 BASR 不能由 final BASR 代表；正式结果需包含 attack-window mean、peak、
   AUC、peak round 和 post-attack recovery。Blend 必须用 triggered-no-poison control
   评估 attack established/weak/not established。

## 5. 拟实施的 NIABD v2 数学定义

对固定 proxy 顺序、`Z[k,p,c]` 和同一输入的 `S[p,c]`，持久状态为：

```text
mu[p,c]       : [P,C]
variance[p,c] : [P,C]
theta[c]      : [C]
```

初始化沿教师维度做 median/MAD：

```text
center = median_k Z[k,p,c]
scale  = max(1.4826*median_k abs(Z[k,p,c]-center), minimum_standard_deviation)
variance = scale^2
```

post-warmup 首先读取上一轮状态：

```text
D_hist[k,p,c] = abs(Z[k,p,c]-mu_prev[p,c])/(sqrt(v_prev[p,c])+epsilon)
G[p,c]        = median_k Z[k,p,c]
R[p,c]        = max(1.4826*MAD_k(Z[k,p,c]), minimum_standard_deviation)
D_cons[k,p,c] = abs(Z[k,p,c]-G[p,c])/(R[p,c]+epsilon)
```

教师级 eligibility 使用 anomaly fraction、`q_memory` high quantile、mean excess
和 consensus quantile，并对每个指标先做 median/MAD/IQR robust-z，再取四者最大值：

```text
teacher_memory_score[k] = max(upper_robust_z(anomaly_fraction),
                               upper_robust_z(high_quantile_deviation),
                               upper_robust_z(mean_excess),
                               upper_robust_z(consensus_deviation))
eligible[k] iff score <= beta
  and high_quantile_deviation <= benign_deviation_limit
  and anomaly_fraction <= maximum_memory_anomaly_fraction
  and consensus_deviation <= benign_deviation_limit
```

`benign_deviation_limit` 的新语义是高分位历史/共识偏差上界，不是整套 proxy 集
最大值上界。默认新增参数为 `q_memory=0.95`、`maximum_memory_anomaly_fraction=0.10`、
`teacher_score_beta=3.0`、`teacher_score_scale_floor=1e-3`、
`minimum_consensus_teachers=4`、`consensus_recovery_fraction=0.75`、
`threshold_exposure_quantile=0.75`。

净化使用更新前状态：

```text
excess = relu(D_hist - theta_prev[c])
weight = exp(-excess^2/(2*transition_smoothness^2))
reference = mu_prev (prototype) or S (student)
Z_purified = weight*Z + (1-weight)*reference
```

只有足够安全的 memory-eligible 原始 logits 才更新 memory。设其稳健中心和方差为
`eligible_center`、`eligible_variance`，则：

```text
mu_new = (1-eta)*mu_prev + eta*eligible_center
v_new  = (1-eta)*(v_prev + (mu_prev-mu_new)^2)
        + eta*(eligible_variance + (eligible_center-mu_new)^2)
v_new  = max(v_new, minimum_standard_deviation^2)
```

阈值只使用 eligible teachers 的 `threshold_exposure_quantile` 暴露：

```text
exposure[c] = quantile_{eligible k,p}(D_hist[k,p,c], 0.75)
delta[c] = potentiation_balance*(exposure[c]-theta_prev[c])  if exposure > theta_prev
         = -(1-potentiation_balance)*threshold_decay            otherwise
theta_new = clamp(theta_prev + threshold_learning_rate*delta,
                  minimum_threshold, maximum_threshold)
```

### 每轮更新顺序与状态机

1. 严格校验 `[P,C]` shape、`P>0`、`C>1`、round、finite 和 memory shape；memory
   shape 变化必须显式 `reset()`，不自动 reshape/broadcast/truncate/reset。
2. warmup 未初始化时使用稳健 center/scale；教师数不足则 pass-through，记录
   `freeze_insufficient_teachers`，下一轮重试。
3. post-warmup 用上一轮 memory/threshold 计算历史偏差、当前共识、教师指标、评分和
   eligibility。
4. 用上一轮状态完成连续软净化。
5. 在正常 eligible 数量满足安全要求时更新 memory；否则仅当 `K >= minimum_consensus_teachers`、
   至少 `max(minimum_consensus_teachers, ceil(consensus_recovery_fraction*K))` 个候选
   形成紧凑共识时记录 `consensus_drift_update`；否则冻结并记录
   `freeze_no_safe_candidate` 或精确 shape/nonfinite reason。
6. 仅在本轮实际 memory update 后增加 observation；threshold 更新在净化之后，且
   freeze/all-ineligible 轮不得提高 threshold。

全体不 eligible 时不吸收全部教师、不污染 prototype、variance、threshold，增加
`consecutive_frozen_rounds`；下一轮重新计算，输入恢复后允许安全更新。该 drift 恢复
依赖准入集合仍有足够比例一致良性教师的假设，不声称抵抗任意多数恶意场景；无法
安全判定时冻结。

## 6. 威胁模型、安全假设、复杂度和不变量

攻击者可控制本地投毒数据和本地教师行为，但不能让 Server 读取 Client 参数。生产
防御不使用 malicious ID、attack type、trigger/target label、poison count、BASR 或
triggered test 数据；这些只在防御完成后的离线诊断 join 中使用。NIABD 不读取
`proxy_labels` 数值，接口参数仅为兼容保留，并由测试验证 labels 变化不影响输出。

持久状态复杂度为 `O(P*C)` prototype/variance 与 `O(C)` thresholds；不保存全部历史
teacher logits。单轮时间复杂度 `O(K*P*C)`。实现保持有限数量的 round-local 临时张量；
若启用 chunk，必须与非 chunk 结果做等价测试。同步和 process runtime 使用同一 NIABD
对象语义；未消费 packet 不进入 NIABD，late packet 只在 coordinator consume 后进入 VCAA/NIABD。

Observation 继续是 teacher-proxy observations：
`prototype_observations += eligible_teacher_count * proxy_sample_count`。freeze、retry、
duplicate packet 和 mailbox 重复查看不增加 observations；新增
`niabd_eligible_teacher_observations` 与 `niabd_memory_update_rounds`。

## 7. 已实施修改与测试覆盖

已修改：`niabd.py`、`defense.py`、`experiment_runner.py`、`federated_runtime.py`、
`process_runtime.py`、结果收集/完整性脚本、README、本报告、
`AUTHORIZED_CORE_FIXES.json`、`GPU_VALIDATION_RUNBOOK.md`，以及现有 NIABD/运行时/输出
回归测试和新的 NIABD v2 测试。

原则上不修改 `vcaa.py`、`federated_server.py`、transport/coordinator、models、
numeric integrity、attacks 和 formal JSON；若接口完整性确需修改，必须在授权清单和
最终报告中精确记录 hash、原因、复现命令和测试。

测试覆盖：旧极值漏洞复现、`20x5000x10` 良性 warmup、proxy-size invariance、16+4
异常教师、多数 drift、少数异常簇、样本语义、student reference、proxy-label
independence、threshold poisoning、all-ineligible recovery、sync/process 联合链、
duplicate/late packet、CSV schema、preservation negative tests、tiny-model TCP/RPC
和 240-job dry-run。

## 8. 实际验证记录

以下记录使用工作区中的 `D:\conda_envs\receiversync-viz\python.exe`，退出码均为实际
进程退出码。正式配置文件未被修改，SHA-256（大写）仍为：
`7458562AC63FABA21497799EA74E98CC7E0A0E21801320386B37704A40E204D9`。

### 8.1 保护、语法与回归

| 检查 | 命令/结果 |
| --- | --- |
| 原始主线保护 | `python scripts/verify_preserved_main.py`；exit 0；54 个原始主线文件存在，9 个核心变更均有精确授权。 |
| 原始完整回归 | `python -m pytest -q --basetemp .pytest_tmp_niabd_fix`；exit 0；`101 passed in 68.17s`。 |
| NIABD v2 定向测试 | `python -m pytest -q tests/test_niabd.py tests/test_niabd_v2.py`；exit 0；`19 passed`。覆盖 `[P,C]`、20×5000×10、极值隔离、共识漂移/冻结、student reference、label independence、shape/nonfinite 和参数校验。 |
| 同步/过程运行时回归 | `python -m pytest -q tests/test_simulation.py tests/test_runner_outputs.py`；exit 0；`16 passed`；`python -m pytest -q tests/test_backdoor_runtime.py tests/test_backdoor_diagnostics.py tests/test_process_runtime.py tests/test_process_backdoor_runtime.py`；exit 0；`16 passed in 29.03s`。 |
| 保护脚本负例 | `python -m pytest -q tests/test_preservation_verifier.py`；exit 0；`3 passed`。 |
| 快速缓存/上传检查 | `python experiment_example.py`；exit 0；`logits=(2, 10), upload_bytes=160`。 |
| 全仓库编译 | 直接 `python -m compileall -q .`；exit 1，仅因既有 `scripts\\__pycache__` 下多个 `.pyc.*` 文件拒绝写入；设置工作区隔离的 `PYTHONPYCACHEPREFIX=.pycache_niabd_validation` 后同一 compileall；exit 0。 |

### 8.2 合成 CPU 实测与复杂度

命令使用 20 个 teacher、5000 个 proxy 样本、10 个类别，在 CPU 上完成 warmup 后
执行一个 post-warmup NIABD round；结果为：

```text
niabd_cpu_benchmark_seconds=0.470936
niabd_cpu_benchmark_shape=20x5000x10
niabd_cpu_benchmark_reason=normal_eligible_update
```

本实现的单轮计算复杂度为 `O(K*P*C)`，持久状态为 `O(P*C+C)`。若按 `P=5000`
估算，`C=10` 时 prototype/variance/threshold 约 390.7 KiB，20 个 teacher 的
stacked logits 约 3.81 MiB；`C=200` 时分别约 7.63 MiB 和 76.3 MiB。后者是
round-local 峰值估计，不是额外持久历史存储；实际峰值还取决于 PyTorch 临时张量。

### 8.3 正式矩阵规划与未验证项

```text
python scripts/run_main_backdoor_matrix.py --dataset-roots dataset_roots.json \
  --config configs/main_backdoor_experiment.json --dry-run
exit=0
job_lines=240
last_job=[240/240] ... --method vcaa-niabd --attack dynamic ... --seeds 4 ... --device cuda
```

因此 4 数据集 × 5 攻击 × 3 方法 × 4 seed 的正式运行规划完整展开为 240 个 job；
dry-run 不等于训练成功。当前工作站的真实 CUDA/CoreX 设备、CUDA 依赖、正式主线矩阵、
攻击窗口 BASR、peak/AUC/recovery、线程级 TCP/RPC 长时间运行均标记为“未验证”。
`tiny-imagenet-200` 本地存在 100200 个 train 文件；由于没有为本次修复复制/缩小真实
数据集，也没有将这项大规模 CPU 训练冒充 smoke 结果。合成数据和运行时测试只证明
接口、状态机、序列化边界和输出 schema，不证明正式 GPU 指标。

正式 GPU 执行须遵循 [GPU_VALIDATION_RUNBOOK.md](GPU_VALIDATION_RUNBOOK.md)，输出到新的
`experiment_results_main_backdoor_niabd_v2` 根目录，不覆盖历史结果。完成后必须先运行
`scripts/check_result_completeness.py`，再运行 `scripts/collect_main_backdoor_results.py`；
缺失、重复、版本不一致或非完整 attack-window 的结果应 fail-closed。

## 9. 结果字段与审计版本

每轮和汇总结果新增 `niabd_algorithm_version=niabd-v2-proxy-conditioned-robust-memory`
以及 `result_schema_version=fedagg-results-v2`，并记录 memory candidate/eligible、score、
consensus drift、freeze reason、eligible observations、memory update rounds。正式攻击汇总
新增 attack-window mean、peak BASR、peak round、attack-window BASR AUC 和 post-attack
recovery BASR；这些字段在正式运行未完成前不得填入或解释为正式指标。

## 10. 剩余风险与安全假设

NIABD 的 drift recovery 依赖准入集合中存在足够比例的紧凑良性共识，不能证明抵抗任意
恶意多数。其输入边界仍是假设 packet 已由现有 transport/coordinator 做 hash、shape、
dtype、finite、round、duplicate 和 retry 校验；NIABD 本身不提供差分隐私，也不声称从
logits 反推出的模型行为具备隐私保证。VCAA 仍是质量/时效准入，不被解释为后门检测器。
攻击诊断继续与生产防御路径隔离，NIABD 不读取 `proxy_labels` 的数值，也不读取攻击真值。
