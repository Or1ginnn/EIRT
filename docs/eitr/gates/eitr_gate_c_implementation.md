# EITR Gate C：训练实现与验收协议

日期：2026-08-06

> 迁移状态：Gate C 已重新移植到官方 Search-R1 单 query 底座。旧的 Parallel Search Step900 不再作为初始化模型；当前只有实现与测试，尚无干净基线上的训练结论。

## 1. Gate C 要回答什么

Gate A/B 说明了语言距离与检索环境后果存在错位，并且环境距离更能预测下游收益差异。Gate C 不再做相关性诊断，而是直接检验：把 environment-induced geometry 接进 GRPO 后，能否稳定改善最终准确率、收敛速度、训练坍塌率或 stability-improvement trade-off。

Gate C 通过会明显增强 EITR 的论文可信度，但不等于 ICLR 稳中。正式投稿仍需至少 3 个随机种子、跨数据集验证、关键消融和总环境调用预算对齐。

## 2. 当前实现

### 2.1 Online same-state probes

当前实现不把普通 sibling trajectories 直接当作同状态 query。原 Search-R1 会先生成 `<think>`，不同 trajectory 的 think prefix 不相同，直接复用会破坏“固定同一状态”的定义。

训练时对每个问题执行：

1. 正常 rollout 生成到第一个 `<search>`；
2. 固定该轨迹从 prompt 到 `<search>` 的完整 token prefix；
3. 直接复用 vLLM 原始 token prefix，从完全相同的状态额外采样 query continuations；
4. 保留共 `K=4` 个有效单 query probe；
5. 每个 probe 只调用一次真实 retriever，不继续 answer suffix，不产生 task reward；
6. 缓存 action token、old-policy sequence log-prob、top-k document id 和 score。

为了抵消无效或重复 query，默认额外 oversample 2 个候选；同状态内 query 去重后，再取前 4 个有效 probe。
检索前即截断为最多 `K-1` 个额外 probe，并记录 `eitr/probe_retrieval_call_count`，用于按环境调用预算做公平对照。
Probe 使用独立且可复现的随机种子流，避免额外采样无意推进正常 rollout 的 RNG；paired smoke 同时关闭 dataloader shuffle，固定问题顺序。

Collector 使用完整已解码响应的上下文前缀定位 `<search>` token 边界，不再用孤立编码的标签做子串匹配。运行时同时记录 state group、首轮搜索、边界定位、probe query 和 retrieval effect 的逐层计数；coverage 不达标时，异常会携带 collector rejection 统计。

### 2.2 Environment distribution

每个 query 的 top-k score 先经 `softmax(score / tau_R)` 变成 document distribution。在组内所有文档的 union support 上计算：

```text
p_old(d|s)     = uniform mixture of cached probe document distributions
p_current(d|s) = SNIS-weighted mixture using exp(logp_current - logp_old)
D_B            = JS(p_old || p_current)
```

retriever 不参与反向传播；梯度只经过 current policy 对 probe action 的 sequence log-prob。

Probe coverage 只说明成功生成了 query，不说明这些 query 带来了不同检索结果。因此训练还会计算同状态 probe 两两之间的 retrieval-effect JS：

- `eitr/informative_probe_state_rate`：在有效 probe 组中，至少一对检索分布 JS 超过 `0.01` 的比例；
- `eitr/probe_effect_pairwise_js_mean/max`：真实检索分歧的平均值和最大值；
- `eitr/probe_effect_top1_disagreement_rate`：同组 query 的 top-1 文档发生变化的比例。

如果 query 只是同义改写且检索结果相同，该状态的分歧为零，不产生有效 EITR geometry。Smoke 默认要求 informative state rate 至少为 `0.1`，防止在几乎没有环境分歧信号时误跑完整训练。

### 2.3 Actor objective 与 dual

任务 reward、GRPO advantage 和 PPO ratio 保持不变。Actor 最小化目标中额外加入：

```text
L = L_GRPO + beta * D_B + weak_token_KL
beta <- clip(beta + dual_lr * (mean(D_B) - target_js), 0, beta_max)
```

`- beta * target_js` 对 actor 梯度是常数，因此实现中只把 `beta * D_B` 加入 loss；target 用于 dual 更新和 W&B 诊断。

### 2.4 多卡一致性

Probe group 被封装在每个问题的固定代表行中，避免普通 data-parallel 切分拆散 group。EITR 开启时关闭 sequence-length rebalance，并要求：

- `train_batch_size` 可被 actor world size 整除；
- 每个 uid 恰有连续的 `n_agent` 条 rollout；
- `use_dynamic_bsz=false`；
- 所有 rank 即使遇到无效 probe group，也执行零梯度 dummy probe forward，保证 FSDP collective 顺序一致。
- EITR loss 按跨 rank 的有效状态总数加权，避免各卡 probe 覆盖率不同时产生 rank-level bias。

EITR 默认关闭；关闭后 rollout 不采 probe，actor 不增加额外 forward，沿用原 GRPO 路径。

## 3. 代码位置

- `verl/trainer/ppo/eitr.py`：probe batch、doc distribution、SNIS、induced JS、dual update；
- `search_r1/llm_agent/generation.py`：固定 prefix、在线 query probe 采样、真实检索结果缓存；
- `verl/trainer/ppo/ray_trainer.py`：probe old log-prob、训练数据流和配置校验；
- `verl/workers/actor/dp_actor.py`：EITR loss、W&B 指标和 beta 更新；
- `verl/workers/fsdp_workers.py`：probe sampling 参数及 log-prob 默认元信息；
- `scripts/train/train_eitr_nq_gate_c_smoke.sh`：同一脚本运行 baseline/EITR smoke；
- `scripts/train/train_eitr_nq_gate_c_full.sh`：显式指定训练预算后运行正式 Gate C；
- `tests/test_eitr.py`：数学、梯度、probe 对齐和多卡布局单测。

## 4. Smoke 运行

先准备原 Search-R1 `base` prompt 的 NQ parquet，并确认 retriever 可用。脚本顶部只需设置模型、数据、GPU 和 retriever 路径。

只做预检、不启动训练：

```bash
CHECK_ONLY=true bash scripts/train/train_eitr_nq_gate_c_smoke.sh
```

EITR smoke：

```bash
bash scripts/train/train_eitr_nq_gate_c_smoke.sh
```

同 seed/no-op baseline：

```bash
EITR_ENABLED=false \
EXPERIMENT_NAME=eitr-nq-gate-c-baseline-smoke \
bash scripts/train/train_eitr_nq_gate_c_smoke.sh
```

脚本设置 `total_training_steps=11`，对应当前 trainer 计数逻辑下的 10 次实际 update。

Smoke 通过后，正式训练必须显式给出与 baseline 对齐的 update 预算，避免误用 smoke 默认值：

```bash
TOTAL_TRAINING_STEPS=1005 \
bash scripts/train/train_eitr_nq_gate_c_full.sh
```

在线 probe 的生成状态与计算 log-prob 的状态必须逐 token 相同。因此
`max_probe_prompt_tokens` 不得小于 `data.max_prompt_length`；配置和运行脚本会在启动前拒绝不一致设置，不再静默截断状态。

## 5. Smoke 验收门槛

必须同时满足：

1. 完成 10 次 update，无 CUDA/Ray OOM、NaN、FSDP hang 或 tensor-shape 错误；
2. `eitr/probe_state_coverage >= 0.5`；
3. `eitr/informative_probe_state_rate >= 0.1`，证明至少一部分 query 组产生了真实检索分歧；
4. `actor/eitr_global_induced_js`、`actor/eitr_beta`、`actor/eitr_probe_ess` 全部有限；
5. ESS 位于 `[1, 4]`，`log_ratio_clipfrac` 不应长期接近 1；
6. 至少从第二个 optimizer mini-batch 起 induced JS 出现非零值；
7. EITR 关闭时不出现 probe 指标，答案 reward 与原 GRPO 一致；
8. 抽查 probe：同组 4 个输入 prefix token 完全一致，query 不同，retrieval 结果来自真实 `/retrieve`。

## 6. 2026-08-06 旧链路试跑记录

服务器曾从旧的 Parallel Search 迁移提交 `41e3b58` 启动一次先导试跑：

- baseline 完成 10 次 update，64 条验证子集最终 EM 为 `0.46875`；
- EITR 在第 1 步 rollout 后、参数更新前退出；
- online probe coverage 为 `0.000`，32 个问题组均报告 `missing_online_probe_group`；
- 未产生 EITR loss、induced JS、beta、ESS 或 EITR validation；
- 无 OOM、NaN、FSDP hang、retriever 500 或 checkpoint。

这次结果不能作为 Gate C 对照结论。旧脚本默认初始化模型为 `models/parallel_search_qwen25_3b_step900`，代码也包含 LiteCoA 多 query rollout；它不是本文要求的官方 Search-R1 单 query 干净底座。`missing_online_probe_group` 发生在 tensor builder 之前，说明上游 collector 没有建立 probe group。为消除已发现的脆弱点，当前实现已改为上下文感知的 search token 边界定位，并增加 collector 分层诊断。

下一次试跑必须同时确认：

1. 工程来自官方 Search-R1 单 query 底座；
2. 初始化模型不是 Parallel Search Step900 或 Finance checkpoint；
3. 训练 parquet 使用原 Search-R1 base prompt；
4. baseline 与 EITR 使用同一模型、问题顺序、seed、retriever 和 rollout 预算；
5. 先只复跑 10-update paired smoke，不保存 checkpoint。

## 7. Gate C 正式实验

Smoke 通过后再跑正式对照，不直接凭 10 step validation 下结论：

1. Standard GRPO；
2. GRPO + 当前 weak token KL；
3. EITR-GRPO hybrid；
4. 可选 behavior-only 消融。

每组至少 3 seeds，保持初始 checkpoint、训练问题顺序、task rollouts、temperature、reward、retriever 和验证协议一致。报告：

- full-test EM/F1；
- 达到固定 EM 所需 update / wall-clock；
- task retrieval calls 与 probe retrieval calls；
- grad norm variance、PPO clipfrac、token KL、induced JS、beta、ESS；
- collapse 次数；
- performance improvement 对 token drift / behavioral drift 的联合曲线。

Gate C 至少要在 final accuracy、convergence speed、collapse rate、gradient variance或同 behavioral drift 下的优化幅度之一得到跨 seed 稳定改善。若都没有改善，则停止扩大模型规模。
