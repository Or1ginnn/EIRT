# Conditional EITR Phase 2：V5.1 实现与安全检查

日期：2026-08-08

> 当前状态：V5.1 代码与 CPU 单元测试已完成；两卡 FSDP paired smoke 尚未运行，因此本文档不声称训练收益或服务器端运行通过。

## 1. Phase 2 的作用

Gate A/B 已提供语言距离与检索环境后果不一致的前提证据。Phase 2 只完成 Conditional EITR 的可训练实现：模型没有合法搜索时继续普通 GRPO；出现合法搜索后，才对 query-induced retrieval drift 做一次固定系数的 correction。

当前 3B MVP 的探测单位是：

- 每条真实 rollout 的第一个完整、非空 `<search>query</search>`；
- search 可以出现在任意 turn，不再限制为第一个生成 turn；
- 后续 search state 记录为 `deferred`，暂不增加 probe 成本；
- 默认 `p_probe=1`；
- 默认 `K=4`，包括1条真实 query 和3条额外 query；
- `K_eff>=2` 即可计算，额外 probe 失败不会终止训练。

这与7B多轮版本的“每个 search turn 都探测”不同。多 state/FSDP packing 留到扩展实验，不在本次3B Phase 2中冒充已完成。

## 2. 合法性与容错

只有完整且非空的 action 才是合法 search：

```text
<search>who wrote Hamlet</search>   valid
<search></search>                   invalid
<search>who wrote Hamlet            invalid
```

执行规则：

```text
没有合法真实 search
→ 普通 GRPO

真实 search 合法，但未被 p_probe 选中
→ 普通 GRPO

被选中，但所有额外 probe 均不合法
→ 普通 GRPO，并记录失败

真实 query + 至少1条合法额外 probe
→ 普通 GRPO 更新后执行 EITR correction
```

默认不 oversample、不因 malformed 而重采到凑满 K。duplicate query 是从旧策略得到的合法独立样本，保留并记录，不做去重条件化。

额外 probe 会在整条真实 trajectory 完成后统一生成和检索，避免 probe 请求影响后续真实 turn 的随机采样。
同一个 state 的 `K-1` 条额外 query 分成 `K-1` 轮 batched sampling，每轮使用不同 seed；不能把相同 state 复制多行后共享一个 vLLM seed，否则会退化为重复生成同一 query。

## 3. 三种运行模式

```text
off         普通 GRPO，不收集 probe
probe_only  收集、检索并重评分 probe，但不反向传播 EITR
eitr        收集相同 probe，并执行固定 lambda 的 EITR correction
```

`probe_only` 用于环境调用预算对照。旧的 `enabled` 开关只作为兼容 fallback；新实验统一使用 `mode`。

## 4. 优化顺序

Adaptive dual beta 已删除。系数固定为 `lambda_env`。

一次 rollout batch 的顺序是：

```text
1. 冻结 pi_old 并收集真实 trajectories / probes
2. 计算旧策略 query sequence log-prob
3. 执行 actor.ppo_epochs 次普通 GRPO（默认1次）
4. candidate 对缓存 query 重新评分
5. probe_only：只记录 JS/ESS/ratio，不更新参数
6. eitr：只对 lambda_env * D_env 做 correction optimizer step
```

Correction 不是第二次 reward-bearing GRPO epoch。这样零有效 search 时，该 batch 的参数与 optimizer 更新都退化为原始普通 GRPO；EITR 也不会凭空获得额外任务梯度。

Online probes 会在普通 sequence balancing 前打包成 batch-aligned tensors，因此 `off` 与 Phase 2 使用同一条 GRPO 重排路径。Correction 再按 state chunk 执行；默认 `K=4, probe_micro_batch_size=4` 时每次只保留一个 state 的 probe activation graph，完成 backward 后立即释放，避免把整个 rollout microbatch 的长前缀图同时留在显存。

## 5. 主要配置

```yaml
eitr:
  mode: eitr
  probe_source: online_same_state
  probe_probability: 1.0
  probe_count: 4
  min_valid_probe_count: 2
  probe_oversample: 0
  probe_micro_batch_size: 4
  probe_logprob_micro_batch_size: 4
  # 与真实单轮生成预算一致；每个 state 实际只使用 <search> 后的剩余预算
  max_query_tokens: 500
  max_action_tokens: 500
  max_probe_prompt_tokens: 8692
  correction_passes: 1
  lambda_env: 0.1
  log_ratio_clip: 10.0
```

训练入口同时设置 `actor_rollout_ref.rollout.max_model_len=9192`。真实一轮仍使用
`500` token 总生成预算；构造 exact search state 后，probe 只继承该轮在
`<search>` 后剩余的 token budget。batched vLLM 输出也会按各自 state 的剩余预算裁切后再解析，因此真实 query 与额外 query 不再使用不同的 action support。

检索 observation 的单轮保留上限为 `1024` token，完整 trajectory 的累计保存上限与
每轮提供给模型的 rolling prompt 均为 `8192` token。两者使用独立配置，当前取值相同。

三种 paired 模式统一继承 Search-R1 Qwen2.5 GRPO 的 actor 优化参数：
`lr=1e-6`、`lr_warmup_steps_ratio=0.285`。EITR 不单独调整学习率。

三种 paired 模式统一使用 `rollout.top_p=1.0, top_k=-1`。EITR 的 SNIS ratio 使用 actor 完整 softmax 下的 query sequence log-prob，因此采样也必须来自同一个未截断分布；`probe_only/eitr` 若配置 nucleus 或 top-k 截断会在启动时直接拒绝。

`min_state_coverage` 和 `min_informative_state_rate` 现在只产生诊断标记，不再抛异常。

## 6. 关键指标

### 6.1 统一的 step 口径

Search-R1 原始训练循环存在 off-by-one，并且在 dataloader 的 epoch 容量小于
`trainer.total_training_steps` 时会静默提前结束。Phase 2 将训练预算统一定义为：

```text
1 outer update = 1个训练batch的rollout + reward/advantage + GRPO更新
                 + 该模式可能执行的EITR correction
```

显式设置 `trainer.total_training_steps=T` 时，trainer 会按需重新迭代 dataloader，
并且恰好完成 `T` 个 outer updates；`global_step` 表示已经完成的 outer updates。
初始验证记录在 step 0，训练日志记录在 step 1...T，最终验证与第 T 次更新合并在
同一个 W&B step，不再制造一个没有训练更新的终点 step。

LR scheduler 也以 outer update 为时间单位，因此 paired 的 `off/probe_only/eitr`
在相同 outer step 使用相同学习率。一次 outer update 内部的 AdamW 次数单独报告，
不再与 W&B global step 混用：

- `trainer/outer_update_step`
- `trainer/target_outer_updates`
- `actor/grpo_optimizer_step_count`
- `actor/eitr_correction_optimizer_step_count`
- `actor/optimizer_step_count`
- 对应的 `*_cumulative` 累计指标

在默认 smoke 参数下，每个 outer update 有 5 次 GRPO optimizer steps；只有
`eitr` 模式会在有效 state 上增加 0～5 次 correction optimizer steps。

### 6.2 EITR 与环境指标

- `eitr/real_valid_search_count`
- `eitr/probe_selected_state_count`
- `eitr/probe_candidate_valid_rate`
- `eitr/active_state_rate_given_selected`
- `eitr/effective_probe_count_mean/min/max`
- `eitr/deferred_additional_search_state_count`
- `actor/eitr_global_induced_js`
- `actor/eitr_global_probe_ess`
- `actor/eitr_pass_*_raw_logprob_grad_norm`
- `actor/eitr_pass_*_applied_logprob_grad_norm`
- `actor/eitr_correction_optimizer_step_count`
- `env/number_of_executed_search`
- `env/final_generation_ratio`
- `env/final_unexecuted_search_ratio`

Collector 的具体拒绝原因仍以 `eitr/collector_*` 记录，例如 missing close tag、empty query 和 empty retrieval effect。

## 7. 运行入口

训练奖励采用 Search-R1 v0.3 的格式奖励入口 `main_ppo_format`：完整且答案
正确为 1.0，答案正确但轨迹结构非法为 0.8，答案错误但完整轨迹结构合法为
0.2，只有最终 `<answer>...</answer>` 格式合法为 0.1，其余为 0。默认
`retrieval_score=0`，因此不额外奖励“检索结果中包含答案”。周期验证仍使用纯答案
EM，不加入格式 shaping。该奖励与 EITR correction 独立：GRPO 使用上述标量奖励，
EITR 仍在随后的 correction pass 中最小化环境诱导分布漂移。

正式训练默认读取 `data/nq_hotpotqa_train/train.parquet`，并开启 dataloader
shuffle；该文件由NQ与HotpotQA顺序拼接而成，不打乱会让短预算训练偏向文件前部的
NQ。验证单独读取 `data/nq_search/test.parquet` 中固定抽样的256条NQ，因此训练集与
验证集不再共享同一个 `DATA_DIR`。混合训练还使用 `data_source::index` 作为GRPO组
ID，避免两个数据源中相同的局部编号被误合并。

默认所有缓存、日志、checkpoint、W&B 与临时文件都写入：

```text
/mnt/data1/zar/eitr_storage
```

启动脚本会拒绝 `/mnt/data1/zar` 之外的 `STORAGE_ROOT`，并显式把 Ray session、Hydra 输出、HF/Torch/Triton/CUDA/Numba cache、W&B 和 `TMPDIR` 指向数据盘。`CHECK_ONLY` 也会先创建并校验这些目录，再报告预检通过。

预检：

```bash
CHECK_ONLY=true bash scripts/train/train_eitr_nq_gate_c_smoke.sh
```

EITR safety smoke：

```bash
EITR_MODE=eitr bash scripts/train/train_eitr_nq_gate_c_smoke.sh
```

Probe-only 对照：

```bash
EITR_MODE=probe_only \
EXPERIMENT_NAME=eitr-nq-phase2-probe-only-smoke \
bash scripts/train/train_eitr_nq_gate_c_smoke.sh
```

普通 GRPO：

```bash
EITR_MODE=off \
EXPERIMENT_NAME=eitr-nq-phase2-grpo-smoke \
bash scripts/train/train_eitr_nq_gate_c_smoke.sh
```

正式训练仍需显式给出预算：

```bash
TOTAL_TRAINING_STEPS=<与Search-R1预算一致> \
bash scripts/train/train_eitr_nq_gate_c_full.sh
```

## 8. Safety smoke 验收

1. 三种 mode 都能完成短跑，无 CUDA/Ray/FSDP hang、OOM、NaN 或 shape error；
2. 零 valid search 的 batch 不运行 correction，也不停止 GRPO；
3. malformed probe 只降低 `K_eff`；
4. 有 active state 时 candidate-vs-old JS 和 raw query-logprob gradient 出现非零；
5. `eitr` 至少出现一次 correction optimizer step；
6. `probe_only` 不出现 correction optimizer step；
7. off 与 probe_only 在同 seed 下的真实 rollout 不因 probe 路径发生变化；
8. 所有运行产物留在数据盘。

旧版 Gate C smoke 因硬 coverage 阈值和 probe close-tag 失败在第1步终止，不能作为算法结论；V5.1 的 paired smoke 必须重新运行。
