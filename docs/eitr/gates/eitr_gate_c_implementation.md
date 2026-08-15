# Conditional EITR Phase 2：V6 实现与安全检查

日期：2026-08-09

> 当前状态：V6 coverage-weighted correction、独立 SGD 与 CPU 单元测试已完成；新的两卡 FSDP smoke 尚未运行，因此本文档不声称训练收益或服务器端运行通过。

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
6. eitr：在完整 rollout batch 上累积 lambda_env * D_env 梯度
7. 使用独立、无 momentum/weight decay 的 SGD 执行一次 correction
8. 按配置周期重新评分相同 cached probes，记录 D_pre/D_post/delta
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
  correction_optimizer: sgd
  # 固定 LR 现在是单次 correction 的上限；大梯度 batch 自动缩短该步。
  correction_lr: 3e-5
  correction_max_update_norm: 3e-6
  correction_min_update_norm: 3e-6
  post_diagnostic_freq: 10
  lambda_env: 0.1
  log_ratio_clip: 10.0
```

训练入口同时设置 `actor_rollout_ref.rollout.max_model_len=9192`。真实一轮仍使用
`500` token 总生成预算；构造 exact search state 后，probe 只继承该轮在
`<search>` 后剩余的 token budget。batched vLLM 输出也会按各自 state 的剩余预算裁切后再解析，因此真实 query 与额外 query 不再使用不同的 action support。

检索 observation 的单轮保留上限为 `1024` token，完整 trajectory 的累计保存上限与
每轮提供给模型的 rolling prompt 均为 `8192` token。两者使用独立配置，当前取值相同。

三种 paired 模式的 GRPO 统一使用 Search-R1 v0.3 actor AdamW 参数。EITR
correction 使用独立 SGD，`momentum=0, weight_decay=0`。当前把
`correction_lr=3e-5` 作为单次 correction 的基础 LR 上限，并在更新前根据全局参数梯度范数计算：

```text
eta_effective = eta_base * min(1, tau / (eta_base * ||g_clipped||))
tau = correction_max_update_norm = 3e-6
```

因此预测参数更新范数不超过 `3e-6`。小梯度 batch 与原固定 LR 完全相同；大梯度
batch 只把同一负梯度方向等比例缩短。该实现仍然只有一次 EITR backward 和一次独立
SGD step，不增加 probe、检索、模型重评分或参数快照。W&B 记录基础 LR、实际 LR、
normalization scale 与预测更新范数；`actor/eitr_effective_step_scale` 使用实际 LR 计算。

归一化10-step复验显示：归一化前预测更新达到上限的6个batch中5个下降，而低于上限的
4个batch全部反弹。因此加入显式的弱更新门：仅当
`eta_base * ||g_clipped|| >= correction_min_update_norm=3e-6` 时提交独立SGD；否则保留
GRPO proposal，记录 `correction_skipped_weak_update=1`、真实 correction step为0。若该步
按频率需要post诊断，由于参数未变，可直接精确记录 `D_post=D_pre`，并用
`post_diagnostic_noop=1` 与一次实际下降的correction区分。这个门只刻画本batch预测更新是否
达到当前经验阈值，不等价于query质量或理论置信度；其必要性需要新的20-step随机batch验证。

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
- `actor/eitr_correction_normalization_scale`
- `actor/eitr_correction_unnormalized_update_norm`
- `actor/eitr_correction_predicted_update_norm`
- `actor/eitr_correction_skipped_weak_update`
- `actor/eitr_post_diagnostic_noop`
- `actor/optimizer_step_count`
- 对应的 `*_cumulative` 累计指标

在默认 smoke 参数下，每个 outer update 有 5 次 GRPO AdamW steps；只有
`eitr` 模式会在完整 rollout batch 梯度累积结束后增加 0或1次 EITR SGD step。

### 6.2 EITR 与环境指标

- `eitr/real_valid_search_count`
- `eitr/probe_selected_state_count`
- `eitr/probe_candidate_valid_rate`
- `eitr/active_state_rate_given_selected`
- `eitr/effective_probe_count_mean/min/max`
- `eitr/deferred_additional_search_state_count`
- `actor/eitr_global_induced_js`
- `actor/eitr_env_drift_pre`
- `actor/eitr_env_drift_post`
- `actor/eitr_env_drift_delta`
- `actor/eitr_env_drift_relative_reduction`（正值表示 correction 降低了 drift）
- `actor/eitr_global_probe_ess`
- `actor/eitr_pass_*_raw_logprob_grad_norm`
- `actor/eitr_pass_*_applied_logprob_grad_norm`
- `actor/eitr_correction_optimizer_step_count`
- `env/number_of_executed_search`
- `env/final_generation_ratio`
- `env/final_unexecuted_search_ratio`

console 对上述 drift 指标使用高精度科学计数法，避免有效但很小的 correction 被
格式化成 `0.000/-0.000`。20-step smoke 还记录
`env/trajectory_valid_search_rate`（每条 trajectory 只判断是否至少执行过一次有效搜索）和
`eitr/real_valid_search_per_rollout`（允许多轮搜索，因此可大于1），用于判断搜索行为
是否在 correction 下被明显拖慢。前者直接来自真实环境执行记录，因此在
`off/probe_only/eitr` 三种模式下口径一致，不使用 probe 选中率冒充搜索行为率。

Collector 的具体拒绝原因仍以 `eitr/collector_*` 记录，例如 missing close tag、empty query 和 empty retrieval effect。

## 7. 运行入口

训练仍使用 `main_ppo_format` 入口，但明确区分四种 reward profile：

- `pure_em`：周期验证与纯终局答案对照；
- `official_v03`：Search-R1 v0.3 格式奖励对照，要求 `retrieval_score=0`；
- `evidence_shaping`：保留早期一次性 `retrieval_score` hook 的兼容对照；
- `mandatory_search`：当前 formal runner 的 Search-Agent 训练奖励。

`mandatory_search` 只信环境真实执行后写入的逐轨迹 `executed_search_count`，并只从
`info_mask` 标记的环境 observation 中判断 evidence；仅在文本里留下未执行的
`<search>` 或模型伪造 `<information>` 都不能通过门控。没有真实搜索、工具轨迹不一致
或模型伪造information时
reward为0；最终必须有唯一、非空、正确闭合且位于轨迹末尾的`answer`，否则同样归零。
通过这两个硬门后，`think`格式0.2、`answer`基础分0.1和答案EM 1.2组成训练reward。
训练reward不再使用answer-bearing evidence或严格协议联合奖励，满分仍为1.5。缺少
`think`只损失0.2，不会清空answer基础分或答案分。真实搜索即使返回空information也能通过搜索
门；evidence命中只作为诊断，不改变reward，多轮命中也不累计。
`off/probe_only/eitr`必须使用
同一个显式profile，不能跨reward比较；EITR probe检索绝不能满足真实搜索门控。

W&B 额外记录 `reward/answer_em_rate`、`reward/format_valid_rate`、
`reward/configured_max_score`、`reward/mandatory_search_profile`、
`reward/soft_format_components`、`reward/answer_hard_gate`、
`reward/think_format_score`、`reward/answer_format_score`、
`reward/evidence_score`、`reward/answer_em_score`、`reward/joint_success_bonus`、
`reward/think_format_valid_rate`、`reward/answer_format_valid_rate`、
`reward/tool_trace_consistent_rate`、`reward/hard_reward_gate_pass_rate`、
`reward/generated_information_rate`、`reward/nonzero_rate`、
`reward/score_std`、`reward/unique_level_count`、
`reward/nonzero_advantage_group_rate`、`reward/zero_advantage_group_rate`、
`reward/answer_bearing_evidence_rate`与`reward/answer_bearing_given_search`。
周期验证仍使用
纯答案EM，范围保持0到1，不加入hard-search reward。该profile是所有方法共享的训练
奖励变量，不属于EITR correction；论文报告必须把它与exact-v0.3对照分开命名。

V6 correction采用rollout coverage加权。若一个训练mini-batch共有 `B` 条
rollout，其中mask有效的EITR state为 `m_i=1`，则实际目标为
`sum(m_i * D_env_i) / B`，而不是除以active state数量。名义
`lambda_env=0.1`保持固定，batch级有效尺度记录为
`actor/eitr_effective_lambda = lambda_env * actor/eitr_coverage`。完全没有有效
state时仍严格跳过correction。所有 state chunk 都使用同一个完整 rollout batch
分母并先累积梯度，之后只执行一次独立 SGD，因此 `D_pre` 对应同一个 GRPO 后策略，
也不会推进 GRPO AdamW 的 momentum、second moment 或 weight decay。正式训练默认
每10个 outer updates 额外执行一次无梯度 post-correction re-score；smoke 每步执行。

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
Ray根目录固定使用较短的 `$STORAGE_ROOT/r`，避免长实验名让AF_UNIX socket路径
超过Linux的107字节限制；实验名仍完整保留在日志、Hydra和checkpoint目录中。

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

正式训练默认上限为8000个outer updates；仍可通过环境变量覆盖：

```bash
TOTAL_TRAINING_STEPS=8000 \
bash scripts/train/train_eitr_nq_gate_c_full.sh
```

默认每100步保存一次轻量模型checkpoint。手动中断不会额外保存中断时刻的权重，
因此应尽量在100的整数倍checkpoint完成写盘后停止。

8000步正式预算中，GRPO actor 使用 `lr_warmup_steps_ratio=0.03575`，即286个
outer updates。这保留Search-R1 v0.3原1005步配置中的约286步actor warmup，
而不是保留0.285比例；后者会把actor warmup错误拉长到2280步。

独立 EITR SGD correction 不使用这条 warmup schedule：`EITR_LR=3e-5` 从训练
开始保持固定。Gate C 最终方向审计同样在固定 `EITR_LR=3e-5` 下通过。正式训练
早期的 EITR 实际作用由 GRPO proposal drift 与 rollout coverage 自然控制，而不是
再次对 correction LR 乘 actor warmup factor。

正式默认继续保持 `PPO_MICRO_BATCH_SIZE=4`、
`EITR_PROBE_MICRO_BATCH_SIZE=4` 和
`EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE=4`。虽然更大的8/8/8只改变同一批张量的
分块方式，不改变全局 train batch=32、PPO mini-batch=32、K=4、loss 或 optimizer
step数，但真实4/4/4任务已在第8个随机长batch的EITR correction backward发生OOM，
因此在释放训练GPU上的Retriever显存前，8/8/8不再是可接受候选。

共享三卡profile进一步把全局 `PPO_MICRO_BATCH_SIZE` 降为3，即每个data-parallel
rank一次只处理一条普通GRPO trajectory；全局train batch=30、mini-batch=30与每个
outer update的5次GRPO AdamW steps不变。EITR仍以完整K=4 state为最小评分单元。

另一个已定位的峰值来源是probe correction先前统一使用 `model.eval()`：虽然配置已
执行Hugging Face `gradient_checkpointing_enable()`，Qwen decoder仍会以
`module.training` 作为层级checkpoint开关，因而eval-mode correction保留了全部长序列
激活。当前EITR模式改为对cached-old/current/no-op/post probe统一使用确定性train-mode
评分，从而真正启用已有的layer checkpointing。该路径启动时会检查actor确实启用了
gradient checkpointing，并拒绝任何非零dropout设置；`use_cache=False`继续保持。
普通old/ref log-prob仍使用eval模式。此改动只改变激活的保存/重计算方式，不改变
query、retrieval effect、SNIS/JS、coverage、K、loss、LR或optimizer step。

新增 `actor/eitr_probe_gradient_checkpointing_active` 指标作为运行时证据。该显存修复
在本地仅通过了单元与静态检查，必须先在真实三卡FSDP任务重新通过anchor、1-step方向
审计与短期显存smoke，不能仅凭代码检查宣称正式训练已稳定。

性能诊断继续记录 `timing_s/real_retrieval`、
`timing_s/eitr_probe_generation`、`timing_s/eitr_probe_retrieval`、
`timing_s/eitr_probe_old_logprob`、`timing_s/grpo_update` 和
`timing_s/eitr_correction`，用真实分阶段耗时区分 EITR 计算成本与 CPU offload 成本。

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
