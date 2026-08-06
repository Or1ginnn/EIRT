# EITR NQ Gate B: Branch-to-Return Protocol

> 适用范围：该协议最初在受单 query 约束的 Parallel Search Step900 策略上执行。新项目保留协议与历史结果，并将补做干净 Search-R1 基线复现。

## 目标

验证检索环境距离是否比语言动作距离更能预测同一状态下不同 query 分支的最终收益差异。该实验只验证 EITR 的下游相关性，不训练 EITR-GRPO。

## 固定设置

- 数据：Gate A 已采集的 1000 个 NQ search states。
- 动作：每个状态最多 8 条已采样单 query；不包含并行 query。
- 检索：复用 Gate A 已缓存的 E5 Flat top-3 结果，不重新检索。
- 状态控制：同一问题、同一确定性前缀，仅 query 和其诱导的 information 不同。
- 终局续写：temperature=0；每条分支只允许已有的一次 search，随后生成 answer。
- 任务收益：normalized semantic Answer EM，正确为 1，错误或无终局回答为 0。Gate A 的动作采样 prompt 只约束单次 search，没有规定最终 `<answer>` 标签，因此主收益同时接受 `<answer>...</answer>` 与终局 `Answer: ...`；严格 tag EM 作为格式诊断单独报告。
- 统计单位：query pair；置信区间按 state 聚类 bootstrap，避免把同一问题内 pair 当作独立样本。

## 预注册主比较

- 语言距离：Qwen policy hidden-state query embedding cosine distance。
- 环境距离：top-k 检索分数分布的 Jensen-Shannon distance。
- 预测目标：`|EM_i - EM_j|`，即两个分支是否产生不同任务收益。
- 主指标：AUROC。

辅助报告 token Jaccard、编辑距离、sequence logprob gap、document Jaccard、RBO、evidence embedding distance，以及 answer disagreement 和 evidence-hit difference。

## Gate B 验收标准

必须同时满足：

1. 至少 100 个 state 出现组内 EM 变化；
2. 至少 500 个 pair 的 `|EM_i - EM_j| = 1`；
3. 主环境距离相对主语言距离的 AUROC 提升至少 0.03；
4. state-cluster bootstrap 的 AUROC 提升 95% CI 下界大于 0。

若样本变化不足，结论为数据不充分；若数据充分但环境距离不优于语言距离，Gate B 不通过，不进入 EITR-GRPO 方法实现。

## Smoke 后协议修正

20-state smoke 中有 24 条语义完整的 `Answer: ...` 被 strict-tag parser 误记为无答案。该问题在正式实验前修正：不改变 prompt、query、retrieval 或模型续写，只对已生成的终局回答增加 plain-answer fallback，并保留 strict-tag 指标。这样 Gate B 衡量的是 query 对任务正确性的影响，而不是 Gate A prompt 未规定 answer tag 造成的格式噪声。
