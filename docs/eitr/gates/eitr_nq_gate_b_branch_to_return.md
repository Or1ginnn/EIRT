# EITR NQ Gate B：Branch-to-Return 诊断报告

> 历史实验说明：本报告沿用 Gate A 的 Parallel Search Step900 策略，但所有动作均限制为单 query。结果支持 EITR 的下游相关性前提，不可替代原始 Search-R1 基座上的正式复现。

## 结论

**Gate B 通过。** 在同一 search state 下，不同单 query 诱导的检索环境距离，比 query 的语言表示距离更能预测两个分支是否产生不同的最终 Answer EM。该优势经过 state-level cluster bootstrap 后仍显著，并且在排除无答案和格式失败后依然成立。

该结果支持 EITR 的第二层动机：语言空间中的“近或远”不仅不能完整表示检索行为，而且这种表示错位与最终任务收益有关。

本实验仍不能证明 EITR-GRPO 方法有效，也不能证明现象已跨数据集、跨检索器普遍成立。

## 实验设置

- 数据：NQ Gate A 的 1000 个固定 search states。
- 模型：`parallel_search_qwen25_3b_step900`，在本实验中限制为单 query。
- 动作：复用 Gate A 采样的 7595 条有效 query，不重新采样。
- 检索：复用 E5 Flat retriever 的 top-3 缓存结果。
- 分支控制：同一问题、同一确定性前缀，仅 query 及其诱导的 `<information>` 不同。
- 终局续写：greedy，`temperature=0`，不允许第二次检索。
- 任务收益：normalized semantic Answer EM。
- 统计：25175 个 state 内 query pairs；500 次 state-cluster bootstrap。

### Smoke 后口径修正

Gate A 的动作采样 prompt 只约束单次 search，没有规定最终 `<answer>` 标签。20-state smoke 中有 24 条完整的 `Answer: ...` 被 strict-tag parser 误判为无答案。

正式实验前固定为：

- 主收益同时接受 `<answer>...</answer>` 和终局 `Answer: ...`；
- strict-tag answer rate / EM 继续单独记录；
- 不修改 prompt、query、retrieval 或模型生成结果。

因此 Gate B 测量 query 对语义任务正确性的影响，而不是 prompt 缺少 answer-tag 约束造成的格式噪声。

## 数据充分性

| 指标 | 结果 |
| --- | ---: |
| States | 1000 |
| 有效 query branches | 7595 |
| Query pairs | 25175 |
| 出现组内 EM 变化的 states | 452 / 1000（45.20%） |
| `|EM_i-EM_j|=1` 的 pairs | 4636 / 25175（18.42%） |
| Branch-level semantic EM | 2618 / 7595（34.47%） |
| Semantic answer rate | 6248 / 7595（82.26%） |
| Strict-tag answer rate | 4517 / 7595（59.47%） |
| 第二次 search 被终止 | 1254 / 7595（16.51%） |
| Response clipped | 0 |

预注册的数据充分性要求为至少 100 个变收益 states 和 500 个变收益 pairs，实际结果均显著超过门槛。

## 主结果

主比较固定为：

- 语言距离：Qwen policy query embedding cosine distance；
- 环境距离：top-k retrieval score distribution 的 Jensen-Shannon distance；
- 目标：预测 query pair 的 `|EM_i-EM_j|`；
- 主指标：AUROC。

| 指标 | 环境 JS | Policy embedding | 环境优势 |
| --- | ---: | ---: | ---: |
| AUROC | **0.6599** | 0.5943 | **+0.0655** |
| Average Precision | **0.2673** | 0.2226 | **+0.0447** |
| Spearman | **0.2159** | 0.1267 | **+0.0893** |

AUROC 增益的 state-cluster bootstrap 结果：

- 平均增益：`+0.0665`；
- 95% CI：`[+0.0477, +0.0878]`；
- bootstrap 中环境距离胜出的比例：`100%`。

预注册要求为 AUROC 至少提升 0.03 且 95% CI 下界大于 0，**正式结果通过 Gate B**。

## 最强基线比较

所有语言代理中表现最好的是 normalized edit distance，AUROC 为 `0.6302`。环境 JS 仍达到 `0.6599`：

- 增益：`+0.0297`；
- 95% CI：`[+0.0099, +0.0446]`；
- bootstrap 胜出比例：`100%`。

这项比较是看过正式结果后选出的 strongest observed baseline，因此作为探索性稳健性结果，不替代预注册主比较。

## 稳健性检查

### 两个分支都有语义答案

排除第二次搜索、无答案等终止失败后，保留 18472 个 pairs，其中 2800 个 pair 的 EM 不同：

- 环境 JS AUROC：`0.7410`；
- 最强语言距离 AUROC：`0.6822`；
- 增益：`+0.0588`；
- 95% CI：`[+0.0359, +0.0830]`。

### 两个分支都有严格 `<answer>` 标签

只保留 10604 个 strict-format pairs，其中 1547 个 pair 的 EM 不同：

- 环境 JS AUROC：`0.7490`；
- 最强语言距离 AUROC：`0.6887`；
- 增益：`+0.0603`；
- 95% CI：`[+0.0396, +0.0832]`。

因此主结论不是由格式失败或无答案样本制造的。上述筛选会条件化终止行为这一中间变量，所以它们仅作为 robustness check，主估计仍使用全部分支。

## 代表案例

### 语言很远，但环境和收益等价

问题：`when is the womens ice skating for the olympics?`

- Query A：自然语言问句；
- Query B：对应 Wikipedia URL；
- Policy embedding distance：`0.9718`；
- Document Jaccard distance：`0`；
- Retrieval JS：`0.0013`；
- 两个分支均回答 `21 February`，EM 均为 1。

语言 trust region 会把这两个动作视为相距很远，但它们诱导几乎相同的环境转移和任务收益。

### 语言很近，但环境和收益分叉

问题：`who monitor the recovery of the location during a disaster?`

- Query A：`who monitor the recovery of the location during a disaster?`
- Query B：`what they call authorities to monitor the recovery of a disaster？`
- Policy embedding distance：`0.0275`；
- Document Jaccard distance：`1.0`；
- Retrieval JS：`0.8326`；
- Query A 回答 `management team`，EM=1；
- Query B 回答 `government`，EM=0。

语言表示认为两个动作非常接近，但它们实际进入不同检索状态并产生不同任务收益。

## 限制

1. 当前只有 NQ、一个 E5 retriever 和一个 3B checkpoint。
2. 模型原本经过 parallel-search 训练，本实验通过协议将其限制为单 query；后续需使用原生 single-query checkpoint 复现。
3. 终局 continuation 使用 greedy，结论尚未覆盖 sampling-induced return variance。
4. Retriever 使用已缓存的 deterministic top-3 结果，尚未验证 BM25 或其他 dense retriever。
5. AUROC 虽显著提升，但绝对预测能力仍为中等，不能把 retrieval distance 当作完整 value model。

## 决策

- Gate A：NQ 上存在广泛局部 language/environment geometry mismatch，已通过。
- Gate B：environment distance 对 downstream return difference 的预测显著优于 language proxy，已通过。
- 下一步：在 HotpotQA、2WikiMultiHopQA、MuSiQue 上复现 Gate A/B，并至少增加 BM25 对照；通过跨数据集复现后再进入最小 EITR-GRPO 实现。

## 产物

- 协议：`docs/eitr/gates/eitr_nq_gate_b_protocol.md`
- 脚本：`scripts/eval/eitr_nq_gate_b.py`
- 结果摘要：`results/eitr/gate_b/analysis/summary.json`
- 代表案例：`results/eitr/gate_b/analysis/representative_examples.json`
- 完整 branch 与 pair outcomes 属于大体积实验产物，未迁移到代码仓库；仓库仅保留统计摘要和代表案例。
