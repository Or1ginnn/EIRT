# EITR Gate A：NQ 单查询先导实验

日期：2026-08-05

> 历史实验说明：本报告使用已训练的 Parallel Search Step900 策略，并在采样时强制限制为单 query。它用于验证经验前提，不是原始 Search-R1 基座上的最终论文结果。正式实验需要在本项目的干净 Search-R1 设置上复现。

## 1. 实验目的

在不进行 RL 训练的前提下，检验 EITR 的第一个经验前提：对于同一个 Search Agent 状态，不同单查询动作在模型语言空间中的接近程度，是否能够可靠代表它们经过检索器后产生的环境后果。

本实验仅是 NQ 先导，不作为“跨数据集普遍现象”的最终证据。

## 2. 实验协议

- 数据：NQ test，共 3610 条；固定随机种子 `20260805` 抽取 1000 条。
- 模型：`parallel_search_qwen25_3b_step900`，vLLM BF16 推理。
- 单查询约束：prompt 明确禁止 `<plan>` 和 `||`；先贪心生成到 `<search>`，再从完全相同的前缀独立采样 8 次，并在 `||` 处停止。
- 采样：`temperature=1.0`，`top_p=1.0`，每个 state 采样 8 个 query。
- 检索器：Search-R1 同款 `intfloat/e5-base-v2`，Wikipedia 2018 语料，共 21,015,324 passages，FAISS Flat，`topk=3`。
- 资源：GPU 0 用于 Qwen 采样和离线 embedding；GPU 2、3 用于 E5 检索；GPU 1 全程留空。
- 统计：所有置信区间按 state 做 500 次 cluster bootstrap，不把同一 state 内的 query pair 当作独立样本。

模型侧距离：

- Qwen policy hidden-state cosine distance，作为主语言距离；
- token Jaccard distance；
- normalized edit distance；
- 每 token sequence log-prob gap；
- E5 query embedding distance，仅作为与 retriever 耦合的对照项。

环境侧距离：

- document-set Jaccard distance，作为主环境距离；
- Rank-Biased Overlap distance；
- normalized retrieval-score JS distance；
- retrieved-evidence embedding distance。

## 3. 数据质量

| 指标 | 结果 |
|---|---:|
| States | 1000 |
| 计划采样 query | 8000 |
| 有效单 query | 7595（94.94%） |
| 空/无效 search action | 405（5.06%） |
| 每个 state 平均不同 query 数 | 5.128 |
| 有效 query 的 gold evidence hit | 61.20% |
| 有效 query pairs | 25,175 |

405 条无效动作是模型直接结束 `<search>` 等真实行为，实验将其单独记录，没有将其修补为人工 query。

## 4. 主要结果

### 4.1 全局几何并非完全错位

主指标 Qwen policy embedding distance 对 document Jaccard distance：

- overall Spearman rho：**0.4350**；
- state-cluster bootstrap 95% CI：**[0.4146, 0.4552]**；
- state-level rho 中位数：**0.3996**；
- state-level IQR：**[0.1961, 0.6317]**。

因此，预先设置的强门槛 `|rho| < 0.3` 且置信区间端点小于 0.4 **没有通过**。在 NQ 单跳自然采样上，语言变化与检索变化存在中等程度的单调相关，不能声称两种几何整体近乎无关。

### 4.2 局部错位稳定存在

| 错位类型 | Pair 数 | Pair 比例 | 覆盖 state |
|---|---:|---:|---:|
| Qwen embedding 远、document set 近（四分位定义） | 944 | 3.75% | 255 / 1000 |
| Qwen embedding 近、document set 远（四分位定义） | 278 | 1.10% | 126 / 1000 |
| token Jaccard 距离 >= 0.5，但 top-3 文档集合完全相同 | 870 | 3.46% | 262 / 1000 |
| token Jaccard 距离 <= 0.25，但 document Jaccard 距离 >= 0.8 | 906 | 3.60% | 241 / 1000 |

两类直观错位合计覆盖 **480 / 1000 states（48.0%）**。此外：

- 29.63% 的全部 pair 返回完全相同的 top-3 文档集合；
- 22.71% 的全部 pair 连 top-3 排序也完全相同；
- 模型语言空间最近邻同时属于文档空间最近邻集合的比例为 69.63%，即约 30.37% 的局部最近邻关系不一致。

这说明 NQ 不支持“全局低相关”的强叙述，但支持更严谨的表述：**语言几何能够解释一部分检索变化，却不能完整编码检索器形成的局部等价类与局部不连续性。**

## 5. 典型反例

### 5.1 语言差异大，环境后果相同

问题：`when is the womens ice skating for the olympics?`

Query A：`When is the womens ice skating for the Olympics?`

Query B：`https://en.wikipedia.org/wiki/Women's_ice_skating_at_the_2018_ Winter_Olympics`

两者 Qwen embedding distance 很大，但返回相同的 top-3 文档集合。这是“不同表述、相同工具后果”的直接例子。

### 5.2 文本几乎相同，环境后果完全不同

问题：`who helped them recapture mycenae once they were old enough to fight?`

Query A：`who helped them recapture mycenae once they were old enough to fight?`

Query B：`who helped them recapture mycenae once they were old enough to fight`

距离与后果：

- normalized token/edit distance：0；
- Qwen policy embedding distance：0.0014；
- E5 query embedding distance：0.0406；
- document Jaccard distance：1.0；
- RBO distance：1.0；
- retrieval-score JS distance：0.8326。

Query A 返回 `Wars of the Delian League / Mycenae / First Messenian War`，Query B 返回三个不同的 `Mycenae` passages。该例已从原始 trajectory 和 retriever 返回中复核，不是 pair 映射错误。

## 6. 为什么 E5-E5 相关性不能作为主结论

若同时使用 E5 表示 query 和 retrieved evidence，query embedding distance 对 document Jaccard distance 的 rho 达到 0.8166。这主要反映了 E5 retriever 自身的几何：检索正是由同一个 E5 query vector 驱动，因此两者存在结构性耦合。

正式主指标改用 Qwen policy hidden-state distance 对 document-set distance；E5-E5 结果只保留为 coupled control。

## 7. Gate A 判断

当前判断：**NQ 单数据集强 Gate A 未通过，但局部错位前提得到支持，值得继续做跨数据集验证。**

不能据此直接开始 EITR-GRPO，也不能停止整个 idea。原因是：

1. NQ 主要是单跳问答，不是 proposal 最关心的多跳 search state；
2. 当前模型是 NQ Step900 并行检索模型，通过 action 约束取单 query，尚不是独立训练的单查询 checkpoint；
3. sequence log-prob gap 只是采样概率代理，不是真正的 old/new policy KL；
4. 只验证了固定 E5 retriever，尚未验证 BM25 或其他 dense retriever。

## 8. 下一步

1. 用相同协议复现 HotpotQA、2WikiMultiHopQA、MuSiQue，检验错位是否在多跳状态显著增强。
2. 至少增加 BM25 对照，判断现象是否跨 retriever 稳定。
3. 加入 old/new checkpoint cross-scoring，计算真实 token-policy drift proxy，而不是只看单 checkpoint 内的 log-prob gap。
4. Gate A 跨数据集成立后，再做 branch continuation，比较 language distance 与 retrieval-effect distance 对 terminal return change 的解释力，即 Diagnostic 2。

## 9. 产物

- 统一脚本：`scripts/eval/eitr_nq_gate_a.py`
- 结果摘要：`results/eitr/gate_a/summary.json`
- 典型样例：`results/eitr/gate_a/mismatch_examples.json`
- 完整 states 与 pairs 属于大体积实验产物，未迁移到代码仓库；仓库仅保留统计摘要和代表案例。
