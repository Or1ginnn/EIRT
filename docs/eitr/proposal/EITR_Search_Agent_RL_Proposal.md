# Beyond Token-Space Proximity: Environment-Induced Trust Regions for Search-Agent Reinforcement Learning

> 项目主版本现位于独立的 EITR-Search-Agent 仓库，以官方 Search-R1 单 query 训练链路为实验底座。Parallel Search 与 Finance Agent 仅作为前期观察来源，不再作为 Gate C 的训练初始化。

> **研究方案 / Proposal v0.1**
> 日期：2026-08-05
> 暂定简称：**EITR**（Environment-Induced Trust Region）
> 目标：做一篇以 **RL algorithm / policy optimization** 为核心创新，而不是 reward shaping 的 Search-Agent 后训练论文。

---

## 0. 一句话版本

现有 Search-Agent RL（PPO / GRPO 及其变体）主要在 **token / action text space** 中定义“新旧 policy 是否足够接近”，但 Search Agent 的工具动作真正影响任务的方式，是通过

$$
\text{query} \rightarrow \text{retriever / browser} \rightarrow \text{observation} \rightarrow \text{future state}
$$

产生环境后果。

本项目的核心假设是：

> **对语言 Agent 而言，“小的 policy update”应该更多由它对环境 transition / observation distribution 的改变来定义，而不应仅由 token-space proximity 定义。**

因此，我们希望把 Search Agent 的 proximal policy optimization 从

$$
D(\pi_{\theta}(a\mid s),\pi_{\text{old}}(a\mid s))
$$

改造成或扩展为

$$
D\big(P_{\theta}(o'\mid s),P_{\text{old}}(o'\mid s)\big),
$$

其中

$$
P_{\theta}(o'\mid s)
=\int T(o'\mid s,a)\pi_{\theta}(a\mid s)\,da
$$

是 policy 经过 tool/environment transition kernel 后诱导出的 **pushforward / environment-induced distribution**。

训练 reward 保持最简单的 outcome reward，不增加 process reward、PRM 或 dense shaping。算法创新发生在 **policy update geometry / trust-region constraint** 本身。

---

# 1. 研究目标与选题标准

这篇工作需要满足以下标准：

1. **不靠 reward shaping** 获得主要收益。
2. reward、任务和 retriever 在主要对比中尽量保持固定。
3. 主要修改 **RL 的优化目标、proximal constraint 或 gradient geometry**。
4. 能指出现有 PPO / GRPO 在 Agentic RL 中一个结构性的假设错位，而不是单纯说“训练不稳定”。
5. Search Agent 是最清晰的实验场，但原理应能推广到 browser、coding、OS/tool agents。
6. 方法需要有一个可以被快速证伪的核心 hypothesis；如果 empirical premise 不成立，应尽早停止，而不是堆工程。

期望的论文定位不是：

> “我们提出一个更好的 Search-R1 trick。”

而是：

> **“What is the right notion of a small policy update for language agents that act through tools?”**

---

# 2. 为什么现有 token-space proximal optimization 可能错位

## 2.1 Search Agent 的决策粒度不是普通 next-token generation

一个典型 Search Agent trajectory 可以写成：

```text
Question
  ↓
reasoning
  ↓
search query q1
  ↓
retrieved documents o1
  ↓
reasoning
  ↓
search query q2
  ↓
retrieved documents o2
  ↓
...
  ↓
final answer
  ↓
terminal reward
```

对于普通语言生成，token 本身就是输出。

而对 Search Agent，工具 action 的作用需要经过外部环境：

$$
a_t=q_t
\xrightarrow{\;T\;}
o_{t+1}
\xrightarrow{}s_{t+1}.
$$

因此，一个 query 在语言上变化多大，与它把 Agent 带到多不同的信息状态，并不是同一个概念。

---

## 2.2 两类关键现象

### 现象 A：语言差异大，但环境后果接近

例如：

```text
Christopher Nolan spouse
Who is the wife of the director of Oppenheimer?
```

两个 query 的 lexical / sequence-level difference 可以很大，但可能返回高度重叠的文档集合。

从环境角度，它们可能是近似等价动作。

如果 proximal constraint 只看语言空间，就可能对这些“行为等价但表述不同”的方向过度保守。

### 现象 B：局部文本改动很小，但单次环境后果变化显著

例如：

```text
Christopher Nolan wife
Jonathan Nolan wife
```

只改一个实体词，返回结果可能完全变化。

这里需要非常严谨地区分两个陈述：

- **样本级 / 局部 token 变化小** 并不保证某次 retrieval outcome 变化小；
- 但对于完整 action distribution 的精确 KL，经过固定 Markov kernel 后满足 data-processing contraction，因此不能简单声称“精确 action KL 很小而 induced KL 任意大”。

所以本工作的理论动机不能建立在一个错误命题上。

更准确的主张是：

> 现有 LLM RL 实际使用的 token-level clipping、sampled importance ratios 和局部 KL proxy，并不直接编码工具 action 的环境等价关系；与此同时，完整 action-space KL 对环境行为 divergence 是一个通常较保守的上界，会惩罚被环境 kernel 消除掉的语言差异。

这个“**proxy misalignment + conservative geometry**”才是我们需要验证和利用的核心。

---

# 3. 与现有工作的边界：什么已经有人做了，什么还没有

## 3.1 Search-R1：证明 outcome-only RL 可以学出搜索行为

Search-R1 让模型通过 RL 学会 interleaved reasoning + multi-turn search，并使用简单 outcome reward。它是本项目最自然的基础训练环境之一。

本项目不挑战“RL 是否能学搜索”，而挑战：

> **Search-Agent RL 应该在哪个 policy geometry 上做 proximal update？**

---

## 3.2 SAPO：仍然在 token-policy space 修稳定性

SAPO（2026）指出 GRPO Search Agent 训练中存在 Importance Sampling Distribution Drift，并通过 conditional token-level KL 稳定低概率 positive tokens。

它解决的是：

$$
\text{如何更好地约束 token-policy drift}
$$

而本项目的问题是：

$$
\text{token-policy drift 是否是 tool agent 最合适的行为距离？}
$$

两者可以直接形成清晰的 related-work 对照。

---

## 3.3 APPO / T²PO / AT²PO / BPO：主要改 exploration、branching、credit 或决策粒度

- **APPO**：寻找高影响 decision points，做 fine-grained branching 与 procedure-level advantage scaling。
- **T²PO**：从 token 与 turn 层控制无效探索。
- **AT²PO**：turn-level tree expansion + turn-wise credit + turn-based objective。
- **BPO**：利用可 snapshot 的环境在中间状态 branch，并从 sibling returns 构造低方差 advantage。

这些工作说明 Agentic RL 不能简单照搬 RLHF rollout topology，但它们并未把 **environment-induced transition distribution** 作为 proximal / trust-region 的核心几何对象。

---

## 3.4 BiPACE：一个必须正面区分的最近工作

BiPACE（2026-06）已经把 bisimulation / behavioral equivalence 引入 LLM Agent RL。

但其核心是：

- 用 actor hidden-state geometry 做近似 bisimulation clustering；
- 改善 step grouping；
- 构造 action-conditioned counterfactual baseline；
- 本质上修改 **advantage / credit estimator**。

因此，我们绝不能把 novelty 写成：

> “第一次在 LLM Agent 中考虑 behavioral equivalence。”

我们的边界必须是：

> **BiPACE changes which samples are comparable for credit assignment; EITR changes what constitutes a proximal policy update.**

即：

$$
\text{BiPACE: credit geometry}
\qquad\neq\qquad
\text{EITR: update geometry}.
$$

---

## 3.5 经典 Behavior-Guided Policy Gradient：另一个必须认真处理的 prior

Pacchiano et al. 的 Behavior-Guided Policy Gradient（ICML 2020）已经提出在 latent behavioral space 中用 Wasserstein distance 比较 policy，并把 behavioral regularizer 用于 policy optimization。

所以本项目也不能声称：

> “第一次用 behavior-space distance 做 policy optimization。”

真正可以争取的 novelty 是更具体的三点：

1. **Agentic/tool RL-specific environment pushforward**：behavior 不由手工 trajectory descriptor 定义，而由 tool/environment transition kernel 直接诱导；
2. **Search-specific tractable estimator**：利用 query → retriever output 的结构，在不对 retriever 反传的情况下估计 induced divergence；
3. **proximal/trust-region interpretation**：把环境后果 divergence 作为 old/new policy update constraint，而非额外鼓励 novelty / behavior shaping 的 regularizer。

若这三点不能在理论和实验上做实，项目 novelty 会变弱。

---

# 4. 问题形式化

## 4.1 Tool-turn Semi-MDP / POMDP

把 Search Agent 在工具调用层建模为一个 semi-MDP。

状态：

$$
s_t = (x, h_t, o_{\le t}),
$$

其中：

- $x$：原始任务 / question；
- $h_t$：当前 Agent 内部可见历史；
- $o_{\le t}$：此前工具返回的 observations。

工具动作：

$$
a_t=q_t\in\mathcal A_s,
$$

其中 $q_t$ 是一段可执行 query / structured tool call。

环境 transition：

$$
o_{t+1}\sim T_o(\cdot\mid s_t,q_t).
$$

随后新的 Agent 状态由：

$$
s_{t+1}=F(s_t,q_t,o_{t+1})
$$

生成。

EITR 算法本身仅要求一个固定的 task reward。用于支撑“无额外 process reward”主张的
primary profile 只使用 terminal task reward：

$$
R(\tau)\in\{0,1\}
$$

或标准 EM / F1 / verifier reward。为诊断 Search-R1 v0.3 的 no-search collapse，可以
另设一个明确命名的 evidence-shaping profile；但它必须作为单独实验，不能冒充上述
outcome-only primary profile。

当前工程 formal runner 另设 `mandatory_search` profile，用于阻止NQ+HotpotQA训练中的
no-search shortcut。它用真实环境执行作为硬门，并采用

$$
R=G_{tool}F_{answer}\left(0.2F_{think}+0.1+1.2C\right),
$$

其中 $G_{tool}$ 只在真实Retriever调用、环境轨迹一致且模型没有伪造information时成立；
$F_{answer}$ 要求唯一、非空、正确闭合且位于末尾的最终答案，是第二个硬门；
$F_{think}$ 是软格式分，$C$ 是答案EM。证据是否命中不参与训练reward，只保留为诊断
指标；缺少think不会清空答案基础分或答案分，完整成功满分仍为1.5。
该组是明确的process-reward实验，不能作为“无reward
shaping”主张的唯一证据；
必须同时保留exact-v0.3对照，并让off/probe_only/eitr三组共享同一profile。

**核心要求：主要实验中 reward definition 对所有 RL baseline 保持相同。**

---

## 4.2 Policy-induced environment distribution

LLM policy 在工具 action space 上为：

$$
\pi_\theta(q\mid s).
$$

它通过 retriever / tool kernel 诱导 observation distribution：

$$
\bar\pi_\theta(o\mid s)
:=
\int T_o(o\mid s,q)\pi_\theta(q\mid s)dq.
$$

这可以视为 policy distribution 经环境 kernel 的 pushforward。

为了避免“只看 observation 而忽略 immediate cost / invalid action”等漏洞，也可以定义更一般的 **environment effect variable**：

$$
e=(o,c,r_{\text{imm}},v),
$$

其中：

- $o$：tool observation；
- $c$：tool-call cost / latency / token cost；
- $r_{\text{imm}}$：若环境有 immediate reward；
- $v$：validity / execution status。

定义 effect kernel：

$$
B_s(e\mid a).
$$

对应 policy-induced effect distribution：

$$
\bar\pi^B_\theta(e\mid s)
=
\int B_s(e\mid a)\pi_\theta(a\mid s)da.
$$

理论上以 $\bar\pi^B$ 为对象；Search 实现中可以主要用 retrieval observation distribution 近似。

---

# 5. 核心方法：Environment-Induced Trust Region

## 5.1 从 action-space trust region 到 effect-space trust region

标准 TRPO 类思路约束：

$$
\mathbb E_s
\left[
D_{KL}
\left(
\pi_{old}(\cdot\mid s)
\|\
\pi_\theta(\cdot\mid s)
\right)
\right]
\le \delta.
$$

EITR 希望约束：

$$
\boxed{
\mathbb E_s
\left[
D_B
\left(
\bar\pi^B_{old}(\cdot\mid s),
\bar\pi^B_\theta(\cdot\mid s)
\right)
\right]
\le \delta_B
}
$$

其中 $D_B$ 可以先从 JS / TV / kernel-MMD 等可实现形式开始。

核心语义是：

> 新 policy 可以在语言表述上变化较大，只要它没有不受控地改变 Agent 实际访问的信息状态；反之，如果某些 query variation 会显著改变环境访问分布，policy update 应对这种方向更加谨慎。

---

## 5.2 推荐使用 Hybrid Trust Region，而不是纯 behavioral constraint

完全放弃 token-space constraint 有风险，因为：

1. reasoning text 本身会进入后续 context；
2. query surface form 可能影响未来自回归状态；
3. 行为 kernel 近似不可能捕捉全部 downstream effect；
4. 可能出现 syntax / tool-format degeneration。

因此第一版方法建议采用 **decoupled hybrid constraint**：

### 对非工具 reasoning tokens

继续保留标准 token-level PPO / GRPO proximal machinery。

### 对 tool-action turn

主要加入 environment-induced trust-region constraint，并保留一个很弱的 token-level safety constraint。

整体形式：

$$
\max_\theta L_{PG}(\theta)
$$

subject to

$$
\mathbb E[D_B]\le \delta_B,
$$

以及可选的

$$
\mathbb E[D_{tok}^{tool}]\le \delta_{tok}^{max},
$$

其中 $\delta_{tok}^{max}$ 可以明显比普通 PPO 更宽松。

这个设计允许我们做一个关键实验：

> **在保持相同 behavioral drift 的条件下，EITR 是否允许更大的语言-space update，并获得更快 / 更稳的 policy improvement？**

---

## 5.3 Lagrangian 版本

定义：

$$
\mathcal L(\theta,\beta)
=
L_{PG}(\theta)
-
\beta
\left(
\hat D_B(\theta)-\delta_B
\right),
$$

其中 $\beta\ge 0$ 是 dual variable。

更新：

$$
\theta
\leftarrow
\theta + \eta_\theta
\nabla_\theta \mathcal L,
$$

$$
\beta
\leftarrow
\left[
\beta+\eta_\beta(\hat D_B-\delta_B)
\right]_+.
$$

这比固定 penalty coefficient 更符合“trust region / constrained policy optimization”的论文定位，也能避免被误解成 reward shaping。

注意：

- reward 本身没有改变；
- $D_B$ 不是 trajectory reward；
- 它只限制 **一次 policy update 允许改变多少 environment behavior**。

---

# 6. 一个重要理论事实：为什么 behavioral trust region 有可能比 action KL 更合理

## 6.1 Data-processing / contraction

如果 $B_s(e\mid a)$ 是固定 Markov kernel，那么对于很多 f-divergence（包括 KL），存在 data-processing inequality：

$$
D_f
\left(
B_{s\#}\pi_{old},
B_{s\#}\pi_\theta
\right)
\le
D_f
\left(
\pi_{old},
\pi_\theta
\right).
$$

这意味着：

> action-space divergence 会惩罚一些最终被环境 kernel “压缩掉”的区别。

如果两个语言动作虽表面不同，但环境效果近似相同，那么严格 action KL 可以明显比 behavioral divergence 更大。

因此 EITR 的主要理论价值，不应该宣传成“更安全地捕捉比 action KL 更大的 drift”，而应该是：

> **在保持环境行为变化受控的前提下，减少 action-space trust region 的不必要保守性。**

这是一个比早期直觉更严谨、更可 defend 的理论故事。

---

## 6.2 行为等价动作

给定状态 $s$，若两个动作满足：

$$
B_s(\cdot\mid a_i)=B_s(\cdot\mid a_j),
$$

并且环境 reward / cost 也一致，则它们在 one-step effect 上等价。

若这种等价在后续 Markov dynamics 中保持，则可以建立 action equivalence / bisimulation-style argument：

> 在等价类内部重新分配 policy probability mass，不应显著改变任务 value。

而普通 token/action KL 仍然可能对这种 redistribution 收取很大“距离成本”。

这给 EITR 提供了最直观的理论例子。

---

# 7. 希望证明的理论结果

下面是 proposal 阶段建议争取的定理，不应在还没推导完成前写成已证明结论。

## Theorem Target 1：Environment-equivalent policy invariance

若两个 policy $\pi,\pi'$ 在所有相关状态满足：

$$
\bar\pi^B(\cdot\mid s)
=
\bar\pi'^B(\cdot\mid s),
$$

且 reward 仅依赖于 environment effect / resulting state，则希望证明：

$$
V^\pi(s)=V^{\pi'}(s).
$$

意义：

> policy 的语言 realization 可以不同，但只要诱导 dynamics 相同，其 control behavior 等价。

---

## Theorem Target 2：Value difference bound by induced transition divergence

希望从 simulation lemma / performance difference lemma 出发，证明类似：

$$
|J(\pi')-J(\pi)|
\le
C_1 \epsilon_r
+
C_2
\mathbb E_{s\sim d_\pi}
\left[
D_{TV}
\left(
P_{\pi'}(s'\mid s),
P_\pi(s'\mid s)
\right)
\right],
$$

其中

$$
P_\pi(s'\mid s)
=
\int P(s'\mid s,a)\pi(a\mid s)da.
$$

如果 reward 也纳入 effect kernel，则可以进一步统一成对 effect-distribution discrepancy 的 bound。

意义：

> 真正影响 value drift 的一个自然对象是 policy-induced transition drift，而不是语言动作本身的表面距离。

---

## Theorem Target 3：Action KL 是 induced KL 的保守上界

利用 data-processing inequality：

$$
D_{KL}
(\bar\pi^B_{old}\|\bar\pi^B_\theta)
\le
D_{KL}
(\pi_{old}\|\pi_\theta).
$$

进一步给出一个 constructed example：

- 多个 query 文本完全不同；
- retriever 将它们映射到相同文档 distribution；
- action KL 可以非零甚至很大；
- induced divergence 为 0；
- environment return 完全相同。

这个例子非常适合放在论文 Figure 1 / Proposition 1 附近。

---

# 8. 最大的实现问题：我们无法枚举所有 query

理论定义：

$$
\bar\pi_\theta(o\mid s)
=
\sum_q
T(o\mid s,q)\pi_\theta(q\mid s)
$$

几乎不可直接计算，因为 query 是开放文本。

同时我们也不希望：

- 对搜索引擎反向传播；
- 为每个 gradient step 重新调用大量搜索 API；
- 建一个额外 reward model。

因此方法是否真正成立，很大程度取决于能否设计一个 **cached, differentiable-through-policy-only** estimator。

---

# 9. 推荐的可实现 estimator：Behavioral Probe + Self-Normalized Importance Reweighting

## 9.1 在 rollout 阶段采样少量“behavioral probes”

对一个真实 search state $s$：

1. 正常 trajectory 产生主 query；
2. 额外从 old policy 采样 $K-1$ 个 query probe：

$$
q_i\sim \pi_{old}(\cdot\mid s),\quad i=1,...,K;
$$

3. 对每个 $q_i$ 只执行一次 retrieval，不继续完整 suffix rollout；
4. 缓存：

$$
(q_i,\log\pi_{old}(q_i\mid s),R_i),
$$

其中 $R_i$ 表示 retriever 的 top-k results / scores。

关键区别：

> 这不是 BPO / APPO 那种“branch 后一路 rollout 到 terminal”——probe 只用于估计 local environment geometry，不产生额外 task reward。

因此成本主要是额外 retrieval，而不是额外长轨迹生成。

---

## 9.2 当前 policy 对 probe 的 self-normalized weights

在 policy update 时，不需要再次调用 retriever。

计算：

$$
\rho_i(\theta)
=
\exp
\left(
\log\pi_\theta(q_i\mid s)
-
\log\pi_{old}(q_i\mid s)
\right).
$$

归一化：

$$
\tilde w_i(\theta)
=
\frac{\rho_i(\theta)}
{\sum_j\rho_j(\theta)}.
$$

old policy 下 probe 权重近似为：

$$
w_i^{old}=1/K.
$$

这样我们可以在 **不对 retriever 求梯度** 的情况下，估计 current policy 对已缓存环境后果的 probability mass redistribution。

梯度只通过：

$$
\log\pi_\theta(q_i\mid s)
$$

回传。

---

# 10. Search 场景下最推荐的 induced retrieval distribution

假设 retriever 对 query $q_i$ 返回 top-k docs：

$$
R_i=\{(d_{ij},z_{ij})\}_{j=1}^k,
$$

其中 $z_{ij}$ 是 retrieval score。

先转成一个归一化 doc distribution：

$$
p_R(d\mid q_i)
=
\operatorname{softmax}(z_{ij}/\tau_R).
$$

若是 web API 没有原始 score，可以用 rank-based weights：

$$
p_R(d_j\mid q_i)
\propto
\frac{1}{(j+c)^\alpha}.
$$

则 old induced retrieval distribution：

$$
\hat p_{old}(d\mid s)
=
\frac1K\sum_i p_R(d\mid q_i).
$$

current-policy induced distribution：

$$
\hat p_\theta(d\mid s)
=
\sum_i \tilde w_i(\theta)p_R(d\mid q_i).
$$

在所有 probe 返回文档的 union support 上计算：

$$
\hat D_B
=
JS
\left(
\hat p_{old}(d\mid s),
\hat p_\theta(d\mid s)
\right).
$$

优点：

1. 可微性只依赖 policy weights；
2. retriever 可完全 black-box；
3. query paraphrases 返回相同 docs 时自然视为近似等价；
4. 不需要额外 reward model；
5. 可以精确缓存和重放。

---

# 11. 第二种 estimator：Kernel Behavioral MMD

当 document ID overlap 太低、web 页面动态变化、或 observation 本身是长文本时，可以定义 behavior kernel：

$$
k_B(e_i,e_j).
$$

例如：

$$
k_B
=
\lambda_1\cdot \text{Jaccard@K}
+
\lambda_2\cdot \text{RBO}
+
\lambda_3\cdot \cos(E(o_i),E(o_j)).
$$

令 Gram matrix：

$$
K_{ij}=k_B(e_i,e_j).
$$

则 weighted empirical MMD 可以写成：

$$
\widehat{MMD}^2
=
(w_\theta-w_{old})^T
K
(w_\theta-w_{old}).
$$

这里：

$$
w_{old}=[1/K,...,1/K],
$$

$$
w_\theta=[\tilde w_1,...,\tilde w_K].
$$

这也是完全可微的。

建议：

- **理论正文用 induced TV / KL / JS**；
- **主实验优先用 doc-distribution JS**；
- MMD 作为 general-tool extension 或 ablation。

这样论文会更干净。

---

# 12. EITR-GRPO：第一版完整算法

## 输入

- 初始 policy $\pi_{\theta_0}$；
- fixed retriever / search environment；
- outcome verifier；
- 每个 prompt 的正常 rollout 数 $N$；
- 每个被探测 search state 的 query probes $K$；
- behavioral trust-region target $\delta_B$；
- dual learning rate $\eta_\beta$。

## Rollout Phase

对每个 prompt：

1. 用 $\pi_{old}$ 采样 $N$ 条正常 Search-Agent trajectories；
2. 得到 terminal outcome reward；
3. 用普通 GRPO / REINFORCE / PPO 方式构造 trajectory advantage；
4. 对选中的 search turns：
   - 固定完全相同的 prefix state $s_t$；
   - 从 $\pi_{old}$ 额外采样少量 query probes；
   - 调用 retriever；
   - 缓存 query logprob 与 retrieval distribution；
   - **不继续 probe suffix rollout**。

## Update Phase

对每个 SGD epoch：

1. 计算标准 RL surrogate：

$$
L_{PG}(\theta)
$$

2. 对 probe queries 重算 current sequence logprob；
3. 用 self-normalized importance weights 得到 $\hat p_\theta(d\mid s)$；
4. 计算：

$$
\hat D_B
=
JS(\hat p_{old},\hat p_\theta);
$$

5. 优化：

$$
L_{EITR}
=
L_{PG}
-
\beta(\hat D_B-\delta_B);
$$

6. 更新 dual：

$$
\beta
\leftarrow
[\beta+\eta_\beta(\hat D_B-\delta_B)]_+.
$$

7. 对 reasoning tokens 保持标准 proximal mechanism；对 query action chunk 使用 EITR 主约束 + weak token safety constraint。

---

# 13. 伪代码

```text
Algorithm: EITR-GRPO

Initialize policy πθ
Initialize dual variable β ≥ 0

for training iteration k = 1 ... K_train:
    πold ← πθ

    # 1. Normal on-policy rollouts
    trajectories ← rollout_search_agent(πold, prompts)
    rewards ← terminal_verifier(trajectories)
    advantages ← group_relative_advantage(rewards)

    # 2. Behavioral probing
    probe_buffer ← {}
    for selected search state s in trajectories:
        queries q1...qK ~ πold(.|s)
        for qi in queries:
            retrieval Ri ← Search(qi)
            old_logp_i ← log πold(qi|s)
            cache(s, qi, old_logp_i, Ri)

    # 3. Multiple policy-update epochs
    for epoch = 1 ... E:
        L_pg ← standard_GRPO_surrogate(trajectories, advantages, πθ, πold)

        D_behavior ← 0
        for probed state s:
            for cached qi:
                logp_i ← log πθ(qi|s)
                ρi ← exp(logp_i - old_logp_i)
            wθ ← normalize(ρ)

            p_old_docs ← mixture(retrieval_distributions, uniform_weights)
            p_new_docs ← mixture(retrieval_distributions, wθ)

            D_behavior += JS(p_old_docs, p_new_docs)

        L ← L_pg - β * (D_behavior - δB)
        θ ← optimizer_step(∇θ L)

    β ← max(0, β + ηβ * (D_behavior - δB))
```

---

# 14. 为什么这不是 reward shaping

这是论文必须反复强调的点。

Reward shaping 会改变：

$$
R(\tau)
$$

例如：

$$
R'=R_{answer}+\lambda R_{search-step}.
$$

EITR correction 本身不改变任务 reward。在 `pure_em` primary profile 中：

$$
R'=R_{answer}.
$$

`official_v03` 是Search-R1发布的格式shaping对照，并不是纯二元outcome reward。

如果实验显式使用answer-bearing evidence shaping或`mandatory_search`硬门控，则只能
声称“所有方法共享相同reward，EITR不额外修改它”，不能再把该组实验描述成
outcome-only。必须同时保留`official_v03 / retrieval_score=0`对照，区分reward效果与
EITR效果。`mandatory_search`的训练reward范围为0到1.5，周期验证仍报告0到1的纯EM。

它改变的是一次 policy optimization step 的 feasible set：

$$
\theta_{k+1}
\in
\left\{
\theta:
D_B(\pi_{\theta_k},\pi_\theta)\le\delta_B
\right\}.
$$

因此论文主贡献属于：

- policy optimization；
- trust-region geometry；
- agentic RL training algorithm；

而不是：

- reward engineering；
- process supervision；
- PRM；
- intermediate credit shaping。

---

# 15. 最关键的第一阶段：先证明 premise，而不是直接训大模型

这个项目最好的地方是：**可以低成本证伪。**

完整 RL 训练之前必须先做三个 diagnostic。

---

## Diagnostic 1：Language distance 与 retrieval-effect distance 到底有多一致？

固定同一个 search state $s$，采样：

$$
q_1,...,q_K.
$$

计算多种 linguistic distance：

- token edit distance；
- query embedding cosine；
- sequence-level old-policy logprob difference；
- sampled token KL proxy；
- entity-level edit distance。

计算 behavioral distance：

- 1 - Jaccard@K；
- Rank-Biased Overlap；
- JS over normalized retrieval scores；
- retrieval-result embedding distance。

看 pairwise correlation。

**不是要求相关性为 0。**

真正要找的是：

1. 是否存在大量 high-linguistic / low-behavioral pairs；
2. token-level proxy 是否无法很好排序 behavioral equivalence；
3. 这种 mismatch 是否在多跳 Search 状态中稳定存在。

---

## Diagnostic 2：哪种 distance 更能预测 downstream consequence？

对小规模 subset，在同一个 state 从不同 query branch 后继续 rollout 到 terminal。

测：

$$
\Delta R_{ij}=|R_i-R_j|
$$

或：

$$
|V(s'_{i})-V(s'_{j})|.
$$

比较以下变量对 $\Delta R$ 的预测能力：

- linguistic distance；
- token KL proxy；
- retrieval JS；
- effect MMD。

如果 environment-induced distance 对 downstream return change 的预测力并不优于 token proxy，则 EITR 的核心故事会明显变弱。

---

## Diagnostic 3：训练中的 failure 是否更与 behavioral drift 相关？

复现 GRPO / Search-R1 / SAPO 的训练轨迹，记录：

- reward curve；
- token KL；
- importance-ratio statistics；
- retrieval behavioral drift；
- gradient norm；
- entropy；
- answer accuracy；
- search-call distribution。

分析：

> performance drop / collapse / sudden strategy shift 是更早被 token drift 还是 behavioral drift 预测？

如果 behavioral drift 没有任何额外解释力，那么做 trust region 的必要性会下降。

---

# 16. Go / No-Go 证伪门槛

建议在项目最早期设置明确 kill criteria。

## Gate A：现象是否真实

若不同 query 的 retrieval behavior 基本与语言距离高度一致，没有明显等价类结构：

> **停止 EITR。**

## Gate B：behavioral distance 是否与 downstream value 更相关

若 retrieval-effect distance 对 return change 的解释力不优于 token proxy：

> **降级该 idea，转 off-policy / replay 方向。**

## Gate C：小模型训练是否产生优化收益

在 1.5B / 3B 规模和固定 corpus 上，EITR 至少需要在以下某一项出现稳定改善：

- final accuracy；
- convergence speed；
- collapse rate；
- gradient variance；
- 相同 behavioral drift 下允许更大 optimization step。

如果所有指标均无改善：

> **不继续扩大规模。**

---

# 17. 实验设计

## 17.1 Primary benchmark：固定 corpus 的 Search-Agent RL

优先选择 deterministic / reproducible retriever，因为 EITR 依赖环境 effect 的可靠测量。

推荐：

1. Search-R1 风格 Wikipedia corpus；
2. multi-hop QA：HotpotQA / 2WikiMultiHopQA / MuSiQue；
3. single-hop + long-tail：NQ / TriviaQA / PopQA；
4. 如果工程可承受，加入 BrowseComp-Plus / controlled deep-search benchmark。

固定 corpus 的优势：

- retrieval transition 可缓存；
- 训练可复现；
- probe 成本低；
- document IDs 稳定；
- induced doc distribution 定义清楚。

---

## 17.2 Secondary benchmark：验证能否推广出 Search 之外

至少加入一个非 Search 环境，否则 reviewer 可能认为只是 retriever-specific heuristic。

候选：

### WebShop

动作：search / click / option selection。
环境 effect：page / URL / DOM observation。

### ALFWorld

动作：text command。
环境 effect：world observation / symbolic state。

### Browser / WebArena-lite

动作：browser action。
环境 effect：page state / URL / DOM embedding。

建议第一篇不要同时把所有环境做复杂。最优策略是：

> **Search 做完整方法与理论验证 + 一个 tool environment 做 generalization proof-of-concept。**

---

# 18. 模型规模

推荐分三级：

### Phase 1：1.5B / 3B

用于：

- diagnostic；
- estimator debug；
- 超参数；
- falsification。

### Phase 2：7B / 8B

用于正式主结果。

### Phase 3：14B+（可选）

如果主结果已经明确，再验证 scaling consistency。

论文的主要贡献不依赖“大模型绝对 SOTA”，因此不应该一开始就烧资源。

---

# 19. Baselines

需要分成三类。

## 标准 RL baseline

- GRPO / Search-R1；
- PPO；
- REINFORCE/RLOO（如果训练框架方便）。

## Search-Agent RL stability baseline

- SAPO；
- T²PO。

## Agentic policy-optimization / branching baseline

根据代码可用性选择：

- AT²PO；
- APPO；
- BPO。

## Behavioral / structural baseline

- BiPACE（如果能在 Search 环境合理复现）；
- token-KL / sequence-KL regularized GRPO；
- random / semantic-only behavior kernel ablation。

最重要的公平性条件：

> **所有主要 RL baseline 使用相同 final reward。**

---

# 20. 主指标

不仅看 final EM / F1。

## Capability

- EM / F1 / success rate；
- BrowseComp-style accuracy；
- multi-hop completion。

## Efficiency

- tool calls / answer；
- retrieval calls；
- generated tokens；
- training wall-clock；
- reward per environment interaction。

## Optimization

- gradient norm variance；
- importance-ratio distribution；
- fraction of clipped tokens；
- token KL；
- environment-induced divergence；
- policy entropy；
- training-collapse frequency across seeds。

## Geometry-specific metric

最关键的图之一：

$$
\text{performance improvement}
\quad vs \quad
\text{token drift}
\quad vs \quad
\text{behavioral drift}.
$$

理想结果是：

> EITR 在相似 behavioral drift 下允许更大 token-space movement，并因此更快获得 task improvement。

---

# 21. 关键消融

## Ablation A：Token-only vs Behavior-only vs Hybrid

1. Standard GRPO / PPO；
2. token KL only；
3. behavior constraint only；
4. token + behavior hybrid。

预测：

- pure behavior 可能有 surface degeneration；
- hybrid 最稳；
- token-only 最保守或对 paraphrase-equivalent query update 不够高效。

---

## Ablation B：Behavior representation

比较：

- doc ID overlap；
- rank-weighted doc distribution；
- result-text embedding；
- kernel MMD；
- random hash / random projection control。

如果任何 embedding 都能带来同样效果，则说明方法可能只是额外 regularization，不是真正 environment geometry。

---

## Ablation C：Probe 数 K

测试：

$$
K\in\{2,4,8,16\}.
$$

目标是在低 probe cost 下找到稳定 estimator。

---

## Ablation D：Probe 哪些 turns

- 每个 search turn；
- 随机 25%；
- 高 entropy turns；
- 高 retrieval-novelty turns。

注意：主方法不要过度依赖 entropy-guided branching，否则容易和 APPO / BPO / T²PO 纠缠。

最好证明：

> 随机/均匀 subsampling 就已有效；智能 probe selection 只是 efficiency extension。

---

## Ablation E：Dual constraint vs fixed penalty

- fixed $\lambda D_B$；
- adaptive dual target $\delta_B$。

如果 adaptive trust-region 显著更稳，可以加强“constrained optimization”定位。

---

## Ablation F：固定 retriever vs retriever shift

训练时 BM25 / dense retriever，测试时换：

- 不同 index；
- 不同 dense retriever；
- 轻微 corpus update。

这个实验回答：

> EITR 学到的是 current retriever-specific geometry，还是更一般的 search behavior？

---

# 22. Stress Tests：专门证明 token geometry 的不足

构造两个 controlled test suites。

## Suite 1：Semantic Paraphrase Equivalence

对同一 search intent 自动生成多个表述：

```text
Nolan spouse
Christopher Nolan married to whom
wife of Oppenheimer director
```

筛选出 retrieval result 高度一致的 query pairs。

观察标准 PPO/GRPO 是否对这些语言变化施加明显 proximal penalty。

---

## Suite 2：Entity-Sensitive Perturbation

只替换一个关键实体 / 年份 / relation：

```text
Christopher Nolan wife
Jonathan Nolan wife
```

或：

```text
2024 Nobel chemistry winner
2023 Nobel chemistry winner
```

观察 local token proxies 与 environment consequence 的排序一致性。

这两个 suite 会让 Figure 1 非常直观。

---

# 23. 预期 Figure 1

论文开头最好不是 architecture diagram，而是一个 empirical phenomenon。

```text
             Environment / Retrieval Drift
                       ↑
                       │     ●  small text edit,
                       │        large retrieval change
                       │
                       │
                       │ ● ● ●
                       │
                       │              ●  paraphrase:
                       │                 large text distance,
                       │                 same retrieval
                       └────────────────────────→
                           Linguistic / Token Drift
```

旁边放两个真实 query case。

主标题可以是：

> **Language-space proximity does not faithfully express tool-behavior equivalence.**

但正文要同时注明 data-processing nuance，避免过度声称精确 distribution-level KL 的反例。

---

# 24. 预期 Figure 2：方法图

```text
                    π_old
                      │
              same search state s
                      │
          ┌───────────┼───────────┐
          │           │           │
         q1          q2          qK
          │           │           │
       Search      Search      Search
          │           │           │
         R1          R2          RK
          └───────────┼───────────┘
                      │
          cached environment effects
                      │
     ┌────────────────┴────────────────┐
     │                                 │
old uniform weights             current IS weights
     │                                 │
 p_old(retrieval|s)            p_θ(retrieval|s)
     └────────────────┬────────────────┘
                      │
               induced JS / MMD
                      │
                trust-region dual
                      │
              policy-gradient update
```

这个图要突出：

> **retriever 不需要 gradient，结果只在 rollout 时缓存一次。**

---

# 25. Optimization 分析：我们真正希望看到什么

EITR 最理想的结果并不是简单 “accuracy +2”。

而是出现以下训练动力学：

### 标准 PPO / GRPO

- token KL 较快上升；
- clip fraction 高；
- 很多 paraphrase-equivalent query direction 被限制；
- update 变得保守或梯度利用率低。

### SAPO

- token importance-ratio collapse 得到改善；
- 但约束仍然主要基于语言分布。

### EITR

- token drift 可以更大；
- retrieval behavior drift 维持目标范围；
- reward 单调性 / 稳定性更好；
- 在相同 environment interaction budget 下学得更快。

如果只能看到“多了个 regularizer 所以稍微更稳”，论文力度会不足。

---

# 26. 潜在更强版本：Behavioral Natural Gradient

如果 EITR 的 constrained objective 证明有效，可以进一步从 local second-order geometry 推导一个更纯粹的 RL algorithm。

普通 natural policy gradient 使用 Fisher：

$$
F_{action}
=
\mathbb E
\left[
\nabla\log\pi(a|s)
\nabla\log\pi(a|s)^T
\right].
$$

我们可以考虑 induced effect distribution 的 Fisher：

$$
F_B
=
\mathbb E
\left[
\nabla\log \bar\pi^B(e|s)
\nabla\log \bar\pi^B(e|s)^T
\right].
$$

然后求：

$$
\Delta\theta
\propto
F_B^{-1}g.
$$

直观上：

> 参数更新沿“不会强烈改变 environment behavior”的方向可以更大胆；沿高 behavioral sensitivity 方向更加谨慎。

这个版本理论味更强，但实现和数值稳定性明显更难。

**建议不要把它作为第一版必须完成的主方法。**

第一篇先把 induced trust region 证明成立，再考虑 behavioral natural gradient 作为 extension。

---

# 27. 一个可能更好的数学 framing：Quotient Action Space

给定状态 $s$，定义近似行为等价关系：

$$
a_i\sim_s a_j
\quad\Longleftrightarrow\quad
B_s(\cdot|a_i)\approx B_s(\cdot|a_j).
$$

则语言 action space：

$$
\mathcal A
$$

被投影到一个 behavioral quotient space：

$$
\mathcal A / \sim_s.
$$

传统 LLM proximal RL 在：

$$
\mathcal A
$$

中测量 policy movement。

EITR 希望更接近在：

$$
\mathcal A / \sim_s
$$

上测量 movement。

这个 framing 很适合理论部分，但需要谨慎使用“bisimulation”术语，因为已有大量经典工作和 BiPACE。

建议论文正文使用：

- **environment-induced equivalence**；
- **effect-space policy geometry**；
- **pushforward policy**；

把 bisimulation 放 related work，而不是标题中心。

---

# 28. 这篇论文最危险的五个 reviewer 问题

## Q1. “Behavior-space policy optimization 不是 ICML 2020 就有了吗？”

必须回答：

- 是，有 Behavior-Guided Policy Gradient；
- 我们不 claim 首次 behavioral distance；
- 本文是 tool/environment-kernel-induced trust region；
- behavior representation 不是手工 trajectory descriptor，而是可执行工具 transition；
- estimator 利用 Search action → result distribution，且是 old/new proximal constraint；
- 需要在实验中证明这种结构带来超越 generic behavior regularizer 的收益。

如果无法做出这一差异，novelty 不够。

---

## Q2. “为什么 retrieval result 就代表完整 environment behavior？”

回答不能说“因为直觉”。

需要：

1. 理论定义用一般 effect kernel；
2. Search 中 retrieval distribution 是 tractable proxy；
3. hybrid token constraint 保留未建模部分；
4. 做 result-only / result+query / result+value feature ablation。

---

## Q3. “你的 probe 本质上也是 branching，和 BPO / APPO 有什么区别？”

关键区别：

- BPO/APPO branch 的目的是获得 sibling terminal returns / credit；
- EITR probe 的目的是估计 **local transition geometry**；
- probe 不继续 suffix rollout；
- probe 不产生 step reward / terminal reward；
- 主要创新在 policy constraint，不在 branch advantage。

实验中要报告额外 environment cost，证明 probe 远低于 terminal branch rollout。

---

## Q4. “为什么不直接用 sequence-level KL？”

这是最关键的 baseline。

需要证明：

- sequence KL 对 behavior-equivalent paraphrases 过度保守；
- EITR 能在相同 induced drift 下获得更高 return improvement；
- 单纯调大 PPO clip / KL coefficient 无法获得同等稳定性-进步 tradeoff。

---

## Q5. “是不是又一个 regularizer trick？”

必须有三层回答：

1. constrained formulation + adaptive dual；
2. theoretical invariance / transition-bound motivation；
3. optimization-dynamics evidence：不是只看 final score，而是展示相同 behavior budget 下的更优 policy improvement。

---

# 29. 最大风险与补救

## Risk 1：经典 prior 太接近

Behavior-Guided PG 已经存在。

### 补救

尽早读透，并把创新落到：

- executable transition-kernel induced metric；
- trust region rather than novelty shaping；
- black-box environment estimator；
- LLM tool action structure；
- integration with modern GRPO / agentic rollout pipeline。

如果最后方法只是“用 retrieval embedding 的 Wasserstein penalty”，建议放弃。

---

## Risk 2：query 本身会进入未来 context，所以同 retrieval ≠ 真正等价

### 补救

三种设置逐步验证：

1. **query-hidden environment**：后续 context 只保留 tool result，不回填原 query；理论最干净；
2. 标准 Search-R1 context；
3. behavior feature 加入 canonicalized query semantics。

如果仅设置 1 有收益、标准 setting 无收益，则 general claim 需要显著收缩。

---

## Risk 3：probe estimator 方差大

### 补救

- self-normalized IS；
- K probe sensitivity；
- old/current mixed proposal；
- ESS threshold；
- top-k truncation；
- kernel smoothing。

可以定义 probe effective sample size：

$$
ESS
=
\frac{(\sum_i\rho_i)^2}{\sum_i\rho_i^2}.
$$

ESS 过低时：

- 提前停止 policy epoch；
- 收紧 update；
- 或刷新 probes。

这个机制甚至可能成为实用训练 stabilizer，但第一版不要把故事做太散。

---

## Risk 4：额外 retrieval cost 抵消收益

### 补救

- 固定 corpus + local retriever；
- probe 只在部分 search turns；
- K=2/4 起步；
- retrieval cache；
- probe 不 rollout suffix；
- 以 total environment-call budget 做公平比较。

---

## Risk 5：behavior constraint 太宽，语言 policy 发生 degeneration

### 补救

采用 hybrid constraint：

$$
D_B\le\delta_B
$$

同时保留一个宽松的

$$
D_{token}\le\delta_{max}.
$$

这里 token constraint 是 safety rail，不是主要 optimization geometry。

---

# 30. 最小可行实验（MVP）

建议先做非常小的版本。

## MVP-1：不训练

1. 取一个现成 Search-R1 checkpoint；
2. 抽 500-1000 个真实 search states；
3. 每个状态采 8 个 query；
4. 跑固定 retriever；
5. 画 language-distance vs retrieval-distance；
6. 找真实 mismatch cases。

如果图不好看，立即暂停。

---

## MVP-2：小规模 branch-to-terminal diagnostic

对 100-300 个 states：

- query branches 继续 rollout 到最终答案；
- 比较 token/retrieval distance 对 return difference 的预测能力。

如果 retrieval distance 没有优势，暂停。

---

## MVP-3：只实现一个 induced-JS penalty

在 Search-R1 / veRL 中加：

- K=4 probes；
- doc-score distribution；
- induced JS；
- dual constraint。

模型：1.5B / 3B。

只比较：

1. GRPO；
2. GRPO + sequence KL；
3. SAPO；
4. EITR-GRPO。

如果 EITR 没明显训练动力学优势，不急着上大模型。

---

# 31. 工程模块划分

如果基于 Search-R1 / veRL，建议拆成五个独立模块。

## `behavior_probe_collector`

负责：

- 捕获 tool-call prefix state；
- 采 K 个 query；
- 保存 old sequence logprob。

## `retrieval_effect_cache`

负责：

- query → top-k docs；
- scores / ranks normalization；
- cache key 与版本控制。

## `induced_distribution_estimator`

负责：

- SNIS weights；
- doc-mixture distribution；
- JS / MMD；
- ESS。

## `behavioral_constraint`

负责：

- batch/state aggregation；
- dual variable；
- target divergence；
- diagnostics。

## `trainer`

负责把：

$$
L_{PG}
$$

与：

$$
D_B
$$

组合成 constrained update。

这种模块化很重要，因为后面方案二的 replay buffer 可以直接复用 effect cache。

---

# 32. 论文可能的标题

## 偏 general RL framing

**Beyond Token-Space Proximity: Environment-Induced Trust Regions for Agentic Reinforcement Learning**

## 偏问题导向

**What Is a Small Policy Update for a Language Agent?**

副标题：

*Environment-Induced Policy Optimization for Tool-Using LLMs*

## 偏 Search Agent

**EITR: Environment-Induced Trust Regions for Reinforcement Learning of Search Agents**

## 偏理论 / geometry

**Policy Optimization in Environment-Effect Space for Language Agents**

当前最推荐第一个。

---

# 33. 论文 Abstract 草案

> Reinforcement learning has become a central post-training paradigm for language agents that interact with search engines and other external tools. Existing proximal policy optimization methods, however, measure policy change primarily in token or action-text space, even though tool actions affect future returns through their induced environmental transitions. We study this mismatch in search agents and show that linguistically different queries can induce near-equivalent retrieval behavior, causing token-space proximity constraints to penalize behaviorally redundant policy changes. Motivated by this observation, we introduce Environment-Induced Trust Regions (EITR), a policy optimization framework that constrains the divergence between environment-effect distributions induced by consecutive policies rather than relying solely on linguistic policy divergence. We develop a black-box estimator that uses cached search outcomes and self-normalized importance reweighting, requiring no gradient through the retriever and no additional process reward. EITR can be integrated with standard outcome-only GRPO while retaining a weak token-space safety constraint. Our goal is to establish environment-induced policy geometry as a more appropriate notion of proximity for tool-using language agents and to improve training stability and efficiency across search and interactive-agent benchmarks.

这只是 proposal abstract；真正投稿时必须根据实验结果改写，尤其不能提前 claim “show” 尚未完成的结论。

---

# 34. 预期 Contribution 写法

如果实验成立，贡献最好压成四点：

1. **Empirical diagnosis**：系统揭示 Search-Agent RL 中 linguistic policy drift 与 environment-effect drift 的错位，并证明后者更能描述部分 downstream behavioral change。
2. **RL formulation**：提出 environment-induced / pushforward policy trust region，把 tool transition 而非纯 token distribution 作为 proximal geometry 的核心对象。
3. **Practical algorithm**：提出无需 differentiable retriever 的 probe + SNIS estimator，可直接嵌入 GRPO/PPO 风格训练。
4. **Empirical validation**：在固定 reward 下，提高训练稳定性、policy-improvement efficiency 或最终 Search-Agent capability，并展示向另一类 tool environment 的推广。

不要写第五个 reward / credit contribution，保持故事聚焦。

---

# 35. Related Work 阅读优先级

以下是开始实验前应该精读的工作。

## 必须精读

1. **Trust Region Policy Optimization** — Schulman et al., ICML 2015
   https://proceedings.mlr.press/v37/schulman15.html
2. **Proximal Policy Optimization Algorithms** — Schulman et al., 2017
   https://arxiv.org/abs/1707.06347
3. **Learning to Score Behaviors for Guided Policy Optimization** — Pacchiano et al., ICML 2020
   https://proceedings.mlr.press/v119/pacchiano20a.html
4. **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning**
   https://arxiv.org/abs/2503.09516
5. **Improving Search Agent with One Line of Code (SAPO)**
   https://arxiv.org/abs/2603.10069
6. **BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents**
   https://arxiv.org/abs/2606.25556

## Agentic RL 邻近工作

7. **T²PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning** — ICML 2026 Spotlight
   https://arxiv.org/abs/2605.02178
8. **APPO: Agentic Procedural Policy Optimization**
   https://arxiv.org/abs/2606.12384
9. **AT²PO: Agentic Turn-based Policy Optimization via Tree Search** — ACL 2026 Long Paper
   https://aclanthology.org/2026.acl-long.1106/
10. **Branching Policy Optimization: Sandbox-Native Language Agent Reinforcement Learning**
    https://arxiv.org/abs/2607.14171
11. **Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning**
    https://arxiv.org/abs/2606.22995

## 后续方案二相关

12. **Off-Policy Value-Based Reinforcement Learning for Large Language Models (ReVal)**
    https://arxiv.org/abs/2603.23355
13. **Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)** — ICLR 2026
    https://arxiv.org/abs/2602.01601
14. **Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning (SAO)**
    https://arxiv.org/abs/2607.07508

---

# 36. 当前 novelty 判断

截至 2026-08-05 的检索结果，已经存在：

- behavioral-space RL；
- bisimulation-guided Agentic RL；
- Agentic branching / tree optimization；
- token-level Search-Agent stabilization；
- off-policy LLM RL。

因此，这个 idea **不是一个“没人想到 behavior”式的空白领域**。

它是否值得做，取决于能不能证明下面这个更窄但更强的命题：

> **For tool-using LLMs, the proximal geometry used for policy optimization should be defined by the policy-induced environment transition/effect distribution; in search agents, this geometry can be estimated cheaply from black-box retrieval outcomes and yields a better stability–improvement tradeoff than token-space proximity alone.**

如果实验最后只表现为：

> retrieval embedding regularization +1%

则不建议投稿主会。

如果能同时得到：

1. 一个清楚的 mismatch phenomenon；
2. 一个 transition/value 理论解释；
3. 一个真正改变 update feasible set 的算法；
4. 在固定 reward 下稳定提升；
5. 对 Search 之外至少一个 tool agent 有迁移；

那么这个项目才有较好的 ICML / NeurIPS 方法论文骨架。

---

# 37. 推荐研究路径

建议按以下顺序推进：

```text
Literature collision check
        ↓
No-training behavioral diagnostic
        ↓
Does environment distance predict downstream change?
        ↓
Small-scale EITR-GRPO
        ↓
Optimization/stability analysis
        ↓
7B main experiments
        ↓
Generalization to one non-search tool environment
        ↓
Theory tightening
        ↓
Full paper
```

不要反过来先搭完整大规模训练框架。

---

# 38. 最终要整合的方案二：Behavior-Aware Off-Policy / Replay-Based Search RL

方案一如果成立，最终不应该停在 on-policy trust region。

它最自然的下一步，就是与之前的 **方案二：Off-Policy Search RL / Replay-Based Actor-Critic** 整合。

这是因为方案一实际上给方案二提供了一个此前缺失的核心概念：

> **一条旧 trajectory 到底“离当前 policy 有多远”，不能只从语言 likelihood drift 判断，还应该看它所代表的环境行为是否已经发生实质变化。**

---

## 38.1 方案二原始问题

现有 GRPO / PPO 式 Agentic RL 非常浪费：

```text
昂贵长轨迹
+ 多次 search calls
+ 大量生成 token
        ↓
使用一轮 / 少量 epoch
        ↓
丢弃
```

Search Agent 尤其适合 replay，因为每一个工具 turn 都可以存成：

$$
(s_t,q_t,o_{t+1},\mu(q_t|s_t),R,\text{metadata}).
$$

ReVal 已经表明 value-based off-policy LLM RL 与 replay buffer 是可行方向；SAO 也说明长程 Agentic RL 中 policy lag / asynchronous off-policy drift 是现实问题。

但直接把经典 replay 套到 Search Agent 上会遇到：

- query sequence likelihood 很快 stale；
- old/current token ratio 极端；
- 长文本 action 的 IS variance 高；
- 明明旧 query 与当前 query 访问的是同一信息区域，却因为语言概率变化而被判定为“严重 off-policy”。

这里正好可以用方案一的 environment-induced geometry。

---

## 38.2 最安全的整合方式：Behavioral Replay Admission + 正统 Off-Policy Correction

第一版整合不要直接用 behavioral ratio 替换 importance ratio，因为那样可能破坏 policy-gradient 无偏性。

更稳妥的是：

### Replay buffer

$$
\mathcal B
=
\{(s,a,o,\mu(a|s),R,B_s(a))\}.
$$

其中 $B_s(a)$ 保存或可恢复 action 的 environment effect，例如 retrieval distribution。

### Replay admission / gating

旧 transition 是否允许复用，不只看：

$$
D_{token}(\pi_\theta,\mu),
$$

而增加：

$$
D_B
\left(
\bar\pi^B_\theta,
\bar\pi^B_\mu
\right).
$$

例如只有满足：

$$
D_B\le\delta_{replay}
$$

的旧数据进入当前 critic / auxiliary update。

### Off-policy correction

对 actor update 仍保留理论上合法的：

- action-level importance ratio；
- V-trace / Retrace；
- clipped IS；
- 或直接走 value-based off-policy critic。

behavior distance 的作用是：

> **判断哪些 stale trajectories 在环境意义上仍值得重用，以及应该重用多少。**

这样既利用方案一，又不会为了“看起来新”而牺牲 off-policy correctness。

---

## 38.3 最终整合算法的形态

可以形成如下训练循环：

```text
                  ┌──────────────────────────────┐
                  │      Replay Buffer B         │
                  │ old search transitions       │
                  │ cached retrieval effects     │
                  └──────────────┬───────────────┘
                                 │
                    behavioral replay gating
                                 │
                                 ↓
Fresh on-policy rollouts ──→ Off-policy critic / value learner
          │                      │
          │                      │
          └──────────┬───────────┘
                     ↓
               advantage / value
                     ↓
              current policy update
                     │
          Environment-Induced Trust Region
                     │
                     ↓
                 π_{k+1}
```

即：

$$
\boxed{
\text{EITR on-policy actor}
+
\text{behavior-aware replay}
+
\text{off-policy value learning}
}
$$

---

## 38.4 为什么两个方案组合后比单独做方案二更强

单独方案二容易被 reviewer 问：

> “Replay buffer、Retrace、V-trace、off-policy actor-critic 都是经典 RL，你在 LLM Agent 上新在哪里？”

整合方案一以后，回答可以变成：

> **Language-agent replay suffers from a unique mismatch between linguistic policy staleness and environmental behavioral staleness. We use environment-induced policy geometry both to constrain fresh updates and to decide when past tool interactions remain reusable.**

这样整个 research program 的统一主线变成：

$$
\boxed{
\textbf{Agentic RL should measure policy shift in environment-behavior space, not only token space.}
}
$$

方案一解决：

$$
\text{How far may the current policy move?}
$$

方案二解决：

$$
\text{How old can an experience be before it should no longer be reused?}
$$

两个问题实际上由同一个 behavioral geometry 连接。

最终理想路线是：

### Stage A / 第一篇核心工作

**Environment-Induced Trust Regions for Agentic RL**

重点：

- on-policy；
- fixed outcome reward；
- proximal update geometry；
- Search Agent 为主；
- 理论 + optimizer + diagnostics。

### Stage B / 最终整合方案二

**Behavior-Aware Off-Policy Reinforcement Learning for Search Agents**

重点：

- replay buffer；
- behavior-aware replay admission；
- off-policy value learning；
- policy-lag handling；
- environment interaction sample efficiency。

最终完整版方法可以概括为：

$$
\boxed{
\textbf{Behavioral Proximity for Updates}
+
\textbf{Behavioral Proximity for Replay}
}
$$

也就是不仅回答“**policy 应该怎么更新**”，还回答“**过去的 Agent experience 什么时候仍然有效**”。

如果方案一的 empirical premise 被验证，我建议最终就沿这条统一路线推进，而不是把方案二做成一篇独立、泛化的 replay-buffer 工程论文。
