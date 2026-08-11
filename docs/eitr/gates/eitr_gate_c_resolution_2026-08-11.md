# EITR Gate C 方向审计：失败、修复与最终通过记录

日期：2026-08-11

代码基座：官方 Search-R1 单 query 路径 + Qwen2.5-3B

最终机制提交：`f6e3f08c4c915756a630f976f7d57852830c4188`

正式训练显存修复：`126d07d0ac809d9e6353a0ccf15b1be1303bd809`

> 最终结论：Gate C 的同 batch、同 cached probe 方向审计已经通过。GRPO proposal
> 之后，真实 EITR SGD 方向降低了 environment-induced drift，反方向诊断则提高了
> drift；query 空间、参数空间、old-policy anchor、缓存一致性和参数恢复同时通过。
> 这证明当前 correction 的目标、反向传播方向和实际参数更新链路成立，但不等价于
> 任务 EM/F1 提升，也不等价于论文主实验完成。

## 1. Gate C 实际要验证什么

Gate A/B 只说明语言空间变化与检索环境后果之间存在值得研究的错位。Gate C 不再
重复证明问题存在，而是检查我们写出的 EITR correction 是否真的沿预期方向工作。

记：

- `theta_old`：本轮 rollout 和 probe 采样时的旧策略；
- `theta_grpo`：完成普通 GRPO AdamW 更新后的 proposal；
- `g = grad D_env(theta_grpo)`：在相同 cached probes 上计算的 EITR 梯度；
- `D_noop = D_env(theta_grpo)`：不执行 correction 时的 proposal drift；
- `D_minus = D_env(theta_grpo - eta * g)`：真实 SGD 下降方向；
- `D_plus = D_env(theta_grpo + eta * g)`：只用于审计的反方向候选。

理想的双向机制判据是：

```text
D_minus < D_noop < D_plus
```

实际 hard gate 只把算法真正执行的 minus correction 当作必须条件：

```text
D_minus < D_noop
g dot delta_theta_minus < 0
cosine(delta_theta_minus, -g) > 0
```

query-logprob 空间仍要求严格的双向符号检查；参数空间的 `D_plus` 保留为平滑性诊断，
因为 FSDP/BF16 下有限参数扰动可能是分段量化的。最终通过的那次 run 额外满足了
`D_plus > D_noop`，因此证据比 hard gate 的最低要求更强。

同时必须满足：

1. old-policy anchor 对齐；
2. query-logprob 空间的负梯度方向下降；
3. 真实参数更新非零，且 `g dot delta_theta` 符号正确；
4. pre/noop/minus/plus 使用同一批 query、old log-prob、retrieval effects 和 mask；
5. candidate 更新不互相累计；
6. 审计结束后参数能够精确恢复；
7. 所有两卡 FSDP rank 的事件顺序、统计归约和 forward/backward 次数一致。

生产训练里的 `D_pre` 对应这里的 `D_noop`，`D_post` 对应真实 correction 后的
`D_minus`。反方向 `D_plus` 只用于机制审计，不进入正式算法。

## 2. 最终保留的 EITR 方案

一次 outer update 的真实顺序是：

```text
theta_old rollout / real retrieval / same-state probes
-> 旧策略 probe sequence log-prob
-> 5 次 GRPO AdamW mini-batch steps
-> 在 theta_grpo 上计算完整 batch 的 D_pre 与梯度
-> 1 次独立 EITR SGD correction
-> 用相同 cached probes 重新计算 D_post
```

核心设计没有在排错过程中被替换：

- 每条 rollout 最多选第一个可用 search state；
- `K=4`，包括真实 query 和额外同状态 query；
- `p_probe=1`；
- `K_eff>=2` 才形成有效 EITR state；
- 使用真实 E5 + FAISS Wikipedia Retriever；
- 根据旧/当前 query sequence log-prob 做 SNIS 重权；
- 在文档效果分布上计算 Jensen-Shannon divergence；
- 使用 rollout 分母的 coverage-weighted 目标：

```text
L_EITR = lambda_env * (1 / B) * sum_i m_i * D_env_i
```

- `lambda_env=0.1` 固定；
- 当前正式 correction base LR 固定为 `3e-5`，并与 actor 共享 warmup factor；
- correction 使用独立 `SGD(momentum=0, weight_decay=0)`；
- GRPO AdamW 的 momentum、second moment 和 weight decay 不被 EITR 推进；
- 没有有效 state 时严格退化为普通 GRPO。

最终没有采用的做法包括：翻转 loss 符号、把 EITR 塞进 reward、重新引入 adaptive
beta、让 correction 复用 AdamW、动态搜索正式 LR、换新版 veRL 后重新实现整套系统。

## 3. 解决过程与真实挫折

### 3.1 第一次失败：`coverage=0` 被硬阈值直接终止

原始 Qwen2.5-3B 刚开始训练时 search 格式不稳定，额外 probe 经常缺少
`</search>`。第一版 Gate C 要求每个 state 必须凑齐4条 probe，并把 coverage 当
硬门槛，因此训练在第一个 update 的 probe 收集阶段停止。

这次失败证明的是启动协议不适合 base model，不是 EITR idea 失效。

修复：

- 非法 probe 只记录，不再令整批训练失败；
- 允许 `K_eff=2/3/4`；
- 没有有效 state 时继续普通 GRPO；
- coverage 从硬门槛改为 batch loss 的自然权重；
- 只要 rollout 中出现一个可执行 search，就可选择其第一个可用 state。

### 3.2 Search-R1 继承问题：step、长度、reward 与解析口径

在进入真正方向审计前，还修复了几类会污染实验的底座问题：

- 统一 outer update 的 step 定义，避免 off-by-one 和 dataloader 提前结束；
- observation 超长时只截断检索正文，再重新拼接 `</information>`；
- observation 上限扩展到1024，完整轨迹和 rolling context 扩展到8192；
- reward 只解析模型生成部分，不能把 prompt 示例中的 `<answer>` 当成答案；
- 使用 Search-R1 v0.3 的格式 shaping，验证仍保持纯答案 EM；
- NQ + HotpotQA 完整训练集开启 shuffle，NQ 固定子集单独验证；
- 所有缓存、Ray、Hydra、W&B、日志和 checkpoint 指向数据盘。

这些修复不改变 EITR 数学目标，但如果不先处理，会让“训练能跑”和“机制有效”都
无法被可信判断。

### 3.3 优化器混杂：不能用第二次 AdamW 冒充局部 correction

早期实现把 EITR 与 GRPO optimizer state 混在一起，无法判断 drift 变化来自环境
目标还是 AdamW 的历史动量。之后将其重构为：先完成所有 GRPO AdamW steps，再在
完整 rollout batch 上累积 EITR 梯度，最后只执行一次独立无动量 SGD。

这一修改建立了干净的因果边界：

```text
theta_old -> theta_grpo -> theta_eitr
```

也使 `D_pre/D_post` 第一次具备明确含义。

### 3.4 盲调学习率没有解决问题

最初的单步 smoke 出现过以下现象：

- 很小 LR：`D_post == D_pre`，看不到可分辨更新；
- `1e-4`、`1e-3`：`D_post > D_pre`，表现为过冲或反向；
- 同 batch 的 `1e-5/3e-5/1e-4` 三点候选一度全部上升。

这些结果阻止了直接启动8000步训练，但不能直接判定 EITR 无效，因为当时还没有
证明 scorer、anchor、FSDP 参数候选和数值精度属于同一口径。项目因此停止继续做
LR ladder，转而实现确定性的同 batch 方向审计。

### 3.5 方向审计本身也经历了多次工程失败

审计路径需要对3B FSDP 参数做 snapshot、候选更新、统计和恢复，过程中暴露了多个
与算法无关、但会制造假结论的问题：

1. 对超大 FlatParameter 直接调用 `quantile()`，输入过大而崩溃；
2. 已经全局归约的 active-state 统计再次 SUM，双卡后变成2倍；
3. worker 在 metrics 返回前直接抛错，导致 W&B 没有留下失败证据；
4. plus/minus 候选若不从同一个 `theta_grpo` snapshot 恢复，会发生累计更新；
5. 全参数 checksum、delta 和 quantile 临时张量可能制造额外 CPU/GPU 内存峰值。

对应修复：

- 使用有界、固定预算的参数 delta 采样；
- global additive stats 只由一个 rank 持有，再统一 SUM；
- 科学验收失败先返回 driver、写入 W&B，再安全退出；
- 每个候选都从同一 snapshot 恢复，最终检查 restore error；
- snapshot 统计改为分块计算，避免再次因审计工具 OOM。

### 3.6 假方向失败：warmup 零 proposal、score path 错位和 FP32 数值底噪

一次审计得到负的 JS 和几乎为零的参数变化：

```text
D_zero ~= -1e-8
parameter update norm ~= 4.5e-11
```

JS 理论上非负，因此该结果不是“idea 方向反了”，而是没有可分辨 proposal signal，
同时落入 FP32 相消误差。进一步检查发现：

- 零 warmup 配置的 scheduler 首步仍可能把 actor LR 初始化为0；
- old/current probe score 必须在相同 sequence balance、micro-batch 和 eval mode 下计算；
- sequence log-prob 累加、SNIS 和 JS 的小张量计算使用 FP32 时，在近零处会出现负噪声；
- `loss_applied=1` 只能说明调用过 step，不能证明参数真的发生了可分辨变化。

最终修复：

- `warmup_steps<=0` 时从第一个 optimizer step 使用 base LR；
- old/current probe scoring 对齐到相同 row、rank、chunk 和 model mode；
- SNIS/JS 小张量链路改为 FP64，并只对理论上不可能的微小负 JS 做 `clamp_min(0)`；
- 引入 proposal-signal gate，将零信号记为 `INCONCLUSIVE/NO_SIGNAL`，不能冒充 FAIL；
- correction step count 只在真实非零 SGD 被执行时增加；
- 同时记录 query-space 方向、真实 `g dot delta_theta`、cosine、changed fraction、
  cache hash 和 restore error。

### 3.7 一次非对称候选不能直接推翻真实 SGD

修复数值链路后，一次审计出现：真实 minus 方向下降，但手工 plus 方向也下降。query
空间方向正确，`g dot delta_theta` 符号也正确。这说明 FSDP/BF16 的有限参数扰动不是
理想光滑对称曲线，不能要求每个手工反方向候选都完美镜像真实 optimizer step。

最终 Gate C 把核心验收聚焦到真实 correction 必须明确下降，同时继续保留 plus 作为
诊断。随后在相同 corrected score path 上得到严格的双向关系，Gate C 才正式通过。

## 4. 最终 PASS 证据

最终服务器方向审计报告：

- W&B run ID：`einpe5dt`；
- 事件顺序完整：

```text
probe_old_logp@v0
-> GRPO_STEP_1 ... GRPO_STEP_5
-> D_PRE_ALL_STATES@v5
-> EITR_SGD_STEP_1
-> D_POST_ALL_STATES@v6
```

关键数值：

| 指标 | 结果 | 含义 |
|---|---:|---|
| `D_noop` | `8.911459e-04` | GRPO proposal 的 environment drift |
| `D_minus` | `7.196204e-04` | 真实 EITR SGD 后的 drift |
| `D_plus` | `9.955575e-04` | 反方向诊断候选的 drift |
| `D_minus` 相对下降 | 约 `19.25%` | correction 产生清晰、超出噪声的下降 |
| query direction | PASS | query-logprob 空间负梯度方向正确 |
| parameter direction | PASS | 实际参数空间方向正确 |
| anchor | PASS | old/current scorer 口径对齐 |
| cache hash | PASS | pre/post 使用同一 cached probe batch |
| restore error | `0` | 审计结束后参数精确恢复 |
| minus cosine | `0.777` | 实际参数 delta 与预期下降方向正对齐 |
| active states | `103` | 160条 rollout 中103条形成有效 EITR state |
| coverage | `0.644` | `103 / 160`，不是固定超参数 |
| ESS | `3.844` | 接近当前 `K_eff` 的有效 probe 支持 |
| valid retrieval calls | `160` | 真实 search 检索执行数 |

最终满足：

```text
7.196204e-04 < 8.911459e-04 < 9.955575e-04
```

并且5次 GRPO AdamW 与1次独立 EITR SGD 均实际完成。因此这次 PASS 不是“测试代码
能运行”，而是对真实两卡 FSDP、Qwen2.5-3B、真实 Retriever 训练链路的机制验证。

## 5. coverage=0.644 的准确解释

`coverage=0.644` 不是固定系数，也不表示模型已有64.4%的完整严格格式正确率。

它表示本 batch 的160条 rollout 中，有103条至少形成了一个可执行 search state，并
成功得到 `K_eff>=2` 的 probe 支持。当前 parser 对 executable search 的要求是完整、
非空的 `<search>...</search>`；它不要求整条 trajectory 的 think/search/answer 顺序都
严格正确，也不判断 query 是否高质量。

因此 coverage 只控制“这个 batch 有多少 rollout 能应用 EITR”，不能被当成格式学习
或任务准确率。由于每条 rollout 最多贡献一个 active state，coverage 严格位于
`[0,1]`，后期不会无限放大；从0.644增长到1时，名义 EITR batch 强度最多再增长约
`1 / 0.644 = 1.55`倍。

## 6. Gate C 之后的正式训练 OOM

Gate C PASS 后首次启动混合 NQ+HotpotQA 正式训练，第一批长轨迹在 GRPO backward
发生 OOM：

- 训练 worker 峰值约 `66.04 GiB`；
- Retriever 在同一卡占约 `12.73 GiB`；
- 另有少量既有服务；
- 仅余约 `101 MiB` 时，backward 还需申请约 `1.96 GiB`。

固定 audit batch 能通过而正式首批 OOM，是因为 shuffle 后抽到的真实混合 batch 有更
长的 token/trajectory 激活；这属于系统容量问题，不推翻 Gate C 的方向结论。

提交 `126d07d` 做了最小显存修复：

- actor FSDP 参数继续常驻 GPU；
- 不参与 actor backward 的 reference FSDP 参数只在 ref log-prob 阶段加载，算完立即
  卸载到 CPU；
- 正式 `PPO_MICRO_BATCH_SIZE=4`；
- probe score/correction micro-batch 保持4；
- 不修改 train batch=32、K=4、loss、reward、optimizer 或学习率。

该提交有66项单元/静态回归测试通过并已推送。正式长跑的持续稳定性应由新的服务器
run、W&B ID、update 1/2状态和显存峰值另行补录；不能仅凭代码测试声称8000步已经
完成。

## 7. 这次 PASS 能证明和不能证明什么

### 已经证明

- EITR probe、真实 Retriever、old/current scoring 和 FSDP actor 更新链路能够闭环；
- GRPO 后的 environment drift 在当前 batch 上非零且可分辨；
- 独立 EITR SGD 的真实下降方向正确；
- correction 明确降低同一 cached-probe 目标；
- coverage weighting、variable `K_eff` 和零有效 state fallback 可运行；
- 失败诊断能够区分真正方向错误、无 proposal signal、数值噪声和系统 OOM。

### 尚未证明

- EITR 提升 NQ/HotpotQA 的最终 EM/F1；
- EITR 比普通 GRPO 或 probe-only 更好；
- 收益不是额外计算、额外 optimizer step 或普通 query-token KL 带来的；
- 单 seed 的增益能够跨 seed、模型规模和 benchmark 稳定复现；
- 方法达到 SOTA 或已经满足论文主张。

换句话说：Gate C 通过说明“我们实现的 correction 确实在做它数学上声称要做的事”，
而不是“论文效果已经成立”。

## 8. 下一阶段的证据顺序

1. 先让正式 EITR 长跑稳定通过 update 1/2，并持续记录训练曲线；
2. 完成一个有足够长度的 EITR pilot，检查 EM、valid-search rate、format rate、
   coverage、`D_pre/D_post` 和梯度尺度是否稳定；
3. 从同一初始化、同一数据顺序运行 `off / probe_only / eitr` 配对实验；
4. 增加 correction-budget/update-norm 匹配的 token-KL 或 SAPO-like 强基线；
5. 再扩展到多 seed、完整训练和跨 benchmark；
6. 只有配对主实验稳定改善任务指标，才能把机制 PASS 上升为方法有效性结论。

## 9. 关键提交索引

| 提交 | 作用 |
|---|---|
| `5c71543` | 将 EITR correction 与 GRPO AdamW optimizer state 分离 |
| `919dda3` | 加入同 batch correction 尺度诊断 |
| `5495352` | 加入 score-path 与参数更新方向审计 |
| `f8a4c48` | 保证 FSDP audit backward/collective 安全 |
| `e3a5419` | 将超大参数 delta quantile 改为有界采样 |
| `51c84f7` | 区分 query 空间与参数空间方向，保证先记录再失败 |
| `f5a7ee4` | FP64 SNIS/JS、零信号判定与 zero-warmup 修复 |
| `f6e3f08` | 以真实 correction descent 完成 Gate C 验收 |
| `126d07d` | 正式训练仅卸载 reference FSDP 参数，缓解 backward OOM |

## 10. 最重要的复盘结论

真正解决 Gate C 的不是“终于找到一个神奇学习率”，而是把一个模糊的单步训练现象
拆成了可审计的因果链：

```text
同一个旧策略与 probe
-> 明确的 GRPO proposal
-> 同口径 D_pre
-> 可验证的真实 SGD 参数方向
-> 同 cached batch 的 D_post
```

早期很多 FAIL 来自格式启动协议、优化器混杂、warmup 零更新、FP32 数值底噪、FSDP
统计和审计工具本身，而不是 environment-aware geometry 已被证伪。最终只有在这些
干扰被逐一排除后，真实 minus correction 的下降才成为可信的 Gate C PASS；最终 run
还额外给出了 `D_minus < D_noop < D_plus` 的完整双向证据。
