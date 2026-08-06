# EITR Search Agent

EITR studies a basic question in tool-agent reinforcement learning:

> What should count as a small policy update when a language-model action changes the world through a search engine?

PPO and GRPO constrain policy updates mainly in token space. For a search agent, however, a query first passes through a retriever and changes the next observation. EITR therefore measures and constrains the policy shift after the query is mapped to a retrieval-result distribution.

## Experimental Scope

- Clean base: official Search-R1 at commit `598e61b`.
- Policy: original Qwen2.5-3B base model, not a Parallel Search or Finance checkpoint.
- Agent format: original Search-R1 single-query `<search>...</search>` loop.
- Reward: the original outcome-only answer reward.
- Main comparison: standard Search-R1 GRPO versus Search-R1 GRPO with EITR.

The current repository contains the Gate C implementation and historical Gate A/B diagnostics. Gate C has not yet produced a clean-baseline training result.

## Current Evidence

- Gate A: the strong global low-correlation claim did not pass on NQ, but local language/environment mismatch covered 48.0% of sampled states.
- Gate B: retrieval-environment distance predicted branch return differences better than the main policy-language distance, with AUROC gain `+0.0655` and a positive cluster-bootstrap confidence interval.
- Gate C: implementation and unit tests are ready; paired Search-R1 baseline/EITR smoke is pending.

Historical Gate A/B runs used a Parallel Search Step900 policy constrained to one query. They support the premise but must be reproduced on the clean Search-R1 setup before becoming final paper evidence.

## Repository Map

- `verl/trainer/ppo/eitr.py`: induced retrieval distribution, JS constraint, SNIS estimator, and dual update.
- `search_r1/llm_agent/generation.py`: exact same-prefix single-query probes and real retriever calls.
- `scripts/eval/eitr_nq_gate_a.py`: language-distance versus retrieval-distance diagnostic.
- `scripts/eval/eitr_nq_gate_b.py`: branch-to-return diagnostic.
- `scripts/train/train_eitr_nq_gate_c_smoke.sh`: paired Gate C smoke entry point.
- `tests/test_eitr.py`: estimator, gradient, grouping, and configuration tests.
- `docs/eitr/`: proposal, gate reports, and migration notes.
- `results/eitr/`: compact historical summaries and representative examples only.

## Gate C Smoke

Prepare NQ with the original Search-R1 `base` prompt and start the original E5 Wikipedia retriever. Then run both experiments from the same Qwen2.5-3B base model:

```bash
CUDA_VISIBLE_DEVICES=1,2 \
EITR_ENABLED=false \
EXPERIMENT_NAME=eitr-nq-gate-c-baseline-smoke \
bash scripts/train/train_eitr_nq_gate_c_smoke.sh

CUDA_VISIBLE_DEVICES=1,2 \
EITR_ENABLED=true \
EXPERIMENT_NAME=eitr-nq-gate-c-eitr-smoke \
bash scripts/train/train_eitr_nq_gate_c_smoke.sh
```

Set `BASE_MODEL`, `DATA_DIR`, and `RETRIEVER_URL` to local server paths when needed. Do not use a previously trained Parallel Search checkpoint for this comparison.

## Upstream

This project is built on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and VeRL. The original Search-R1 README is preserved as `README_SEARCH_R1.md`; upstream licensing and attribution remain unchanged.
