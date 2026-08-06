# EITR Clean-Base Migration Manifest

Date: 2026-08-06

## New Source of Truth

- Project: `EITR-Search-Agent`
- Base repository: `PeterGriffinJin/Search-R1`
- Base commit: `598e61bd1d36895726d28a8d06b3a15bed19f5d3`
- Training policy: original Search-R1 single-query agent
- Initial model for Gate C: original Qwen2.5-3B base model

## Migrated

- EITR estimator, induced-JS constraint, dual coefficient update, and actor integration.
- Exact same-prefix query probe collection using the real retriever.
- Gate A and Gate B diagnostic scripts.
- Gate C smoke runner and unit tests.
- Paper proposal, gate protocols, reports, compact result summaries, and representative examples.

## Explicitly Excluded

- Parallel Search Step900 weights and LiteCoA prompt/plan format.
- Multi-query `q1 || q2` rollout behavior.
- Finance Agent data, rewards, calculator actions, and report-aware retrieval.
- SFT/LoRA weights, datasets, checkpoints, trajectories, W&B caches, and API keys.

## Historical Result Boundary

Gate A/B historical results were measured with a Parallel Search Step900 checkpoint constrained to one query. They remain useful evidence that motivated EITR, but they are not clean Search-R1 replications. Reports now state this limitation explicitly.

Gate C must compare two runs initialized from the same Qwen2.5-3B base model:

1. Original Search-R1 GRPO.
2. The same Search-R1 GRPO with EITR enabled.

All other data, prompt, reward, retriever, rollout, sampling, and optimizer settings must remain identical.
