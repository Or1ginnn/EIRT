# EITR Documentation

## Canonical Documents

- `eitr/proposal/EITR_Search_Agent_RL_Proposal.md`: full paper proposal and method framing.
- `eitr/gates/eitr_nq_gate_a_pilot.md`: NQ language/environment geometry diagnostic.
- `eitr/gates/eitr_nq_gate_b_protocol.md`: preregistered branch-to-return protocol.
- `eitr/gates/eitr_nq_gate_b_branch_to_return.md`: Gate B result report.
- `eitr/gates/eitr_gate_c_implementation.md`: EITR-GRPO implementation and smoke acceptance rules.
- `eitr/migration_manifest.md`: clean-base migration boundary and provenance.

## Status

Gate A and Gate B are historical premise diagnostics. They used a Parallel Search Step900 policy restricted to one query. Gate C is now based on the official Search-R1 single-query path and must start from the original Qwen2.5-3B base model.

No Gate C training result should be claimed until the paired baseline/EITR smoke passes on the clean project.
