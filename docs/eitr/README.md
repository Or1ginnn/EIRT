# EITR Documentation

## Canonical Documents

- `eitr/proposal/EITR_Search_Agent_RL_Proposal.md`: full paper proposal and method framing.
- `eitr/gates/eitr_nq_gate_a_pilot.md`: NQ language/environment geometry diagnostic.
- `eitr/gates/eitr_nq_gate_b_protocol.md`: preregistered branch-to-return protocol.
- `eitr/gates/eitr_nq_gate_b_branch_to_return.md`: Gate B result report.
- `eitr/gates/eitr_gate_c_implementation.md`: Conditional EITR V5.1 Phase-2 implementation and safety-smoke rules.
- `eitr/migration_manifest.md`: clean-base migration boundary and provenance.

## Status

Gate A and Gate B are premise diagnostics. Phase 2 now implements Conditional EITR on the official Search-R1 single-query path and starts from the original Qwen2.5-3B base model.

No EITR training result should be claimed until the off/probe-only/EITR paired safety smoke passes on the clean project.
