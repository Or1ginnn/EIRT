# EITR Documentation

## Canonical Documents

- `eitr/proposal/EITR_Search_Agent_RL_Proposal.md`: full paper proposal and method framing.
- `eitr/gates/eitr_nq_gate_a_pilot.md`: NQ language/environment geometry diagnostic.
- `eitr/gates/eitr_nq_gate_b_protocol.md`: preregistered branch-to-return protocol.
- `eitr/gates/eitr_nq_gate_b_branch_to_return.md`: Gate B result report.
- `eitr/gates/eitr_gate_c_implementation.md`: Conditional EITR V5.1 Phase-2 implementation and safety-smoke rules.
- `eitr/gates/eitr_gate_c_resolution_2026-08-11.md`: Gate C direction-audit failures, fixes, final mechanism PASS, and claim boundary.
- `eitr/migration_manifest.md`: clean-base migration boundary and provenance.

## Status

Gate A and Gate B are premise diagnostics. Phase 2 implements Conditional EITR on the official Search-R1 single-query path and starts from the original Qwen2.5-3B base model. The same-batch Gate C direction audit now passes: the real EITR SGD candidate reduces cached-probe environment drift while the reverse diagnostic increases it.

This is a mechanism result, not a task-performance result. No EITR method benefit should be claimed until compute-aware `off / probe_only / eitr` paired experiments and strong matched baselines pass on the clean project.
