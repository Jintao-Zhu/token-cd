# Prompt-v1 implementation audit

Result: **PASS**

- Audited observations: 9 (required: 9; three per task).
- Query: exact non-special instruction tokens; multimodal positions are text indices + 256.
- Visual keys: positions 1–256, exactly 256 projector tokens in row-major 16×16 order.
- Language layers: 32 total; Prompt-v1 uses 0-based layers 16–31.
- Attention: model post-softmax weights; equal mean across layers, heads, queries; no visual re-normalization.
- Independent repeated extraction maximum absolute difference: 0.
- Original closed-loop selector replay: 27300 control steps; Top-m mismatches=0; stored-score SHA mismatches=0.
- Clean logits before/after attention extraction maximum absolute difference: 0; greedy unchanged in all audit observations.
- Diagnostic source trajectories: all 30 reruns matched the original closed-loop clean action and Standard mask at the initial step (fail-closed check in the collector).
- Same-state recomputation: clean greedy action and Standard mask match the diagnostic Standard driver on the identical cached RGB in all 90 selected states.
- Coverage: Standard, Prompt-v1 and Random each use exactly the same state-local m; no duplicate positions.
- Reconstruction: shared 16×16 four-neighbor Dirichlet harmonic solve, beta=0/gamma=1; outside-mask tokens bit-identical.
- CD: shared clean teacher-forced prefix; lambda=0.5 on dimensions 0–5; gripper dimension 6 remains clean.

See `query_index_audit.csv` for token IDs/positions and `patch_index_reference.png` for the spatial index convention.
