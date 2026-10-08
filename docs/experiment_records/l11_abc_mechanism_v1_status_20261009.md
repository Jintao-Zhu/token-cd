# L11 A/B/C mechanism study: run record

**Status recorded:** 2026-10-09 (Asia/Shanghai)
**Run state:** in progress; this is a checkpoint, not a final result.

## Locked protocol

- Model: OpenVLA-7B, Fractal SIMPLER checkpoint.
- Tasks: Open Drawer, Close Drawer, Pick Coke Can, Move Near.
- Seeds: 100–199 for reused source trajectories and paired region arms; fresh seeds 300–499 for reconstruction arms.
- Renderer: RT full, depth 8, for Open/Close Drawer; RT depth 0 for Pick Coke Can/Move Near. 32 spp with denoiser.
- Compute allocation: GPUs 1, 2, and 3 only; three tmux workers per GPU.
- Source full arm: L11-Matched, harmonic projector reconstruction, lambda 0.5.

## Planned phases

1. Replay the 400 existing full trajectories and capture up to four valid quartile states per episode, without padding early terminations. Save RGB, SAPIEN instance segmentation, simulator/proprioceptive state, RNG state, selected token IDs, and Matched budget.
2. A: analyze independent segmentation-derived support measures and run a same-state offline budget-dose scan.
3. B: compare Target-Enriched and Target-Sparing against the reusable Full arm; 800 new closed-loop episodes after the 40-episode preflight.
4. C: compare Harmonic, Zero, and Global Mean on fresh paired seeds 300–499; 2,400 closed-loop episodes after the 60-episode preflight.

## Progress checkpoint

At this checkpoint, the nine state-library workers and the pipeline orchestrator were active in tmux. The Open Drawer state library had 88/100 complete four-frame seed records; Close Drawer had begun. No final A, B, or C conclusions are available yet. The state-library replay phase is data extraction from recorded actions, not a new policy evaluation.

The small preflight rollouts for the four tasks passed their initial-state/RGB pairing and renderer checks. A separate long replay spot-check found one episode whose restored initial state and RGB hashes matched the source, but whose replayed final success differed. This replay-fidelity discrepancy is recorded for follow-up; replayed success labels must not replace the original rollout labels.

## Data handling

Raw RGB/segmentation/state arrays, checkpoints, and large exports remain in local ignored artifact directories. This repository record contains the protocol and progress only; it does not claim that the planned experiments have completed.
