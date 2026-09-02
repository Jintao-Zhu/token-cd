# vanilla_recon_shr_canonical_0_299_v2

Canonical paired rollout release for Vanilla, semantic Recon, and spatial SHR.

- 9 tasks, seeds 0-299 per task
- 2,700 paired simulator scenarios
- 8,100 compact episode summaries
- 2,700 reusable canonical simulator snapshots
- zero snapshot/state/RGB hash mismatches

`FINAL_RESULTS.json` contains aggregate success rates and paired McNemar tests.
`SNAPSHOT_MANIFEST.jsonl` records both the serialized-file SHA256 and the
canonical/state/RGB hashes for every `(task, seed)`.

The `episodes/` summaries omit only the large per-step `selector_trace` and raw
logit/action array reference. The 8,100 local NPZ arrays were validated during
export but intentionally excluded from Git because they are regenerable and
occupy several GiB.

To reuse the exact scenes, copy `snapshots/` into a new artifact directory and
run a compatible rollout driver with the same OpenVLA checkpoint and simulator
environment. Drivers must restore the pickle rather than recapture from seed,
and should verify the canonical, initial-state, and initial-RGB hashes against
the manifest before executing an arm.
