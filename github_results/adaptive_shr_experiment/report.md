# Adaptive-SHR v1 report

## Method

The negative branch is the locked SHR implementation: semantic KMeans K=8 selection and beta=0 four-neighbor harmonic reconstruction. At every control step, the candidate SHR action uses lambda=0.5. The shift ratio is the fraction of changed tokens over action dimensions 0..5 (gripper excluded). If S > 0.33, the final action is recomputed with lambda=0.25; otherwise lambda=0.5 is retained.

All Adaptive-SHR episodes restore the exact prior Vanilla/Recon/SHR canonical snapshot and verify snapshot, initial-state, and initial-RGB hashes.

## Overall success

| Arm | Success | Rate |
|---|---:|---:|
| vanilla | 507/2700 | 18.778% |
| semantic_recon_k8_m10 | 614/2700 | 22.741% |
| shr_harmonic | 646/2700 | 23.926% |
| adaptive_shr | 590/2700 | 21.852% |

## Adaptive-SHR vs SHR paired transitions

- Rescue (SHR fail, Adaptive success): 149
- Harm (SHR success, Adaptive fail): 205
- Net paired gain: -56

## Lambda triggering

- Episode lambda=0.5 (no step triggered): 7
- Episode lambda=0.25 (at least one step triggered): 2693
- Episode trigger rate: 99.741%
- Triggered subset: SHR 646/2693, Adaptive 590/2693, rescue 149, harm 205, net -56

## Task-level results

| Task | N | SHR | Adaptive | Δ pp | Rescue | Harm | Triggered |
|---|---:|---:|---:|---:|---:|---:|---:|
| google_robot_pick_coke_can | 300 | 32.333% | 33.333% | +1.00 | 45 | 42 | 293 |
| google_robot_open_drawer | 300 | 46.000% | 28.000% | -18.00 | 18 | 72 | 300 |
| google_robot_close_drawer | 300 | 69.667% | 65.333% | -4.33 | 35 | 48 | 300 |
| google_robot_move_near | 300 | 65.667% | 61.667% | -4.00 | 30 | 42 | 300 |
| google_robot_place_apple_in_closed_top_drawer | 300 | 0.000% | 0.333% | +0.33 | 1 | 0 | 300 |
| widowx_carrot_on_plate | 300 | 1.333% | 2.667% | +1.33 | 4 | 0 | 300 |
| widowx_put_eggplant_in_basket | 300 | 0.000% | 0.333% | +0.33 | 1 | 0 | 300 |
| widowx_spoon_on_towel | 300 | 0.333% | 0.667% | +0.33 | 2 | 1 | 300 |
| widowx_stack_cube | 300 | 0.000% | 4.333% | +4.33 | 13 | 0 | 300 |

Episode-level lambda selection means whether any control step triggered. Exact step-level lambda, shift ratio, and all three token actions remain in each episode JSON.
