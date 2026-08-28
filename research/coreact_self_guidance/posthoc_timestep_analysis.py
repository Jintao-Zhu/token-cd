from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ARMS = ("A_vanilla", "N0_pure_negative", "W05_shrink", "W15_extrapolate", "W20_extrapolate", "REF_toward_top8")


def mean(values):
    return float(statistics.mean(values)) if values else None


def rms(values):
    return float(math.sqrt(statistics.mean([value * value for value in values]))) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True); args = parser.parse_args()
    art = args.artifact.resolve(); records = [json.loads(path.read_text()) for path in (art / "episodes").glob("*.json")]
    grouped = defaultdict(dict)
    for record in records: grouped[record["pair_id"]][record["arm"]] = record

    step_rows = []
    for task in (4, 7):
        for arm in ARMS:
            for step in range(10):
                values = {"pre": [], "clipped": [], "actual": [], "clip": [], "active": []}
                for pair, arms in grouped.items():
                    if int(arms["A_vanilla"]["task_id"]) != task: continue
                    traces = arms[arm].get("replan_traces", [])
                    for trace in traces:
                        steps = trace.get("step_traces", [])
                        if step >= len(steps): continue
                        s = steps[step]
                        if arm == "REF_toward_top8":
                            values["pre"].append(float(s["negative_delta_norm"]))
                            applied = float(s["applied_guidance_norm"])
                            values["actual"].append(applied)
                            values["clipped"].append(applied / 0.5)
                            values["clip"].append(float(s["clip_scale"]) < 1.0 - 1e-12)
                            values["active"].append(True)
                        elif arm != "A_vanilla":
                            values["pre"].append(float(s["clean_minus_shift_l2_norm_pre_clip"]))
                            values["clipped"].append(float(s["clipped_direction_l2_norm"]))
                            values["actual"].append(float(s["actual_applied_delta_l2_norm"]))
                            values["clip"].append(float(s["trust_region_clipping_active_bool"]))
                            values["active"].append(bool(s["active_timestep_shift_bool"]))
                step_rows.append({"task_id": task, "arm": arm, "flow_step": step, "n": len(values["actual"]), "pre_clip_mean": mean(values["pre"]), "pre_clip_rms": rms(values["pre"]), "clipped_direction_mean": mean(values["clipped"]), "actual_applied_mean": mean(values["actual"]), "actual_applied_rms": rms(values["actual"]), "clipping_fraction": mean([float(x) for x in values["clip"]]), "active_fraction": mean([float(x) for x in values["active"]])})
    with (art / "timestep_posthoc_per_step.csv").open("w", newline="") as stream:
        fields = list(step_rows[0]); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(step_rows)

    rescue_rows = []
    for pair, arms in sorted(grouped.items()):
        for arm in ARMS:
            if arm == "A_vanilla": continue
            rescue_rows.append({"pair_id": pair, "task_id": arms["A_vanilla"]["task_id"], "arm": arm, "vanilla_success": int(arms["A_vanilla"]["success"]), "arm_success": int(arms[arm]["success"]), "arm_rescued_vanilla_failure": int(not arms["A_vanilla"]["success"] and arms[arm]["success"]), "arm_regressed_vanilla_success": int(arms["A_vanilla"]["success"] and not arms[arm]["success"])})
    with (art / "timestep_posthoc_paired_rescue.csv").open("w", newline="") as stream:
        fields = list(rescue_rows[0]); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rescue_rows)

    summary = {"artifact": str(art), "new_rollouts": 0, "direction_cosine": {"status": "unavailable", "reason": "episode traces store norms, sums, hashes, and clip scales, but not correction vectors"}, "dose_response": {}, "paired_rescue": {}, "ref_vs_shift": {}}
    for task in (4, 7):
        summary["dose_response"][str(task)] = {}
        for arm in ("N0_pure_negative", "W05_shrink", "W15_extrapolate", "W20_extrapolate"):
            rows = [row for row in step_rows if row["task_id"] == task and row["arm"] == arm]
            summary["dose_response"][str(task)][arm] = {"actual_applied_rms_by_step": [row["actual_applied_rms"] for row in rows], "clipping_fraction_by_step": [row["clipping_fraction"] for row in rows], "active_fraction_by_step": [row["active_fraction"] for row in rows]}
        for arm in ("N0_pure_negative", "W05_shrink", "W15_extrapolate", "W20_extrapolate", "REF_toward_top8"):
            rows = [row for row in rescue_rows if int(row["task_id"]) == task and row["arm"] == arm]
            summary["paired_rescue"].setdefault(str(task), {})[arm] = {"vanilla_failure_rescued": sum(row["arm_rescued_vanilla_failure"] for row in rows), "vanilla_success_regressed": sum(row["arm_regressed_vanilla_success"] for row in rows), "discordant": sum(row["arm_rescued_vanilla_failure"] + row["arm_regressed_vanilla_success"] for row in rows)}
        summary["ref_vs_shift"][str(task)] = {arm: {"actual_applied_rms_mean_over_steps": mean([row["actual_applied_rms"] for row in step_rows if row["task_id"] == task and row["arm"] == arm]), "clipping_fraction_mean_over_steps": mean([row["clipping_fraction"] for row in step_rows if row["task_id"] == task and row["arm"] == arm])} for arm in ("N0_pure_negative", "W05_shrink", "W15_extrapolate", "W20_extrapolate", "REF_toward_top8")}
    (art / "timestep_posthoc_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
