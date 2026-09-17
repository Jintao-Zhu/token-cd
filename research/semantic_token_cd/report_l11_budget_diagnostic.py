"""Build the final evidence-bounded report for the L11 budget diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--artifact",type=Path,required=True);args=parser.parse_args();root=args.artifact.resolve();stats=root/"statistics"
    rows=list(csv.DictReader((stats/"same_state_metrics.csv").open()))
    phase_rows=[]
    for task in ("open_drawer","close_drawer","pick_coke_can","move_near","Overall"):
        base=rows if task=="Overall" else [row for row in rows if row["task"]==task]
        for phase in ("early","middle","late"):
            chosen=[row for row in base if row["phase"]==phase]
            for arm in ("l11_top_p80","l11_top_p85"):
                delta=np.asarray([float(row[f"{arm}_count_minus_matched"]) for row in chosen])
                action=np.asarray([float(row[f"{arm}_action_dims_vs_matched"]) for row in chosen])
                phase_rows.append({"task":task,"phase":phase,"arm":arm,"states":len(chosen),
                                   "mean_count_delta":delta.mean(),"less_rate":np.mean(delta<0),"more_rate":np.mean(delta>0),
                                   "action_diff_state_rate":np.mean(action>0),"mean_action_dims_different":action.mean()})
    write_csv(stats/"phase_budget_summary.csv",phase_rows)

    causal=[]
    for path in sorted((root/"causal_one_step").rglob("*.json")):
        data=json.loads(path.read_text());m=data["results"]["matched_action_then_matched"];p=data["results"]["top_p85_action_then_matched"]
        causal.append({"seed":data["seed"],"step":data["step"],"phase":data["phase"],"category":data["category"],
                       "original_matched_success":data["original_matched_success"],"matched_branch_success":m["success"],
                       "top_p85_branch_success":p["success"],"technical_valid":data["interpretation_valid"],
                       "outcome_changed_by_one_top_p85_action":bool(data["interpretation_valid"] and m["success"]!=p["success"])})
    write_csv(stats/"causal_one_step_results.csv",causal)
    valid=[row for row in causal if row["technical_valid"]]
    causal_summary={"attempted":len(causal),"technically_valid":len(valid),"excluded_nonreproducing_matched_continuations":len(causal)-len(valid),
                    "valid_outcome_flips":sum(row["outcome_changed_by_one_top_p85_action"] for row in valid),
                    "boundary":"A null one-step result does not prove the full trajectory is unaffected."}
    (stats/"causal_one_step_summary.json").write_text(json.dumps(causal_summary,indent=2)+"\n")

    # Two deliberately contrasting cases: Matched-only and TopP85-only.
    case_paths=[root/"outcome_state_figures/google_robot_open_drawer__seed122__step101.png",
                root/"outcome_state_figures/google_robot_open_drawer__seed124__step056.png"]
    images=[Image.open(path).convert("RGB") for path in case_paths]
    width=1200; resized=[im.resize((width,round(im.height*width/im.width))) for im in images]
    canvas=Image.new("RGB",(width,sum(im.height for im in resized)),"white");y=0
    for im in resized:canvas.paste(im,(0,y));y+=im.height
    canvas.save(root/"key_contrasting_cases.png")

    lines=[
        "# Why Matched beats cumulative Top-p: targeted diagnostic", "",
        "## Scope", "",
        "No new adaptive rule was introduced. The analysis used 60 pre-existing identical observations (15 per task), plus 30 fully replay-verified open-drawer states from 10 outcome-stratified episodes. Matched, TopP80 and TopP85 always used the same L11 ranking; only the cutoff differed.", "",
        "## Main findings", "",
        "1. **Global mean count hides large state-level errors.** Across the 60 common states, TopP80 differed from Matched by 16.0 tokens on average in absolute value despite a signed mean of only +3.1. It selected fewer tokens in 46.7% of states and more in 53.3%. TopP85 differed by 18.3 tokens in absolute value, selecting fewer in 31.7% and more in 66.7%.",
        "2. **Drawer mismatch is phase-dependent, not a fixed under-selection.** In open-drawer, TopP80 was below Matched by 6.6 tokens on average in middle states but above it by 6.8 in late states. TopP85 was below Matched in 53.3% of all sampled open-drawer states, despite a positive overall mean difference.",
        "3. **The differing rank tail often contains structured context, but it is not uniformly beneficial.** Visual inspection shows drawer fronts/edges, adjacent drawer bands, robot/contact vicinity, and occasional background. Seed 122 is a Matched-only case where the extra tail covers drawer structure and changes one action dimension. Seed 124 is the opposite: Matched extends over several drawer bands, robot and background, while the smaller TopP85 set is more concentrated and the full TopP85 rollout succeeds.",
        "4. **Budget mismatch frequently reaches the policy.** On the 60 identical states, TopP85 and Matched produced different guided actions in 56.7% of states (1.17 differing dimensions on average). On the 30 outcome-aligned open-drawer states, they differed in 53.3% (1.0 dimension on average).",
        "5. **Count difference alone does not predict action difference or success.** For the 60 states, correlation between signed TopP85 count difference and number of changed dimensions was approximately zero (-0.006). In the outcome sample, TopP85-only states tended to use fewer tokens, while both-success states used substantially more; therefore neither 'more is better' nor 'less is better' explains Matched.",
        f'6. **One-step causal replacement was null in the technically valid subset.** {causal_summary["technically_valid"]} of {causal_summary["attempted"]} continuations reproduced the Matched control outcome; none changed success after replacing one selected action with TopP85 and then returning to Matched. The remaining {causal_summary["excluded_nonreproducing_matched_continuations"]} were excluded because the restored Matched continuation did not reproduce its original outcome. This suggests the closed-loop difference is accumulated over multiple replans, or that the sampled single step was not decisive.',
        "", "## Conclusion", "",
        "The data support the narrow conclusion that **attention concentration is not a sufficient proxy for the useful intervention budget**. They do not yet prove that KMeans group size directly estimates the correct causal evidence extent. Matched appears to provide a different, phase-sensitive budget signal; the extra or omitted rank segment can contain task-relevant structure, but also irrelevant context. Its benefit is therefore conditional rather than a simple preference for larger masks.",
        "", "The next justified experiment would manipulate only the identified rank segment over a short multi-step window on replayable states. No such new rule or full closed-loop arm was run here.",
    ]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"complete":True,"same_states":60,"outcome_aligned_states":30,"causal":causal_summary},indent=2))


if __name__=="__main__":main()
