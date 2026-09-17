"""Analyze outcome-aligned open-drawer same-state budget evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS=("l11_matched","l11_top_p80","l11_top_p85")
LOCKED_SEEDS={100,102,104,122,124,150,158,180,196,199}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def expanded(mask: np.ndarray, shape: tuple[int,int]) -> np.ndarray:
    return np.repeat(np.repeat(mask.reshape(16,16),shape[0]//16+1,0),shape[1]//16+1,1)[:shape[0],:shape[1]]


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--artifact",type=Path,required=True);args=parser.parse_args()
    root=args.artifact.resolve(); rows=[]; payloads=[]
    for path in sorted((root/"outcome_states").rglob("step_*.json")):
        data=json.loads(path.read_text());a=np.load(path.with_suffix(".npz"));m=data["metrics"]
        if int(data["seed"]) not in LOCKED_SEEDS:
            continue
        category=("matched_only" if data["closed_loop_outcomes"]["l11_matched"] and not data["closed_loop_outcomes"]["l11_top_p85"] else
                  "top_p85_only" if data["closed_loop_outcomes"]["l11_top_p85"] and not data["closed_loop_outcomes"]["l11_matched"] else
                  "both_success" if data["closed_loop_outcomes"]["l11_matched"] else "both_fail")
        row={"state_id":f'{data["task"]}__seed{data["seed"]:03d}__step{data["control_step"]:03d}',
             "seed":data["seed"],"phase":data["phase"],"category":category}
        for arm in ARMS:
            row[f"{arm}_m"]=m[arm]["actual_selected_count"]
            row[f"{arm}_residual"]=m[arm]["centered_logit_residual_norm"]
            row[f"{arm}_perturbation"]=m[arm]["feature_perturbation_relative"]
            row[f"{arm}_clean_flips"]=m[arm]["guided_changed_dims"]
        for arm in ARMS[1:]:
            row[f"{arm}_count_minus_matched"]=row[f"{arm}_m"]-row["l11_matched_m"]
            row[f"{arm}_action_dims_vs_matched"]=sum(x!=y for x,y in zip(m[arm]["final_token_ids"][:6],m["l11_matched"]["final_token_ids"][:6]))
        rows.append(row);payloads.append((row,data,a))
    if len(rows)!=30: raise RuntimeError(f"expected 30 outcome states, found {len(rows)}")
    stats=[]
    for category in ("matched_only","top_p85_only","both_success","both_fail","Overall"):
        chosen=rows if category=="Overall" else [row for row in rows if row["category"]==category]
        for arm in ARMS[1:]:
            d=np.asarray([row[f"{arm}_count_minus_matched"] for row in chosen],float)
            ac=np.asarray([row[f"{arm}_action_dims_vs_matched"] for row in chosen],float)
            stats.append({"category":category,"arm":arm,"states":len(chosen),"mean_count_delta":d.mean(),
                          "median_count_delta":np.median(d),"less_rate":np.mean(d<0),"more_rate":np.mean(d>0),
                          "mean_abs_count_delta":np.abs(d).mean(),"action_diff_state_rate":np.mean(ac>0),
                          "mean_action_dims_different":ac.mean()})
    statistics=root/"statistics";figures=root/"outcome_state_figures";statistics.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    write_csv(statistics/"outcome_same_state_metrics.csv",rows);write_csv(statistics/"outcome_category_summary.csv",stats)
    selected={}
    for seed in sorted({row["seed"] for row in rows}):
        candidates=[row for row in rows if row["seed"]==seed]
        # Prioritize a state where budget changes the action, then the largest budget gap.
        chosen=max(candidates,key=lambda row:(row["l11_top_p85_action_dims_vs_matched"]>0,
                                               row["l11_top_p85_action_dims_vs_matched"],
                                               abs(row["l11_top_p85_count_minus_matched"])))
        selected[str(seed)]={"state_id":chosen["state_id"],"phase":chosen["phase"],"category":chosen["category"],
                             "action_dims_different":chosen["l11_top_p85_action_dims_vs_matched"],
                             "count_delta":chosen["l11_top_p85_count_minus_matched"]}
    (statistics/"causal_state_manifest.json").write_text(json.dumps(selected,indent=2)+"\n")

    for row,data,a in payloads:
        image=a["image"];fig,axes=plt.subplots(2,3,figsize=(13,8));axes[0,0].imshow(image);axes[0,0].set_title(f'{row["category"]} seed {row["seed"]} {row["phase"]}')
        attention=a["l11_matched__prompt_attention"].reshape(16,16);axes[0,1].imshow(image);axes[0,1].imshow(attention,cmap="magma",alpha=.62,extent=(0,image.shape[1],image.shape[0],0));axes[0,1].set_title("L11 attention")
        for ax,arm,title in ((axes[0,2],"l11_matched","Matched"),(axes[1,0],"l11_top_p80","TopP80"),(axes[1,1],"l11_top_p85","TopP85")):
            base=image.astype(float)/255;mask=expanded(a[f"{arm}__selected_mask"],image.shape[:2]);base[mask>0]=.52*base[mask>0]+.48*np.array([1,.1,.1]);ax.imshow(base);ax.set_title(f'{title} m={row[f"{arm}_m"]}')
        mm=a["l11_matched__selected_mask"].astype(bool);pp=a["l11_top_p85__selected_mask"].astype(bool);color=np.zeros((256,3));color[mm&~pp]=(.1,.3,1);color[pp&~mm]=(1,.55,.05);mask=expanded((mm^pp).astype(np.uint8),image.shape[:2]);ec=np.repeat(np.repeat(color.reshape(16,16,3),image.shape[0]//16+1,0),image.shape[1]//16+1,1)[:image.shape[0],:image.shape[1]];base=image.astype(float)/255;base[mask>0]=.45*base[mask>0]+.55*ec[mask>0];axes[1,2].imshow(base);axes[1,2].set_title("blue=Matched only; orange=TopP85 only")
        for ax in axes.ravel():ax.axis("off")
        fig.tight_layout();fig.savefig(figures/f'{row["state_id"]}.png',dpi=150);plt.close(fig)
    print(json.dumps({"states":len(rows),"summary":stats,"causal_manifest":selected},indent=2))


if __name__=="__main__":main()
