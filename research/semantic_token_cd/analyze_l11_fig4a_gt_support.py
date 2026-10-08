#!/usr/bin/env python3
"""Episode-clustered correlations between Matched m_t and GT projected areas."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

TASKS=("google_robot_open_drawer","google_robot_close_drawer",
       "google_robot_pick_coke_can","google_robot_move_near")
METRICS=("gt_target_area_fraction","gt_target_bbox_area_fraction")

def rho(x,y):
    value=float(spearmanr(x,y).statistic)
    return value if np.isfinite(value) else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=2000)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261005)
    report={"protocol":"FIG4A_GT_INDEPENDENT_SUPPORT_V1","bootstrap_unit":"episode; both early/late states resampled together",
            "bootstrap_replicates":args.bootstrap,"metrics":{},"episodes_per_task":{},"states_per_task":{}}
    rows_all=[]
    for task in TASKS:
        path=args.input_dir/f"{task}.csv"
        with path.open() as f: rows=list(csv.DictReader(f))
        if len(rows)!=200: raise RuntimeError(f"expected 200 states for {task}; found {len(rows)}")
        by={}
        for row in rows:
            row["seed"]=int(row["seed"]);row["m_t"]=float(row["m_t"])
            for key in METRICS: row[key]=float(row[key])
            by.setdefault(row["seed"],[]).append(row)
        if len(by)!=100 or any(len(v)!=2 for v in by.values()): raise RuntimeError(f"bad episode/state topology for {task}")
        report["episodes_per_task"][task]=len(by);report["states_per_task"][task]=len(rows)
        task_metrics={}
        seeds=np.array(sorted(by))
        for metric in METRICS:
            x=np.array([r["m_t"] for r in rows]);y=np.array([r[metric] for r in rows])
            state_rho=rho(x,y)
            ep_x=np.array([np.mean([r["m_t"] for r in by[s]]) for s in seeds])
            ep_y=np.array([np.mean([r[metric] for r in by[s]]) for s in seeds])
            episode_mean_rho=rho(ep_x,ep_y)
            boot_state=[];boot_episode=[]
            for _ in range(args.bootstrap):
                chosen=rng.choice(seeds,size=len(seeds),replace=True)
                bx=[];byy=[];ex=[];ey=[]
                for seed in chosen:
                    group=by[int(seed)]
                    bx.extend(r["m_t"] for r in group);byy.extend(r[metric] for r in group)
                    ex.append(float(np.mean([r["m_t"] for r in group])))
                    ey.append(float(np.mean([r[metric] for r in group])))
                boot_state.append(rho(np.asarray(bx),np.asarray(byy)))
                boot_episode.append(rho(np.asarray(ex),np.asarray(ey)))
            task_metrics[metric]={
                "state_level_rho":state_rho,
                "state_level_episode_bootstrap_95ci":[float(np.quantile(boot_state,.025)),float(np.quantile(boot_state,.975))],
                "episode_mean_rho":episode_mean_rho,
                "episode_mean_bootstrap_95ci":[float(np.quantile(boot_episode,.025)),float(np.quantile(boot_episode,.975))],
                "n_episodes":len(by),"n_states":len(rows),
            }
        report["metrics"][task]=task_metrics
        rows_all.extend(rows)
    (args.output_dir/"gt_support_correlations.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.output_dir/"gt_support_correlations.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["task","metric","state_level_rho","state_level_cluster_bootstrap_ci_low","state_level_cluster_bootstrap_ci_high","episode_mean_rho","episode_mean_bootstrap_ci_low","episode_mean_bootstrap_ci_high","episodes","states"])
        for task in TASKS:
            for metric in METRICS:
                x=report["metrics"][task][metric];writer.writerow([task,metric,x["state_level_rho"],*x["state_level_episode_bootstrap_95ci"],x["episode_mean_rho"],*x["episode_mean_bootstrap_95ci"],x["n_episodes"],x["n_states"]])
    try:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        labels={"google_robot_open_drawer":"Open drawer","google_robot_close_drawer":"Close drawer","google_robot_pick_coke_can":"Pick Coke can","google_robot_move_near":"Move near"}
        for ax,task in zip(axes.flat,TASKS):
            rr=[r for r in rows_all if r["task"]==task]
            for segment,color in (("early","#277da1"),("late","#f9844a")):
                part=[r for r in rr if r["sample_segment"]==segment]
                ax.scatter([r["gt_target_area_fraction"] for r in part],[r["m_t"] for r in part],s=16,alpha=.65,label=segment,color=color)
            item=report["metrics"][task]["gt_target_area_fraction"]
            lo,hi=item["state_level_episode_bootstrap_95ci"]
            ax.set_title(f"{labels[task]}  ρ={item['state_level_rho']:.2f} [{lo:.2f}, {hi:.2f}]")
            ax.set_xlabel("GT target area fraction");ax.set_ylabel("Matched mₜ");ax.grid(alpha=.2)
        axes.flat[0].legend(frameon=False)
        fig.savefig(args.output_dir/"gt_support_correlation.png",dpi=180)
        plt.close(fig)
    except Exception as exc:
        report["plot_error"]=repr(exc)
        (args.output_dir/"gt_support_correlations.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(args.output_dir),"episodes":report["episodes_per_task"],"metrics":report["metrics"]},indent=2))

if __name__=="__main__":main()
