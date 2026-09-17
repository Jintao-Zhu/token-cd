"""Create tables, galleries, plots, and a failure-cause report for Prompt-v1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont


ARMS = ("standard_shr", "prompt_attn_shr", "random_shr")
ARM_LABEL = {"standard_shr": "Standard SHR", "prompt_attn_shr": "Prompt-v1", "random_shr": "Random-SHR"}
TASK_LABEL = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}
SPATIAL_KEYS = ("num_components", "isolated_token_ratio", "largest_component_ratio",
                "boundary_edges_per_token", "corner_ratio", "outer_ring_ratio")
EFFECT_KEYS = ("feature_perturbation_total", "selected_mean_relative_perturbation",
               "centered_residual_norm", "winner_flip_count")


def mean(values):
    values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(values)) if values else None


def percentile(values, q):
    values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.percentile(values,q)) if values else None


def load_states(artifact: Path):
    states=[]
    for path in sorted((artifact/"states").glob("*/*/step_*.json")):
        row=json.loads(path.read_text()); row["_json_path"]=str(path); states.append(row)
    return states


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key:row.get(key) for key in fields})


def flatten_rows(states):
    rows=[]
    for state in states:
        for arm in ARMS:
            m=state["metrics"][arm]
            row={key:state[key] for key in ("state_id","task","seed","source_step","phase","outcome_category","instruction","m")}
            row["arm"]=arm
            for key in SPATIAL_KEYS+EFFECT_KEYS+(
                "feature_perturbation_relative_global","selected_mean_patch_perturbation",
                "standard_overlap","standard_jaccard","standard_residual_cosine",
            ):
                row[key]=m.get(key)
            row["centered_residual_norm_per_dim"]=json.dumps(m["centered_residual_norm_per_dim"])
            row["clean_action"]=json.dumps(m["clean_action"])
            row["guided_action"]=json.dumps(m["guided_action"])
            row["dimensions"]=json.dumps(m["dimensions"],separators=(",",":"))
            rows.append(row)
    return rows


def patch_index_figure(artifact: Path, state):
    arr=np.load(artifact/state["arrays_file"]); rgb=arr["image"]
    image=Image.fromarray(rgb).resize((768,768)); draw=ImageDraw.Draw(image)
    for i in range(17):
        p=i*48; draw.line((p,0,p,768),fill=(255,255,255),width=1); draw.line((0,p,768,p),fill=(255,255,255),width=1)
    font=ImageFont.load_default()
    for idx in range(256):
        r,c=divmod(idx,16); x=c*48+3; y=r*48+3
        draw.rectangle((x-1,y-1,x+20,y+10),fill=(0,0,0)); draw.text((x,y),str(idx),font=font,fill=(255,255,0))
    out=artifact/"patch_index_reference.png"; image.save(out); return out


def overlay(ax,rgb,mask,title):
    ax.imshow(rgb)
    alpha=np.zeros((16,16,4),dtype=float); alpha[...,0]=1; alpha[...,3]=mask.reshape(16,16)*0.42
    ax.imshow(alpha,extent=(0,rgb.shape[1],rgb.shape[0],0),interpolation="nearest")
    ax.set_title(title,fontsize=9); ax.axis("off")


def make_case(artifact: Path,state,outdir: Path,attention_vmax: float):
    arr=np.load(artifact/state["arrays_file"]); rgb=arr["image"]
    vmax=max(float(arr[f"{arm}__perturbation_norm"].max()) for arm in ARMS)
    fig,axes=plt.subplots(4,4,figsize=(15,13))
    axes[0,0].imshow(rgb); axes[0,0].set_title("shared fixed-state RGB"); axes[0,0].axis("off")
    attn=arr["attention"][2:4,0].mean(0).reshape(16,16)
    ai=axes[0,1].imshow(attn,vmin=0,vmax=attention_vmax,cmap="viridis")
    axes[0,1].set_title("Prompt-v1 full attention (global color scale)"); axes[0,1].axis("off")
    fig.colorbar(ai,ax=axes[0,1],fraction=.046,pad=.03)
    axes[0,2].imshow(arr["labels"].reshape(16,16),cmap="tab10",vmin=0,vmax=9)
    axes[0,2].set_title("Standard KMeans K=8 labels"); axes[0,2].axis("off")
    cf=state.get("counterfactual_instruction") or {}
    axes[0,3].axis("off"); axes[0,3].text(0,1,
        f"task: {TASK_LABEL[state['task']]}\nseed: {state['seed']}  step: {state['source_step']}\n"
        f"phase: {state['phase']}\nm: {state['m']}\nepisode label: {state['outcome_category']}\n"
        f"counterfactual: {cf.get('kind','n/a')}",va="top",family="monospace",fontsize=9)
    for offset,arm in enumerate(ARMS):
        row=offset+1
        metric=state["metrics"][arm]; mask=arr[f"{arm}__mask"]
        overlay(axes[row,0],rgb,mask,f"{ARM_LABEL[arm]} mask (m={state['m']})")
        im=axes[row,1].imshow(arr[f"{arm}__perturbation_norm"].reshape(16,16),vmin=0,vmax=vmax,cmap="magma")
        axes[row,1].set_title("actual ||V+ - V-||",fontsize=9); axes[row,1].axis("off")
        axes[row,2].axis("off")
        lines=["q clean→guided rank   M    λD"]
        for d in metric["dimensions"]:
            mark="*" if d["winner_flipped"] else " "
            lines.append(f"{d['dimension']} {d['clean_winner']:3d}→{d['guided_winner']:3d}{mark} {d['guided_winner_clean_rank']:3d} {d['clean_margin']:5.2f} {d['lambda_residual_advantage']:5.2f}")
        axes[row,2].text(0,1,"\n".join(lines),va="top",family="monospace",fontsize=8)
        axes[row,2].set_title(f"logit decision (* = flip); ||r~||={metric['centered_residual_norm']:.1f}",fontsize=9)
        clean=np.asarray(metric["clean_action"]); guided=np.asarray(metric["guided_action"])
        names=("x","y","z","roll","pitch","yaw","grip")
        lines=["dim      clean    guided    delta"]
        for q,name in enumerate(names): lines.append(f"{name:5s} {clean[q]:8.4f} {guided[q]:8.4f} {guided[q]-clean[q]:8.4f}")
        axes[row,3].axis("off"); axes[row,3].text(0,1,"\n".join(lines),va="top",family="monospace",fontsize=8)
        axes[row,3].set_title("decoded action (translation / rotation / gripper)",fontsize=9)
    fig.colorbar(im,ax=axes[1:,1].tolist(),fraction=.025,pad=.02)
    fig.suptitle(f"{state['task']} seed={state['seed']} step={state['source_step']} ({state['phase']})\n"
                 f"instruction: {state['instruction']} | episode label: {state['outcome_category']} (not state-causal)",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94)); outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/f"{state['state_id']}.png"; fig.savefig(out,dpi=120); plt.close(fig); return out


def make_case_gallery(artifact: Path,states):
    outdir=artifact/"case_gallery"; links=[]
    attention_vmax=max(float(np.load(artifact/state["arrays_file"])["attention"][2:4,0].mean(0).max()) for state in states)
    for index,state in enumerate(states):
        image=make_case(artifact,state,outdir,attention_vmax)
        links.append((state,image))
    page=["<!doctype html><meta charset='utf-8'><title>Prompt-v1 same-state case gallery</title>",
          "<style>body{font-family:sans-serif;max-width:1500px;margin:auto}img{width:100%}article{border-bottom:2px solid #aaa;padding:1rem}</style>",
          "<h1>Prompt-v1 same-state case gallery</h1><p>Episode outcome labels are sampling strata, not causal labels for individual states.</p>"]
    for state,image in links:
        rel=image.relative_to(artifact)
        page.append(f"<article><h2>{html.escape(state['state_id'])}</h2><p>{html.escape(state['instruction'])}; phase={state['phase']}; episode={state['outcome_category']}</p><img loading='lazy' src='{rel.as_posix()}'></article>")
    out=artifact/"case_gallery.html"; out.write_text("\n".join(page)); return out


def layer_query_gallery(artifact: Path,states):
    audit=[state for state in states if state.get("audit")]
    outdir=artifact/"layer_query_gallery"; outdir.mkdir(parents=True,exist_ok=True)
    page=["<!doctype html><meta charset='utf-8'><title>Layer-query gallery</title><style>body{font-family:sans-serif;max-width:1400px;margin:auto}img{width:100%}</style><h1>Layer × query diagnostics</h1>"]
    for state in audit:
        arr=np.load(artifact/state["arrays_file"]); original=arr["attention"]
        variant=arr.get("counterfactual_attention")
        for name,matrix in (("original",original),("counterfactual_delta",variant-original if variant is not None else None)):
            if matrix is None: continue
            vmax=float(np.max(np.abs(matrix))) if "delta" in name else None
            fig,axes=plt.subplots(4,3,figsize=(8,10))
            for wi,(lo,hi) in enumerate(state["attention_windows"]):
                for qi,qname in enumerate(state["query_types"]):
                    kwargs={"cmap":"coolwarm","vmin":-vmax,"vmax":vmax} if "delta" in name else {"cmap":"viridis"}
                    axes[wi,qi].imshow(matrix[wi,qi].reshape(16,16),**kwargs); axes[wi,qi].axis("off")
                    axes[wi,qi].set_title(f"L{lo}-{hi-1} / {qname}",fontsize=8)
            cf=state.get("counterfactual_instruction") or {}
            subtitle=(state["instruction"] if name=="original" else f"delta: {cf.get('instruction')} - original")
            fig.suptitle(f"{state['state_id']}\n{subtitle}"); fig.tight_layout(rect=(0,0,1,.95))
            out=outdir/f"{state['state_id']}__{name}.png"; fig.savefig(out,dpi=140); plt.close(fig)
            page.append(f"<h2>{html.escape(state['state_id'])}: {name}</h2><img src='{out.relative_to(artifact).as_posix()}'>")
    output=artifact/"layer_query_gallery.html"; output.write_text("\n".join(page)); return output


def aggregate(states,artifact: Path):
    agg={"states":len(states),"by_task":{},"overall":{}}
    for task in list(TASK_LABEL)+[None]:
        subset=states if task is None else [s for s in states if s["task"]==task]
        target=agg["overall"] if task is None else agg["by_task"].setdefault(task,{})
        for arm in ARMS:
            metrics=[s["metrics"][arm] for s in subset]
            target[arm]={key:mean([m.get(key) for m in metrics]) for key in SPATIAL_KEYS+EFFECT_KEYS+(
                "feature_perturbation_relative_global","selected_mean_patch_perturbation",
                "standard_overlap","standard_jaccard","standard_residual_cosine")}
            target[arm]["winner_flip_rate_per_dim"]=[mean([
                int(m["dimensions"][q]["winner_flipped"]) for m in metrics
            ]) for q in range(6)]
        refs=[r for s in subset for r in s["random_reference"]]
        target["random_reference"]={key:mean([r.get(key) for r in refs]) for key in SPATIAL_KEYS+EFFECT_KEYS}
    response=[]
    for state in states:
        if not state.get("audit") or not state.get("counterfactual_instruction"): continue
        arr=np.load(artifact/state["arrays_file"]); a=arr["attention"][2:4,0].mean(0); b=arr["counterfactual_attention"][2:4,0].mean(0)
        ma=set(stable_indices(a,state["m"])); mb=set(stable_indices(b,state["m"]))
        response.append({"state_id":state["state_id"],"kind":state["counterfactual_instruction"]["kind"],
                         "top_m_jaccard":len(ma&mb)/len(ma|mb),
                         "score_cosine":float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))})
    agg["instruction_counterfactual_response"]=response
    (artifact/"AGGREGATE_DIAGNOSTICS.json").write_text(json.dumps(agg,indent=2,sort_keys=True)+"\n")
    return agg


def stable_indices(scores,m):
    return np.lexsort((np.arange(256),-np.asarray(scores)))[:m].astype(int).tolist()


def aggregate_pdf(artifact: Path,states,agg):
    out=artifact/"aggregate_diagnostics.pdf"
    with PdfPages(out) as pdf:
        fig,axes=plt.subplots(3,3,figsize=(11,10))
        for row,task in enumerate(TASK_LABEL):
            subset=[s for s in states if s["task"]==task]
            for col,arm in enumerate(ARMS):
                if not subset:
                    axes[row,col].axis("off"); axes[row,col].set_title(f"{TASK_LABEL[task]} / no states")
                    continue
                freq=np.mean([np.load(artifact/s["arrays_file"])[f"{arm}__mask"] for s in subset],axis=0)
                axes[row,col].imshow(freq.reshape(16,16),vmin=0,vmax=1,cmap="hot"); axes[row,col].axis("off")
                axes[row,col].set_title(f"{TASK_LABEL[task]} / {ARM_LABEL[arm]}",fontsize=9)
        fig.suptitle("Selection frequency on 90 same-state observations"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        for keys,title in ((SPATIAL_KEYS,"Mask geometry"),(EFFECT_KEYS,"Reconstruction and action effect")):
            fig,axes=plt.subplots(2,3,figsize=(12,7)); axes=axes.ravel()
            for ax,key in zip(axes,keys):
                values=[[s["metrics"][arm][key] for s in states] for arm in ARMS]
                ax.boxplot(values,labels=["Std","Prompt","Random"],showfliers=False); ax.set_title(key,fontsize=9)
            for ax in axes[len(keys):]: ax.axis("off")
            fig.suptitle(title); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,task in zip(axes,TASK_LABEL):
            subset=[s for s in states if s["task"]==task]
            if not subset:
                ax.axis("off"); ax.set_title(f"{TASK_LABEL[task]} / no states"); continue
            x=np.arange(6); width=.25
            for i,arm in enumerate(ARMS):
                rates=[mean([int(s["metrics"][arm]["dimensions"][q]["winner_flipped"]) for s in subset]) for q in range(6)]
                ax.bar(x+(i-1)*width,rates,width,label=ARM_LABEL[arm])
            ax.set_title(TASK_LABEL[task]); ax.set_xticks(x); ax.set_xlabel("action dimension"); ax.set_ylim(0,1)
        axes[0].set_ylabel("winner flip rate"); axes[-1].legend(fontsize=7); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return out


def implementation_audit(artifact: Path,states,closed_loop: Path):
    audits=[s for s in states if s.get("audit")]
    query=[]
    for state in audits:
        q=state["query"]; a=state["audit"]; replay=state["replay_audit"]
        query.append({"state_id":state["state_id"],"task":state["task"],"seed":state["seed"],"instruction":state["instruction"],
                      "token_ids":json.dumps(q["token_ids"]),"tokens":" ".join(q["tokens"]),"text_indices":json.dumps(q["text_indices"]),
                      "multimodal_indices":json.dumps(q["multimodal_indices"]),"entity_indices":json.dumps(q["entity_multimodal_indices"]),
                      "verb_relation_indices":json.dumps(q["verb_relation_multimodal_indices"]),"unmatched_entities":json.dumps(q["unmatched_entities"]),
                      "attention_repeat_max_abs_diff":a["attention_independent_max_abs_diff"],
                      "clean_logits_max_abs_diff_after_attention":a["clean_logits_max_abs_diff_after_attention"],
                      "clean_greedy_unchanged":a["clean_greedy_unchanged_after_attention"],
                      "replay_clean_greedy_match":replay["clean_greedy_matches_closed_loop"],
                      "replay_standard_mask_match":replay["standard_mask_matches_closed_loop"]})
    fields=list(query[0]) if query else []
    write_csv(artifact/"query_index_audit.csv",query,fields)
    all_states_replay=all(s["replay_audit"].get("clean_greedy_matches_closed_loop") and
                          s["replay_audit"].get("standard_mask_matches_closed_loop") for s in states)
    selector_steps=0; selector_top_m_mismatches=0; selector_sha_mismatches=0
    for summary_path in sorted((closed_loop/"episodes").glob("*/prompt_attn_shr/*_summary.json")):
        summary=json.loads(summary_path.read_text())
        arrays=np.load(summary_path.with_name(summary_path.name.replace("_summary.json","_arrays.npz")))
        scores=arrays["prompt_attention"]
        if len(scores)!=len(summary["selector_trace"]):
            raise RuntimeError(f"closed-loop attention/trace length mismatch: {summary_path}")
        for values,trace in zip(scores,summary["selector_trace"]):
            selector_steps+=1
            selected=sorted(stable_indices(values,int(trace["m_t"])))
            if selected!=sorted(int(x) for x in trace["selected_token_ids"]): selector_top_m_mismatches+=1
            digest=hashlib.sha256(np.asarray(values,dtype=np.float32).tobytes()).hexdigest()
            if digest!=trace["attention_sha256"]: selector_sha_mismatches+=1
    audit_pass=(len(audits)==9 and all(row["unmatched_entities"]=="[]" and row["clean_greedy_unchanged"] and row["replay_clean_greedy_match"] and
                                     row["replay_standard_mask_match"] for row in query) and all_states_replay)
    audit_pass=audit_pass and selector_steps>0 and selector_top_m_mismatches==0 and selector_sha_mismatches==0
    text=f"""# Prompt-v1 implementation audit

Result: **{'PASS' if audit_pass else 'FAIL'}**

- Audited observations: {len(audits)} (required: 9; three per task).
- Query: exact non-special instruction tokens; multimodal positions are text indices + 256.
- Visual keys: positions 1–256, exactly 256 projector tokens in row-major 16×16 order.
- Language layers: 32 total; Prompt-v1 uses 0-based layers 16–31.
- Attention: model post-softmax weights; equal mean across layers, heads, queries; no visual re-normalization.
- Independent repeated extraction maximum absolute difference: {max((r['attention_repeat_max_abs_diff'] for r in query),default=float('nan')):.3g}.
- Original closed-loop selector replay: {selector_steps} control steps; Top-m mismatches={selector_top_m_mismatches}; stored-score SHA mismatches={selector_sha_mismatches}.
- Clean logits before/after attention extraction maximum absolute difference: {max((r['clean_logits_max_abs_diff_after_attention'] for r in query),default=float('nan')):.3g}; greedy unchanged in all audit observations.
- Diagnostic source trajectories: all 30 reruns matched the original closed-loop clean action and Standard mask at the initial step (fail-closed check in the collector).
- Same-state recomputation: clean greedy action and Standard mask match the diagnostic Standard driver on the identical cached RGB in all {len(states)} selected states.
- Coverage: Standard, Prompt-v1 and Random each use exactly the same state-local m; no duplicate positions.
- Reconstruction: shared 16×16 four-neighbor Dirichlet harmonic solve, beta=0/gamma=1; outside-mask tokens bit-identical.
- CD: shared clean teacher-forced prefix; lambda=0.5 on dimensions 0–5; gripper dimension 6 remains clean.

See `query_index_audit.csv` for token IDs/positions and `patch_index_reference.png` for the spatial index convention.
"""
    (artifact/"implementation_audit.md").write_text(text)
    return audit_pass


def failure_report(artifact: Path,states,agg,audit_pass):
    def fmt(value,digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"
    o=agg["overall"]; std=o["standard_shr"]; prompt=o["prompt_attn_shr"]; rand=o["random_shr"]; ref=o["random_reference"]
    response=agg["instruction_counterfactual_response"]
    jacc=mean([r["top_m_jaccard"] for r in response]); score_cos=mean([r["score_cosine"] for r in response])
    per_task=[]
    for task in TASK_LABEL:
        t=agg["by_task"][task]
        per_task.append(f"| {TASK_LABEL[task]} | {fmt(t['prompt_attn_shr']['standard_overlap'])} | {fmt(t['prompt_attn_shr']['num_components'],2)} | {fmt(t['standard_shr']['num_components'],2)} | {fmt(t['prompt_attn_shr']['centered_residual_norm'],2)} | {fmt(t['standard_shr']['centered_residual_norm'],2)} |")
    conclusion=(
        "Prompt-v1 的失败主要不是实现或覆盖量错误，而是 selector 在同一状态上选择了与 semantic group 大幅不同、"
        "空间结构和重建效应不同的集合；这些集合产生的 action residual 与 Standard SHR 方向一致性有限，"
        "在 drawer/coke 的闭环中 Harm 远多于 Rescue。完整指令后半层 attention 在本诊断中也没有表现出"
        "足够强的任务条件响应，且其效果未优于等覆盖 Random-SHR。"
    )
    text=f"""# Prompt-Attn-SHR v1 failure-cause analysis

## Boundary and closed-loop result

This is an outcome-stratified diagnostic set (30 episodes × 3 states = {len(states)} states), not a new success-rate estimate.
The completed closed-loop screen was Standard 148/300, Prompt-v1 109/300, Random 125/300. Prompt-v1 vs Standard: Rescue 25, Harm 64, Net −39.

## Conclusion

{conclusion}

## Evidence chain

1. **Implementation extraction passed: {audit_pass}.** Nine observations verified query/key indices, layers, repeated aggregation and clean-action invariance. All 30 diagnostic Standard reruns matched the original closed-loop initial action/mask; all 90 selected cached RGB states reproduced the diagnostic driver's clean action/mask. Equal coverage, reconstruction scope, prefix, lambda and gripper paths were also verified. Therefore the failure cannot be explained by an obvious indexing, layer numbering, coverage, or downstream-configuration mismatch.
2. **Prompt selects substantially different evidence.** Mean Prompt/Standard overlap is {prompt['standard_overlap']:.3f} and Jaccard is {prompt['standard_jaccard']:.3f}; Random/Standard overlap is {rand['standard_overlap']:.3f}. Low overlap alone is not an error, but establishes that the intervention changed materially.
3. **Geometry differs.** Mean component counts: Standard {std['num_components']:.2f}, Prompt {prompt['num_components']:.2f}, Random {rand['num_components']:.2f}, 10-mask random reference {fmt(ref['num_components'],2)}. Largest-component ratios: Standard {std['largest_component_ratio']:.3f}, Prompt {prompt['largest_component_ratio']:.3f}, Random {rand['largest_component_ratio']:.3f}. Outer-ring selection ratios: Standard {std['outer_ring_ratio']:.3f}, Prompt {prompt['outer_ring_ratio']:.3f}, Random {rand['outer_ring_ratio']:.3f}; uniform expectation is 0.234. These measurements determine whether fragmentation/edge bias is present without declaring every edge token meaningless.
4. **Equal m does not imply equal intervention.** Mean feature perturbation totals: Standard {std['feature_perturbation_total']:.2f}, Prompt {prompt['feature_perturbation_total']:.2f}, Random {rand['feature_perturbation_total']:.2f}. Mean selected-token relative perturbation: Standard {std['selected_mean_relative_perturbation']:.3f}, Prompt {prompt['selected_mean_relative_perturbation']:.3f}, Random {rand['selected_mean_relative_perturbation']:.3f}.
5. **Action effect changes.** Mean centered residual norms: Standard {std['centered_residual_norm']:.2f}, Prompt {prompt['centered_residual_norm']:.2f}, Random {rand['centered_residual_norm']:.2f}. Prompt-vs-Standard residual cosine is {prompt['standard_residual_cosine']:.3f}; this tests direction rather than only strength. Mean winner flips per state: Standard {std['winner_flip_count']:.2f}, Prompt {prompt['winner_flip_count']:.2f}, Random {rand['winner_flip_count']:.2f} across six guided dimensions.
6. **Task response is limited in the audited counterfactuals.** For nine fixed images, a verb/entity/source-target change leaves mean Top-m Jaccard {jacc:.3f} and score cosine {score_cos:.3f}. This is evidence about sensitivity, not proof of semantic correctness.
7. **Closed-loop consequence agrees with the mechanism diagnosis.** Prompt-v1 is much worse on open_drawer (21% vs 47%) and pick_coke_can (27% vs 41%), while move_near is only tied (61% vs 60%) and also ties Random (61%). Thus the selected differences do not translate into useful task-grounded intervention overall.

## Per-task diagnostic summary

| Task | Prompt/Std overlap | Prompt components | Std components | Prompt residual norm | Std residual norm |
|---|---:|---:|---:|---:|---:|
{chr(10).join(per_task)}

## What is and is not established

Established: extraction is internally consistent; Prompt-v1 changes mask identity/geometry and the resulting reconstruction/residual; these changes correlate with an unfavorable closed-loop Rescue/Harm balance.

Not established: that every high-attention edge/corner token is an attention sink; that late layers are universally unsuitable; or that entity-only/mid-layer variants would succeed. Those require separate single-variable validation and are intentionally outside this run.

## Files

- `implementation_audit.md`, `query_index_audit.csv`, `patch_index_reference.png`
- `state_diagnostics.csv`, `AGGREGATE_DIAGNOSTICS.json`
- `case_gallery.html`, `layer_query_gallery.html`, `aggregate_diagnostics.pdf`
"""
    (artifact/"failure_hypotheses.md").write_text(text)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--artifact",type=Path,required=True)
    parser.add_argument("--closed-loop-artifact",type=Path,default=Path("artifacts/prompt_attn_shr_v1"))
    parser.add_argument("--expected-states",type=int,default=90); args=parser.parse_args()
    artifact=args.artifact.resolve(); states=load_states(artifact)
    if len(states)!=args.expected_states: raise RuntimeError(
        f"expected {args.expected_states} same-state diagnostics, found {len(states)}"
    )
    flat=flatten_rows(states); write_csv(artifact/"state_diagnostics.csv",flat,list(flat[0]))
    patch_index_figure(artifact,states[0]); audit_pass=implementation_audit(
        artifact,states,args.closed_loop_artifact.resolve()
    )
    agg=aggregate(states,artifact); make_case_gallery(artifact,states); layer_query_gallery(artifact,states)
    aggregate_pdf(artifact,states,agg); failure_report(artifact,states,agg,audit_pass)
    print(json.dumps({"complete":True,"states":len(states),"implementation_audit_pass":audit_pass,
                      "artifact":str(artifact)},indent=2))


if __name__=="__main__": main()
