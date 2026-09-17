"""Create compact layer/control and frozen-candidate diagnostic galleries."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def bounds(indices):
    rows, cols = zip(*(divmod(int(x), 16) for x in indices))
    return min(cols), min(rows), max(cols) + 1, max(rows) + 1


def add_box(ax, indices, color, label=None):
    x0, y0, x1, y1 = bounds(indices)
    ax.add_patch(patches.Rectangle((x0-.5, y0-.5), x1-x0, y1-y0,
                                   fill=False, edgecolor=color, linewidth=1.0, label=label))


def layer_grids(artifact: Path, row: dict, arrays: dict, outdir: Path):
    control = row["target_control"]
    original = arrays["original"] / np.maximum(arrays["original"].sum(1, keepdims=True), 1e-12)
    target = arrays["target_switch"] / np.maximum(arrays["target_switch"].sum(1, keepdims=True), 1e-12)
    for name, matrix, cmap in (("original", original, "viridis"),
                               ("target_switch_delta", target-original, "coolwarm")):
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        vmax = float(np.max(np.abs(matrix))) if "delta" in name else float(np.percentile(matrix, 99.5))
        for layer, ax in enumerate(axes.flat):
            kwargs = {"vmin": -vmax, "vmax": vmax} if "delta" in name else {"vmin": 0, "vmax": vmax}
            ax.imshow(matrix[layer].reshape(16, 16), cmap=cmap, **kwargs)
            add_box(ax, control["old_target"], "red")
            add_box(ax, control["new_target"], "lime")
            ax.set_title(f"L{layer}", fontsize=8); ax.axis("off")
        fig.suptitle(f"{row['state_id']} — {name}\nred=old target, green=new target\n"
                     f"{row['instruction']}  →  {control['new_instruction']}")
        fig.tight_layout(rect=(0, 0, 1, .93))
        path = outdir / f"{row['state_id']}__{name}.png"
        fig.savefig(path, dpi=150); plt.close(fig)


def mask_overlay(ax, rgb, mask, title):
    ax.imshow(rgb)
    rgba = np.zeros((16,16,4)); rgba[...,0]=1; rgba[...,3]=mask.reshape(16,16)*.42
    ax.imshow(rgba, extent=(0,rgb.shape[1],rgb.shape[0],0), interpolation="nearest")
    ax.set_title(title, fontsize=9); ax.axis("off")


def candidate_figure(artifact: Path, source: Path, row: dict, arrays: dict, output: Path):
    task, seed, step = row["task"], row["seed"], row["source_step"]
    src = source / "states" / task / f"seed_{seed:03d}" / f"step_{step:03d}.npz"
    old = np.load(src)
    candidate_path = artifact / "candidate_state_eval" / task / f"seed_{seed:03d}" / f"step_{step:03d}.npz"
    cand = np.load(candidate_path); rgb = arrays["image"]
    masks = [old["standard_shr__mask"], old["prompt_attn_shr__mask"],
             cand["prompt_single__mask"], cand["prompt_sparse__mask"]]
    names = ["Standard SHR", "Prompt-v1 L16-31", "Prompt-Single L11", "Prompt-Sparse L11+L14"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i, (mask, name) in enumerate(zip(masks, names)):
        mask_overlay(axes[0,i], rgb, mask, f"{name} mask")
    configs = [arrays["original"][16:32].mean(0), arrays["original"][11],
               arrays["original"][[11,14]].mean(0)]
    axes[1,0].imshow(rgb); axes[1,0].set_title("Fixed-state RGB"); axes[1,0].axis("off")
    for i, (score, name) in enumerate(zip(configs, names[1:]), start=1):
        axes[1,i].imshow(score.reshape(16,16), cmap="viridis")
        axes[1,i].set_title(f"{name} score"); axes[1,i].axis("off")
    control = row["target_control"]
    fig.suptitle(f"{row['state_id']}\n{row['instruction']} → {control['new_instruction']}")
    fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(output,dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--artifact",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True); args=parser.parse_args()
    outdir=args.artifact/"layer_control_gallery"; outdir.mkdir(parents=True,exist_ok=True)
    candidate_dir=args.artifact/"candidate_case_gallery"; candidate_dir.mkdir(parents=True,exist_ok=True)
    pages=[]; csv_rows=[]
    for path in sorted((args.artifact/"states").glob("*/*/step_*.json")):
        row=json.loads(path.read_text())
        if not row.get("target_control"): continue
        arrays=dict(np.load(path.with_suffix(".npz")))
        layer_grids(args.artifact,row,arrays,outdir)
        candidate_out=candidate_dir/f"{row['state_id']}.png"
        candidate_figure(args.artifact,args.source,row,arrays,candidate_out)
        pages.append((row,candidate_out))
        old=row["target_control"]["old_target"]; new=row["target_control"]["new_target"]
        for layer in range(32):
            a=arrays["original"][layer]; b=arrays["target_switch"][layer]
            a=a/max(float(a.sum()),1e-12); b=b/max(float(b.sum()),1e-12)
            csv_rows.append({"state_id":row["state_id"],"split":row["split"],"task":row["task"],"layer":layer,
                             "new_target_gain":float(b[new].sum()-a[new].sum()),
                             "old_target_drop":float(a[old].sum()-b[old].sum())})
    with (args.artifact/"target_control_per_layer.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    page=["<!doctype html><meta charset='utf-8'><title>Frozen layer candidate cases</title>",
          "<style>body{font-family:sans-serif;max-width:1500px;margin:auto}img{width:100%}</style>",
          "<h1>Prompt-Attn frozen layer candidate cases</h1>"]
    for row,image in pages:
        page += [f"<h2>{html.escape(row['state_id'])} ({row['split']})</h2>",
                 f"<img src='{image.relative_to(args.artifact).as_posix()}'>"]
    (args.artifact/"candidate_case_gallery.html").write_text("\n".join(page))
    print(json.dumps({"controls":len(pages),"layer_heatmaps":len(pages)*2,"candidate_cases":len(pages)}))


if __name__=="__main__": main()
