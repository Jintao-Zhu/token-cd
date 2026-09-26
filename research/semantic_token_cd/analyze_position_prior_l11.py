#!/usr/bin/env python3
"""Offline position-prior and debiased-L11 diagnostic on existing LIBERO traces.

This analysis reads only stored per-token attention scores, matched m, and
stored canonical top-m IDs. It never reads episode outcomes or launches policy
inference / simulator rollouts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
DEV = REPO / "artifacts/libero_action_value_trace_pilot_v2_osmesa_ext40_init8_15_20260925/episodes"
EVAL = REPO / "artifacts/libero_action_value_egl_gpu6_n40_20260925/episodes"
OUT = REPO / "artifacts/libero_position_prior_offline_v1_20260926"
STATES_PER_EPISODE = 10
NULL_DRAWS = 100
EPS = 1e-12


def episode_rows(root: Path):
    paths = sorted(root.glob("*/matched/episode.json"))
    rows = []
    for p in paths:
        meta = json.loads(p.read_text())
        scores = np.load(p.parent / "step_arrays.npz")["l11_attention_scores"]
        trace = meta["trace"]
        if scores.shape != (len(trace), 256):
            raise RuntimeError(f"score/trace mismatch: {p}")
        steps = sorted(set(np.rint(np.linspace(0, len(trace) - 1, STATES_PER_EPISODE)).astype(int)))
        if len(steps) != STATES_PER_EPISODE:
            raise RuntimeError(f"unexpected sampled state count: {p}")
        for s in steps:
            t = trace[int(s)]
            m = int(t["matched_m"])
            raw = np.asarray(scores[int(s)], dtype=np.float64)
            if not np.isfinite(raw).all() or abs(float(raw.sum()) - 1.0) > 2e-4:
                raise RuntimeError(f"invalid attention distribution: {p} step {s}")
            top = np.asarray(t["attention_top256_order"][:m], dtype=np.int64)
            selected = np.asarray(t["selected_token_ids"], dtype=np.int64)
            if len(top) != m or set(top.tolist()) != set(selected.tolist()):
                raise RuntimeError(f"stored canonical top-m mismatch: {p} step {s}")
            rows.append({"case_id": meta["case_id"], "task_id": int(meta["task_id"]),
                         "step": int(s), "m": m, "scores": raw, "raw_ids": top})
    return paths, rows


def prior_from(rows, task_id=None):
    chosen = [r["scores"] for r in rows if task_id is None or r["task_id"] == task_id]
    if not chosen:
        raise RuntimeError(f"no development rows for task {task_id}")
    mu = np.mean(np.stack(chosen), axis=0)
    mu = np.maximum(mu, EPS)
    mu /= mu.sum()
    return mu


def rng_for(case_id, step, scope):
    h = hashlib.sha256(f"position-null-v1:{scope}:{case_id}:{step}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def sampled_null(rng, prior, m, draws):
    # Probability-proportional-to-position-prior sampling without replacement.
    # Overlap with canonical L11 is intentionally allowed.
    return [rng.choice(256, size=m, replace=False, p=prior).astype(np.int64) for _ in range(draws)]


def analyze_scope(rows, prior_source, scope):
    out = []
    for r in rows:
        mu = prior_source[r["task_id"]] if isinstance(prior_source, dict) else prior_source
        scores, m = r["scores"], r["m"]
        raw_ids = r["raw_ids"]
        residual = scores - mu
        debiased_ids = np.argsort(-residual, kind="stable")[:m]
        position_ids = np.argsort(-mu, kind="stable")[:m]
        nulls = sampled_null(rng_for(r["case_id"], r["step"], scope), mu, m, NULL_DRAWS)
        raw_mass = float(scores[raw_ids].sum())
        raw_excess = float(residual[raw_ids].sum())
        deb_mass = float(scores[debiased_ids].sum())
        deb_excess = float(residual[debiased_ids].sum())
        pos_mass = float(scores[position_ids].sum())
        null_mass = np.asarray([scores[x].sum() for x in nulls], dtype=np.float64)
        null_excess = np.asarray([residual[x].sum() for x in nulls], dtype=np.float64)
        raw_pct = (1 + int(np.count_nonzero(null_mass <= raw_mass))) / (NULL_DRAWS + 1)
        deb_pct = (1 + int(np.count_nonzero(null_excess <= deb_excess))) / (NULL_DRAWS + 1)
        null_overlap = np.asarray([len(set(x.tolist()) & set(raw_ids.tolist())) for x in nulls], dtype=np.float64)
        out.append({
            "case_id": r["case_id"], "task_id": r["task_id"], "step": r["step"], "m": m,
            "raw_vs_position_topm_overlap": int(len(set(raw_ids.tolist()) & set(position_ids.tolist()))),
            "raw_vs_debiased_topm_overlap": int(len(set(raw_ids.tolist()) & set(debiased_ids.tolist()))),
            "raw_attention_mass": raw_mass, "position_topm_attention_mass": pos_mass,
            "debiased_topm_attention_mass": deb_mass,
            "raw_excess_over_position_prior": raw_excess,
            "debiased_excess_over_position_prior": deb_excess,
            "null_attention_mass_mean": float(null_mass.mean()),
            "null_attention_mass_p95": float(np.quantile(null_mass, .95)),
            "raw_attention_mass_null_percentile": float(raw_pct),
            "raw_overlap_with_spatial_null_mean": float(null_overlap.mean()),
            "raw_overlap_with_spatial_null_p95": float(np.quantile(null_overlap, .95)),
            "null_excess_mean": float(null_excess.mean()),
            "debiased_excess_null_percentile": float(deb_pct),
            "null_overlap_with_raw_mean": float(null_overlap.mean()),
        })
    return out


def summarize(rows):
    def vals(k): return np.asarray([r[k] for r in rows], dtype=np.float64)
    def med(k): return float(np.median(vals(k)))
    return {
        "states": len(rows),
        "tasks": sorted({r["task_id"] for r in rows}),
        "median_m": med("m"),
        "median_raw_position_topm_overlap": med("raw_vs_position_topm_overlap"),
        "median_raw_debiased_topm_overlap": med("raw_vs_debiased_topm_overlap"),
        "median_null_raw_overlap": med("raw_overlap_with_spatial_null_mean"),
        "raw_attention_mass_gt_null_p95_fraction": float(np.mean(vals("raw_attention_mass_null_percentile") > .95)),
        "raw_attention_mass_null_percentile_median": med("raw_attention_mass_null_percentile"),
        "debiased_excess_null_percentile_median": med("debiased_excess_null_percentile"),
        "raw_excess_positive_fraction": float(np.mean(vals("raw_excess_over_position_prior") > 0)),
        "debiased_excess_positive_fraction": float(np.mean(vals("debiased_excess_over_position_prior") > 0)),
        "median_raw_attention_mass_minus_null_mean": float(np.median(vals("raw_attention_mass") - vals("null_attention_mass_mean"))),
        "median_debiased_excess_minus_null_mean": float(np.median(vals("debiased_excess_over_position_prior") - vals("null_excess_mean"))),
    }


def main():
    dev_paths, dev = episode_rows(DEV)
    eval_paths, ev = episode_rows(EVAL)
    dev_tasks = sorted({r["task_id"] for r in dev})
    eval_tasks = sorted({r["task_id"] for r in ev})
    if len(dev_paths) != 40 or len(eval_paths) != 40 or dev_tasks != eval_tasks:
        raise RuntimeError(f"unexpected cohorts: dev={len(dev_paths)}/{dev_tasks}, eval={len(eval_paths)}/{eval_tasks}")

    priors = {"global": prior_from(dev)}
    task_priors = {tid: prior_from(dev, tid) for tid in dev_tasks}
    results = {}
    for name, prior_source in [("global", priors["global"]), ("task_specific", task_priors)]:
        rows = analyze_scope(ev, prior_source, name)
        results[name] = {"summary": summarize(rows), "rows": rows,
                         "prior_16x16": ({str(k): v.reshape(16, 16).tolist() for k, v in task_priors.items()}
                                         if isinstance(prior_source, dict) else prior_source.reshape(16, 16).tolist())}
    for tid in dev_tasks:
        rows = [r for r in results["task_specific"]["rows"] if r["task_id"] == tid]
        results[f"task_{tid}"] = {"summary": summarize(rows), "rows": rows,
                                  "prior_16x16": task_priors[tid].reshape(16, 16).tolist()}

    payload = {
        "protocol": "OFFLINE_POSITION_PRIOR_RAW_VS_DEBIASED_L11_V1",
        "created_from_existing_data_only": True,
        "outcomes_read": False,
        "policy_inference_or_rollouts_run": False,
        "development_source": str(DEV),
        "evaluation_source": str(EVAL),
        "development_episodes": len(dev_paths), "development_sampled_states": len(dev),
        "evaluation_episodes": len(eval_paths), "evaluation_sampled_states": len(ev),
        "states_per_episode": STATES_PER_EPISODE,
        "null_draws_per_state": NULL_DRAWS,
        "null_sampling": "probability proportional to frozen development mean attention, without replacement; canonical overlap allowed",
        "position_prior": "mean normalized L11 attention per spatial token over sampled development states; global and task-specific",
        "debiased_score": "current_state_attention - frozen_development_position_prior",
        "caveat": "This is an offline attention-ranking diagnostic, not a behavioral or semantic-ground-truth test. Cohorts are disjoint by init-state IDs but reuse the same five tasks; EGL cohort has been used in earlier analyses.",
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "POSITION_PRIOR_ANALYSIS.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Position Prior and Debiased L11: Offline Diagnostic",
        "",
        "## Scope",
        "",
        f"Used existing OSMesa development traces ({len(dev_paths)} episodes, {len(dev)} sampled states) to estimate frozen mean-attention position priors, then evaluated existing EGL traces ({len(eval_paths)} episodes, {len(ev)} sampled states). No success labels were read; no inference, simulator replay, or new episodes were run.",
        "",
        "The cohorts use disjoint init-state IDs but the same five LIBERO tasks. The EGL cohort is not globally untouched: earlier unrelated offline analyses used it. This is therefore a frozen-prior split for this analysis, not an independent benchmark replication.",
        "",
        "## Results",
        "",
        "Null masks were drawn without replacement with token probabilities proportional to the frozen prior; overlap with canonical L11 was allowed. Position-only Top-m is deterministic. Debiased L11 ranks current attention minus the frozen prior.",
        "",
        "| Prior | median m | median Raw∩Position Top-m | median Raw∩Debiased Top-m | mean spatial-null overlap with Raw |",
        "|---:|---:|---:|---:|---:|",
    ]
    for name in ("global", "task_specific"):
        block = results[name]
        s = block["summary"]
        lines.append(f"| {name} | {s['median_m']:.0f} | {s['median_raw_position_topm_overlap']:.1f}/{s['median_m']:.0f} | {s['median_raw_debiased_topm_overlap']:.1f}/{s['median_m']:.0f} | {s['median_null_raw_overlap']:.1f}/{s['median_m']:.0f} |")
    lines += ["", "### Task-specific prior summaries", "", "| Task ID | States | median m | Raw∩Position Top-m | Raw∩Debiased Top-m | Mean spatial-null overlap with Raw |", "|---:|---:|---:|---:|---:|---:|"]
    for tid in dev_tasks:
        s = results[f"task_{tid}"]["summary"]
        lines.append(f"| {tid} | {s['states']} | {s['median_m']:.0f} | {s['median_raw_position_topm_overlap']:.1f}/{s['median_m']:.0f} | {s['median_raw_debiased_topm_overlap']:.1f}/{s['median_m']:.0f} | {s['median_null_raw_overlap']:.1f}/{s['median_m']:.0f} |")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This only tests whether current-state attention ranking or the simple additive residual ranking departs from an attention-derived spatial prior. It does not establish semantic correctness or behavioral value. The prior explains a meaningful portion of raw Top-m placement, especially with a task-specific prior. Subtracting the frozen prior changes some selected tokens but leaves substantial overlap with raw L11. This is evidence that current-state ranking contains structure beyond this simple position prior; it is not evidence that the extra structure is semantic or behaviorally useful. Raw Top-m attention mass exceeding prior-weighted null draws is mechanically expected because Raw Top-m directly maximizes current attention, so that statistic is retained in JSON for audit but not used as a conclusion. Any behavioral claim requires a later preregistered closed-loop comparison.",
        "",
        "Full per-state results and 16x16 priors are in `POSITION_PRIOR_ANALYSIS.json`.",
        "",
    ]
    # A representative map panel: broad prior versus state-specific raw and
    # prior-subtracted attention for a fixed evaluation state.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        example = next(r for r in ev if r["case_id"] == "task49__init000")
        tid = example["task_id"]
        mu = task_priors[tid]
        raw = example["scores"]
        delta = raw - mu
        m = example["m"]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
        panels = [(priors["global"], "Global position prior", "viridis"),
                  (mu, f"Task {tid} position prior", "viridis"),
                  (raw, f"Raw L11 ({example['case_id']}, m={m})", "viridis"),
                  (delta, "Raw L11 − task prior", "coolwarm")]
        for ax, (values, title, cmap) in zip(axes, panels):
            im = ax.imshow(values.reshape(16, 16), cmap=cmap,
                           vmin=float(values.min()) if cmap == "coolwarm" else None,
                           vmax=float(values.max()) if cmap == "coolwarm" else None)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        fig.savefig(OUT / "POSITION_PRIOR_MAPS.png", dpi=180)
        plt.close(fig)
        lines += ["", f"Example 16×16 maps for `{example['case_id']}` are saved in `POSITION_PRIOR_MAPS.png`."]
    except ImportError:
        lines += ["", "Matplotlib was unavailable; the exact 16×16 prior maps are preserved in the JSON output."]
    lines.append("")
    (OUT / "POSITION_PRIOR_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({k: v["summary"] for k, v in results.items()}, indent=2))
    print(f"report={OUT / 'POSITION_PRIOR_REPORT.md'}")


if __name__ == "__main__":
    main()
