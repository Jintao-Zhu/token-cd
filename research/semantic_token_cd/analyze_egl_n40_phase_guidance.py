#!/usr/bin/env python3
"""Independent phase-stratified action-value check on fresh EGL LIBERO N=40."""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/libero_action_value_egl_gpu6_n40_20260925"
OUT_JSON = ROOT / "PHASE_GUIDANCE_VALIDATION.json"
OUT_MD = ROOT / "PHASE_GUIDANCE_VALIDATION.md"


def exact_auc_p(rescue: list[float], harm: list[float]):
    if not rescue or not harm:
        return None, None
    auc = sum((r > h) + 0.5 * (r == h) for r in rescue for h in harm) / (len(rescue) * len(harm))
    values = np.asarray(rescue + harm, dtype=float)
    n_rescue = len(rescue)
    permuted = []
    for chosen in itertools.combinations(range(len(values)), n_rescue):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(chosen)] = True
        r, h = values[mask], values[~mask]
        permuted.append(sum((a > b) + 0.5 * (a == b) for a in r for b in h) / (len(r) * len(h)))
    p = sum(abs(x - 0.5) >= abs(auc - 0.5) - 1e-12 for x in permuted) / len(permuted)
    return float(auc), float(p)


def main():
    audit = json.loads((ROOT / "FINAL_AUDIT.json").read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("Refusing analysis: N=40 integrity audit is not PASS")
    pairs = [json.loads(p.read_text()) for p in sorted((ROOT / "pairs").glob("*.json"))]
    rows = []
    for pair in pairs:
        cid = pair["case_id"]
        ep_path = ROOT / "episodes" / cid / "matched" / "episode.json"
        ep = json.loads(ep_path.read_text())
        if ep["initial_state_sha256"] != pair["initial_state_sha256"]:
            raise RuntimeError(f"initial-state audit mismatch: {cid}")
        arrays = np.load(ep_path.parent / "step_arrays.npz")
        pos = arrays["positive_action_logits"].astype(np.float64)
        neg = arrays["negative_action_logits"].astype(np.float64)
        phase_values = defaultdict(list)
        for t, trace in enumerate(ep["trace"]):
            phase = str(trace["task_phase"])
            for q in range(min(6, pos.shape[1])):
                order = np.argsort(pos[t, q])[::-1]
                a1, a2 = int(order[0]), int(order[1])
                g = (pos[t, q, a1] - neg[t, q, a1]) - (pos[t, q, a2] - neg[t, q, a2])
                phase_values[phase].append(float(g))
        rows.append({
            "case_id": cid,
            "task_id": int(pair["task_id"]),
            "outcome": pair["outcome"],
            "phase_median_G": {k: float(np.median(v)) for k, v in phase_values.items() if v},
        })

    phases = sorted({p for row in rows for p in row["phase_median_G"]})
    summary = {}
    for phase in phases:
        values = {
            o: [r["phase_median_G"][phase] for r in rows
                if r["outcome"] == o and phase in r["phase_median_G"]]
            for o in ("rescue", "harm", "concordant")
        }
        auc, p = exact_auc_p(values["rescue"], values["harm"])
        summary[phase] = {
            "n_rescue": len(values["rescue"]), "n_harm": len(values["harm"]),
            "n_concordant": len(values["concordant"]),
            "rescue_median_of_episode_medians": float(np.median(values["rescue"])) if values["rescue"] else None,
            "harm_median_of_episode_medians": float(np.median(values["harm"])) if values["harm"] else None,
            "rescue_episode_values": values["rescue"], "harm_episode_values": values["harm"],
            "rescue_higher_auc": auc, "two_sided_exact_permutation_p": p,
        }
    result = {
        "protocol": "FRESH_EGL_N40_PHASE_GUIDANCE_VALIDATION_V1",
        "n_pairs": len(rows),
        "outcome_counts": dict(Counter(r["outcome"] for r in rows)),
        "definition": "For each episode and phase, median across steps and first-six action dims of G=(z+-z-)[clean winner]-(z+-z-)[clean runner-up]; episode is unit.",
        "phase_summary": summary,
        "cases": rows,
        "limits": [
            "Only 5 Rescue and 3 Harm episodes in the full N=40 cohort; phase-specific counts are smaller.",
            "This is a validation of the phase interaction suggested by separate N=50 data, not a causal intervention test.",
            "Phase availability is post-treatment selected; task mix and renderer differ from the N=50 cohort.",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Fresh EGL N=40 phase-guidance check",
        "",
        "Independent cohort relative to the unseen-task N=50 diagnostic. Audited N=40: 40 pairs, 5 Rescue, 3 Harm. The episode is the analysis unit; for each episode/phase, G is summarized as a median across time and the first six action dimensions.",
        "",
        "| Phase | Rescue n / median G | Harm n / median G | Rescue-higher AUC | Exact two-sided p |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase, s in summary.items():
        r = "—" if s["rescue_median_of_episode_medians"] is None else f"{s['rescue_median_of_episode_medians']:.3f}"
        h = "—" if s["harm_median_of_episode_medians"] is None else f"{s['harm_median_of_episode_medians']:.3f}"
        auc = "—" if s["rescue_higher_auc"] is None else f"{s['rescue_higher_auc']:.3f}"
        p = "—" if s["two_sided_exact_permutation_p"] is None else f"{s['two_sided_exact_permutation_p']:.3f}"
        lines.append(f"| {phase} | {s['n_rescue']} / {r} | {s['n_harm']} / {h} | {auc} | {p} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The N=50 approach pattern does not replicate clearly here: approach AUC is .533 (5/3 episodes, p=1.000). Grasp and transport point in the opposite direction from the N=50 pattern (AUC .700 and .800), but with only 2 Harm episodes reaching each phase and exact p=.571/.286. This independent cohort therefore does not support a stable phase-by-G interaction; it also cannot rule one out at this sample size.",
        "",
        "**Hypothesis update:** the apparent phase reversal in N=50 may be sampling noise or task-cohort dependence. **Decision: REFINE, low confidence.** Do not implement a phase gate from these data. A confirmatory phase interaction needs more independently sampled discordant episodes and task-stratified replication; no closed-loop phase gate is justified yet.",
        "",
        "The early/late renderer ablation remains a separate negative result against generic temporal gating. This phase diagnostic does not establish the SIMPLER/LIBERO causal bottleneck.",
        "",
        "Reproducer: [analysis script](../../research/semantic_token_cd/analyze_egl_n40_phase_guidance.py).",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_pairs": len(rows), "outcomes": result["outcome_counts"], "phase_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
