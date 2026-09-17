"""Reuse verification and *valid stratified* three-arm overlap audit.

A = current-harness Vanilla   (formal_1200, seeds 0-99)
B = VLA-Pruner P50            (formal_1200, seeds 0-99)
C = Prompt-CD baseline        (prompt_attn_layer_selection_v1 prompt_single, seeds 0-99)

Important: a Prompt-CD Harm (A=1,C=0) and a P50 Rescue (A=0,B=1)
are structurally disjoint because they condition on opposite values of A.
Their intersection and an unstratified Fisher test are therefore invalid.
Association between B and C is evaluated separately within A=0 and A=1.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
FORMAL = REPO / "artifacts/vla_pruner_openvla_reproduction/formal_1200"
CD_SRC = {
    "google_robot_open_drawer": "closed_loop",
    "google_robot_pick_coke_can": "closed_loop",
    "google_robot_move_near": "closed_loop",
    "google_robot_close_drawer": "closed_loop_remaining6",
}
CD_ROOT = REPO / "artifacts/prompt_attn_layer_selection_v1"
TASKS = ["google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near"]


def load_formal():
    rows = {}
    for line in (FORMAL / "episode_results.csv").read_text().splitlines()[1:]:
        f = line.split(",")
        if not f:
            continue
        rows[(f[0], int(f[1]), f[2])] = {"success": f[3] == "True"}
    return rows


def load_cd():
    out, hashes = {}, {}
    for task in TASKS:
        d = CD_ROOT / CD_SRC[task] / "episodes" / task / "prompt_single"
        for p in sorted(d.glob("episode_*_summary.json")):
            s = json.loads(p.read_text())
            seed = int(s.get("evaluation_seed", s.get("seed")))
            out[(task, seed)] = {
                "success": bool(s["success"]),
                "technical_pass": bool(s.get("technical_pass", True)),
                "m_t": s.get("mean_m_t"),
                "state_sha": s.get("initial_state_sha256"),
                "rgb_sha": s.get("initial_rgb_sha256"),
                "canon_sha": s.get("canonical_snapshot_sha256"),
                "failure_reason": s.get("failure_reason"),
            }
    return out


def formal_hashes():
    h = {}
    for task in TASKS:
        for p in sorted((FORMAL / "episodes" / task / "vanilla").glob("episode_*_summary.json")):
            s = json.loads(p.read_text())
            h[(task, int(s["seed"]))] = (s["initial_state_sha256"], s["initial_rgb_sha256"],
                                         s["canonical_snapshot_sha256"])
    return h


def main():
    F = load_formal()
    C = load_cd()
    H = formal_hashes()

    print("=" * 78)
    print("A. REUSE ELIGIBILITY (C vs current-harness vanilla)")
    print("=" * 78)
    n_match = n_mismatch = n_missing = 0
    mism = []
    for (task, seed), c in C.items():
        fh = H.get((task, seed))
        if fh is None:
            n_missing += 1; continue
        if (c["state_sha"], c["rgb_sha"]) == (fh[0], fh[1]):
            n_match += 1
        else:
            n_mismatch += 1; mism.append((task, seed))
    print(f"C episodes: {len(C)}   state+rgb hash identical to vanilla: {n_match}"
          f"   mismatch: {n_mismatch}   missing vanilla: {n_missing}")
    if mism:
        print("  mismatches:", mism[:10])
    print(f"  canonical_snapshot_sha256 unique: {len({v['canon_sha'] for v in C.values()})}"
          f"   vanilla canonical unique: {len({v[2] for v in H.values()})}")
    print(f"  technical_pass: {sum(v['technical_pass'] for v in C.values())}/{len(C)}")

    print()
    print("=" * 78)
    print("B. SUCCESS RATES (seeds 0-99)")
    print("=" * 78)
    print(f"{'task':32s} {'A vanilla':>10s} {'B P50':>8s} {'C PromptCD':>11s}")
    tot = defaultdict(int)
    for task in TASKS:
        a = sum(F[(task, s, 'vanilla')]['success'] for s in range(100))
        b = sum(F[(task, s, 'vla_pruner_prune50')]['success'] for s in range(100))
        c = sum(C[(task, s)]['success'] for s in range(100))
        tot['a'] += a; tot['b'] += b; tot['c'] += c
        print(f"{task:32s} {a:5d}/100 {b:4d}/100 {c:6d}/100  ({c}%)")
    print(f"{'OVERALL':32s} {tot['a']:5d}/400 {tot['b']:4d}/400 {tot['c']:6d}/400"
          f"  A={tot['a']/4:.1f}% B={tot['b']/4:.1f}% C={tot['c']/4:.1f}%")

    print()
    print("=" * 78)
    print("C. VALID STRATIFIED ASSOCIATION AUDIT")
    print("=" * 78)
    rows = []
    for t in TASKS:
        for s in range(100):
            rows.append((F[(t, s, 'vanilla')]['success'],
                         F[(t, s, 'vla_pruner_prune50')]['success'],
                         C[(t, s)]['success']))

    for label, baseline_value in (("vanilla fail (rescue side)", False),
                                  ("vanilla success (harm side)", True)):
        sub = [(b, c) for a, b, c in rows if a is baseline_value]
        table = np.asarray([
            [sum(b and c for b, c in sub), sum(b and not c for b, c in sub)],
            [sum((not b) and c for b, c in sub), sum((not b) and (not c) for b, c in sub)],
        ], dtype=np.int64)
        odds, p = stats.fisher_exact(table)
        print(f"{label}: n={len(sub)}")
        print("             CD+  CD-")
        print(f"  P50+      {table[0,0]:3d}  {table[0,1]:3d}")
        print(f"  P50-      {table[1,0]:3d}  {table[1,1]:3d}")
        print(f"  OR={odds:.3f}  Fisher p={p:.6g}")

    print("\nInvalid comparison intentionally omitted: "
          "H_prompt (A=1,C=0) versus R_p50 (A=0,B=1).")


if __name__ == "__main__":
    main()
