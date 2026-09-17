"""Deeper cross-tab: how do P50 and Prompt-CD partition the outcome space?"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
FORMAL = REPO / "artifacts/vla_pruner_openvla_reproduction/formal_1200"
CD_SRC = {"google_robot_open_drawer": "closed_loop", "google_robot_pick_coke_can": "closed_loop",
          "google_robot_move_near": "closed_loop", "google_robot_close_drawer": "closed_loop_remaining6"}
CD_ROOT = REPO / "artifacts/prompt_attn_layer_selection_v1"
TASKS = ["google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near"]

F = {}
for line in (FORMAL / "episode_results.csv").read_text().splitlines()[1:]:
    f = line.split(",")
    F[(f[0], int(f[1]), f[2])] = f[3] == "True"
C = {}
Mt = {}
for task in TASKS:
    for p in sorted((CD_ROOT / CD_SRC[task] / "episodes" / task / "prompt_single").glob("episode_*_summary.json")):
        s = json.loads(p.read_text())
        seed = int(s.get("evaluation_seed", s.get("seed")))
        C[(task, seed)] = bool(s["success"])
        Mt[(task, seed)] = s.get("mean_m_t")

def cells(task=None):
    rows = []
    for t in (TASKS if task is None else [task]):
        for s in range(100):
            rows.append((t, s, F[(t, s, 'vanilla')], F[(t, s, 'vla_pruner_prune50')], C[(t, s)]))
    return rows

R = cells()
print("=" * 96)
print("FULL 3-WAY OUTCOME TABLE  (A vanilla, B P50, C Prompt-CD), 400 episodes")
print("=" * 96)
print(f"{'A':>5s} {'B':>5s} {'C':>5s}  {'n':>4s}   interpretation")
import itertools
for a, b, c in itertools.product([False, True], repeat=3):
    n = sum(1 for r in R if (r[2], r[3], r[4]) == (a, b, c))
    tag = []
    if not a and b and not c: tag.append("P50-only rescue")
    if not a and c and not b: tag.append("CD-only rescue")
    if not a and b and c: tag.append("BOTH rescue")
    if a and not b and not c: tag.append("both harm")
    if a and not b and c: tag.append("P50-only harm")
    if a and b and not c: tag.append("CD-only harm")
    if not a and not b and not c: tag.append("nobody rescues")
    if a and b and c: tag.append("all succeed")
    print(f"{str(a):>5s} {str(b):>5s} {str(c):>5s}  {n:>4d}   {', '.join(tag)}")

print()
print("On the 242 van-fail episodes:")
sub = [r for r in R if not r[2]]
tab = np.array([[sum(1 for r in sub if r[3] and r[4]), sum(1 for r in sub if r[3] and not r[4])],
                [sum(1 for r in sub if not r[3] and r[4]), sum(1 for r in sub if not r[3] and not r[4])]])
print("        CD+   CD-")
print(f"P50+  {tab[0,0]:4d}  {tab[0,1]:4d}")
print(f"P50-  {tab[1,0]:4d}  {tab[1,1]:4d}")
phi = stats.fisher_exact(tab)
print(f"odds ratio={phi[0]:.3f}  fisher p={phi[1]:.4g}")
try:
    chi2 = stats.chi2_contingency(tab, correction=False)
    print(f"chi2={chi2[0]:.2f} p={chi2[1]:.4g}  phi={np.sqrt(chi2[0]/tab.sum()):.3f}")
except Exception as e:
    print(e)

print()
print("On the 158 van-success episodes (harm side):")
sub2 = [r for r in R if r[2]]
tab2 = np.array([[sum(1 for r in sub2 if not r[3] and not r[4]), sum(1 for r in sub2 if not r[3] and r[4])],
                 [sum(1 for r in sub2 if r[3] and not r[4]), sum(1 for r in sub2 if r[3] and r[4])]])
print("        CD-   CD+")
print(f"P50-  {tab2[0,0]:4d}  {tab2[0,1]:4d}")
print(f"P50+  {tab2[1,0]:4d}  {tab2[1,1]:4d}")
print("fisher p =", stats.fisher_exact(tab2)[1])

print()
print("=" * 96)
print("PER-TASK: disjointness of rescue sets")
print("=" * 96)
print(f"{'task':32s} {'vanfail':>7s} {'Rp50':>5s} {'Rcd':>5s} {'both':>5s} {'overlap?':>9s}")
for t in TASKS + ["ALL"]:
    sub = [r for r in cells(None if t == "ALL" else t) if not r[2]]
    rp = sum(1 for r in sub if r[3]); rc = sum(1 for r in sub if r[4]); bo = sum(1 for r in sub if r[3] and r[4])
    print(f"{t:32s} {len(sub):>7d} {rp:>5d} {rc:>5d} {bo:>5d} {str(bo == 0):>9s}")

print()
print("Hypothetical union/intersection if B and C were combined with an ORACLE selector:")
sub = [r for r in R if not r[2]]
print(f"  vanilla-fail: {len(sub)}")
print(f"  rescued by P50 alone {sum(1 for r in sub if r[3] and not r[4])}"
      f" + by CD alone {sum(1 for r in sub if r[4] and not r[3])}"
      f" + by both {sum(1 for r in sub if r[3] and r[4])}")
print(f"  => union ceiling on van-fail = {sum(1 for r in sub if r[3] or r[4])}/{len(sub)}")
harm = [r for r in R if r[2]]
print(f"  vanilla-success: {len(harm)}; P50 breaks {sum(1 for r in harm if not r[3])},"
      f" CD breaks {sum(1 for r in harm if not r[4])},"
      f" both break {sum(1 for r in harm if not r[3] and not r[4])}")
print()
mt = np.array([Mt[(t, s)] for t in TASKS for s in range(100)])
print(f"Prompt-CD mean_m_t: overall {mt.mean():.1f}  min {mt.min():.0f}  max {mt.max():.0f}")
for t in TASKS:
    v = np.array([Mt[(t, s)] for s in range(100)])
    print(f"  {t:32s} mean_m_t={v.mean():6.1f}  (P50 keep=128 => G\\K expected large)")
