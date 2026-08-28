"""Analysis for GENERIC_SUBSPACE_PROBE_V1.

Answers the reviewer's core question:

    The ~93% of r_sem that is orthogonal to the single `uniform` basis — is it
    really semantic-specific, or just other generic perturbation directions?

The generic residual set is R_G = [r_1, ..., r_M] where r_i = pos - masked_i
(raw logit differences, 256-token action vocabulary). We ask whether r_sem
lives in the subspace spanned by the generic residuals.

Three controls plus a null, and K selected ONLY from generic residuals (never
from semantic), per the reviewer's protocol:

  [A] Does generic form a low-dim subspace?
        Basis built from "train" masks; held-out generic residuals projected
        onto it. r^2 (fraction of squared norm explained) at each K. K is the
        smallest K with held-out generic r^2 >= 0.80.

  [B] How much of r_sem survives the generic subspace?
        rho_sem(K) = ||(I - P_gen^K) r_sem|| / ||r_sem||   (NORM ratio, matching
        the existing orthogonal_cd.py `ortho_ratio`). Compared against:
          - the single-basis anchor: projecting r_sem onto only the uniform
            residual (lattice_phase_00) reproduces the known ~0.92-0.94.
          - the K-dim generic subspace (drop tells how much "unique semantic"
            was in fact generic).

  [C] Random isotropic null:
        Same rank K, same metric, but a random orthonormal subspace. If the
        generic subspace removes r_sem no better than a random K-dim subspace,
        then any rho_sem drop with K is just dimensionality, not structure.

The subspace is built on CENTERED generic residuals (centered by the train mean),
so it captures the *diversity* of generic perturbation directions rather than the
trivially-shared `pos` component. A secondary raw (uncentered) span is also
reported for completeness.

All residuals are raw logit differences. D = N * 7 * 256 (states x action tokens
x action-vocabulary bins), stacked into one vector per mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# r^2 threshold on held-out generic residuals used to pick K (generic-only).
HELDOUT_R2_TARGET = 0.80
# Number of random-subspace draws for the Control C null.
N_NULL = 300
# Index of the uniform (strided_2x2) generic mask in the fixed mask set.
UNIFORM_LABEL = "lattice_phase_00"


def load_residuals(path: Path) -> dict:
    z = np.load(path)
    return {
        "positive": z["positive"].astype(np.float64),   # [N, 7, 256]
        "generic": z["generic"].astype(np.float64),     # [M, N, 7, 256]
        "semantic": z["semantic"].astype(np.float64),   # [N, 7, 256]
        "mask_labels": [str(s) for s in z["mask_labels"]],
    }


def generic_residuals(d: dict) -> np.ndarray:
    """r_i = pos - masked_i, stacked as [M, N, 7, 256]."""
    return d["positive"][None] - d["generic"]           # [M, N, 7, 256]


def sem_residual(d: dict) -> np.ndarray:
    return d["positive"] - d["semantic"]                # [N, 7, 256]


def flatten(mat: np.ndarray) -> np.ndarray:
    """Collapse trailing [N,7,256] dims into one vector per leading row."""
    return mat.reshape(mat.shape[0], -1)


def pca_basis(centered: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of a centered [rows, D] matrix -> [k, D]."""
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh[:k]


def explained_variance_fraction(basis: np.ndarray, vecs: np.ndarray) -> float:
    """Fraction of squared norm of `vecs` rows captured by `basis` rows (r^2)."""
    proj = vecs @ basis.T @ basis
    num = float(np.sum(proj * proj))
    den = float(np.sum(vecs * vecs))
    return num / den if den > 0 else 0.0


def residual_norm_ratio(basis: np.ndarray, vecs: np.ndarray) -> float:
    """||(I - P) v|| / ||v|| (norm ratio, matching orthogonal_cd.ortho_ratio)."""
    proj = vecs @ basis.T @ basis
    resid_norm_sq = float(np.sum((vecs - proj) ** 2))
    total_norm_sq = float(np.sum(vecs ** 2))
    return float(np.sqrt(resid_norm_sq / total_norm_sq)) if total_norm_sq > 0 else 0.0


def per_token_ortho_ratio(r_geom: np.ndarray, r_sem: np.ndarray, eps: float = 1e-8) -> float:
    """Reproduce orthogonal_cd.py's `ortho_ratio` (per action token, then mean).

    Args are [N, 7, 256] residual arrays. r_geom is the single uniform residual
    direction; r_sem the semantic residual. Matches compute_orthogonal_dual_logits.
    """
    dot = np.sum(r_sem * r_geom, axis=-1, keepdims=True)
    geom_norm_sq = np.sum(r_geom * r_geom, axis=-1, keepdims=True)
    coef = dot / (geom_norm_sq + eps)
    r_sem_ortho = r_sem - coef * r_geom
    ortho_norm = np.linalg.norm(r_sem_ortho, axis=-1)
    sem_norm = np.linalg.norm(r_sem, axis=-1)
    ratio = ortho_norm / (sem_norm + eps)
    return float(ratio.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-train-random", type=int, default=0,
                    help="number of random masks folded into the train split "
                         "(default 0: train = 4 deterministic lattice phases only)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = load_residuals(args.data)
    labels = d["mask_labels"]
    r_all = generic_residuals(d)                        # [M, N, 7, 256]
    r_sem = sem_residual(d)                             # [N, 7, 256]
    M, N, _, _ = r_all.shape
    D = N * 7 * 256

    lattice_idx = [i for i, l in enumerate(labels) if l.startswith("lattice")]
    random_idx = [i for i, l in enumerate(labels) if l.startswith("random")]
    uniform_idx = labels.index(UNIFORM_LABEL)

    train_idx = lattice_idx + random_idx[: args.n_train_random]
    held_idx = [i for i in random_idx if i not in train_idx]
    held_idx = held_idx or random_idx[-1:]             # guard against empty

    # ---------------- Anchor: single-basis ortho_ratio (reproduce ~0.93) ----
    r_geom = r_all[uniform_idx]                         # [N, 7, 256]
    anchor_ortho_ratio = per_token_ortho_ratio(r_geom, r_sem)

    # ---------------- Subspace construction on centered generic residuals ----
    R_flat = flatten(r_all)                             # [M, D]
    if len(train_idx) < 2 or len(held_idx) < 1:
        raise RuntimeError(
            f"Need >=2 train masks and >=1 held-out mask, got "
            f"{len(train_idx)} train / {len(held_idx)} held-out"
        )
    train_mean = R_flat[train_idx].mean(axis=0, keepdims=True)
    train_centered = R_flat[train_idx] - train_mean     # [T, D]
    held_centered = R_flat[held_idx] - train_mean       # [H, D] (train-centered)
    sem_flat = r_sem.reshape(1, -1)                     # [1, D]
    sem_centered = sem_flat - train_mean                # [1, D] (train-centered)

    # ---- Control A: held-out generic r^2 vs K; K chosen from generic only ----
    k_max = min(len(train_idx), len(held_idx))
    a_curve = {}
    K_sel = 1
    for k in range(1, k_max + 1):
        basis = pca_basis(train_centered, k)
        a_curve[k] = explained_variance_fraction(basis, held_centered)
    for k in sorted(a_curve):
        if a_curve[k] >= HELDOUT_R2_TARGET:
            K_sel = k
            break
    else:
        K_sel = max(1, k_max)

    # ---- Control B: rho_sem under K-dim generic subspace (centered) ----
    basis_sel = pca_basis(train_centered, K_sel)
    rho_sem_centered = residual_norm_ratio(basis_sel, sem_centered)
    r2_sem_centered = explained_variance_fraction(basis_sel, sem_centered)

    # Secondary: raw (uncentered) span of all generic residuals.
    basis_raw = pca_basis(R_flat[train_idx], K_sel)
    rho_sem_raw = residual_norm_ratio(basis_raw, sem_flat)

    # rho_sem curve over K (centered), reported not used to pick K.
    rho_curve = {}
    for k in range(1, k_max + 1):
        rho_curve[k] = residual_norm_ratio(
            pca_basis(train_centered, k), sem_centered
        )

    # ---- Control C: random isotropic K-dim subspace null ----
    rng = np.random.RandomState(0)
    rho_rand = []
    for _ in range(N_NULL):
        Q = rng.randn(K_sel, D)
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        Q, _ = np.linalg.qr(Q.T)
        Q = Q.T[:K_sel]
        rho_rand.append(residual_norm_ratio(Q, sem_centered))
    rho_rand_mean = float(np.mean(rho_rand))
    rho_rand_lo = float(np.percentile(rho_rand, 2.5))
    rho_rand_hi = float(np.percentile(rho_rand, 97.5))
    # Analytic expectation for a random K-dim subspace: E[rho^2] = 1 - K/D.
    rho_rand_expect = float(np.sqrt(max(0.0, 1.0 - K_sel / D)))

    report = {
        "protocol_id": "GENERIC_SUBSPACE_PROBE_V1",
        "n_states": N,
        "n_generic_masks": M,
        "lattice_masks": [labels[i] for i in lattice_idx],
        "train_masks": [labels[i] for i in train_idx],
        "held_out_masks": [labels[i] for i in held_idx],
        "dimension_D": D,
        "centering": "generic residuals centered by train mean (diversity subspace)",
        "anchor": {
            "uniform_basis": UNIFORM_LABEL,
            "single_basis_ortho_ratio": anchor_ortho_ratio,
            "note": "should reproduce orthogonal_cd ortho_ratio ~0.92-0.94",
        },
        "control_A_heldout_generic_r2_by_K": a_curve,
        "K_selected_by_generic_only": K_sel,
        "K_selection_rule": f"min K with held-out generic r^2 >= {HELDOUT_R2_TARGET}",
        "control_B_rho_sem": {
            "centered_generic_K": rho_sem_centered,
            "centered_generic_r2": r2_sem_centered,
            "raw_span_K": rho_sem_raw,
            "drop_vs_single_basis": anchor_ortho_ratio - rho_sem_centered,
        },
        "control_B_rho_sem_curve_by_K": rho_curve,
        "control_C_random_subspace_null": {
            "K": K_sel,
            "rho_sem_random_mean": rho_rand_mean,
            "rho_sem_random_ci95": [rho_rand_lo, rho_rand_hi],
            "rho_sem_random_expectation": rho_rand_expect,
            "generic_advantage_over_random": rho_rand_mean - rho_sem_centered,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
