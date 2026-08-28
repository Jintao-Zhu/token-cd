"""ALE-CD SLED-style layer-trajectory scoring (pure numpy, no model).

Pre-registered in artifacts/ar_ale_cd_phase0_v1/CONFIG_LOCK.yaml.

For each action position j (independently), given the full layer trajectory of
256-bin logits z[l, j, c] (l = 0..31), convert to probabilities
p_l = softmax_256(z_l) and score every candidate token c by three components:

  A(c) = mean_{l=0..30} sign(p_{l+1}(c) - p_l(c))          monotonic maturity
  B(c) = mean_{l=24..31} p_l(c) - std_{l=24..31} p_l(c)    late presence+stability
  C(c) = mean_{l=0..23} p_l(c) - p_31(c)                   final-disagreement bonus

Each component is z-scored over the 256 candidates, averaged with equal weight,
then softmaxed into the latent distribution q (sum = 1).
"""
from __future__ import annotations

import numpy as np

N_LAYERS = 32
FINAL = N_LAYERS - 1
LATE_START = 24          # layers 24..31 for late-layer stability
EARLY_MID_END = 24       # layers 0..23 for early/middle dominance


def softmax_256(z):
    """z [..., 256] -> row-softmax over last axis (float64)."""
    z = z.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def zscore(x, axis=-1):
    """z-score along `axis` (population std), with epsilon guard."""
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-8
    return (x - mu) / sd


def trajectory_score_raw(z, layer_perm=None):
    """z [N, 32, 7, 256] logits -> (score [N,7,256], components dict).

    layer_perm (optional): length-32 permutation of layer order applied before
    scoring (used by the random-trajectory control to destroy temporal structure).
    """
    if layer_perm is not None:
        z = z[:, layer_perm]
    z = z.astype(np.float64)
    p = softmax_256(z)                                  # [N,32,7,256]

    delta = p[:, 1:] - p[:, :-1]                        # [N,31,7,256]
    A = np.sign(delta).mean(axis=1)                     # [N,7,256]

    late = p[:, LATE_START:]                            # [N,8,7,256]
    B = late.mean(axis=1) - late.std(axis=1)            # [N,7,256]

    C = p[:, :EARLY_MID_END].mean(axis=1) - p[:, FINAL]  # [N,7,256]

    A_zs = zscore(A)
    B_zs = zscore(B)
    C_zs = zscore(C)
    score = (A_zs + B_zs + C_zs) / 3.0
    return score, {"A": A, "B": B, "C": C, "A_zs": A_zs, "B_zs": B_zs, "C_zs": C_zs}


def trajectory_score(z, layer_perm=None):
    """z [N,32,7,256] -> q [N,7,256] latent distribution (softmax of score)."""
    score, _ = trajectory_score_raw(z, layer_perm=layer_perm)
    return softmax_256(score)
