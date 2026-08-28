"""CW-LPCD offline residual metrics (STEP 9 / STEP 11).

Primary cosine: per action position, over the 256 action-vocab bins, then pooled
across all Selection states x 7 positions and summarized by the median. This is
the literal reading of spec STEP 9 ("all Selection states / 7 action positions").
A secondary per-state cosine (7x256 = 1792-dim flattened) is reported for
continuity with the historical stage_a number.
"""
from __future__ import annotations

import numpy as np


def _cos_pos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine of [*, 256] arrays -> [*]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a2 = a.reshape(a.shape[0], -1)
    b2 = b.reshape(b.shape[0], -1)
    dot = np.einsum("ij,ij->i", a2, b2)
    na = np.linalg.norm(a2, axis=1)
    nb = np.linalg.norm(b2, axis=1)
    denom = na * nb
    out = np.full(a2.shape[0], np.nan)
    mask = denom > 0
    out[mask] = dot[mask] / denom[mask]
    return out


def per_position_cosine(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """r_a, r_b [7, 256] -> [7] per-position cosines."""
    return _cos_pos(r_a, r_b)


def state_cosine(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """r_a, r_b [7, 256] -> scalar 1792-dim cosine."""
    return float(_cos_pos(r_a.reshape(1, -1), r_b.reshape(1, -1))[0])


def norm_ratio(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Per-position ||r_a|| / ||r_b|| -> [7]."""
    a = r_a.astype(np.float64)
    b = r_b.astype(np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return na / nb


def projection(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Per-position <r_a, r_b> / ||r_b||^2 -> [7]."""
    a = r_a.astype(np.float64)
    b = r_b.astype(np.float64)
    dot = np.einsum("ij,ij->i", a, b)
    nb2 = np.einsum("ij,ij->i", b, b)
    out = np.full(a.shape[0], np.nan)
    mask = nb2 > 0
    out[mask] = dot[mask] / nb2[mask]
    return out


def sign_agreement(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Per-position fraction of 256 bins with matching sign -> [7]."""
    return np.mean(np.sign(r_a) == np.sign(r_b), axis=1)


def top1_shift_agreement(z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    """Per-position argmax-over-action-vocab agreement -> [7] (0/1)."""
    return (z_a.argmax(axis=1) == z_b.argmax(axis=1)).astype(np.float64)


def _nanmedian_pool(pool: np.ndarray) -> float:
    pool = pool[np.isfinite(pool)]
    return float(np.median(pool)) if pool.size else float("nan")
