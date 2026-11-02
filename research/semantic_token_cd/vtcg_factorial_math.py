"""Pure NumPy math for the new VTCG factorial protocol; no model/GPU imports.

KMeans full ranking is NEW: it is not the legacy entity-winner cluster union.
TopP always uses normalized attention mass, independent of selection ranking.
Noise matching is in supplied feature space, before any model dtype cast.
"""
from __future__ import annotations

import hashlib
import json
import numbers

import numpy as np

N_VISUAL = 256


def _matrix(value, name, rows=None):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if rows is not None and out.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def validate_count(count):
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, numbers.Integral):
        raise ValueError("count must be an integer")
    if not 1 <= int(count) <= N_VISUAL:
        raise ValueError("count must be in [1,256]")
    return int(count)


def _indices(indices, *, permutation=False):
    out = np.asarray(indices)
    if out.ndim != 1 or not np.issubdtype(out.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    if len(out) == 0 or len(np.unique(out)) != len(out):
        raise ValueError("indices must be nonempty and unique")
    if (out < 0).any() or (out >= N_VISUAL).any():
        raise ValueError("indices out of visual grid")
    if permutation and len(out) != N_VISUAL:
        raise ValueError("ranking must be a 256-token permutation")
    return out.astype(np.int64, copy=False)


def validate_ranking(ranking):
    return _indices(ranking, permutation=True)


def fixed_count(count):
    return validate_count(count)


def matched_count(entity_union):
    return validate_count(len(_indices(entity_union)))


def attention_ranking(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (N_VISUAL,) or not np.isfinite(scores).all():
        raise ValueError("attention scores must be finite [256]")
    if (scores < 0).any() or scores.sum() <= 0:
        raise ValueError("attention scores must be nonnegative with positive mass")
    return np.lexsort((np.arange(N_VISUAL), -scores))


def top_p_count(attention_scores, p, min_count=1, max_count=256):
    """Smallest k reaching p of visual-normalized attention, then explicit clip."""
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, numbers.Real) or not 0 < p < 1:
        raise ValueError("p must be a real number strictly between 0 and 1")
    lo, hi = validate_count(min_count), validate_count(max_count)
    if lo > hi:
        raise ValueError("minimum exceeds maximum")
    order = attention_ranking(attention_scores)
    probability = np.asarray(attention_scores, dtype=np.float64)
    probability = probability / probability.sum()
    raw = int(np.searchsorted(np.cumsum(probability[order]), p, side="left") + 1)
    if not 1 <= raw <= N_VISUAL:
        raise ValueError("invalid cumulative attention count")
    return int(np.clip(raw, lo, hi))


def select_top_count(ranking, count):
    """Return spatially sorted mask; never confuse this with the full ranking."""
    return np.sort(validate_ranking(ranking)[:validate_count(count)])


def _unit_rows(matrix, name):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError(f"{name} contains zero/invalid norm vectors")
    return matrix / norm


def kmeans_full_ranking(features, labels, entity_embeddings):
    """Rank clusters by max entity/centroid cosine, then members by cosine.

    Cluster order: descending max entity cosine; smallest member token ID breaks
    ties (label-permutation invariant). Members: descending cosine to centroid,
    ascending token ID on ties. Returns a complete [256] int64 permutation.
    Centroids are computed from the supplied clean features, not normalized
    token features. KMeans fitting and its K/seed/n_init are caller-owned.
    """
    features = _matrix(features, "features", N_VISUAL)
    entities = _matrix(entity_embeddings, "entity embeddings")
    if entities.shape[1] != features.shape[1]:
        raise ValueError("entity and feature dimensions differ")
    labels = np.asarray(labels)
    if labels.shape != (N_VISUAL,) or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError("labels must be nonnegative integers [256]")
    groups = np.unique(labels)
    members = [np.flatnonzero(labels == group) for group in groups]
    centroids = np.stack([features[ids].mean(axis=0) for ids in members])
    cn = _unit_rows(centroids, "centroids")
    en = _unit_rows(entities, "entity embeddings")
    fn = _unit_rows(features, "features")
    group_scores = (cn @ en.T).max(axis=1)
    group_order = np.lexsort((np.array([ids[0] for ids in members]), -group_scores))
    chunks = []
    for group in group_order:
        ids = members[group]
        cosine = fn[ids] @ cn[group]
        chunks.append(ids[np.lexsort((ids, -cosine))])
    return validate_ranking(np.concatenate(chunks))


def deterministic_seed(*parts):
    """Stable seed; caller supplies task/episode/control-step and a purpose salt."""
    if not parts or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (str, numbers.Integral)) for x in parts):
        raise ValueError("seed parts must be strings or integers")
    text = json.dumps([int(x) if isinstance(x, numbers.Integral) else x for x in parts], separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def norm_matched_random_replacement(features, selected, harmonic_replacement, seed):
    """Use same mask, replacing each harmonic displacement by a random direction.

    Returns [m,D] float64 replacements in ascending mask order, plus audit.
    `selected` MUST already be ascending, matching harmonic_reconstruct's rows.
    Norms are matched individually, not by global Frobenius norm. Zero target
    residual produces an exact unchanged token. No orthogonalization is used.
    """
    features = _matrix(features, "features", N_VISUAL)
    selected = _indices(selected)
    if not np.array_equal(selected, np.sort(selected)):
        raise ValueError("selected must be ascending to align reconstruction rows")
    harmonic = _matrix(harmonic_replacement, "harmonic replacement", len(selected))
    if harmonic.shape[1] != features.shape[1]:
        raise ValueError("harmonic and feature dimensions differ")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    residual = harmonic - features[selected]
    target = np.linalg.norm(residual, axis=1)
    if not np.isfinite(target).all():
        raise ValueError("invalid target residual norm")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    direction = _unit_rows(rng.standard_normal(residual.shape), "random directions")
    replacement = features[selected] + direction * target[:, None]
    actual = np.linalg.norm(replacement - features[selected], axis=1)
    if not np.isfinite(replacement).all() or not np.allclose(actual, target, rtol=1e-10, atol=1e-12):
        raise ValueError("per-token residual norm matching failed")
    full = features.copy()
    full[selected] = replacement
    outside = np.ones(N_VISUAL, dtype=bool)
    outside[selected] = False
    audit = {
        "seed": int(seed), "rng": "PCG64", "selected_token_ids": selected.tolist(),
        "target_residual_norms": target.tolist(), "actual_residual_norms": actual.tolist(),
        "max_abs_norm_error": float(np.max(np.abs(target - actual))),
        "non_target_bit_identical": bool(np.array_equal(full[outside], features[outside])),
        "zero_residual_count": int((target == 0).sum()),
        "matching_space": "float64 supplied features before model dtype cast",
    }
    return replacement, audit
