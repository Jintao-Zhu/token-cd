"""SEMANTIC_TOKEN_CD Phase-0 offline grouping (no training).

Three grouping methods over the 256 visual patch tokens:
  A_kmeans        K-means on raw projector features h_i (Euclidean)
  B_cosine_agglo  Agglomerative clustering on cosine affinity
  C_attention_spec Spectral clustering on visual self-attention graph

Plus group-vs-object-mask metrics (precision / recall / IoU / compactness).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def group_indices(method: str, h: np.ndarray, A: np.ndarray | None, K: int, seed: int = 0) -> list[list[int]]:
    """Return K groups (list of patch-token index lists) for one method/K.

    h: [256, 4096] projector features. A: [256, 256] attention graph (None unless
    method == 'spectral').
    """
    if method == "kmeans":
        km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(h.astype(np.float64))
        labels = km.labels_
    elif method == "agglomerative":
        ac = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average").fit(_norm(h.astype(np.float64)))
        labels = ac.labels_
    elif method == "spectral":
        assert A is not None
        S = (A + A.T) / 2.0
        sc = SpectralClustering(n_clusters=K, affinity="precomputed", random_state=seed, assign_labels="kmeans").fit(S.astype(np.float64))
        labels = sc.labels_
    else:
        raise ValueError(f"unknown method {method}")
    return [np.flatnonzero(labels == k).astype(int).tolist() for k in range(K)]


def group_overlap(group: list[int], obj_ids: list[int], n_total: int = 256) -> dict:
    """precision / recall / IoU of a group vs the object mask tokens."""
    g = set(int(x) for x in group)
    obj = set(int(x) for x in obj_ids)
    inter = len(g & obj)
    s = len(g)
    n_obj = len(obj)
    precision = inter / s if s else 0.0
    recall = inter / n_obj if n_obj else 0.0
    union = len(g | obj)
    iou = inter / union if union else 0.0
    chance_precision = n_obj / n_total
    chance_iou = (s * chance_precision) / (s + n_obj - s * chance_precision) if (s + n_obj) else 0.0
    return {"precision": precision, "recall": recall, "iou": iou,
            "chance_precision": chance_precision, "chance_iou": chance_iou,
            "size": s, "n_obj": n_obj}


def group_compactness(group: list[int], patch_grid: int = 16) -> float:
    """Mean spatial variance of the patch coordinates in a group (lower = more compact)."""
    if not group:
        return float("nan")
    xs = np.array([p % patch_grid for p in group], dtype=np.float64)
    ys = np.array([p // patch_grid for p in group], dtype=np.float64)
    return float(np.var(xs) + np.var(ys))
