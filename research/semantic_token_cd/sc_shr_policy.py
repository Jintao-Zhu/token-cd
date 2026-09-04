"""Spatially Coherent SHR (SC-SHR): a spatial ablation of SHR.

Only the reconstruction region changes. The K=8 semantic selector and CD are
bit-identical to ``SpatialHarmonicReconCDInference``. After the semantic union
mask ``M_sem`` is built, its 4-neighbor connected components are computed and
only the largest component (single-entity tasks) or the top-2 components
(multi-entity tasks) are kept; everything else stays clean in the negative
branch. Harmonic inpainting then runs on the filtered region.
"""

from __future__ import annotations

import numpy as np

from research.semantic_token_cd.spatial_harmonic_recon_policy import (
    SpatialHarmonicReconCDInference,
    _connected_components,
    _grid_neighbors,
    harmonic_inpaint,
)


def keep_largest_components(token_ids, n_keep: int) -> tuple[np.ndarray, dict]:
    """Return (filtered_ids, meta) keeping the n_keep largest 4-neighbor
    connected components of ``token_ids`` (sorted by size, descending)."""
    ids = np.asarray(sorted(set(int(i) for i in token_ids)), dtype=np.int64)
    components = _connected_components(ids, _grid_neighbors(256))
    components.sort(key=len, reverse=True)
    kept = components[: max(0, n_keep)]
    filtered = (
        np.asarray(sorted(int(i) for c in kept for i in c), dtype=np.int64)
        if kept
        else ids
    )
    meta = {
        "sc_n_components": len(components),
        "sc_component_sizes": [len(c) for c in components],
        "sc_n_kept": len(kept),
        "sc_kept_component_sizes": [len(c) for c in kept],
        "sc_before_tokens": int(ids.size),
        "sc_after_tokens": int(filtered.size),
    }
    if filtered.size == 0:
        raise RuntimeError("SC-SHR: component filter emptied the semantic region")
    return filtered, meta


class ScShrHarmonicReconCDInference(SpatialHarmonicReconCDInference):
    """SHR-CD whose masked region is spatially cleaned before inpainting."""

    def _reconstruct(self, h: np.ndarray, labels: np.ndarray, selected_groups: list[int]):
        union = np.asarray(
            sorted(int(i) for g in selected_groups for i in np.flatnonzero(labels == g)),
            dtype=np.int64,
        )
        n_entities = len(getattr(self, "_entities", None) or [])
        n_keep = 1 if n_entities <= 1 else 2
        filtered, sc_meta = keep_largest_components(union, n_keep)
        h_tilde, diag = harmonic_inpaint(h, filtered)
        diag["recon_finite"] = diag["shr_finite"]
        diag.update(sc_meta)
        if not diag["shr_finite"]:
            raise FloatingPointError("SC-SHR reconstruction produced non-finite features")
        return h_tilde, diag
