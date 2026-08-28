"""Selection geometries for the Spatial Grid vs Random Attention-CD study."""

from __future__ import annotations

import numpy as np

from research.semantic_token_cd.attention_mask_intervention import (
    block_semantic_tokens_globally,
    validate_global_token_mask,
)
from research.semantic_token_cd.attention_mask_policy import (
    AttentionMaskEntityCDInference,
)


GRID_SIZE = 16


def get_grid_mask_indices(mask_type: str) -> list[int]:
    if mask_type == "strided_2x2":
        indices = [r * GRID_SIZE + c for r in range(0, GRID_SIZE, 2) for c in range(0, GRID_SIZE, 2)]
    elif mask_type == "checkerboard":
        indices = [r * GRID_SIZE + c for r in range(GRID_SIZE) for c in range(GRID_SIZE) if (r + c) % 2 == 0]
    elif mask_type == "shifted_strided_2x2":
        indices = [r * GRID_SIZE + c for r in range(1, GRID_SIZE, 2) for c in range(1, GRID_SIZE, 2)]
    else:
        raise ValueError(f"Unknown grid mask type: {mask_type}")
    expected = 128 if mask_type == "checkerboard" else 64
    if len(indices) != expected or len(set(indices)) != expected:
        raise RuntimeError(f"Invalid {mask_type} geometry")
    return indices


class SpatialGridAttentionCDInference(AttentionMaskEntityCDInference):
    """Attention-CD with fixed grid, fixed-size random, or semantic geometry."""

    spatial_selection_mode: str

    def _select(self, h: np.ndarray) -> tuple[list[int], dict]:
        mode = self.spatial_selection_mode
        random_seed = None
        semantic_reference: list[int] = []
        semantic_meta: dict = {}

        if mode in {"semantic_hard", "semantic_sparse"}:
            semantic_reference, semantic_meta = super()._select(h)
            selected = semantic_reference
            if mode == "semantic_sparse":
                random_seed = int(
                    np.random.SeedSequence(
                        [self._episode_seed, self._selector_step - 1, 0x5E5A25E]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                rng = np.random.default_rng(random_seed)
                count = max(1, len(semantic_reference) // 2)
                selected = sorted(
                    int(index)
                    for index in rng.choice(semantic_reference, size=count, replace=False)
                )
        elif mode == "random_64":
            random_seed = int(
                np.random.SeedSequence(
                    [self._episode_seed, self._selector_step, 0xA77A64]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            selected = sorted(
                int(index)
                for index in np.random.default_rng(random_seed).choice(256, size=64, replace=False)
            )
            self._selector_step += 1
        elif mode == "grid_strided":
            selected = get_grid_mask_indices("strided_2x2")
            self._selector_step += 1
        elif mode == "grid_checkerboard":
            selected = get_grid_mask_indices("checkerboard")
            self._selector_step += 1
        else:
            raise ValueError(f"Unknown spatial selection mode: {mode}")

        if len(selected) != len(set(selected)) or not selected:
            raise RuntimeError("Spatial selector returned invalid indices")
        meta = {
            **semantic_meta,
            "selection_mode": mode,
            "instruction": self._selector_instr,
            "selected_entities": list(self._entities),
            "selected_token_ids": selected,
            "num_tokens": len(selected),
            "random_seed": random_seed,
            "semantic_reference_token_ids": semantic_reference,
            "semantic_reference_num_tokens": len(semantic_reference),
            "grid_shape": [16, 16],
        }
        return selected, meta


class GlobalTokenMaskCDInference(SpatialGridAttentionCDInference):
    """Semantic CD with a global token-mask negative branch.

    Same selector, seed, lambda, and layer interval as the Attention-CD arm;
    only the intervention differs: the selected visual tokens are made
    unreadable as key/value to *all* queries on the selected layers, rather
    than blocking only the Action Query -> selected visual key connections.
    """

    def _attention_intervention_context(self, selected, action_query_start):
        return block_semantic_tokens_globally(
            self.vla,
            selected,
            layer_start=self.attention_layer_start,
            layer_end=self.attention_layer_end,
            mask_value=self.attention_mask_value,
        )

    def _attention_trace_audit(self, attention_trace, n_negative):
        validate_global_token_mask(attention_trace, n_negative)
        return attention_trace.as_dict()
