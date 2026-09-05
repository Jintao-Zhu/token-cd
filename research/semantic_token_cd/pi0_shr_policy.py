"""SHR inference for the open-pi-zero flow-matching policy.

The positive and negative branches share the same noisy action state at every
Euler step.  SHR is applied to the first six action-velocity dimensions; the
gripper velocity stays exactly equal to the positive branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans


GRID = 16
KMEANS_K = 8
KMEANS_SEED = 0
KMEANS_N_INIT = 10
EPS = 1e-8

PREPOSITIONS = {
    "into", "in", "on", "onto", "near", "to", "next", "from", "under",
    "over", "behind", "beside", "above", "below", "inside", "at", "with",
}
DETERMINERS = {"the", "a", "an"}


def extract_entities(instruction: str) -> list[str]:
    """Return the deduplicated source/target noun phrases used by SHR."""
    words = instruction.lower().split()
    if not words:
        raise ValueError("instruction must be non-empty")
    rest = words[1:]
    positions = [index for index, word in enumerate(rest) if word in PREPOSITIONS]
    if not positions:
        obj = " ".join(word for word in rest if word not in DETERMINERS)
        return [obj] if obj else [instruction.lower()]
    first, last = positions[0], positions[-1]
    source = " ".join(word for word in rest[:first] if word not in DETERMINERS)
    target = " ".join(word for word in rest[last + 1:] if word not in DETERMINERS)
    entities: list[str] = []
    for entity in (source, target):
        if entity and entity not in entities:
            entities.append(entity)
    return entities or [instruction.lower()]


def harmonic_reconstruct(features: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Solve the beta=0 four-neighbor Dirichlet problem on a 16x16 grid."""
    region = np.asarray(sorted(set(int(index) for index in region)), dtype=np.int64)
    if not region.size:
        raise ValueError("cannot reconstruct an empty region")
    if features.ndim != 2 or features.shape[0] != GRID * GRID:
        raise ValueError(f"expected visual features [256,D], got {features.shape}")
    position = {int(token): row for row, token in enumerate(region)}
    matrix = np.zeros((region.size, region.size), dtype=np.float64)
    rhs = np.zeros((region.size, features.shape[1]), dtype=np.float64)
    clean = features.astype(np.float64, copy=False)
    for row, token in enumerate(region):
        r, c = divmod(int(token), GRID)
        neighbors = []
        if r > 0:
            neighbors.append(token - GRID)
        if r + 1 < GRID:
            neighbors.append(token + GRID)
        if c > 0:
            neighbors.append(token - 1)
        if c + 1 < GRID:
            neighbors.append(token + 1)
        matrix[row, row] = len(neighbors)
        for neighbor in neighbors:
            column = position.get(int(neighbor))
            if column is None:
                rhs[row] += clean[neighbor]
            else:
                matrix[row, column] -= 1.0
    try:
        solved = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("singular SHR harmonic region") from exc
    if not np.isfinite(solved).all():
        raise FloatingPointError("non-finite SHR reconstruction")
    return solved.astype(np.float32)


def shr_guided_velocity(
    positive: torch.Tensor,
    negative: torch.Tensor,
    lambd: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SHR while leaving the last (gripper) dimension positive."""
    if positive.shape != negative.shape or positive.ndim != 3:
        raise ValueError("positive and negative velocities must share shape [B,H,A]")
    if positive.shape[-1] < 2 or lambd < 0:
        raise ValueError("SHR requires at least two action dimensions and lambda >= 0")
    residual = positive.float() - negative.float()
    guided = positive.float().clone()
    guided[..., :-1] += float(lambd) * residual[..., :-1]
    if not torch.isfinite(guided).all():
        raise FloatingPointError("non-finite SHR velocity")
    return guided.to(positive.dtype), residual


@dataclass
class ConditionBranch:
    caches: dict[str, Any]
    visual_features: torch.Tensor


class Pi0SHRInference:
    """A thin inference wrapper around a loaded ``PiZeroInference`` model."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        lambd: float = 0.5,
        kmeans_k: int = KMEANS_K,
        kmeans_seed: int = KMEANS_SEED,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lambd = float(lambd)
        self.kmeans_k = int(kmeans_k)
        self.kmeans_seed = int(kmeans_seed)
        self._entity_cache: dict[str, np.ndarray] = {}

    def _image_text_embedding(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        model = self.model
        token_embeddings = model.embed_tokens(input_ids)
        batch, length = input_ids.shape
        output = torch.full(
            (batch, length, image_features.shape[-1]),
            model.pad_token_id,
            dtype=dtype,
            device=input_ids.device,
        )
        text_mask = (input_ids != model.image_token_index) & (input_ids != model.pad_token_id)
        image_mask = input_ids == model.image_token_index
        output[text_mask] = token_embeddings[text_mask]
        scaled = image_features / (model.image_text_hidden_size ** 0.5)
        for index in range(batch):
            locations = image_mask[index].nonzero(as_tuple=True)[0]
            output[index, locations] = scaled[index, : len(locations)]
        return output

    def _condition_branch(
        self,
        inputs: dict[str, torch.Tensor],
        image_features: torch.Tensor,
    ) -> ConditionBranch:
        model = self.model
        caches = model.joint_model.build_mixture_caches()
        embeddings = self._image_text_embedding(
            inputs["input_ids"], image_features, inputs["pixel_values"].dtype
        )
        proprio = model.proprio_encoder(inputs["proprios"])
        _, caches = model.joint_model(
            attention_mask=inputs["image_text_proprio_mask"],
            position_ids_all={
                "vlm": inputs["vlm_position_ids"],
                "proprio": inputs["proprio_position_ids"],
            },
            embeds_all={"vlm": embeddings, "proprio": proprio},
            kv_caches=caches,
            return_caches=True,
        )
        return ConditionBranch(caches=caches, visual_features=image_features)

    def _phrase_embedding(self, phrase: str, device: torch.device) -> np.ndarray:
        cached = self._entity_cache.get(phrase)
        if cached is not None:
            return cached
        ids = self.tokenizer(phrase, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = self.tokenizer(phrase, add_special_tokens=True)["input_ids"]
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        value = self.model.embed_tokens(tokens)[0].mean(0).detach().float().cpu().numpy()
        self._entity_cache[phrase] = value
        return value

    def _negative_features(
        self,
        positive: torch.Tensor,
        instruction: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        features = positive[0].detach().float().cpu().numpy()
        labels = KMeans(
            n_clusters=self.kmeans_k,
            random_state=self.kmeans_seed,
            n_init=KMEANS_N_INIT,
        ).fit_predict(features.astype(np.float64))
        group_vectors = np.stack(
            [features[labels == group].mean(0) for group in range(self.kmeans_k)]
        )
        group_vectors /= np.linalg.norm(group_vectors, axis=1, keepdims=True) + EPS
        entities = extract_entities(instruction)
        groups: list[int] = []
        scores: list[float] = []
        for entity in entities:
            embedding = self._phrase_embedding(entity, positive.device)
            embedding = embedding / (np.linalg.norm(embedding) + EPS)
            similarities = group_vectors @ embedding
            group = int(np.argmax(similarities))
            groups.append(group)
            scores.append(float(similarities[group]))

        negative = features.copy()
        claimed: set[int] = set()
        regions = []
        for entity, group in zip(entities, groups):
            indices = np.asarray(
                [int(i) for i in np.flatnonzero(labels == group) if int(i) not in claimed],
                dtype=np.int64,
            )
            if indices.size:
                claimed.update(map(int, indices))
                negative[indices] = harmonic_reconstruct(features, indices)
            regions.append({"entity": entity, "group": group, "num_tokens": int(indices.size)})
        if not claimed:
            raise RuntimeError("Pi0-SHR semantic selector produced an empty region")
        tensor = torch.from_numpy(negative).to(device=positive.device, dtype=positive.dtype)[None]
        return tensor, {
            "selected_entities": entities,
            "selected_group_ids": groups,
            "per_entity_score": scores,
            "selected_token_ids": sorted(claimed),
            "num_tokens": len(claimed),
            "entity_regions": regions,
        }

    def _velocity(
        self,
        action: torch.Tensor,
        time: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        branch: ConditionBranch,
    ) -> torch.Tensor:
        model = self.model
        time_condition = model.time_embedding(time)
        if model.action_expert_adaptive_mode:
            embeddings = model.action_encoder(action)
        else:
            embeddings = model.action_encoder(action, time_condition)
        hidden = model.joint_model(
            attention_mask=inputs["action_mask"],
            position_ids_all={"action": inputs["action_position_ids"]},
            embeds_all={"action": embeddings},
            time_cond=time_condition,
            kv_caches=branch.caches,
            cache_mode="append_non_active",
        )["action"]
        return model.action_decoder(hidden)

    @torch.inference_mode()
    def infer(
        self,
        inputs: dict[str, torch.Tensor],
        instruction: str,
        *,
        arm: str,
        noise_seed: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if arm not in {"pi0_vanilla", "pi0_shr_harmonic"}:
            raise ValueError(f"unknown Pi0 arm: {arm}")
        model = self.model
        visual = model.multi_modal_projector(model.vision_tower(inputs["pixel_values"]))
        positive = self._condition_branch(inputs, visual)
        selector: dict[str, Any] | None = None
        negative: ConditionBranch | None = None
        if arm == "pi0_shr_harmonic":
            negative_visual, selector = self._negative_features(visual, instruction)
            negative = self._condition_branch(inputs, negative_visual)

        generator = torch.Generator(device=inputs["pixel_values"].device)
        generator.manual_seed(int(noise_seed))
        action = torch.randn(
            (inputs["pixel_values"].shape[0], model.horizon_steps, model.action_dim),
            generator=generator,
            device=inputs["pixel_values"].device,
            dtype=inputs["pixel_values"].dtype,
        )
        initial_noise = action.detach().float().cpu()
        delta_t = 1.0 / model.num_inference_steps
        time = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
        steps = []
        for step in range(model.num_inference_steps):
            positive_velocity = self._velocity(action, time, inputs, positive)
            if negative is None:
                guided_velocity = positive_velocity
                residual = torch.zeros_like(positive_velocity, dtype=torch.float32)
                negative_velocity = None
            else:
                negative_velocity = self._velocity(action, time, inputs, negative)
                guided_velocity, residual = shr_guided_velocity(
                    positive_velocity, negative_velocity, self.lambd
                )
            action = action + delta_t * guided_velocity
            steps.append({
                "flow_step": step,
                "time": float(time[0].item()),
                "positive_velocity": positive_velocity.detach().to(torch.float16).cpu(),
                "negative_velocity": None if negative_velocity is None else negative_velocity.detach().to(torch.float16).cpu(),
                "guided_velocity": guided_velocity.detach().to(torch.float16).cpu(),
                "residual_norm": float(torch.linalg.vector_norm(residual[..., :-1]).item()),
                "gripper_velocity_positive_unchanged": bool(
                    torch.equal(guided_velocity[..., -1], positive_velocity[..., -1])
                ),
            })
            time = time + delta_t
        if model.final_action_clip_value is not None:
            action = torch.clamp(
                action, -model.final_action_clip_value, model.final_action_clip_value
            )
        trace = {
            "arm": arm,
            "noise_seed": int(noise_seed),
            "initial_noise": initial_noise,
            "selector": selector,
            "flow_steps": steps,
            "lambda": 0.0 if negative is None else self.lambd,
            "shared_action_state_between_branches": True,
            "all_gripper_velocity_positive_unchanged": all(
                item["gripper_velocity_positive_unchanged"] for item in steps
            ),
            "reconstruction_finite": selector is None or bool(torch.isfinite(negative.visual_features).all()),
        }
        return action, trace
