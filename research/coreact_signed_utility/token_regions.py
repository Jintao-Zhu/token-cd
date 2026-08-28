"""Outcome-blind local visual-token candidate construction for Phase T0."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from research.coreact_closed_loop.guidance import _full_velocity, _prefix_cache, tensor_sha256
from research.coreact_exploration.instrumentation import PrefixToken, validate_intervention_group


@dataclass(frozen=True)
class TokenRegion:
    camera_id: str
    row: int
    col: int
    prefix_indices: tuple[int, int, int, int]
    visual_indices: tuple[int, int, int, int]

    @property
    def region_id(self) -> str:
        return f"{self.camera_id}__r{self.row:02d}c{self.col:02d}"


def local_2x2_regions(
    span_map: Sequence[PrefixToken], camera_ids: Sequence[str], grid_size: int = 8
) -> list[TokenRegion]:
    regions: list[TokenRegion] = []
    for camera_id in camera_ids:
        by_visual = {
            token.visual_token_index: token.index
            for token in span_map
            if token.modality == "visual"
            and token.camera_id == camera_id
            and token.intervention_allowed
        }
        if set(by_visual) != set(range(grid_size * grid_size)):
            raise RuntimeError(f"{camera_id} is not a complete {grid_size}x{grid_size} visual grid")
        for row in range(grid_size - 1):
            for col in range(grid_size - 1):
                visual = (row * grid_size + col, row * grid_size + col + 1,
                          (row + 1) * grid_size + col, (row + 1) * grid_size + col + 1)
                prefix = tuple(by_visual[index] for index in visual)
                validate_intervention_group(span_map, prefix)
                regions.append(TokenRegion(camera_id, row, col, prefix, visual))
    if len(regions) != len(camera_ids) * (grid_size - 1) ** 2:
        raise AssertionError("unexpected local-region count")
    return regions


def regions_overlap(left: TokenRegion, right: TokenRegion) -> bool:
    return left.camera_id == right.camera_id and bool(
        set(left.visual_indices) & set(right.visual_indices)
    )


def replace_region_batch(
    prefix: torch.Tensor,
    regions: Sequence[TokenRegion],
    visual_position_mean: torch.Tensor,
    camera_ids: Sequence[str],
) -> torch.Tensor:
    camera_to_index = {camera: index for index, camera in enumerate(camera_ids)}
    output = prefix.expand(len(regions), -1, -1).clone()
    for batch_index, region in enumerate(regions):
        camera_index = camera_to_index[region.camera_id]
        for prefix_index, visual_index in zip(region.prefix_indices, region.visual_indices, strict=True):
            output[batch_index, prefix_index] = visual_position_mean[camera_index, visual_index].to(
                device=prefix.device, dtype=prefix.dtype
            )
    return output


@torch.no_grad()
def integrate_prefix_batch(
    model,
    prefix: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    batch_size = prefix.shape[0]
    pads = prefix_pad_masks.expand(batch_size, -1)
    atts = prefix_att_masks.expand(batch_size, -1)
    cache = _prefix_cache(model, prefix, pads, atts)
    x_t = noise.expand(batch_size, -1, -1).clone()
    dt = -1.0 / model.config.num_steps
    for step in range(model.config.num_steps):
        timestep = torch.full(
            (batch_size,), 1.0 + step * dt, device=x_t.device, dtype=torch.float32
        )
        velocity = model.denoise_step(pads, cache, x_t, timestep)
        if not bool(torch.isfinite(velocity).all()):
            raise RuntimeError("nonfinite masked flow integration")
        x_t = x_t + dt * velocity
    return x_t


@torch.no_grad()
def sample_fixed_region_actions(
    model,
    images: Sequence[torch.Tensor],
    image_masks: Sequence[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    region_prefix_indices: Sequence[int],
    visual_position_mean: torch.Tensor,
    camera_ids: Sequence[str],
) -> tuple[torch.Tensor, dict]:
    """Directly sample one flow chunk with exactly one preregistered region replaced."""
    prefix, pads, atts = model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)
    from research.coreact_exploration.instrumentation import build_prefix_span_map
    span_map = build_prefix_span_map(
        model, images, image_masks, lang_tokens, lang_masks, pads, camera_ids=camera_ids
    )[0]
    validate_intervention_group(span_map, region_prefix_indices)
    selected = set(region_prefix_indices)
    matching = [
        region for region in local_2x2_regions(span_map, camera_ids)
        if set(region.prefix_indices) == selected
    ]
    if len(matching) != 1:
        raise RuntimeError("selected indices do not define exactly one local 2x2 region")
    masked_prefix = replace_region_batch(prefix, matching, visual_position_mean, camera_ids)
    changed = (prefix != masked_prefix).any(dim=-1).nonzero(as_tuple=False)[:, 1].tolist()
    if sorted(changed) != sorted(region_prefix_indices):
        raise RuntimeError(f"masked prefix changed {changed}, expected {list(region_prefix_indices)}")
    actions = integrate_prefix_batch(model, masked_prefix, pads, atts, noise)
    return actions, {
        "region": asdict(matching[0]),
        "changed_indices": changed,
        "clean_prefix_sha256": tensor_sha256(prefix),
        "masked_prefix_sha256": tensor_sha256(masked_prefix),
        "noise_sha256": tensor_sha256(noise),
        "output_sha256": tensor_sha256(actions),
        "finite": bool(torch.isfinite(actions).all()),
        "protected_tokens_untouched": all(
            span_map[index].modality == "visual" and span_map[index].intervention_allowed
            for index in changed
        ),
    }


@torch.no_grad()
def region_features(
    model,
    prefix: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    span_map: Sequence[PrefixToken],
    regions: Sequence[TokenRegion],
    attention_scores: torch.Tensor,
    noise: torch.Tensor,
    visual_position_mean: torch.Tensor,
    camera_ids: Sequence[str],
    batch_size: int = 8,
) -> tuple[list[dict], torch.Tensor]:
    clean_action = integrate_prefix_batch(
        model, prefix, prefix_pad_masks, prefix_att_masks, noise
    )[0]
    language_indices = [
        token.index for token in span_map
        if token.modality == "language" and token.intervention_allowed
    ]
    if not language_indices:
        raise RuntimeError("no valid instruction tokens")
    language_embedding = prefix[0, language_indices].float().mean(dim=0)
    rows: list[dict] = []
    for start in range(0, len(regions), batch_size):
        block = regions[start:start + batch_size]
        masked_prefix = replace_region_batch(prefix, block, visual_position_mean, camera_ids)
        masked_actions = integrate_prefix_batch(
            model, masked_prefix, prefix_pad_masks, prefix_att_masks, noise
        )
        for region, masked_action in zip(block, masked_actions, strict=True):
            region_embedding = prefix[0, list(region.prefix_indices)].float().mean(dim=0)
            relevance = torch.nn.functional.cosine_similarity(
                region_embedding[None], language_embedding[None]
            )[0]
            delta = clean_action[..., :7] - masked_action[..., :7]
            rows.append({
                **asdict(region),
                "region_id": region.region_id,
                "attention_mean": float(attention_scores[list(region.prefix_indices)].mean()),
                "attention_sum": float(attention_scores[list(region.prefix_indices)].sum()),
                "iss_action_rms": float(torch.sqrt(torch.mean(delta.float().square()))),
                "iss_action_l2": float(torch.linalg.vector_norm(delta.float())),
                "task_relevance_cosine": float(relevance),
                "masked_action_sha256": tensor_sha256(masked_action),
                "finite": bool(torch.isfinite(masked_action).all()),
            })
    return rows, clean_action


def _rank01(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    for rank, index in enumerate(order):
        ranks[index] = rank / denominator
    return ranks


def select_four_candidates(rows: list[dict], seed: int) -> list[dict]:
    if not rows:
        raise ValueError("candidate rows are empty")
    attention_rank = _rank01([row["attention_mean"] for row in rows])
    relevance_rank = _rank01([row["task_relevance_cosine"] for row in rows])
    for index, row in enumerate(rows):
        row["nuisance_score"] = attention_rank[index] * (1.0 - relevance_rank[index])
        row["iss_x_relevance"] = row["iss_action_rms"] * row["task_relevance_cosine"]
        row["nuisance_x_iss"] = row["nuisance_score"] * row["iss_action_rms"]

    selected: list[tuple[str, int]] = []
    used: list[TokenRegion] = []

    def choose(selector: str, order: list[int]) -> None:
        for index in order:
            region = TokenRegion(
                rows[index]["camera_id"], rows[index]["row"], rows[index]["col"],
                tuple(rows[index]["prefix_indices"]), tuple(rows[index]["visual_indices"]),
            )
            if all(not regions_overlap(region, previous) for previous in used):
                selected.append((selector, index)); used.append(region); return
        raise RuntimeError(f"cannot find a non-overlapping region for {selector}")

    choose("attention_max", sorted(range(len(rows)), key=lambda i: (-rows[i]["attention_mean"], i)))
    choose("iss_max", sorted(range(len(rows)), key=lambda i: (-rows[i]["iss_action_rms"], i)))
    choose("attention_high_relevance_low", sorted(range(len(rows)), key=lambda i: (-rows[i]["nuisance_score"], i)))
    digest = hashlib.sha256(str(seed).encode("ascii")).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little"))
    choose("random_control", torch.randperm(len(rows), generator=generator).tolist())

    output = []
    for selector, index in selected:
        output.append({"candidate_id": selector, **rows[index]})
    if len({item["region_id"] for item in output}) != 4:
        raise AssertionError("candidate regions are not unique")
    return output


def select_oracle_candidates(rows: list[dict], seed: int) -> list[dict]:
    """T0 selectors restricted to Attention, ISS, and matched random control."""
    if not rows:
        raise ValueError("candidate rows are empty")
    selected: list[tuple[str, int]] = []
    used: list[TokenRegion] = []

    def choose(selector: str, order: list[int]) -> None:
        for index in order:
            region = TokenRegion(
                rows[index]["camera_id"], rows[index]["row"], rows[index]["col"],
                tuple(rows[index]["prefix_indices"]), tuple(rows[index]["visual_indices"]),
            )
            if all(not regions_overlap(region, previous) for previous in used):
                selected.append((selector, index)); used.append(region); return
        raise RuntimeError(f"cannot find a non-overlapping region for {selector}")

    choose("attention_max", sorted(range(len(rows)), key=lambda i: (-rows[i]["attention_mean"], i)))
    choose("iss_max", sorted(range(len(rows)), key=lambda i: (-rows[i]["iss_action_rms"], i)))
    seed_bytes = hashlib.sha256(str(seed).encode("ascii")).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(seed_bytes[:8], "little"))
    choose("random_control", torch.randperm(len(rows), generator=generator).tolist())
    return [{"candidate_id": selector, **rows[index]} for selector, index in selected]


@torch.no_grad()
def sample_signed_region_actions(
    model, images, image_masks, lang_tokens, lang_masks, state, noise,
    candidate: dict, visual_position_mean: torch.Tensor, camera_ids: Sequence[str],
    *, direction: str, guidance_scale: float = 0.5, trust_region_kappa: float = 0.25,
    action_dim: int = 7,
) -> tuple[torch.Tensor, dict]:
    """Integrate full +/- clipped(full-counterfactual) for one fixed grid region."""
    if direction not in ("toward", "away"):
        raise ValueError(direction)
    prefix, pads, atts = model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)
    from research.coreact_exploration.instrumentation import build_prefix_span_map
    span_map = build_prefix_span_map(
        model, images, image_masks, lang_tokens, lang_masks, pads, camera_ids=camera_ids
    )[0]
    matches = [r for r in local_2x2_regions(span_map, camera_ids)
               if r.camera_id == candidate["camera_id"] and r.row == candidate["row"] and r.col == candidate["col"]]
    if len(matches) != 1:
        raise RuntimeError("candidate grid region cannot be relocated")
    region = matches[0]
    negative_prefix = replace_region_batch(prefix, matches, visual_position_mean, camera_ids)
    changed = (prefix != negative_prefix).any(dim=-1).nonzero(as_tuple=False)[:, 1].tolist()
    if sorted(changed) != sorted(region.prefix_indices):
        raise RuntimeError("unexpected changed token indices")
    clean_cache = _prefix_cache(model, prefix, pads, atts)
    negative_cache = _prefix_cache(model, negative_prefix, pads, atts)
    x_t = noise.clone(); dt = -1.0 / model.config.num_steps; steps = []
    for step in range(model.config.num_steps):
        tau = torch.full((1,), 1.0 + step * dt, device=state.device, dtype=torch.float32)
        clean = model.denoise_step(pads, clean_cache, x_t, tau)
        counterfactual = model.denoise_step(pads, negative_cache, x_t, tau)
        raw = clean - counterfactual
        raw[..., action_dim:] = 0
        clean_norm = torch.linalg.vector_norm(clean[..., :action_dim])
        raw_norm = torch.linalg.vector_norm(raw[..., :action_dim])
        clip_scale = torch.clamp(trust_region_kappa * clean_norm / (raw_norm + 1e-12), max=1.0)
        sign = -1.0 if direction == "toward" else 1.0
        applied = sign * guidance_scale * raw * clip_scale
        guided = clean + applied
        if not bool(torch.isfinite(guided).all()):
            raise RuntimeError("nonfinite signed region guidance")
        x_t = x_t + dt * guided
        steps.append({"step": step, "clean_norm": float(clean_norm), "raw_norm": float(raw_norm),
                      "clip_scale": float(clip_scale),
                      "applied_correction_norm": float(torch.linalg.vector_norm(applied[..., :action_dim])),
                      "finite": True})
    return x_t, {"candidate_id": candidate["candidate_id"], "direction": direction,
                 "region_id": region.region_id, "changed_indices": changed,
                 "protected_tokens_untouched": True, "steps": steps,
                 "noise_sha256": tensor_sha256(noise), "output_sha256": tensor_sha256(x_t)}
