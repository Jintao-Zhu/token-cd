"""L6 matched harmonic contrastive guidance for OpenPI pi0.5.

The adapter keeps OpenPI untouched.  It constructs a counterfactual primary
camera prefix, builds independent clean/counterfactual KV caches, and combines
their flow velocities while both branches share the same noisy action state and
adaRMS timestep condition.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import numpy as np


ROOT = Path(os.environ.get("PI05_WORKSPACE_ROOT", "/home/leju-suzhou/zjt_ws"))
OPENPI_ROOT = Path(os.environ.get("PI05_OPENPI_ROOT", ROOT / "openpi"))
SKLEARN_OVERLAY = Path(
    os.environ.get("PI05_SKLEARN_OVERLAY", ROOT / "pi05_sklearn_overlay")
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "PI05_LIBERO_CHECKPOINT",
        "/home/leju-suzhou/.cache/openpi/openpi-assets/checkpoints/pi05_libero",
    )
)

GRID = 16
N_VISUAL = GRID * GRID
KMEANS_GROUPS = 8
KMEANS_SEED = 0
KMEANS_N_INIT = 10
DEFAULT_LAYER_INDEX = 6
DEFAULT_GUIDANCE_SCALE = 0.5
LIBERO_ACTION_DIMS = 7
LIBERO_GUIDED_DIMS = 6
EPS = 1e-8

PREPOSITIONS = {
    "into", "in", "on", "onto", "near", "to", "next", "from", "under",
    "over", "behind", "beside", "above", "below", "inside", "at", "with",
}
DETERMINERS = {"the", "a", "an"}


def extract_entities(instruction: str) -> list[str]:
    words = instruction.lower().strip().replace("_", " ").split()
    if not words:
        raise ValueError("instruction must be non-empty")
    rest = words[1:]
    positions = [index for index, word in enumerate(rest) if word in PREPOSITIONS]
    if not positions:
        entity = " ".join(word for word in rest if word not in DETERMINERS)
        return [entity] if entity else [instruction.lower()]
    first, last = positions[0], positions[-1]
    source = " ".join(word for word in rest[:first] if word not in DETERMINERS)
    target = " ".join(word for word in rest[last + 1 :] if word not in DETERMINERS)
    entities: list[str] = []
    for entity in (source, target):
        if entity and entity not in entities:
            entities.append(entity)
    return entities or [instruction.lower()]


def stable_top_m(scores: np.ndarray, count: int) -> list[int]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (N_VISUAL,) or not np.isfinite(values).all():
        raise ValueError(f"expected {N_VISUAL} finite scores, got {values.shape}")
    if not 1 <= int(count) <= N_VISUAL:
        raise ValueError(f"invalid selected-token count {count}")
    order = np.lexsort((np.arange(N_VISUAL), -values))
    return sorted(int(index) for index in order[: int(count)])


def harmonic_reconstruct(features: np.ndarray, selected: Iterable[int]) -> np.ndarray:
    clean = np.asarray(features, dtype=np.float64)
    region = np.asarray(sorted(set(int(index) for index in selected)), dtype=np.int64)
    if clean.ndim != 2 or clean.shape[0] != N_VISUAL:
        raise ValueError(f"expected visual features [{N_VISUAL},D], got {clean.shape}")
    if not region.size:
        raise ValueError("cannot reconstruct an empty region")
    if region[0] < 0 or region[-1] >= N_VISUAL:
        raise ValueError("selected token index outside the primary-view grid")

    position = {int(token): row for row, token in enumerate(region)}
    matrix = np.zeros((region.size, region.size), dtype=np.float64)
    rhs = np.zeros((region.size, clean.shape[1]), dtype=np.float64)
    for row, token in enumerate(region):
        grid_row, grid_col = divmod(int(token), GRID)
        neighbours = []
        if grid_row > 0:
            neighbours.append(token - GRID)
        if grid_row + 1 < GRID:
            neighbours.append(token + GRID)
        if grid_col > 0:
            neighbours.append(token - 1)
        if grid_col + 1 < GRID:
            neighbours.append(token + 1)
        matrix[row, row] = len(neighbours)
        for neighbour in neighbours:
            column = position.get(int(neighbour))
            if column is None:
                rhs[row] += clean[neighbour]
            else:
                matrix[row, column] -= 1.0
    try:
        reconstructed = np.linalg.solve(matrix, rhs).astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("singular harmonic reconstruction system") from exc
    if not np.isfinite(reconstructed).all():
        raise FloatingPointError("non-finite harmonic reconstruction")
    return reconstructed


def make_harmonic_negative(features: np.ndarray, selected: Iterable[int]) -> np.ndarray:
    selected_ids = sorted(set(int(index) for index in selected))
    negative = np.asarray(features).copy()
    negative[selected_ids] = harmonic_reconstruct(features, selected_ids)
    if not np.isfinite(negative).all():
        raise FloatingPointError("non-finite negative visual prefix")
    return negative


def guided_velocity_numpy(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    guidance_scale: float,
    guided_dim_mask: np.ndarray,
    active: bool = True,
) -> np.ndarray:
    positive_array = np.asarray(positive)
    negative_array = np.asarray(negative)
    mask = np.asarray(guided_dim_mask, dtype=np.float32)
    if positive_array.shape != negative_array.shape:
        raise ValueError("positive and negative velocities must have the same shape")
    if mask.shape != (positive_array.shape[-1],):
        raise ValueError("guided-dimension mask does not match the action dimension")
    if guidance_scale < 0 or not np.isfinite(guidance_scale):
        raise ValueError("guidance scale must be finite and nonnegative")
    if not active or guidance_scale == 0:
        return positive_array.copy()
    return positive_array + guidance_scale * (positive_array - negative_array) * mask


def libero_guided_dim_mask(action_dim: int) -> np.ndarray:
    if action_dim < LIBERO_ACTION_DIMS:
        raise ValueError(f"LIBERO requires at least {LIBERO_ACTION_DIMS} model action dimensions")
    mask = np.zeros((action_dim,), dtype=np.float32)
    mask[:LIBERO_GUIDED_DIMS] = 1.0
    return mask


def instrument_openpi_for_attention(openpi_root: Path = OPENPI_ROOT) -> Path:
    """Create a temporary OpenPI overlay with a read-only PaliGemma attention tap."""
    temporary = Path(tempfile.mkdtemp(prefix="pi05_matched_flow_cd_"))
    package = temporary / "openpi"
    shutil.copytree(openpi_root / "src/openpi", package)
    path = package / "models/gemma.py"
    source = path.read_text()
    anchor = "        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)\n"
    tap = '''        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        if _PI05_ATTN_CAPTURE is not None:
            query_mask = jnp.asarray(_PI05_ATTN_CAPTURE["query_mask"], dtype=probs.dtype)
            query_mass = jnp.maximum(jnp.sum(query_mask, axis=1), 1.0)
            key_start = int(_PI05_ATTN_CAPTURE.get("key_start", 0))
            primary = probs[..., key_start : key_start + 256]
            selected = jnp.einsum("bkgts,bt->bks", primary, query_mask)
            selected = jnp.sum(selected, axis=1) / query_mass[:, None]
            jax.debug.callback(
                _PI05_ATTN_CAPTURE["callback"],
                selected,
                ordered=bool(_PI05_ATTN_CAPTURE.get("ordered", True)),
            )
'''
    if source.count(anchor) != 1:
        raise RuntimeError("could not locate the unique Gemma attention softmax")
    source = source.replace(anchor, tap)
    header = "\n_PI05_ATTN_CAPTURE = None\n\n"
    if source.count("Variant = Literal[") != 1:
        raise RuntimeError("could not locate Gemma variant declaration")
    source = source.replace("Variant = Literal[", header + "Variant = Literal[", 1)
    path.write_text(source)
    return temporary


def _pi05_guided_sample_actions(
    model,
    rng,
    observation,
    negative_primary_tokens,
    guidance_scale,
    guided_dim_mask,
    active_step_mask,
    *,
    num_steps=10,
    noise=None,
):
    """Model-bound JAX sampler used through ``nnx_utils.module_jit``."""
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    observation = model_lib.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(
            rng, (batch_size, model.action_horizon, model.action_dim)
        )

    positive_prefix, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    negative_prefix = positive_prefix.at[:, :N_VISUAL].set(negative_primary_tokens)
    prefix_attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, positive_cache = model.PaliGemma.llm(
        [positive_prefix, None], mask=prefix_attention_mask, positions=positions
    )
    _, negative_cache = model.PaliGemma.llm(
        [negative_prefix, None], mask=prefix_attention_mask, positions=positions
    )

    dt = -1.0 / num_steps

    def velocity(x_t, time, cache):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attention_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attention_mask = jnp.concatenate(
            [prefix_to_suffix_mask, suffix_attention_mask], axis=-1
        )
        suffix_positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )
        (_, suffix_output), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attention_mask,
            positions=suffix_positions,
            kv_cache=cache,
            adarms_cond=[None, adarms_cond],
        )
        return model.action_out_proj(suffix_output[:, -model.action_horizon :])

    def step(carry):
        x_t, time, step_index = carry
        positive_velocity = velocity(x_t, time, positive_cache)
        negative_velocity = velocity(x_t, time, negative_cache)
        residual = positive_velocity - negative_velocity
        safe_step_index = jnp.minimum(step_index, active_step_mask.shape[0] - 1)
        active = jnp.where(
            step_index < active_step_mask.shape[0],
            active_step_mask[safe_step_index],
            False,
        ).astype(positive_velocity.dtype)
        correction = (
            guidance_scale.astype(positive_velocity.dtype)
            * active
            * residual
            * guided_dim_mask.astype(positive_velocity.dtype)
        )
        guided_velocity = jnp.where(
            guidance_scale == 0, positive_velocity, positive_velocity + correction
        )
        return x_t + dt * guided_velocity, time + dt, step_index + 1

    def condition(carry):
        _, time, _ = carry
        return time >= -dt / 2

    action, _, _ = jax.lax.while_loop(
        condition,
        step,
        (noise, jnp.asarray(1.0, dtype=noise.dtype), jnp.asarray(0, dtype=jnp.int32)),
    )
    return action


def _pi05_pcd_sample_actions(
    model, rng, observation, negative_primary_tokens, noise, alpha, bandwidth_scale, *, num_steps=10
):
    """JAX PCD candidate generation and contrastive KDE resampling.

    Candidate 0 may be the paired vanilla noise; all candidates share each
    candidate's noise across clean and harmonic-counterfactual conditions.
    """
    import einops
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    observation = model_lib.preprocess_observation(None, observation, train=False)
    n_candidates = noise.shape[0]
    batch_size = observation.state.shape[0]
    if batch_size != 1:
        raise ValueError(f"PCD expects one observation, got batch {batch_size}")

    positive_prefix, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    negative_prefix = positive_prefix.at[:, :N_VISUAL].set(negative_primary_tokens)
    prefix_attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, positive_cache = model.PaliGemma.llm(
        [positive_prefix, None], mask=prefix_attention_mask, positions=positions
    )
    _, negative_cache = model.PaliGemma.llm(
        [negative_prefix, None], mask=prefix_attention_mask, positions=positions
    )

    # Prefix caches are identical across candidates. Tile only their batch axis;
    # this keeps the expensive image/language prefix pass at batch size one.
    # OpenPI Gemma caches are [layers, batch, time, kv, head_dim], unlike
    # ordinary batch-first model outputs. Repeat the cache batch axis (1).
    positive_cache = jax.tree.map(
        lambda value: jnp.repeat(value, n_candidates, axis=1), positive_cache
    )
    negative_cache = jax.tree.map(
        lambda value: jnp.repeat(value, n_candidates, axis=1), negative_cache
    )
    candidate_observation = jax.tree.map(
        lambda value: jnp.repeat(value, n_candidates, axis=0), observation
    )
    candidate_prefix_mask = jnp.repeat(prefix_mask, n_candidates, axis=0)
    dt = -1.0 / num_steps

    def velocity(x_t, time, cache):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            candidate_observation, x_t, jnp.full((n_candidates,), time, dtype=x_t.dtype)
        )
        suffix_attention_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(
            candidate_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attention_mask = jnp.concatenate(
            [prefix_to_suffix_mask, suffix_attention_mask], axis=-1
        )
        suffix_positions = (
            jnp.sum(candidate_prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )
        (_, suffix_output), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attention_mask,
            positions=suffix_positions,
            kv_cache=cache,
            adarms_cond=[None, adarms_cond],
        )
        return model.action_out_proj(suffix_output[:, -model.action_horizon :])

    def integrate(cache):
        def body(index, x_t):
            time = 1.0 + index.astype(x_t.dtype) * dt
            return x_t + dt * velocity(x_t, time, cache)
        return jax.lax.fori_loop(0, num_steps, body, noise)

    actions_clean = integrate(positive_cache)
    actions_cf = integrate(negative_cache)

    # Match the reference sampler: product KDE over the first six robot action
    # dimensions and the whole LIBERO action chunk, excluding gripper/padding.
    dims = min(LIBERO_GUIDED_DIMS, model.action_dim)
    clean = actions_clean[:, :, :dims].astype(jnp.float32)
    counterfactual = actions_cf[:, :, :dims].astype(jnp.float32)
    scale = jnp.std(jnp.concatenate([clean, counterfactual], axis=0), axis=0, ddof=1)
    valid = jnp.isfinite(scale) & (scale >= 1e-6)
    bandwidth = jnp.maximum(scale * bandwidth_scale, 1e-6)
    norm = jnp.log(n_candidates * bandwidth * jnp.sqrt(2.0 * jnp.pi))
    clean_diff = (clean[:, None, :, :] - clean[None, :, :, :]) / bandwidth[None, None, :, :]
    cf_diff = (clean[:, None, :, :] - counterfactual[None, :, :, :]) / bandwidth[None, None, :, :]
    log_pc_terms = logsumexp(-0.5 * jnp.square(clean_diff), axis=1) - norm
    log_pn_terms = logsumexp(-0.5 * jnp.square(cf_diff), axis=1) - norm
    valid_mask = valid[None, :, :]
    log_pc = jnp.sum(jnp.where(valid_mask, log_pc_terms, 0.0), axis=(1, 2))
    log_pn = jnp.sum(jnp.where(valid_mask, log_pn_terms, 0.0), axis=(1, 2))
    score = log_pc - log_pn
    score = score - jnp.mean(score)
    score_std = jnp.std(score, ddof=1)
    score = jnp.where(score_std > 1e-12, score / score_std, score)
    score = jnp.where(jnp.sum(valid) > 0, score, jnp.zeros_like(score))
    weights = jax.nn.softmax(alpha * score)
    chosen_index = jax.random.categorical(rng, jnp.log(weights)).astype(jnp.int32)
    chosen = jnp.take(actions_clean, chosen_index, axis=0)[None, ...]
    return (
        chosen,
        weights,
        chosen_index,
        actions_clean,
        actions_cf,
        log_pc,
        log_pn,
        score,
        jnp.sum(valid),
    )


def _pi05_prefix_attention(model, observation):
    """Jitted prefix pass that fires the L6 attention tap for every layer."""

    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    processed = model_lib.preprocess_observation(None, observation, train=False)
    prefix, prefix_mask, prefix_ar_mask = model.embed_prefix(processed)
    attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    model.PaliGemma.llm([prefix, None], mask=attention_mask, positions=positions)
    return prefix


class Pi05MatchedFlowCDPolicy:
    """OpenPI pi0.5-LIBERO policy with matched harmonic flow guidance."""

    def __init__(
        self,
        *,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        gpu: int = 0,
        layer_index: int = DEFAULT_LAYER_INDEX,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        active_steps: Iterable[int] | None = None,
        num_steps: int = 10,
    ) -> None:
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"pi0.5 checkpoint not found: {checkpoint}")
        if not 0 <= int(layer_index) < 18:
            raise ValueError("PaliGemma layer index must be in [0, 17]")
        if guidance_scale < 0 or not np.isfinite(guidance_scale):
            raise ValueError("guidance scale must be finite and nonnegative")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        self.overlay = instrument_openpi_for_attention()
        sys.path.insert(0, str(OPENPI_ROOT / "src"))
        sys.path.insert(0, str(self.overlay))
        if SKLEARN_OVERLAY.is_dir() and str(SKLEARN_OVERLAY) not in sys.path:
            sys.path.insert(0, str(SKLEARN_OVERLAY))

        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_lib
        from openpi.policies import policy_config
        from openpi.shared import nnx_utils
        from openpi.training import config as config_lib

        self.jax = jax
        self.jnp = jnp
        self.model_lib = model_lib
        self.layer_index = int(layer_index)
        self.guidance_scale = float(guidance_scale)
        self.num_steps = int(num_steps)
        self.active_steps = (
            tuple(range(self.num_steps))
            if active_steps is None
            else tuple(sorted(set(int(step) for step in active_steps)))
        )
        if any(step < 0 or step >= self.num_steps for step in self.active_steps):
            raise ValueError("active flow step outside num_steps")

        config = config_lib.get_config("pi05_libero")
        self.policy = policy_config.create_trained_policy(config, checkpoint)
        self.model = self.policy._model
        if not self.model.pi05:
            raise RuntimeError("loaded model is not configured as pi0.5")
        self.config = config
        self.tokenizer = next(
            transform.tokenizer
            for transform in self.policy._input_transform.transforms
            if hasattr(transform, "tokenizer") and hasattr(transform.tokenizer, "_tokenizer")
        )
        import openpi.models.gemma as gemma

        if not hasattr(gemma, "_PI05_ATTN_CAPTURE"):
            raise RuntimeError(
                "OpenPI was imported before the pi0.5 attention overlay; "
                "construct Pi05MatchedFlowCDPolicy in a fresh process"
            )
        self._entity_embedding_cache: dict[str, np.ndarray] = {}
        model_type = type(self.model)
        setattr(model_type, "_pi05_guided_sample_actions", _pi05_guided_sample_actions)
        setattr(model_type, "_pi05_prefix_attention", _pi05_prefix_attention)
        self._sample_guided = nnx_utils.module_jit(
            self.model._pi05_guided_sample_actions
        )
        setattr(model_type, "_pi05_pcd_sample_actions", _pi05_pcd_sample_actions)
        self._sample_pcd = nnx_utils.module_jit(self.model._pi05_pcd_sample_actions)
        # One compiled prefix pass per (instruction, prompt length).  The tapped
        # query mask is a trace-time constant, so each instruction needs its own
        # executable; the callback list is reused (and cleared) per call.
        self._capture_forward: dict[tuple[str, int], Any] = {}
        self._capture_store: dict[tuple[str, int], list[np.ndarray]] = {}
        self.last_trace: dict[str, Any] = {}

    def _phrase_embedding(self, phrase: str) -> np.ndarray:
        cached = self._entity_embedding_cache.get(phrase)
        if cached is not None:
            return cached
        token_ids = self.tokenizer._tokenizer.encode(phrase, add_bos=False)
        if not token_ids:
            token_ids = self.tokenizer._tokenizer.encode(phrase, add_bos=True)
        embedded = self.model.PaliGemma.llm(
            self.jnp.asarray([token_ids], dtype=self.jnp.int32), method="embed"
        )
        value = np.asarray(embedded[0].mean(axis=0), dtype=np.float32)
        self._entity_embedding_cache[phrase] = value
        return value

    def _matched_budget(
        self, visual: np.ndarray, instruction: str
    ) -> tuple[int, list[int], list[int], list[str]]:
        from sklearn.cluster import KMeans

        features = np.asarray(visual, dtype=np.float64)
        labels = KMeans(
            n_clusters=KMEANS_GROUPS,
            init="k-means++",
            n_init=KMEANS_N_INIT,
            max_iter=300,
            algorithm="lloyd",
            tol=1e-4,
            random_state=KMEANS_SEED,
        ).fit_predict(features)
        group_means = np.stack(
            [features[labels == group].mean(axis=0) for group in range(KMEANS_GROUPS)]
        )
        group_means /= np.linalg.norm(group_means, axis=1, keepdims=True) + EPS
        entities = extract_entities(instruction)
        groups: list[int] = []
        for entity in entities:
            embedding = self._phrase_embedding(entity).astype(np.float64)
            embedding /= np.linalg.norm(embedding) + EPS
            group = int(np.argmax(group_means @ embedding))
            if group not in groups:
                groups.append(group)
        matched = sorted(
            int(index) for group in groups for index in np.flatnonzero(labels == group)
        )
        if not matched:
            raise RuntimeError("matched entity clustering selected no visual tokens")
        return len(matched), groups, matched, entities

    def _query_mask(self, item: dict, instruction: str, prefix_length: int) -> np.ndarray:
        token_ids = np.asarray(item["tokenized_prompt"], dtype=np.int32).tolist()
        query_ids = self.tokenizer._tokenizer.encode(
            instruction.strip().replace("_", " "), add_bos=False
        )
        starts = [
            start
            for start in range(len(token_ids) - len(query_ids) + 1)
            if token_ids[start : start + len(query_ids)] == query_ids
        ]
        if not starts:
            raise RuntimeError(f"could not locate task tokens for {instruction!r}")
        image_prefix_length = 3 * N_VISUAL
        query_mask = np.zeros((1, prefix_length), dtype=np.float32)
        for position in range(starts[-1], starts[-1] + len(query_ids)):
            query_mask[0, image_prefix_length + position] = 1.0
        return query_mask

    def _select_negative(self, observation, item: dict, instruction: str, layer_index=None):
        from openpi.shared import nnx_utils
        import openpi.models.gemma as gemma

        selector_layer = self.layer_index if layer_index is None else int(layer_index)
        if not 0 <= selector_layer < 18:
            raise ValueError("selector layer index must be in [0, 17]")
        prompt_ids = np.asarray(item["tokenized_prompt"], dtype=np.int32)
        prefix_length = 3 * N_VISUAL + int(prompt_ids.shape[0])
        query_mask = self._query_mask(item, instruction, prefix_length)
        cache_key = (instruction, int(prompt_ids.shape[0]))
        forward = self._capture_forward.get(cache_key)
        if forward is None:
            forward = nnx_utils.module_jit(self.model._pi05_prefix_attention)
            self._capture_forward[cache_key] = forward
            self._capture_store[cache_key] = []
        store = self._capture_store[cache_key]
        store.clear()
        gemma._PI05_ATTN_CAPTURE = {
            "query_mask": query_mask,
            "key_start": 0,
            "callback": store.append,
            "ordered": False,
        }
        try:
            prefix = forward(observation)
            self.jax.effects_barrier()
        finally:
            gemma._PI05_ATTN_CAPTURE = None
        if len(store) != 18:
            raise RuntimeError(f"captured {len(store)} attention layers, expected 18")
        if prefix.shape[1] != prefix_length:
            raise RuntimeError(
                f"jitted prefix length {prefix.shape[1]} != expected {prefix_length}"
            )
        visual = np.asarray(prefix[0, :N_VISUAL], dtype=np.float32)
        matched_count, groups, matched_ids, entities = self._matched_budget(
            visual, instruction
        )
        scores = np.asarray(store[selector_layer][0], dtype=np.float32)
        selected = stable_top_m(scores, matched_count)
        negative = make_harmonic_negative(visual, selected)
        return self.jnp.asarray(negative[None], dtype=prefix.dtype), {
            "selector_layer_index_0based": selector_layer,
            "selector_layer_label": f"L{selector_layer}",
            "entities": entities,
            "matched_cluster_ids": groups,
            "matched_cluster_token_ids": matched_ids,
            "matched_m_t": matched_count,
            "selected_token_ids": selected,
            "selected_token_count": len(selected),
            "selected_count_equals_matched": len(selected) == matched_count,
            "harmonic_finite": bool(np.isfinite(negative).all()),
            "attention_score_min": float(scores.min()),
            "attention_score_max": float(scores.max()),
        }

    def capture_all_layers(self, raw: dict) -> np.ndarray:
        """Return the instruction→primary-view attention of all 18 layers.

        Shape ``(n_layers, N_VISUAL)``.  Used by the offline layer-selection
        screen; it runs the same tapped prefix pass as the selector but keeps
        every layer instead of the configured one.
        """
        from openpi.shared import nnx_utils
        import openpi.models.gemma as gemma

        instruction = str(raw["prompt"])
        item = self.policy._input_transform(dict(raw))
        batch = self.jax.tree.map(lambda value: self.jnp.asarray(value)[None, ...], item)
        observation = self.model_lib.Observation.from_dict(batch)
        prompt_ids = np.asarray(item["tokenized_prompt"], dtype=np.int32)
        prefix_length = 3 * N_VISUAL + int(prompt_ids.shape[0])
        query_mask = self._query_mask(item, instruction, prefix_length)
        cache_key = (instruction, int(prompt_ids.shape[0]))
        forward = self._capture_forward.get(cache_key)
        if forward is None:
            forward = nnx_utils.module_jit(self.model._pi05_prefix_attention)
            self._capture_forward[cache_key] = forward
            self._capture_store[cache_key] = []
        store = self._capture_store[cache_key]
        store.clear()
        gemma._PI05_ATTN_CAPTURE = {
            "query_mask": query_mask,
            "key_start": 0,
            "callback": store.append,
            "ordered": False,
        }
        try:
            forward(observation)
            self.jax.effects_barrier()
        finally:
            gemma._PI05_ATTN_CAPTURE = None
        if len(store) != 18:
            raise RuntimeError(f"captured {len(store)} attention layers, expected 18")
        return np.stack([np.asarray(layer[0], dtype=np.float32) for layer in store])

    def infer(
        self,
        raw: dict,
        *,
        seed: int = 0,
        guidance_scale: float | None = None,
        noise: np.ndarray | None = None,
        layer_index: int | None = None,
        active_steps: Iterable[int] | None = None,
        num_steps: int | None = None,
    ) -> dict:
        scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        if scale < 0 or not np.isfinite(scale):
            raise ValueError("guidance scale must be finite and nonnegative")
        total_steps = self.num_steps if num_steps is None else int(num_steps)
        if total_steps < 1:
            raise ValueError("num_steps must be positive")
        steps = (
            tuple(range(total_steps))
            if active_steps is None and num_steps is None
            else tuple(sorted({int(step) for step in (active_steps or range(total_steps))}))
        )
        if not steps:
            raise ValueError("active flow steps cannot be empty")
        if any(step < 0 or step >= total_steps for step in steps):
            raise ValueError("active flow step outside num_steps")
        instruction = str(raw["prompt"])
        item = self.policy._input_transform(dict(raw))
        batch = self.jax.tree.map(lambda value: self.jnp.asarray(value)[None, ...], item)
        observation = self.model_lib.Observation.from_dict(batch)
        negative_primary, selector_trace = self._select_negative(
            observation, item, instruction, layer_index=layer_index
        )
        active_mask = np.zeros((total_steps,), dtype=np.bool_)
        active_mask[list(steps)] = True
        action = self._sample_guided(
            self.jax.random.key(seed),
            observation,
            negative_primary,
            self.jnp.asarray(scale, dtype=self.jnp.float32),
            self.jnp.asarray(libero_guided_dim_mask(self.model.action_dim)),
            self.jnp.asarray(active_mask),
            num_steps=total_steps,
            noise=None if noise is None else self.jnp.asarray(noise)[None, ...],
        )
        self.jax.effects_barrier()
        normalized = np.asarray(action[0], dtype=np.float32)
        output = self.policy._output_transform(
            {"state": np.asarray(item["state"]), "actions": normalized}
        )
        actions = np.asarray(output["actions"], dtype=np.float32)
        self.last_trace = {
            **selector_trace,
            "model": "pi0.5",
            "benchmark": "LIBERO",
            "guidance": "v_pos + lambda * (v_pos - v_neg)",
            "guidance_scale": scale,
            "num_flow_steps": total_steps,
            "active_flow_steps": list(steps),
            "guided_action_dims": list(range(LIBERO_GUIDED_DIMS)),
            "gripper_dim": LIBERO_GUIDED_DIMS,
            "padding_dims_unguided": list(range(LIBERO_ACTION_DIMS, self.model.action_dim)),
            "shared_noisy_action_state": True,
            "shared_adarms_timestep_condition": True,
            "output_shape": list(actions.shape),
            "output_finite": bool(np.isfinite(actions).all()),
        }
        return {**output, "pi05_matched_flow_cd_trace": self.last_trace}


    def infer_pcd(
        self,
        raw: dict,
        *,
        seed: int = 0,
        noise: np.ndarray | None = None,
        n_candidates: int = 8,
        alpha: float = 0.5,
        bandwidth_scale: float = 0.5,
        num_steps: int | None = None,
        trace_candidates: bool = False,
    ) -> dict:
        """Generate PCD candidates and resample with clean/CF KDE ratio."""
        if n_candidates < 2:
            raise ValueError("PCD requires at least two candidates")
        if alpha < 0 or not np.isfinite(alpha):
            raise ValueError("PCD alpha must be finite and nonnegative")
        if bandwidth_scale <= 0 or not np.isfinite(bandwidth_scale):
            raise ValueError("PCD bandwidth_scale must be finite and positive")
        total_steps = self.num_steps if num_steps is None else int(num_steps)
        if total_steps < 1:
            raise ValueError("num_steps must be positive")

        instruction = str(raw["prompt"])
        item = self.policy._input_transform(dict(raw))
        batch = self.jax.tree.map(lambda value: self.jnp.asarray(value)[None, ...], item)
        observation = self.model_lib.Observation.from_dict(batch)
        negative_primary, selector_trace = self._select_negative(
            observation, item, instruction
        )
        if noise is None:
            noise = np.asarray(
                self.jax.random.normal(
                    self.jax.random.key(seed),
                    (n_candidates, self.model.action_horizon, self.model.action_dim),
                ),
                dtype=np.float32,
            )
        noise = np.asarray(noise, dtype=np.float32)
        expected = (n_candidates, self.model.action_horizon, self.model.action_dim)
        if noise.shape != expected:
            raise ValueError(f"PCD noise has shape {noise.shape}, expected {expected}")

        result = self._sample_pcd(
            self.jax.random.key(seed + 1),
            observation,
            negative_primary,
            self.jnp.asarray(noise),
            self.jnp.asarray(alpha, dtype=self.jnp.float32),
            self.jnp.asarray(bandwidth_scale, dtype=self.jnp.float32),
            num_steps=total_steps,
        )
        self.jax.effects_barrier()
        (
            action,
            weights,
            chosen_index,
            candidates,
            candidate_cf,
            log_pc,
            log_pn,
            score,
            score_terms,
        ) = result
        normalized = np.asarray(action[0], dtype=np.float32)
        output = self.policy._output_transform(
            {"state": np.asarray(item["state"]), "actions": normalized}
        )
        actions = np.asarray(output["actions"], dtype=np.float32)
        candidate_np = np.asarray(candidates, dtype=np.float32)
        self.last_trace = {
            **selector_trace,
            "method": "pcd_contrastive_resampling",
            "model": "pi0.5",
            "benchmark": "LIBERO",
            "alpha": float(alpha),
            "bandwidth_scale": float(bandwidth_scale),
            "n_candidates": int(n_candidates),
            "num_flow_steps": total_steps,
            "chosen_index": int(np.asarray(chosen_index)),
            "weights": np.asarray(weights, dtype=np.float32).round(6).tolist(),
            "score_terms": int(np.asarray(score_terms)),
            "candidate_spread": float(
                np.linalg.norm(candidate_np - candidate_np[0], axis=(1, 2)).max()
            ),
            "shared_candidate_noise_for_clean_and_cf": True,
            "output_shape": list(actions.shape),
            "output_finite": bool(np.isfinite(actions).all()),
        }
        response = {**output, "pi05_pcd_trace": self.last_trace}
        if trace_candidates:
            response["pi05_vtcg_candidates"] = {
                "noise": noise,
                "actions_clean": candidate_np,
                "actions_cf": np.asarray(candidate_cf, dtype=np.float32),
                "log_pc": np.asarray(log_pc, dtype=np.float32),
                "log_pn": np.asarray(log_pn, dtype=np.float32),
                "normalized_score": np.asarray(score, dtype=np.float32),
                "weights": np.asarray(weights, dtype=np.float32),
                "chosen_index": np.asarray(chosen_index, dtype=np.int32),
            }
        return response
