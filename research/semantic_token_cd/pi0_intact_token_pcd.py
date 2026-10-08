"""Token-SHR + PCD-style action-distribution contrast for INTACT LeRobot Pi0.

The negative condition is made by harmonic reconstruction of prompt-attended
SigLIP patch embeddings at the image-prefix boundary (after the vision tower /
projector and before the PaliGemma language layers).  Contrastive guidance is
applied only after clean and negative action chunks have been sampled: a KDE
approximates both output action distributions and reweights clean candidates by
their PCD likelihood ratio.  No flow/velocity field is combined or modified.

This module is loaded inside ``pi0_intact_runtime`` and wraps an already loaded
LeRobot ``PI0Policy``.  Its ``select_action`` accepts the usual LeRobot batch
and returns one action, preserving Pi0's action-chunk queue semantics.
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


GRID = 16
N_VISUAL = GRID * GRID
KMEANS_K = 8
KMEANS_SEED = 0
KMEANS_N_INIT = 10
DEFAULT_LAYER_INDEX = 2  # Offline screen: Pi0 INTACT BridgeV2 candidate L2 (0-based).
DEFAULT_PCD_ALPHA = 0.2  # PCD paper's best reported alpha for Pi0.
DEFAULT_CANDIDATES = 24  # PCD paper's KDE-PM sample count for Pi0.
EPS = 1e-8

PREPOSITIONS = {
    "into", "in", "on", "onto", "near", "to", "next", "from", "under",
    "over", "behind", "beside", "above", "below", "inside", "at", "with",
}
DETERMINERS = {"the", "a", "an"}


def extract_entities(instruction: str) -> list[str]:
    """Return the deduplicated source/target noun phrases used by canonical SHR."""
    words = instruction.lower().replace("_", " ").split()
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


def stable_top_m(scores: np.ndarray, m: int) -> list[int]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (N_VISUAL,) or not np.isfinite(scores).all():
        raise ValueError(f"expected {N_VISUAL} finite prompt-attention scores, got {scores.shape}")
    if not 1 <= int(m) <= N_VISUAL:
        raise ValueError(f"invalid matched coverage m={m}")
    order = np.lexsort((np.arange(N_VISUAL), -scores))
    return sorted(int(i) for i in order[: int(m)])


def harmonic_reconstruct(features: np.ndarray, selected: list[int]) -> np.ndarray:
    """Canonical beta=0, 4-neighbour Dirichlet reconstruction on a 16x16 grid."""
    region = np.asarray(sorted(set(int(i) for i in selected)), dtype=np.int64)
    if not region.size:
        raise ValueError("cannot reconstruct an empty visual-token set")
    if features.ndim != 2 or features.shape[0] != N_VISUAL:
        raise ValueError(f"expected image-prefix features [256,D], got {features.shape}")
    position = {int(token): row for row, token in enumerate(region)}
    clean = np.asarray(features, dtype=np.float64)
    matrix = np.zeros((region.size, region.size), dtype=np.float64)
    rhs = np.zeros((region.size, clean.shape[1]), dtype=np.float64)
    for row, token in enumerate(region):
        r, c = divmod(int(token), GRID)
        neighbours = []
        if r > 0: neighbours.append(token - GRID)
        if r + 1 < GRID: neighbours.append(token + GRID)
        if c > 0: neighbours.append(token - 1)
        if c + 1 < GRID: neighbours.append(token + 1)
        matrix[row, row] = len(neighbours)
        for neighbour in neighbours:
            col = position.get(int(neighbour))
            if col is None:
                rhs[row] += clean[neighbour]
            else:
                matrix[row, col] -= 1.0
    try:
        result = np.linalg.solve(matrix, rhs).astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("singular harmonic reconstruction system") from exc
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite harmonic reconstruction")
    return result


def _digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _kde_log_densities(
    candidates: np.ndarray, samples: np.ndarray, bandwidth_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Gaussian KDE over chunk-time/action coordinates.

    ``candidates`` and ``samples`` both have shape [N, action_steps, action_dim].
    Scott/Silverman univariate bandwidth is used because the PCD paper specifies
    Gaussian KDE but does not publish a numeric bandwidth in its main method.
    The returned log density is a sum over all valid scalar action coordinates.
    """
    x = np.asarray(candidates, dtype=np.float64)
    ref = np.asarray(samples, dtype=np.float64)
    if x.ndim != 3 or ref.ndim != 3 or x.shape[1:] != ref.shape[1:]:
        raise ValueError(f"KDE expects compatible [N,H,A] arrays, got {x.shape} and {ref.shape}")
    n = len(ref)
    if n < 2 or not np.isfinite(x).all() or not np.isfinite(ref).all():
        raise ValueError("KDE needs at least two finite candidate actions")
    dims = int(np.prod(ref.shape[1:]))
    xf, rf = x.reshape(len(x), dims), ref.reshape(n, dims)
    total = np.zeros(len(x), dtype=np.float64)
    bandwidths = np.zeros(dims, dtype=np.float64)
    log_norm_const = 0.5 * np.log(2.0 * np.pi)
    for d in range(dims):
        values = rf[:, d]
        scale = max(float(values.std(ddof=1)), 1e-3)
        # Normal-reference bandwidth in one dimension.
        bandwidth = max(1e-3, 0.9 * scale * n ** (-0.2) * float(bandwidth_scale))
        bandwidths[d] = bandwidth
        z = (xf[:, d, None] - values[None, :]) / bandwidth
        log_kernel = -0.5 * z * z - log_norm_const - np.log(bandwidth)
        maximum = log_kernel.max(axis=1, keepdims=True)
        total += (maximum[:, 0] + np.log(np.exp(log_kernel - maximum).sum(axis=1)) - np.log(n))
    return total, bandwidths


@dataclass
class BranchCache:
    prefix_pad_masks: torch.Tensor
    past_key_values: dict[int, dict[str, torch.Tensor]]


class Pi0IntactTokenPCDInference:
    """Wrap a loaded LeRobot ``PI0Policy`` with matched token-SHR and KDE PCD."""

    def __init__(
        self,
        base_policy,
        *,
        layer_index: int = DEFAULT_LAYER_INDEX,
        pcd_alpha: float = DEFAULT_PCD_ALPHA,
        n_candidates: int = DEFAULT_CANDIDATES,
        kde_bandwidth_scale: float = 1.0,
        seed: int = 0,
        kmeans_k: int = KMEANS_K,
        kmeans_seed: int = KMEANS_SEED,
    ) -> None:
        self.policy = base_policy
        self.layer_index = int(layer_index)
        self.pcd_alpha = float(pcd_alpha)
        self.n_candidates = int(n_candidates)
        self.kde_bandwidth_scale = float(kde_bandwidth_scale)
        self.seed = int(seed)
        self.kmeans_k = int(kmeans_k)
        self.kmeans_seed = int(kmeans_seed)
        self.step_index = 0
        self.action_queue: deque[torch.Tensor] = deque(maxlen=base_policy.config.n_action_steps)
        self.last_trace: dict[str, Any] | None = None
        self._phrase_cache: dict[str, np.ndarray] = {}
        self._entities: list[str] = []
        self._entity_vectors: list[np.ndarray] = []
        self._instruction: str | None = None
        if not 0 <= self.layer_index < 18:
            raise ValueError("PaliGemma candidate layer must be in [0, 17]")
        if self.pcd_alpha < 0 or not np.isfinite(self.pcd_alpha):
            raise ValueError("PCD alpha must be finite and nonnegative")
        if self.n_candidates < 2:
            raise ValueError("KDE-PM requires at least two action candidates")
        if self.kde_bandwidth_scale <= 0:
            raise ValueError("KDE bandwidth scale must be positive")
        # Avoid this host's unsupported cuDNN SDPA plan in the SigLIP tower.
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)

    def reset(self, task_description: str | None = None, seed: int | None = None) -> None:
        self.policy.reset()
        self.action_queue.clear()
        self.step_index = 0
        self.last_trace = None
        if seed is not None:
            self.seed = int(seed)
        if task_description is not None:
            self._instruction = task_description
            self._entities = extract_entities(task_description)
            self._entity_vectors = [self._embed_phrase(x) for x in self._entities]

    def eval(self):
        self.policy.eval()
        return self

    def __getattr__(self, name):
        # Expose config/device and other benign LeRobot policy attributes.
        if name == "policy":
            raise AttributeError(name)
        return getattr(self.policy, name)

    def _embed_phrase(self, phrase: str) -> np.ndarray:
        if phrase in self._phrase_cache:
            return self._phrase_cache[phrase]
        tokenizer = self.policy.language_tokenizer
        ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tokenizer(phrase, add_special_tokens=True)["input_ids"]
        tokens = torch.tensor([ids], dtype=torch.long, device=next(self.policy.parameters()).device)
        with torch.inference_mode():
            vectors = self.policy.model.paligemma_with_expert.embed_language_tokens(tokens)[0]
        vector = vectors.mean(dim=0).detach().float().cpu().numpy()
        self._phrase_cache[phrase] = vector
        return vector

    def _select_matched_budget(self, image_features: np.ndarray) -> tuple[int, list[int], list[float]]:
        from sklearn.cluster import KMeans

        features = np.asarray(image_features, dtype=np.float64)
        labels = KMeans(
            n_clusters=self.kmeans_k,
            random_state=self.kmeans_seed,
            n_init=KMEANS_N_INIT,
        ).fit_predict(features)
        group_vectors = np.stack([features[labels == group].mean(axis=0) for group in range(self.kmeans_k)])
        group_vectors /= np.linalg.norm(group_vectors, axis=1, keepdims=True) + EPS
        entities = self._entities
        if not entities:
            raise RuntimeError("reset(task_description) must be called before action selection")
        groups: list[int] = []
        scores: list[float] = []
        for entity, vector in zip(entities, self._entity_vectors, strict=True):
            normalized = vector / (np.linalg.norm(vector) + EPS)
            similarities = group_vectors @ normalized
            group = int(np.argmax(similarities))
            scores.append(float(similarities[group]))
            if group not in groups:
                groups.append(group)
        reference = sorted(int(i) for group in groups for i in np.flatnonzero(labels == group))
        if not reference:
            raise RuntimeError("canonical Matched budget selection returned no visual patches")
        return len(reference), groups, scores

    def _query_positions(self, task: str, lang_tokens: torch.Tensor, lang_masks: torch.Tensor) -> list[int]:
        ids = lang_tokens[0].detach().cpu().tolist()
        task_ids = self.policy.language_tokenizer(task, add_special_tokens=False)["input_ids"]
        if not task_ids:
            raise RuntimeError(f"tokenizer returned no tokens for task {task!r}")
        starts = [i for i in range(len(ids) - len(task_ids) + 1) if ids[i:i + len(task_ids)] == task_ids]
        if not starts:
            raise RuntimeError(f"cannot locate task phrase in Pi0 tokens: {task!r}")
        start = starts[-1]
        stop = start + len(task_ids)
        if stop > int(lang_masks[0].sum().item()):
            raise RuntimeError("task-query positions overlap tokenizer padding")
        return list(range(start, stop))

    def _prefill(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        *,
        query_positions: list[int] | None = None,
    ) -> tuple[BranchCache, np.ndarray | None]:
        from lerobot.common.policies.pi0.modeling_pi0 import make_att_2d_masks

        model = self.policy.model
        pg = model.paligemma_with_expert
        attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        captured: list[np.ndarray] = []
        original_attention = pg.eager_attention_forward
        if query_positions is not None:
            num_layers = int(pg.config.paligemma_config.text_config.num_hidden_layers)
            if not 0 <= self.layer_index < num_layers:
                raise RuntimeError(f"selected layer {self.layer_index} outside model depth {num_layers}")
            layer_index = self.layer_index
            call_index = [0]

            def capture_attention(this, mask, batch_size, head_dim, query_states, key_states, value_states):
                index = call_index[0]
                call_index[0] += 1
                num_heads = this.config.paligemma_config.text_config.num_attention_heads
                kv_heads = this.config.paligemma_config.text_config.num_key_value_heads
                groups = num_heads // kv_heads
                seq_len = key_states.shape[1]
                keys = key_states[:, :, :, None, :].expand(batch_size, seq_len, kv_heads, groups, head_dim)
                keys = keys.reshape(batch_size, seq_len, num_heads, head_dim)
                q = query_states.float().transpose(1, 2)
                k = keys.float().transpose(1, 2)
                logits = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
                if mask.ndim == 3:
                    mask = mask[:, None, :, :]
                probs = torch.softmax(torch.where(mask, logits, -2.3819763e38), dim=-1)
                if index == layer_index:
                    qidx = torch.as_tensor(query_positions, device=probs.device, dtype=torch.long) + N_VISUAL
                    captured.append(probs[0, :, qidx, :N_VISUAL].float().mean(dim=(0, 1)).detach().cpu().numpy())
                values = value_states[:, :, :, None, :].expand(batch_size, seq_len, kv_heads, groups, head_dim)
                values = values.reshape(batch_size, seq_len, num_heads, head_dim)
                output = torch.matmul(probs.to(values.dtype), values.permute(0, 2, 1, 3))
                return output.permute(0, 2, 1, 3).reshape(batch_size, -1, kv_heads * groups * head_dim)

            pg.eager_attention_forward = capture_attention.__get__(pg, type(pg))
        try:
            with torch.inference_mode():
                _, cache = pg.forward(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                    fill_kv_cache=True,
                )
        finally:
            pg.eager_attention_forward = original_attention
        scores = captured[0].astype(np.float32) if captured else None
        if query_positions is not None and (scores is None or scores.shape != (N_VISUAL,)):
            raise RuntimeError(f"prompt-attention capture failed: {None if scores is None else scores.shape}")
        return BranchCache(prefix_pad_masks=prefix_pad_masks, past_key_values=cache), scores

    def _branch_actions(
        self,
        branch: BranchCache,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate Pi0's clean or negative conditional flow independently."""
        model = self.policy.model
        count = noise.shape[0]
        pad_masks = branch.prefix_pad_masks.expand(count, -1)
        state_batch = state.expand(count, -1)
        cache = {
            layer: {key: value.expand(count, *value.shape[1:]) for key, value in layer_cache.items()}
            for layer, layer_cache in branch.past_key_values.items()
        }
        x = noise.clone()
        dt = torch.tensor(-1.0 / self.policy.config.num_steps, dtype=torch.float32, device=x.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        while time >= -dt / 2:
            v = model.denoise_step(state_batch, pad_masks, cache, x, time.expand(count))
            # This is vanilla flow integration within each conditioning branch.
            # Positive and negative velocities are never mixed or contrasted.
            x = x + dt * v
            time = time + dt
        action_dim = int(self.policy.config.action_feature.shape[0])
        return x[:, : self.policy.config.n_action_steps, :action_dim]

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, Any], *, seed: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return one unnormalized action chunk and its PCD/mask audit trace."""
        from lerobot.common.constants import OBS_ROBOT
        from lerobot.common.policies.pi0.modeling_pi0 import make_att_2d_masks

        policy = self.policy
        model = policy.model
        batch = dict(batch)
        if policy.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = policy._pi_aloha_decode_state(batch[OBS_ROBOT].clone())
        batch = policy.normalize_inputs(batch)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens, lang_masks = policy.prepare_language(batch)
        task_values = batch["task"]
        if isinstance(task_values, str):
            task_values = [task_values]
        if len(task_values) != 1:
            raise ValueError("Pi0 token-PCD wrapper currently supports batch size 1")
        task = task_values[0].strip()
        if task != self._instruction:
            self._instruction = task
            self._entities = extract_entities(task)
            self._entity_vectors = [self._embed_phrase(x) for x in self._entities]

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if prefix_embs.shape[1] < N_VISUAL or prefix_embs.shape[1] != N_VISUAL + lang_tokens.shape[1]:
            raise RuntimeError(f"Pi0 image/language prefix layout changed: {tuple(prefix_embs.shape)}")
        clean_features = prefix_embs[0, :N_VISUAL].detach().float().cpu().numpy()
        m, group_ids, entity_scores = self._select_matched_budget(clean_features)
        prompt_positions = self._query_positions(task, lang_tokens, lang_masks)
        clean_branch, attention = self._prefill(
            prefix_embs, prefix_pad_masks, prefix_att_masks, query_positions=prompt_positions
        )
        selected = stable_top_m(attention, m)
        reconstructed = harmonic_reconstruct(clean_features, selected)
        negative_prefix = prefix_embs.clone()
        negative_prefix[0, selected] = torch.as_tensor(reconstructed, device=prefix_embs.device, dtype=prefix_embs.dtype)
        keep = np.ones(N_VISUAL, dtype=bool)
        keep[selected] = False
        if not np.array_equal(clean_features[keep], negative_prefix[0, :N_VISUAL].detach().float().cpu().numpy()[keep]):
            raise RuntimeError("negative prefix changed a non-selected visual token")
        negative_branch, _ = self._prefill(negative_prefix, prefix_pad_masks, prefix_att_masks)

        action_seed = self.seed + self.step_index if seed is None else int(seed)
        generator = torch.Generator(device=prefix_embs.device).manual_seed(action_seed)
        noise_shape = (self.n_candidates, self.policy.config.chunk_size, self.policy.config.max_action_dim)
        noise = torch.randn(noise_shape, generator=generator, device=prefix_embs.device, dtype=torch.float32)
        positive_actions = self._branch_actions(clean_branch, state, noise)
        negative_actions = self._branch_actions(negative_branch, state, noise)
        pos_np = positive_actions.detach().float().cpu().numpy()
        neg_np = negative_actions.detach().float().cpu().numpy()
        logp_pos, bandwidths = _kde_log_densities(pos_np, pos_np, self.kde_bandwidth_scale)
        logp_neg, _ = _kde_log_densities(pos_np, neg_np, self.kde_bandwidth_scale)
        log_ratio = logp_pos - logp_neg
        log_weights = self.pcd_alpha * log_ratio
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        weights /= weights.sum()
        selected_candidate = int(np.random.default_rng(action_seed ^ 0x504344).choice(self.n_candidates, p=weights))
        chosen = positive_actions[selected_candidate : selected_candidate + 1]
        # The base policy's output normalizer/robot decoding is preserved.
        chosen = chosen[:, : policy.config.n_action_steps]
        chosen = policy.unnormalize_outputs({"action": chosen})["action"]
        if policy.config.adapt_to_pi_aloha:
            chosen = policy._pi_aloha_encode_actions(chosen)
        if not torch.isfinite(chosen).all():
            raise FloatingPointError("Pi0 token-PCD produced non-finite actions")
        selected_hash = _digest_array(np.asarray(selected, dtype=np.int32))
        self.last_trace = {
            "protocol_id": "PI0_INTACT_TOKEN_SHR_PCD_V1",
            "layer_index_0based": self.layer_index,
            "layer_label": f"L{self.layer_index}",
            "image_token_count": N_VISUAL,
            "query_positions_in_language": prompt_positions,
            "matched_budget_group_ids": group_ids,
            "matched_entity_scores": entity_scores,
            "m_t": m,
            "selected_token_ids": selected,
            "selected_token_ids_sha256": selected_hash,
            "attention_scores_sha256": _digest_array(attention),
            "harmonic_reconstruction_finite": bool(np.isfinite(reconstructed).all()),
            "unselected_visual_features_exact": True,
            "guidance_method": "PCD Eq.(2) action-distribution ratio with KDE-PM over candidate action chunks",
            "flow_velocity_mixing": False,
            "pcd_alpha": self.pcd_alpha,
            "candidate_count": self.n_candidates,
            "paired_common_noise": True,
            "action_distribution_factorization": "independent KDE for each action time-index x action dimension",
            "kde_bandwidth_rule": "Gaussian univariate normal-reference 0.9*std*N^-0.2; bandwidth multiplier logged separately",
            "kde_bandwidth_scale": self.kde_bandwidth_scale,
            "kde_bandwidth_mean": float(np.mean(bandwidths)),
            "pcd_candidate_weights": weights.astype(float).tolist(),
            "chosen_positive_candidate_index": selected_candidate,
            "chosen_actions_finite": bool(torch.isfinite(chosen).all()),
            "positive_candidates_sha256": _digest_array(pos_np),
            "negative_candidates_sha256": _digest_array(neg_np),
            "episode_control_step": self.step_index,
        }
        self.step_index += 1
        return chosen, dict(self.last_trace)

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Any], noise: torch.Tensor | None = None) -> torch.Tensor:
        """LeRobot-style one-action interface with Pi0's action chunk queue."""
        if noise is not None:
            raise ValueError("explicit single-noise Pi0 input is incompatible with KDE-PM's candidate set")
        self.policy.eval()
        if not self.action_queue:
            actions, _ = self.predict_action_chunk(batch)
            self.action_queue.extend(actions.transpose(0, 1))
        return self.action_queue.popleft()
