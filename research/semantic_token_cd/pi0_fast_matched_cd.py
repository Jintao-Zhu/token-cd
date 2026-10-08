"""L6 Prompt-Attention + matched harmonic contrastive decoding for pi0-FAST.

This is an OpenPI adapter, not a change to the OpenPI checkout.  It preserves
the SIMPLER Pi0-FAST input/output transforms and replaces only greedy FAST
token decoding with a shared-prefix clean/negative contrastive decode.
"""
from __future__ import annotations

import json
import os
import sys
import time
import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
CHECKPOINT = ROOT / "checkpoints/openpi_fast_finetuned/fractal"
FRACTAL_STATS = ROOT / "checkpoints/openpi_fast_finetuned/fractal/assets/local/fractal_lerobot/norm_stats.json"
LAYER_INDEX = 6  # Existing project convention: zero-based L-index.
GRID = 16
N_VISUAL = GRID * GRID
LAMBDA = 0.5
KMEANS_GROUPS = 8
KMEANS_SEED = 0
KMEANS_BACKEND = ""


def _kmeans_labels(features: np.ndarray) -> tuple[np.ndarray, str]:
    """Use project-standard sklearn KMeans, with a SciPy-only smoke fallback."""
    try:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=KMEANS_GROUPS, random_state=KMEANS_SEED, n_init=10).fit(features).labels_
        return np.asarray(labels, dtype=np.int32), "sklearn.KMeans(n_init=10)"
    except ModuleNotFoundError as exc:
        if exc.name != "sklearn":
            raise
        # pi05 currently lacks scikit-learn. This fallback keeps technical
        # inference usable while preserving K=8, k-means++, Lloyd updates,
        # ten deterministic starts, and minimum-inertia selection. Scientific
        # runs can install sklearn to use the exact historic implementation.
        from scipy.cluster.vq import kmeans2

        best = None
        best_inertia = np.inf
        for trial in range(10):
            with np.errstate(all="ignore"):
                centers, labels = kmeans2(
                    features, KMEANS_GROUPS, iter=300, minit="++", missing="warn",
                    check_finite=True, seed=KMEANS_SEED + trial,
                )
            if np.any(np.bincount(labels, minlength=KMEANS_GROUPS) == 0):
                continue
            inertia = float(np.sum((features - centers[labels]) ** 2))
            if inertia < best_inertia:
                best, best_inertia = labels, inertia
        if best is None:
            raise RuntimeError("all SciPy KMeans starts produced an empty cluster")
        return np.asarray(best, dtype=np.int32), "scipy.cluster.vq.kmeans2(10 deterministic starts; smoke fallback)"


def _pi0_fast_cd_decode_loop(model, rng, positive_logits, negative_logits, positive_cache,
                             negative_cache, prefill_len, prefix_start, prefill_size,
                             fast_token_min, fast_token_max, pipe_token, header_tokens,
                             char_lengths, payload_chars):
    """NNX-model-bound greedy contrastive loop over FAST action tokens."""
    import jax
    import jax.numpy as jnp

    max_steps = 256
    cache_len = prefill_size + max_steps
    tokens = jnp.zeros((positive_logits.shape[0], max_steps), dtype=jnp.float32)
    # Keep the OpenPI serialization header intact; only score valid FAST action
    # code IDs plus the formatting terminators during guided decoding.
    header_len = header_tokens.shape[0]
    tokens = tokens.at[:, :header_len].set(jnp.broadcast_to(header_tokens[None, :], (tokens.shape[0], header_len)))
    first_pos, first_neg = positive_logits, negative_logits

    def cond(carry):
        _, _, _, _, _, _, all_eos, step, _ = carry
        return jnp.logical_and(jnp.logical_not(all_eos), step < max_steps)

    def body(carry):
        rng, last_pos, last_neg, output, pos_cache, neg_cache, _, step, char_count = carry
        guided = (1.0 + LAMBDA) * last_pos - LAMBDA * last_neg
        vocab_ids = jnp.arange(guided.shape[-1])
        code_ids = fast_token_max - vocab_ids
        in_fast_vocab = jnp.logical_and(vocab_ids >= fast_token_min, vocab_ids <= fast_token_max)
        safe_code_ids = jnp.clip(code_ids, 0, char_lengths.shape[0] - 1)
        token_char_len = char_lengths[safe_code_ids]
        within_payload = jnp.logical_and(token_char_len > 0, token_char_len <= payload_chars - char_count)
        valid_ids = jnp.logical_or(
            jnp.logical_and(in_fast_vocab, within_payload),
            jnp.logical_and(vocab_ids == pipe_token, char_count == payload_chars),
        )
        guided = jnp.where(valid_ids[None, :], guided, -jnp.inf)
        token = jnp.argmax(guided, axis=-1).astype(jnp.int32)[:, None]
        output = output.at[:, step].set(token[:, 0].astype(output.dtype))
        all_eos = jnp.all(jnp.any(token == pipe_token, axis=-1))
        chosen_code = jnp.clip(fast_token_max - token[0, 0], 0, char_lengths.shape[0] - 1)
        chosen_len = char_lengths[chosen_code]
        char_count = char_count + jnp.where(token[0, 0] == pipe_token, 0, chosen_len)
        token_embedding = model.PaliGemma.llm(token, embed_only=True)
        positions = prefill_len[:, None] + step + 1
        columns = jnp.arange(cache_len)[None, None, :]
        mask = jnp.logical_and(
            columns >= prefix_start[:, None, None],
            columns < jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1)),
        )
        p_logits, pos_cache, _ = model.PaliGemma.llm(
            embedded_prefix=token_embedding, mask=mask, positions=positions,
            decode=True, kv_cache=pos_cache,
        )
        n_logits, neg_cache, _ = model.PaliGemma.llm(
            embedded_prefix=token_embedding, mask=mask, positions=positions,
            decode=True, kv_cache=neg_cache,
        )
        return rng, p_logits[:, -1], n_logits[:, -1], output, pos_cache, neg_cache, all_eos, step + 1, char_count

    initial = (
        rng, positive_logits, negative_logits, tokens, positive_cache, negative_cache,
        jnp.asarray(False), jnp.asarray(header_len, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32),
    )
    result = jax.lax.while_loop(cond, body, initial)
    return result[3], result[-1], first_pos, first_neg


def _extract_entities(instruction: str) -> list[str]:
    """Same simple source/target phrase parser used by the semantic CD work."""
    prepositions = {
        "into", "in", "on", "onto", "near", "to", "next", "from", "under",
        "over", "behind", "beside", "above", "below", "inside", "at", "with",
    }
    determiners = {"the", "a", "an"}
    words = instruction.lower().replace("_", " ").split()
    if not words:
        raise ValueError("empty task instruction")
    rest = words[1:]
    prep_positions = [i for i, word in enumerate(rest) if word in prepositions]
    if not prep_positions:
        entity = " ".join(word for word in rest if word not in determiners)
        return [entity] if entity else []
    first, last = prep_positions[0], prep_positions[-1]
    source = " ".join(word for word in rest[:first] if word not in determiners)
    target = " ".join(word for word in rest[last + 1 :] if word not in determiners)
    entities = []
    for entity in (source, target):
        if entity and entity not in entities:
            entities.append(entity)
    return entities or [" ".join(rest)]


def _stable_top_m(scores: np.ndarray, m: int) -> list[int]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (N_VISUAL,) or not np.isfinite(scores).all():
        raise ValueError(f"invalid L6 attention scores {scores.shape}")
    if not 1 <= m <= N_VISUAL:
        raise ValueError(f"invalid matched budget m={m}")
    return sorted(int(i) for i in np.lexsort((np.arange(N_VISUAL), -scores))[:m])


def _harmonic_reconstruct(features: np.ndarray, region: list[int]) -> np.ndarray:
    """Canonical 16x16 four-neighbour Dirichlet solve (beta=0)."""
    region = np.asarray(sorted(set(int(i) for i in region)), dtype=np.int64)
    if region.size == 0:
        raise ValueError("empty harmonic region")
    position = {int(token): row for row, token in enumerate(region)}
    clean = np.asarray(features, dtype=np.float64)
    A = np.zeros((region.size, region.size), dtype=np.float64)
    rhs = np.zeros((region.size, clean.shape[1]), dtype=np.float64)
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
        A[row, row] = len(neighbors)
        for neighbor in neighbors:
            col = position.get(int(neighbor))
            if col is None:
                rhs[row] += clean[neighbor]
            else:
                A[row, col] -= 1.0
    reconstructed = np.linalg.solve(A, rhs).astype(np.float32)
    if not np.isfinite(reconstructed).all():
        raise FloatingPointError("non-finite harmonic reconstruction")
    return reconstructed


def _norm_stats():
    from openpi.shared.normalize import NormStats

    source = json.loads(FRACTAL_STATS.read_text())["norm_stats"]
    return {
        key: NormStats(
            mean=np.asarray(value["mean"], np.float32),
            std=np.asarray(value["std"], np.float32),
            q01=np.asarray(value["q01"], np.float32),
            q99=np.asarray(value["q99"], np.float32),
        )
        for key, value in source.items()
        if key in ("state", "actions")
    }


class Pi0FastMatchedCDPolicy:
    """One-sample greedy autoregressive L6-Matched-CD adapter."""

    def __init__(
        self,
        *,
        gpu: int = 1,
        checkpoint: Path = CHECKPOINT,
        stats_path: Path = FRACTAL_STATS,
        max_decoding_steps: int = 256,
    ):
        # Install the attention tap before importing OpenPI.  The copy lives in
        # a temporary overlay; upstream OpenPI source remains untouched.
        from research.semantic_token_cd.pi0_fast_layer_selection_offline import instrument_openpi

        overlay = instrument_openpi()
        sys.path.insert(0, str(ROOT / "openpi/src"))
        sys.path.insert(0, str(overlay))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        import jax
        import jax.numpy as jnp
        from openpi import transforms
        from openpi.models import model as model_lib
        from openpi.models import pi0_fast as pi0_fast_lib
        from openpi.models import gemma_fast
        from openpi.policies import policy_config
        from openpi.shared import nnx_utils
        from openpi.training import config as config_lib

        self.jax, self.jnp = jax, jnp
        self.model_lib, self.pi0_fast_lib = model_lib, pi0_fast_lib
        self.gemma_fast = gemma_fast
        self.max_decoding_steps = int(max_decoding_steps)
        self.lambda_ = float(LAMBDA)
        # The public Fractal checkpoint is trained with the Fractal data
        # pipeline (see its model card's `pi0_fast_fractal_fft_pt_tokenizer`
        # config).  The local OpenPI checkout does not carry that custom config,
        # so reuse the architecture-compatible pi0_fast_libero model config,
        # but explicitly disable LIBERO's extra absolute->delta action
        # transform.  For PI0_FAST, LiberoInputs/Outputs are inference-equivalent
        # to the checkpoint's FractalInputs/Outputs (same single RGB view,
        # zero padded views, and first-7-action output); the extra delta
        # transform is not equivalent and corrupts Fractal action semantics.
        cfg = config_lib.get_config("pi0_fast_libero")
        if not hasattr(cfg.data, "extra_delta_transform"):
            raise RuntimeError("Expected pi0_fast_libero data config with an explicit delta-transform switch")
        cfg = dataclasses.replace(
            cfg,
            data=dataclasses.replace(cfg.data, extra_delta_transform=False),
        )
        logging.info(
            "Using Fractal-compatible pi0-FAST pipeline: model=%s action_dim=%d horizon=%d "
            "extra_delta_transform=%s; quantile_norm=%s",
            cfg.name,
            cfg.model.action_dim,
            cfg.model.action_horizon,
            cfg.data.extra_delta_transform,
            cfg.model.model_type != model_lib.ModelType.PI0,
        )
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "observation/primary_image",
                        "observation/wrist_image": "observation/wrist_image",
                        "observation/state": "observation/state",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        self.policy = policy_config.create_trained_policy(
            cfg,
            checkpoint,
            repack_transforms=repack,
            norm_stats=_norm_stats() if stats_path == FRACTAL_STATS else self._load_stats(stats_path),
        )
        self.model = self.policy._model
        self.cfg = cfg
        self.overlay = overlay
        self.last_trace: dict[str, Any] = {}
        self.tokenizer = next(
            transform.tokenizer
            for transform in self.policy._input_transform.transforms
            if hasattr(transform, "tokenizer") and hasattr(transform.tokenizer, "_paligemma_tokenizer")
        )
        # module_jit requires a method bound to an NNX.Module. Bind this
        # adapter function to the actual Pi0FAST instance, not this wrapper.
        model_type = type(self.model)
        setattr(model_type, "_pi0_fast_cd_decode_loop", _pi0_fast_cd_decode_loop)
        # The final prefill-size argument determines the fixed cache length.
        self._compiled_decode = nnx_utils.module_jit(
            self.model._pi0_fast_cd_decode_loop, static_argnums=(8,)
        )
        pg = self.tokenizer._paligemma_tokenizer
        fast = self.tokenizer._fast_tokenizer
        fast_vocab_size = getattr(fast, "vocab_size", None)
        if fast_vocab_size is None:
            fast_vocab_size = len(fast)
        fast_vocab_size = int(fast_vocab_size)
        self.fast_token_min = int(pg.vocab_size() - self.tokenizer._fast_skip_tokens - fast_vocab_size)
        self.fast_token_max = int(pg.vocab_size() - 1 - self.tokenizer._fast_skip_tokens)
        bpe = fast.bpe_tokenizer
        self.fast_token_char_lengths = np.asarray(
            [len(bpe.decode([code])) for code in range(fast_vocab_size)], dtype=np.int32
        )
        if not np.any(self.fast_token_char_lengths > 0):
            raise RuntimeError("FAST tokenizer has no nonempty action-code pieces")
        self.fast_payload_chars = int(cfg.model.action_horizon * cfg.model.action_dim)
        pipe_tokens = pg.encode("|", add_bos=False)
        if len(pipe_tokens) != 1:
            raise RuntimeError(f"expected one SentencePiece token for FAST delimiter '|', got {pipe_tokens}")
        self.action_header_tokens = tuple(int(x) for x in pg.encode("Action: ", add_bos=False))
        self.pipe_token = int(pipe_tokens[0])

    @staticmethod
    def _load_stats(path: Path):
        from openpi.shared.normalize import NormStats

        source = json.loads(path.read_text())["norm_stats"]
        return {
            key: NormStats(
                mean=np.asarray(value["mean"], np.float32),
                std=np.asarray(value["std"], np.float32),
                q01=np.asarray(value["q01"], np.float32),
                q99=np.asarray(value["q99"], np.float32),
            )
            for key, value in source.items()
            if key in ("state", "actions")
        }

    def _task_query_mask(self, item: dict, prompt: str, sequence_len: int, image_prefix_len: int) -> np.ndarray:
        state = np.asarray(item["state"], dtype=np.float32)
        discrete = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        cleaned = prompt.lower().strip().replace("_", " ")
        prefix_text = f"Task: {cleaned}, State: {' '.join(map(str, discrete))};\n"
        proto = self.tokenizer._paligemma_tokenizer.encode_as_immutable_proto(prefix_text)
        task_start = len("Task: ")
        task_end = task_start + len(cleaned)
        token_positions = []
        position = 1  # FASTTokenizer prepends the PaliGemma BOS token.
        for piece in proto.pieces:
            if piece.begin >= task_start and piece.end <= task_end:
                token_positions.append(position)
            position += 1
        if not token_positions:
            ids = self.tokenizer._paligemma_tokenizer.encode(cleaned)
            token_ids = np.asarray(item["tokenized_prompt"]).tolist()
            for start in range(len(token_ids) - len(ids) + 1):
                if token_ids[start : start + len(ids)] == ids:
                    token_positions = list(range(start, start + len(ids)))
                    break
        if not token_positions:
            raise RuntimeError(f"could not locate task text tokens for {prompt!r}")
        query = np.zeros((sequence_len,), dtype=np.float32)
        for pos in token_positions:
            absolute = image_prefix_len + int(pos)
            if absolute < sequence_len:
                query[absolute] = 1.0
        return query

    def _entity_embedding(self, phrase: str) -> np.ndarray:
        ids = self.tokenizer._paligemma_tokenizer.encode(phrase, add_bos=False)
        if not ids:
            ids = self.tokenizer._paligemma_tokenizer.encode(phrase, add_bos=True)
        embedded = self.model.PaliGemma.llm(self.jnp.asarray([ids], dtype=self.jnp.int32), embed_only=True)
        return np.asarray(embedded[0].mean(axis=0), dtype=np.float32)

    def _matched_budget(self, visual: np.ndarray, prompt: str) -> tuple[int, list[int], list[int], list[str]]:
        entities = _extract_entities(prompt)
        model_features = np.asarray(visual, dtype=np.float64)
        labels, backend = _kmeans_labels(model_features)
        global KMEANS_BACKEND
        KMEANS_BACKEND = backend
        group_means = np.zeros((KMEANS_GROUPS, model_features.shape[1]), dtype=np.float64)
        for group in range(KMEANS_GROUPS):
            members = np.flatnonzero(labels == group)
            if members.size:
                group_means[group] = model_features[members].mean(axis=0)
        group_means /= np.linalg.norm(group_means, axis=1, keepdims=True) + 1e-8
        groups = []
        for entity in entities:
            emb = self._entity_embedding(entity).astype(np.float64)
            emb /= np.linalg.norm(emb) + 1e-8
            group = int(np.argmax(group_means @ emb))
            if group not in groups:
                groups.append(group)
        selected = sorted(int(i) for g in groups for i in np.flatnonzero(labels == g))
        if not selected:
            raise RuntimeError("matched KMeans/entity budget selected no visual tokens")
        return len(selected), groups, selected, entities

    def infer(self, raw: dict, *, seed: int = 0) -> dict:
        """Return the normal OpenPI output dictionary plus a matched-CD trace."""
        import jax
        import jax.numpy as jnp

        timing = os.environ.get("PI0FAST_TIMING") == "1"
        started = time.monotonic()

        def mark(label: str) -> None:
            if timing:
                print(f"PI0FAST_TIMING {label} +{time.monotonic() - started:.3f}s", flush=True)

        prompt = str(raw["prompt"])
        item = self.policy._input_transform(dict(raw))
        mark("input_transform")
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], item)
        observation = self.model_lib.Observation.from_dict(batch)
        observation = self.model_lib.preprocess_observation(
            jax.random.key(seed), observation, train=False, image_keys=list(observation.images)
        )
        mark("preprocess_observation")
        prefix, prefix_mask, prefix_ar = self.model.embed_inputs(observation)
        mark("embed_inputs")
        if prefix.shape[1] < 3 * N_VISUAL + 1:
            raise RuntimeError(f"unexpected Pi0-FAST prefix length {prefix.shape}")
        image_prefix_len = len(observation.images) * N_VISUAL
        if image_prefix_len != 3 * N_VISUAL:
            raise RuntimeError(f"expected three 256-token views, got {list(observation.images)}")
        visual = np.asarray(prefix[0, :N_VISUAL], dtype=np.float32)
        m, groups, matched_ids, entities = self._matched_budget(visual, prompt)
        mark(f"matched_budget_m{m}")

        # Match the official Pi0-FAST right-aligned prefix/cache convention.
        attn_mask = self.pi0_fast_lib.make_attn_mask(prefix_mask, prefix_ar)
        prefix, prefix_mask, attn_mask = self.pi0_fast_lib.left_to_right_align(prefix, prefix_mask, attn_mask)
        prefix_size = prefix.shape[1]
        prefix_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefix_size - prefix_len
        query_mask = self._task_query_mask(item, prompt, prefix_size, image_prefix_len)
        query_mask = np.roll(query_mask, -int(np.asarray(prefix_len)[0]))[None, :]
        capture: list[np.ndarray] = []
        self.gemma_fast._PI0FAST_ATTN_CAPTURE = {
            "query_mask": query_mask,
            "key_start": int(np.asarray(prefix_start)[0]),
            "callback": lambda x: capture.append(np.asarray(x)),
        }
        padded_mask = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, self.max_decoding_steps)))
        positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        try:
            positive_prefix_logits, positive_cache, _ = self.model.PaliGemma.llm(
                embedded_prefix=prefix, mask=padded_mask, positions=positions, decode=True,
            )
            jax.effects_barrier()
        finally:
            self.gemma_fast._PI0FAST_ATTN_CAPTURE = None
        mark("positive_prefix_attention")
        if len(capture) != 18:
            raise RuntimeError(f"L6 attention capture returned {len(capture)} layers, expected 18")
        l6_scores = np.asarray(capture[LAYER_INDEX][0], dtype=np.float32)
        selected = _stable_top_m(l6_scores, m)
        reconstructed = _harmonic_reconstruct(visual, selected)

        # Replace only primary-view selected patches, keeping both auxiliary
        # views and all nonselected image/text tokens identical to the clean arm.
        # Apply the exact same selected-token replacement with a fixed-shape
        # primary-view write. A variable-length scatter here specializes JAX
        # executables to each matched m_t (often dozens of distinct shapes in
        # one rollout), causing repeated multi-minute compilations. Nonselected
        # visual tokens are copied through unchanged, so this is numerically
        # equivalent to scattering only `selected` rows.
        negative_view = visual.copy()
        negative_view[np.asarray(selected, dtype=np.int32)] = reconstructed
        view_start = int(np.asarray(prefix_start)[0])
        negative_prefix = prefix.at[0, view_start : view_start + N_VISUAL].set(
            jnp.asarray(negative_view, dtype=prefix.dtype)
        )
        negative_prefix_logits, negative_cache, _ = self.model.PaliGemma.llm(
            embedded_prefix=negative_prefix, mask=padded_mask, positions=positions, decode=True,
        )
        jax.effects_barrier()
        mark("negative_prefix")
        final_index = prefix_size - 1  # right-alignment puts the final valid input token here
        clean_next = positive_prefix_logits[:, final_index]
        negative_next = negative_prefix_logits[:, final_index]
        # Prime both branches with the exact normal output-format header. The
        # contrastive policy starts only at serialized FAST action code tokens.
        cache_len = prefix_size + self.max_decoding_steps
        columns = jnp.arange(cache_len)[None, None, :]
        for header_step, header_id in enumerate(self.action_header_tokens):
            header_token = jnp.asarray([[header_id]], dtype=jnp.int32)
            token_embedding = self.model.PaliGemma.llm(header_token, embed_only=True)
            header_positions = prefix_len[:, None] + header_step + 1
            header_mask = jnp.logical_and(
                columns >= prefix_start[:, None, None],
                columns < prefix_size + header_step + 1,
            )
            clean_header_logits, positive_cache, _ = self.model.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=header_mask, positions=header_positions,
                decode=True, kv_cache=positive_cache,
            )
            negative_header_logits, negative_cache, _ = self.model.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=header_mask, positions=header_positions,
                decode=True, kv_cache=negative_cache,
            )
            clean_next, negative_next = clean_header_logits[:, -1], negative_header_logits[:, -1]
        tokens, decoded_steps, first_clean, first_negative = self._compiled_decode(
            jax.random.key(seed), clean_next, negative_next, positive_cache, negative_cache,
            prefix_len, prefix_start, prefix_size, self.fast_token_min, self.fast_token_max,
            self.pipe_token, jnp.asarray(self.action_header_tokens, dtype=jnp.int32),
            jnp.asarray(self.fast_token_char_lengths), self.fast_payload_chars,
        )
        jax.effects_barrier()
        mark("fast_decode")
        tokens = np.asarray(tokens[0], dtype=np.int32)
        serialized_text = self.tokenizer._paligemma_tokenizer.decode(tokens.tolist())
        # Decode the generated FAST BPE ids directly. OpenPI's generic
        # ExtractFASTActions round-trips special PaliGemma pieces through text
        # and strips whitespace, which can corrupt a valid fixed-length FAST
        # coefficient string. The action stream already gives us those IDs.
        stop_positions = np.flatnonzero(tokens == self.pipe_token)
        if not stop_positions.size:
            raise RuntimeError("guided FAST stream ended without its '|' delimiter")
        payload_start = len(self.action_header_tokens)
        payload_ids = tokens[payload_start : int(stop_positions[0])]
        fast_codes = self.fast_token_max - payload_ids
        if np.any(fast_codes < 0) or np.any(fast_codes >= len(self.fast_token_char_lengths)):
            raise RuntimeError("guided stream contains a token outside the FAST code vocabulary")
        from scipy.fft import idct

        fast_processor = self.tokenizer._fast_tokenizer
        payload_text = fast_processor.bpe_tokenizer.decode(fast_codes.tolist())
        raw_payload_chars = len(payload_text)
        action_stream_valid = raw_payload_chars == self.fast_payload_chars
        if action_stream_valid:
            coefficients = np.asarray([ord(char) for char in payload_text], dtype=np.float32)
            coefficients = coefficients.reshape(self.cfg.model.action_horizon, self.cfg.model.action_dim)
            coefficients += float(fast_processor.min_token)
            normalized_actions = idct(
                coefficients / float(fast_processor.scale), axis=0, norm="ortho"
            ).astype(np.float32)
        else:
            # Match OpenPI's documented malformed-stream fallback (zero DCT
            # coefficients) instead of silently padding/truncating generated
            # action tokens into a synthetic action.
            normalized_actions = np.zeros(
                (self.cfg.model.action_horizon, self.cfg.model.action_dim), dtype=np.float32
            )
        if normalized_actions.shape != (self.cfg.model.action_horizon, self.cfg.model.action_dim):
            raise RuntimeError(f"FAST decoder returned unexpected action shape {normalized_actions.shape}")
        result = {"state": item["state"], "actions": normalized_actions}
        for transform in self.policy._output_transform.transforms[1:]:
            result = transform(result)
        actions = np.asarray(result["actions"], dtype=np.float32)
        # Cross-check the direct inverse-DCT path against OpenPI's official
        # tokenizer/output transform for this exact generated stream.
        import contextlib
        import io

        official_stdout = io.StringIO()
        with contextlib.redirect_stdout(official_stdout):
            official_result = self.policy._output_transform(
                {"state": item["state"], "actions": tokens.copy()}
            )
        official_actions = np.asarray(official_result["actions"], dtype=np.float32)
        # OpenPI's FAST decoder intentionally falls back to zero DCT
        # coefficients when a generated code stream has the wrong length.
        # That fallback prints a warning, but is still a valid decode path;
        # agreement is numerical, not the absence of a warning.
        official_decode_agrees = (
            official_actions.shape == actions.shape
            and np.allclose(official_actions, actions, rtol=1e-5, atol=1e-6)
        )
        first_clean_np = np.asarray(first_clean[0], dtype=np.float32)
        first_negative_np = np.asarray(first_negative[0], dtype=np.float32)
        first_guided = (1.0 + self.lambda_) * first_clean_np - self.lambda_ * first_negative_np
        self.last_trace = {
            "selector_layer_index_0based": LAYER_INDEX,
            "selector_layer_label": "L6",
            "entities": entities,
            "matched_cluster_ids": groups,
            "matched_kmeans_backend": KMEANS_BACKEND,
            "matched_m_t": int(m),
            "matched_cluster_token_count": len(matched_ids),
            "selected_token_ids": selected,
            "selected_token_count": len(selected),
            "selected_count_equals_matched": len(selected) == int(m),
            "harmonic_finite": bool(np.isfinite(reconstructed).all()),
            "lambda": self.lambda_,
            "guidance_unit": "FAST action-code tokens only after the shared fixed Action: header",
            "guidance_token_scope": "FAST action-code vocabulary only; preserves fixed Action: header",
            "fast_token_id_range": [self.fast_token_min, self.fast_token_max],
            "fast_payload_char_count": self.fast_payload_chars,
            "first_clean_top_token": int(np.argmax(first_clean_np)),
            "first_negative_top_token": int(np.argmax(first_negative_np)),
            "first_guided_top_token": int(np.argmax(first_guided)),
            "first_logit_residual_norm": float(np.linalg.norm(first_clean_np - first_negative_np)),
            "serialized_output_has_action_header": "Action: " in serialized_text,
            "serialized_output_has_pipe_terminator": "|" in serialized_text,
            "serialized_output_preview": serialized_text[:180],
            "raw_output_token_ids_prefix": tokens[:24].tolist(),
            "fast_code_token_count": int(len(fast_codes)),
            "decoded_fast_payload_chars": int(raw_payload_chars),
            "fast_action_stream_valid": bool(action_stream_valid),
            "fast_action_stream_fallback": not bool(action_stream_valid),
            "openpi_official_decoder_agrees": bool(official_decode_agrees),
            "openpi_official_decoder_warning": official_stdout.getvalue().strip()[:200],
            "decoded_token_steps": int(np.asarray(decoded_steps)),
            "output_action_shape": list(actions.shape),
            "output_action_finite": bool(np.isfinite(actions).all()),
        }
        return {**result, "pi0_fast_matched_cd_trace": self.last_trace}
