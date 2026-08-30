"""SCR-CD: Semantic Context-Reconstruction Contrastive Decoding (Phase 1).

The reconstruction operator replaces each *selected* semantic token v_i (i in G)
with its linear reconstruction from a *context* basis B drawn from C = V \\ G:

    v_i  ->  hat{v}_i = mu_B + B_c alpha_i,
    alpha_i = (B_c^T B_c + rho I)^{-1} B_c^T (v_i - mu_B)

so the negative branch keeps ONLY the part of the semantic region that the
surrounding (non-semantic) visual tokens can linearly explain, and deletes the
region-unique residual e_i = v_i - hat{v}_i. This is a *minimal information
deletion* — the opposite of Merge (which keeps mu_G and deletes within-group
detail) and of Attention-block (which severs the whole access path).

Design rules (locked by the spec):
  * G = source ∪ target KMeans groups (K=8, same frozen selector as Phase 1B).
  * C = V \\ G ; basis B is M=10 tokens selected from C by deterministic FPS on
    cosine distance (L2-normalized features); G ∩ B = ∅ BY CONSTRUCTION.
  * Ridge rho = 1e-3 * tr(B_c^T B_c) / M (not swept).
  * Reconstruction is computed ONCE per control observation; the 7 action tokens
    share it via the guided single-pass forward.
  * Token count stays 256; positions unchanged; non-selected tokens bit-identical.
  * CD identical to Phase 1B: z* = z+ + 0.5(z+ - z-) on action dims 0..5, dim 6
    clean, guided autoregressive prefix.

Two arms share this operator and differ only in how G is chosen:
    semantic_recon_k8_m10  — G = semantic selector's source/target groups.
    random_recon_k8_m10    — G = q *non-semantic* KMeans groups (q = #semantic
                             groups) with total size matched within 20%; the most
                             important control (Semantic≈Random was seen before).

Math note: B_c is stored row-major (M,d) below; the spec's B_c ∈ R^{d×M} is its
transpose. The solve is done in the dual M×M form (B_c B_c^T + rho I) alpha =
B_c (v_i - mu_B), which is numerically cheaper (M=10 << d=4096) and equivalent.
All reconstruction math runs in float64 for stability, then cast back to float32.
"""
from __future__ import annotations

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import (
    AuditedEntityCDInference,
    _action_logits,
)
from research.semantic_token_cd.global_merge_policy import (
    guided_forward_scores,
    projector_merge_intervention,
)


M_BASIS = 10
RHO_SCALE = 1e-3
SIZE_MATCH_TOL = 0.20
_EPS = 1e-12


def fps_on_cosine(h: np.ndarray, c_idx: np.ndarray, M: int) -> np.ndarray:
    """Deterministic farthest-point sampling of M tokens from c_idx.

    Distance = cosine distance on L2-normalized features (equivalently Euclidean
    on the unit sphere). First point = the C token with the largest L2 norm
    (argmax is index-stable on ties), so the whole sequence is reproducible.
    Returns token indices (into the 256-grid), not positions into c_idx.
    """
    X = h[c_idx].astype(np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + _EPS)
    first = int(np.argmax(np.linalg.norm(X, axis=1)))
    chosen = [first]
    D = (1.0 - Xn @ Xn[first]).clip(min=0.0)
    for _ in range(1, M):
        nxt = int(np.argmax(D))
        chosen.append(nxt)
        d_new = (1.0 - Xn @ Xn[nxt]).clip(min=0.0)
        D = np.minimum(D, d_new)
    return c_idx[np.asarray(chosen, dtype=np.int64)]


def _pstats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def reconstruct_groups(
    h: np.ndarray,
    labels: np.ndarray,
    selected_groups: list[int],
    M: int = M_BASIS,
    rho_scale: float = RHO_SCALE,
    return_raw: bool = False,
    strength_mode: str = "full",
) -> tuple[np.ndarray, dict]:
    """Reconstruct every selected-group token from a context basis and return
    (h_tilde [256,d] float32, diagnostics). h is [256,d] float32 projector out.

    The negative branch degrades each selected token v_i toward its context
    reconstruction hat{v}_i by a per-token strength s_i:

        v_i^- = (1 - s_i) v_i + s_i hat{v}_i

    ``strength_mode`` selects how s_i is assigned (Error-Guided Recon spec):
        "full"             — s_i = 1                 (Phase-1 operator: keep only hat{v}).
        "error_guided"     — s_i = clip(e_i, 0, 1)   (weaken by reconstruction error).
        "matched_strength" — s_i = mean_j clip(e_j, 0, 1) (uniform, matches total strength).

    If ``return_raw`` is True the return becomes (h_tilde, diag, raw) where ``raw``
    holds the float64 per-token arrays ``e_rel`` / ``q_ratio`` / ``cos`` / ``s``
    (length |G|). The Phase 0 integrity probe uses this for pooled per-task
    distributions; the rollout passes the default (aggregates only) to keep every
    trace step small.
    """
    h64 = h.astype(np.float64)
    g_idx = np.array(
        sorted(int(i) for g in selected_groups for i in np.flatnonzero(labels == g)),
        dtype=np.int64,
    )
    if g_idx.size == 0:
        raise RuntimeError("Reconstruction selected an empty group set")
    g_set = set(int(i) for i in g_idx)
    c_idx = np.array([i for i in range(h.shape[0]) if i not in g_set], dtype=np.int64)

    basis = fps_on_cosine(h64, c_idx, M)
    assert set(int(b) for b in basis).isdisjoint(g_set), "basis must be disjoint from G"
    B = h64[basis]                       # (M, d) raw features
    mu_B = B.mean(axis=0, keepdims=True)  # (1, d)
    Bc = B - mu_B                        # (M, d) centered basis rows
    G = Bc @ Bc.T                        # (M, M) == B_c^T B_c
    rho = rho_scale * float(np.trace(G)) / float(M)
    A = G + rho * np.eye(M)

    Y = h64[g_idx] - mu_B                 # (|G|, d)
    RHS = Bc @ Y.T                        # (M, |G|) == B_c (v_i - mu_B)
    alpha = np.linalg.solve(A, RHS)       # (M, |G|)
    v_hat = mu_B + alpha.T @ Bc           # (|G|, d)

    v_orig = h64[g_idx]
    vn = np.linalg.norm(v_orig, axis=1)
    vn_hat = np.linalg.norm(v_hat, axis=1)
    e_rel = np.linalg.norm(v_orig - v_hat, axis=1) / (vn + _EPS)
    q_ratio = vn_hat / (vn + _EPS)
    cos = np.sum(v_orig * v_hat, axis=1) / (vn * vn_hat + _EPS)

    # --- degradation strength (Error-Guided Recon spec) ----------------------
    # e_i = ||v_i - v_hat_i|| / (||v_i|| + eps) is the relative reconstruction
    # error the spec calls e_i. Assign per-token strength s_i from it:
    s_hat = np.clip(e_rel, 0.0, 1.0)
    if strength_mode == "full":
        s_used = np.ones_like(e_rel)
        v_deg = v_hat
    elif strength_mode == "error_guided":
        s_used = s_hat
        v_deg = (1.0 - s_used)[:, None] * v_orig + s_used[:, None] * v_hat
    elif strength_mode == "matched_strength":
        s_bar = float(s_hat.mean())
        s_used = np.full_like(e_rel, s_bar)
        v_deg = (1.0 - s_bar) * v_orig + s_bar * v_hat
    else:
        raise ValueError(f"Unknown strength_mode: {strength_mode}")

    h_tilde = h.astype(np.float32).copy()
    h_tilde[g_idx] = v_deg.astype(np.float32)

    # Feature perturbation norm D = ||V~_G - V_G||_F / (||V_G||_F + eps): the
    # spec's single number for "how hard" the negative branch touches the
    # semantic region. full -> ||v_hat-v_orig||_F/||v_orig||_F; matched ->
    # s_bar * that; error_guided -> sqrt(sum_i s_i^2 ||v_hat-v_orig||_i^2)/||V_G||.
    D = float(np.linalg.norm(v_deg - v_orig) / (np.linalg.norm(v_orig) + _EPS))

    diag = {
        "recon_M": int(M),
        "recon_rho": float(rho),
        "recon_basis_token_ids": [int(b) for b in basis],
        "recon_g_basis_disjoint": True,
        "recon_n_g_tokens": int(g_idx.size),
        "recon_n_c_tokens": int(c_idx.size),
        "recon_finite": bool(np.isfinite(v_hat).all() and np.isfinite(v_deg).all()),
        "recon_e_rel": _pstats(e_rel),
        "recon_q_ratio": _pstats(q_ratio),
        "recon_cos": _pstats(cos),
        "recon_strength_mode": strength_mode,
        "recon_s": _pstats(s_used),
        "recon_mean_s": float(s_used.mean()),
        "recon_perturb_norm_D": D,
    }
    if return_raw:
        return h_tilde, diag, {"e_rel": e_rel, "q_ratio": q_ratio, "cos": cos, "s": s_used}
    return h_tilde, diag


def random_non_semantic_groups(
    labels: np.ndarray,
    semantic_groups: list[int],
    rng: np.random.Generator,
    size_tol: float = SIZE_MATCH_TOL,
) -> tuple[list[int], dict]:
    """Pick q non-semantic KMeans groups whose total size matches the semantic
    region within ``size_tol`` (resample up to 200 times; on failure take the
    closest-size draw and flag it). Returns (picked_groups, meta)."""
    q = len(semantic_groups)
    sem_size = int(sum(int(np.sum(labels == g)) for g in semantic_groups))
    K = int(labels.max()) + 1
    non_sem = np.array([g for g in range(K) if g not in set(semantic_groups)], dtype=np.int64)
    if non_sem.size < q:
        raise RuntimeError(f"Not enough non-semantic groups: {non_sem.size} < {q}")
    sizes = {int(g): int(np.sum(labels == g)) for g in non_sem}
    best = None
    best_diff = None
    for _ in range(200):
        picked = np.sort(rng.choice(non_sem, size=q, replace=False))
        size = sum(sizes[int(g)] for g in picked)
        diff = abs(size - sem_size)
        if best_diff is None or diff < best_diff:
            best, best_diff = picked, diff
        if sem_size == 0 or diff / max(1, sem_size) <= size_tol:
            return sorted(int(g) for g in picked), {
                "random_size_match": True,
                "random_sem_size": sem_size,
                "random_picked_size": size,
            }
    return sorted(int(g) for g in best), {
        "random_size_match": False,
        "random_sem_size": sem_size,
        "random_picked_size": sum(sizes[int(g)] for g in best),
    }


class SemanticReconCDInference(AuditedEntityCDInference):
    """KMeans K=8 semantic context-reconstruction CD with a guided prefix.

    ``recon_selection_mode`` is "semantic" (G = source/target groups) or
    "random_recon" (G = q random non-semantic groups, size-matched). Everything
    else — reconstruction operator, CD, guided prefix — is identical.

    ``strength_mode`` selects how each selected token v_i is degraded toward its
    context reconstruction (see ``reconstruct_groups``): "full" (Phase-1, s_i=1),
    "error_guided" (s_i = clip(e_i,0,1)), or "matched_strength" (uniform s_bar).
    """

    recon_selection_mode: str = "semantic"
    strength_mode: str = "full"
    M: int = M_BASIS
    rho_scale: float = RHO_SCALE
    _task_id: int = -1

    def _random_seed(self) -> int:
        # (task_id, episode_seed, timestep) — fully reproducible.
        return int(
            np.random.SeedSequence(
                [self._task_id, self._episode_seed, self._selector_step, 0x5CEED]
            ).generate_state(1, dtype=np.uint32)[0]
        )

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("Positive branch did not produce 7 scores and projector features")

        V = positive_trace.before  # [1,256,d] float32 cpu
        clean_token_ids = clean_scores.argmax(dim=-1)
        h = V[0].numpy()

        labels, semantic_groups, per_entity_score = self._semantic_clusters(h)
        if self.recon_selection_mode == "semantic":
            selected_groups = sorted(int(g) for g in semantic_groups)
            rnd_meta = {"random_size_match": None, "random_sem_size": None, "random_picked_size": None}
        elif self.recon_selection_mode == "random_recon":
            rng = np.random.default_rng(self._random_seed())
            selected_groups, rnd_meta = random_non_semantic_groups(labels, semantic_groups, rng)
        else:
            raise ValueError(f"Unknown recon_selection_mode: {self.recon_selection_mode}")

        selected = sorted(
            int(i) for g in selected_groups for i in np.flatnonzero(labels == g)
        )
        if not selected:
            raise RuntimeError("Reconstruction selected an empty negative branch")

        h_tilde, diag = reconstruct_groups(
            h, labels, selected_groups, self.M, self.rho_scale, strength_mode=self.strength_mode
        )
        if not diag["recon_finite"]:
            raise FloatingPointError("Reconstruction produced non-finite features")

        V_tilde = torch.from_numpy(h_tilde).unsqueeze(0)
        with projector_merge_intervention(self.vla, V_tilde) as negative_trace:
            negative_scores = guided_forward_scores(self.vla, inputs, clean_token_ids, V.shape[1])
        if negative_trace["before"] is None or negative_trace["after"] is None:
            raise RuntimeError("Reconstruction hook was not invoked")
        feature_equal = torch.equal(V, negative_trace["before"])
        if not feature_equal:
            raise RuntimeError("Negative branch projector 'before' != clean V")

        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1 + self.lambd) * clean_scores[:-1] - self.lambd * negative_scores[:-1]
        )
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite Semantic-Recon-CD logits")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        positive = _action_logits(self, clean_scores)
        negative = _action_logits(self, negative_scores)
        residual = (
            torch.log_softmax(torch.from_numpy(positive).float(), dim=-1)
            - torch.log_softmax(torch.from_numpy(negative).float(), dim=-1)
        )
        meta = {
            "recon_selection_mode": self.recon_selection_mode,
            "selection_mode": "semantic" if self.recon_selection_mode == "semantic" else "random_recon",
            "instruction": self._selector_instr,
            "selected_entities": list(self._entities),
            "selected_group_ids": [int(g) for g in selected_groups],
            "semantic_group_ids": [int(g) for g in semantic_groups],
            "selected_token_ids": selected,
            "num_tokens": len(selected),
            "num_groups": len(selected_groups),
            "per_entity_score": [float(s) for s in per_entity_score],
            "language_score": float(max(per_entity_score)) if per_entity_score else 0.0,
            "kmeans_K": int(self.kmeans_K),
            "random_size_match": rnd_meta["random_size_match"],
            "random_sem_size": rnd_meta["random_sem_size"],
            "random_picked_size": rnd_meta["random_picked_size"],
            "positive_token_ids": clean_scores.argmax(-1).detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "residual_norm": float(torch.linalg.vector_norm(residual).item()),
            "lambda": float(self.lambd),
            "feature_equal": feature_equal,
            "feature_shape": list(V.shape),
            "n_tokens_negative": int(V.shape[1]),
            "guided_prefix": True,
            "degenerate": False,
        }
        meta.update(diag)
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        self._selector_step += 1
        return raw_action, actions, meta
