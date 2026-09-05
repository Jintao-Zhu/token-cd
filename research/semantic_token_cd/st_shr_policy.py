"""Spatiotemporal spatial-harmonic reconstruction CD (ST-SHR).

The negative branch replaces each entity-selected KMeans region with a
4-neighbor harmonic reconstruction.  At replans after t=0, the solve may use
the previous *reconstructed* region for the same entity as a translated prior.
Clean semantic features are never stored in temporal state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import AuditedEntityCDInference, _action_logits
from research.semantic_token_cd.global_merge_policy import guided_forward_scores, projector_merge_intervention
from research.semantic_token_cd.semantic_recon_policy import reconstruct_groups


GRID = 16
MAX_DISPLACEMENT = 4.0
EPS = 1e-12


@dataclass
class EntityHistory:
    indices: np.ndarray
    reconstructed: np.ndarray
    centroid: np.ndarray


def centroid(indices: np.ndarray) -> np.ndarray:
    rc = np.column_stack((indices // GRID, indices % GRID)).astype(np.float64)
    return rc.mean(axis=0)


def harmonic_reconstruct(
    features: np.ndarray,
    region: np.ndarray,
    beta: float = 0.0,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Solve (L_GG + beta I) X = -L_GC V_C + beta T on a 16x16 grid."""
    region = np.asarray(sorted(set(int(i) for i in region)), dtype=np.int64)
    if region.size == 0:
        raise ValueError("empty harmonic region")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if beta > 0 and (prior is None or prior.shape != (region.size, features.shape[1])):
        raise ValueError("temporal prior shape mismatch")

    position = {int(token): row for row, token in enumerate(region)}
    A = np.zeros((region.size, region.size), dtype=np.float64)
    rhs = np.zeros((region.size, features.shape[1]), dtype=np.float64)
    clean = features.astype(np.float64, copy=False)
    for row, token in enumerate(region):
        r, c = divmod(int(token), GRID)
        neighbors = []
        if r > 0: neighbors.append(token - GRID)
        if r + 1 < GRID: neighbors.append(token + GRID)
        if c > 0: neighbors.append(token - 1)
        if c + 1 < GRID: neighbors.append(token + 1)
        A[row, row] = len(neighbors) + beta
        for neighbor in neighbors:
            col = position.get(int(neighbor))
            if col is None:
                rhs[row] += clean[neighbor]
            else:
                A[row, col] -= 1.0
        if beta > 0:
            rhs[row] += beta * prior[row]
    try:
        solved = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("singular harmonic region") from exc
    if not np.isfinite(solved).all():
        raise FloatingPointError("non-finite harmonic reconstruction")
    return solved.astype(np.float32)


def translated_prior(
    current_indices: np.ndarray,
    previous: EntityHistory | None,
    max_displacement: float = MAX_DISPLACEMENT,
) -> tuple[np.ndarray | None, dict]:
    """Translate a same-entity reconstructed region using rounded centroid motion."""
    if previous is None:
        return None, {"fallback_reason": "no_history"}
    current_indices = np.asarray(sorted(set(int(i) for i in current_indices)), dtype=np.int64)
    current_centroid = centroid(current_indices)
    displacement = current_centroid - previous.centroid
    norm = float(np.linalg.norm(displacement))
    shift = np.rint(displacement).astype(np.int64)
    if norm > max_displacement:
        return None, {"fallback_reason": "displacement", "centroid_displacement": norm,
                      "centroid_shift": shift.tolist()}
    prev_by_index = {int(i): previous.reconstructed[row] for row, i in enumerate(previous.indices)}
    values = []
    for token in current_indices:
        r, c = divmod(int(token), GRID)
        pr, pc = r - int(shift[0]), c - int(shift[1])
        if not (0 <= pr < GRID and 0 <= pc < GRID):
            return None, {"fallback_reason": "warp_oob", "centroid_displacement": norm,
                          "centroid_shift": shift.tolist()}
        value = prev_by_index.get(pr * GRID + pc)
        if value is None:
            return None, {"fallback_reason": "incomplete_overlap", "centroid_displacement": norm,
                          "centroid_shift": shift.tolist()}
        values.append(value)
    prior = np.asarray(values, dtype=np.float32)
    if not np.isfinite(prior).all():
        return None, {"fallback_reason": "nonfinite_prior", "centroid_displacement": norm,
                      "centroid_shift": shift.tolist()}
    return prior, {"fallback_reason": None, "centroid_displacement": norm,
                   "centroid_shift": shift.tolist()}


class STSHRCDInference(AuditedEntityCDInference):
    beta: float = 1.0

    def _combine_action_scores(
        self, clean_scores: torch.Tensor, negative_scores: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Apply the locked SHR CD rule; subclasses may only control its strength."""
        final_scores = clean_scores.clone()
        final_scores[:-1] = (
            (1 + self.lambd) * clean_scores[:-1]
            - self.lambd * negative_scores[:-1]
        )
        return final_scores, {}

    def reset(self, task_description: str, seed=None) -> None:
        super().reset(task_description, seed)
        self._st_history: dict[str, EntityHistory] = {}
        self._previous_residual: np.ndarray | None = None

    def _entity_groups(self, h: np.ndarray, labels: np.ndarray) -> list[int]:
        group_vectors = np.stack([h[labels == g].mean(axis=0) for g in range(self.kmeans_K)])
        normalized = group_vectors / (np.linalg.norm(group_vectors, axis=1, keepdims=True) + 1e-8)
        groups = []
        for embedding in self._entity_emb:
            emb = embedding / (np.linalg.norm(embedding) + 1e-8)
            groups.append(int(np.argmax(normalized @ emb)))
        return groups

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("positive branch did not produce projector features and 7 scores")
        V = positive_trace.before
        h = V[0].numpy().astype(np.float32)
        labels, _, per_entity_score = self._semantic_clusters(h)
        entity_groups = self._entity_groups(h, labels)

        h_negative = h.copy()
        next_history: dict[str, EntityHistory] = {}
        entity_meta = []
        claimed: set[int] = set()
        for entity, group in zip(self._entities, entity_groups):
            indices = np.asarray([int(i) for i in np.flatnonzero(labels == group)
                                  if int(i) not in claimed], dtype=np.int64)
            if indices.size == 0:
                entity_meta.append({"entity": entity, "group": group,
                                    "temporal_prior_used": False, "fallback_reason": "dedup_empty"})
                continue
            claimed.update(int(i) for i in indices)
            prior, align = translated_prior(indices, self._st_history.get(entity))
            use_temporal = prior is not None
            solved = harmonic_reconstruct(h, indices, self.beta if use_temporal else 0.0, prior)
            h_negative[indices] = solved
            next_history[entity] = EntityHistory(indices.copy(), solved.copy(), centroid(indices))
            entity_meta.append({"entity": entity, "group": group, "num_tokens": int(indices.size),
                                "temporal_prior_used": use_temporal, **align})
        if not claimed:
            raise RuntimeError("ST-SHR selector produced an empty region")
        self._st_history = next_history

        V_tilde = torch.from_numpy(h_negative).unsqueeze(0)
        clean_token_ids = clean_scores.argmax(dim=-1)
        with projector_merge_intervention(self.vla, V_tilde) as negative_trace:
            negative_scores = guided_forward_scores(self.vla, inputs, clean_token_ids, V.shape[1])
        if negative_trace["before"] is None or not torch.equal(V, negative_trace["before"]):
            raise RuntimeError("negative projector input differs from clean features")
        final_scores, guidance_meta = self._combine_action_scores(clean_scores, negative_scores)
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("non-finite ST-SHR logits")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        positive = _action_logits(self, clean_scores)
        negative = _action_logits(self, negative_scores)
        residual = (torch.log_softmax(torch.from_numpy(positive).float(), dim=-1)
                    - torch.log_softmax(torch.from_numpy(negative).float(), dim=-1)).numpy()
        if self._previous_residual is None:
            temporal_cos, temporal_jerk = None, None
        else:
            a, b = residual.ravel(), self._previous_residual.ravel()
            temporal_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
            temporal_jerk = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + EPS))
        self._previous_residual = residual.copy()
        used = sum(int(x.get("temporal_prior_used", False)) for x in entity_meta)
        meta = {
            "selection_mode": "semantic", "selected_entities": list(self._entities),
            "selected_group_ids": entity_groups, "per_entity_score": per_entity_score,
            "selected_token_ids": sorted(claimed), "num_tokens": len(claimed),
            "kmeans_K": int(self.kmeans_K), "beta": float(self.beta), "lambda": float(self.lambd),
            "entity_alignment": entity_meta, "temporal_prior_used_fraction": used / max(1, len(entity_meta)),
            "fallback_fraction": 1.0 - used / max(1, len(entity_meta)),
            "residual_temporal_cos": temporal_cos, "residual_jerk": temporal_jerk,
            "residual_norm": float(np.linalg.norm(residual)), "guided_prefix": True,
            "feature_equal": True, "reconstruction_finite": bool(np.isfinite(h_negative).all()),
            "positive_token_ids": clean_scores.argmax(-1).detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            **guidance_meta,
        }
        self._episode_logits.append({"positive": positive, "negative": negative})
        self._episode_trace.append(meta)
        self._selector_step += 1
        return raw_action, actions, meta


class STReconCDInference(STSHRCDInference):
    """Temporal prior applied to the existing FPS/ridge Recon operator."""

    def _recon(self, h, labels, groups, prior_by_entity):
        selected = np.asarray(sorted(set(int(i) for g in groups for i in np.flatnonzero(labels == g))), dtype=np.int64)
        if not len(selected):
            raise RuntimeError("empty reconstruction region")
        # Reuse the locked FPS/ridge implementation for beta=0 and add the
        # temporal quadratic in the same M-dimensional basis coordinates.
        h64 = h.astype(np.float64); gset=set(map(int,selected)); cidx=np.asarray([i for i in range(256) if i not in gset],dtype=np.int64)
        from research.semantic_token_cd.semantic_recon_policy import fps_on_cosine, M_BASIS, RHO_SCALE
        basis=fps_on_cosine(h64,cidx,M_BASIS); B=h64[basis]; mu=B.mean(0,keepdims=True); Bc=B-mu; G=Bc@Bc.T; rho=RHO_SCALE*np.trace(G)/M_BASIS
        Y=h64[selected]-mu; A=G+rho*np.eye(M_BASIS); rhs=Bc@Y.T
        # Temporal prior is assembled per token; invalid entities use beta=0.
        beta=np.zeros(len(selected)); T=np.zeros_like(Y); pos={int(i):j for j,i in enumerate(selected)}
        for entity, (indices, prior, used) in prior_by_entity.items():
            if not used or prior is None: continue
            for i,val in zip(indices,prior):
                j=pos.get(int(i));
                if j is not None: beta[j]=1.0; T[j]=val
        if np.any(beta):
            # Per-token Sherman-style normal equations in shared basis.
            for j,b in enumerate(beta):
                if b:
                    target=T[j]-mu[0]; rhs[:,j] += b*(Bc@target)
                    A_j=A+b*G
                    coeff=np.linalg.solve(A_j,rhs[:,j]); T[j]=mu[0]+coeff@Bc
        coeff=np.linalg.solve(A,rhs)
        out=h.astype(np.float32).copy(); out[selected]= (mu+coeff.T@Bc).astype(np.float32)
        for j,b in enumerate(beta):
            if b: out[selected[j]]=T[j].astype(np.float32)
        return out, selected

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        # Use ST-SHR's complete guided-CD path, replacing only its harmonic
        # reconstruction with FPS/ridge + temporal quadratic.
        inputs=self.process_inputs(image,task_description=task_description)
        with projector_intervention(self.vla) as pt:
            clean=self._forward_scores(inputs,self.unnorm_key,do_sample=False)
        if clean.shape[0]!=7 or pt.before is None: raise RuntimeError("positive branch incomplete")
        V=pt.before; h=V[0].numpy().astype(np.float32); labels,_,scores=self._semantic_clusters(h); groups=self._entity_groups(h,labels)
        claimed=set(); prior_by={}; meta_align=[]; next_hist={}
        for entity,g in zip(self._entities,groups):
            idx=np.asarray([int(i) for i in np.flatnonzero(labels==g) if int(i) not in claimed],dtype=np.int64); claimed.update(map(int,idx))
            prior,align=translated_prior(idx,self._st_history.get(entity)); used=prior is not None
            prior_by[entity]=(idx,prior,used); meta_align.append({"entity":entity,"group":g,"num_tokens":len(idx),"temporal_prior_used":used,**align})
        out,selected=self._recon(h,labels,groups,prior_by)
        for entity,(idx,_,_) in prior_by.items():
            if len(idx): next_hist[entity]=EntityHistory(idx.copy(),out[idx].copy(),centroid(idx))
        self._st_history=next_hist; ids=clean.argmax(-1)
        with projector_merge_intervention(self.vla,torch.from_numpy(out).unsqueeze(0)) as nt:
            neg=guided_forward_scores(self.vla,inputs,ids,V.shape[1])
        if nt["before"] is None or not torch.equal(V,nt["before"]): raise RuntimeError("feature mismatch")
        final=clean.clone(); final[:-1]=(1+self.lambd)*clean[:-1]-self.lambd*neg[:-1]; toks=final.argmax(-1); raw=self._decode_actions(toks,self.unnorm_key)[None]; raw_action,actions=self.postprocess_actions(raw)
        pos=_action_logits(self,clean); nlog=_action_logits(self,neg); residual=(torch.log_softmax(torch.from_numpy(pos).float(),-1)-torch.log_softmax(torch.from_numpy(nlog).float(),-1)).numpy()
        cos=jerk=None
        if self._previous_residual is not None:
            a=residual.ravel();b=self._previous_residual.ravel();cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+EPS));jerk=float(np.linalg.norm(a-b)/(np.linalg.norm(a)+np.linalg.norm(b)+EPS))
        self._previous_residual=residual.copy(); used=sum(int(x[2]) for x in prior_by.values())
        trace={"selection_mode":"semantic","selected_entities":list(self._entities),"selected_group_ids":groups,"selected_token_ids":sorted(map(int,selected)),"num_tokens":len(selected),"per_entity_score":scores,"beta":1.0,"lambda":float(self.lambd),"entity_alignment":meta_align,"temporal_prior_used_fraction":used/max(1,len(prior_by)),"fallback_fraction":1-used/max(1,len(prior_by)),"residual_temporal_cos":cos,"residual_jerk":jerk,"residual_norm":float(np.linalg.norm(residual)),"guided_prefix":True,"feature_equal":True,"reconstruction_finite":bool(np.isfinite(out).all()),"positive_token_ids":ids.detach().cpu().tolist(),"negative_token_ids":neg.argmax(-1).detach().cpu().tolist(),"final_token_ids":toks.detach().cpu().tolist()}
        self._episode_logits.append({"positive":pos,"negative":nlog});self._episode_trace.append(trace);self._selector_step+=1
        return raw_action,actions,trace
