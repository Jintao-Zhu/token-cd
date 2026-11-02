"""SciPy vs sklearn KMeans on the SAME pi0-FAST visual features.

The OpenVLA L11-Matched runs used sklearn (openvla-ar-h100 has 1.7.2); the pi0-FAST runs fell
back to scipy.cluster.vq.kmeans2 because pi05 has no sklearn. Both are K=8 / k-means++ / 10
restarts, but they are different implementations, so the matched budget m_t - and therefore
the intervention set - can differ. This quantifies that on identical inputs.

Reported per state: latency of each backend, cluster inertia, label agreement (permutation
invariant), the resulting matched budget m_t, and the Jaccard overlap of the resulting Top-m
intervention sets.

Run with the overlay on PYTHONPATH to get sklearn; without it the sklearn side is skipped.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# three processes per GPU in the production layout, so keep BLAS/OpenMP single-threaded
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
sys.path.insert(0, str(ROOT / "openpi/src"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import research.semantic_token_cd.pi0_fast_matched_cd as CD  # noqa: E402
from research.semantic_token_cd.pi0_fast_matched_cd import (  # noqa: E402
    KMEANS_GROUPS,
    KMEANS_SEED,
    N_VISUAL,
    Pi0FastMatchedCDPolicy,
    _extract_entities,
    _stable_top_m,
)

POOLS = [
    REPO / "artifacts/spatialvla_layer_selection_v1b/states",
    REPO / "artifacts/spatialvla_layer_selection_v1/states",
]
OUT = REPO / "artifacts/pi0_fast_kmeans_backend_v1"
LIMIT = int(os.environ.get("KMEANS_LIMIT", "0"))


def scipy_kmeans(features: np.ndarray):
    from scipy.cluster.vq import kmeans2

    best, best_inertia = None, np.inf
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
    return np.asarray(best, dtype=np.int32), best_inertia


def sklearn_kmeans(features: np.ndarray):
    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=KMEANS_GROUPS,
        init="k-means++",
        n_init=10,
        max_iter=300,
        algorithm="lloyd",
        random_state=KMEANS_SEED,
    ).fit(features)
    return np.asarray(km.labels_, dtype=np.int32), float(km.inertia_)


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index without sklearn (so it works when sklearn is absent)."""
    from math import comb

    n = len(a)
    cont = {}
    for x, y in zip(a.tolist(), b.tolist()):
        cont[(x, y)] = cont.get((x, y), 0) + 1
    sa, sb = {}, {}
    for x in a.tolist():
        sa[x] = sa.get(x, 0) + 1
    for y in b.tolist():
        sb[y] = sb.get(y, 0) + 1
    sij = sum(comb(v, 2) for v in cont.values())
    si = sum(comb(v, 2) for v in sa.values())
    sj = sum(comb(v, 2) for v in sb.values())
    exp = si * sj / comb(n, 2)
    mx = 0.5 * (si + sj)
    return (sij - exp) / (mx - exp) if mx != exp else 1.0


def matched_set(labels, features, entities, policy):
    """Replicate _matched_budget's entity->cluster matching to get the Top-m set."""
    group_means = np.zeros((KMEANS_GROUPS, features.shape[1]), dtype=np.float64)
    for g in range(KMEANS_GROUPS):
        members = np.flatnonzero(labels == g)
        if members.size:
            group_means[g] = features[members].mean(axis=0)
    group_means /= np.linalg.norm(group_means, axis=1, keepdims=True) + 1e-8
    groups = []
    for ent in entities:
        emb = policy._entity_embedding(ent).astype(np.float64)
        emb /= np.linalg.norm(emb) + 1e-8
        g = int(np.argmax(group_means @ emb))
        if g not in groups:
            groups.append(g)
    selected = sorted(int(i) for g in groups for i in np.flatnonzero(labels == g))
    return len(selected), groups, selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import sklearn  # noqa: F401
        have_sk = True
    except ModuleNotFoundError:
        have_sk = False
    print(f"[km] sklearn available: {have_sk}", flush=True)
    if not have_sk:
        raise SystemExit("run with PYTHONPATH pointing at pi05_sklearn_overlay")

    states = []
    seen = set()
    for pool in POOLS:
        if not pool.is_dir():
            continue
        for task in sorted(os.listdir(pool)):
            tdir = pool / task
            if not tdir.is_dir():
                continue
            for sd in sorted(os.listdir(tdir)):
                if not sd.startswith("seed_"):
                    continue
                key = (task, int(sd.split("_")[1]))
                if key in seen or not (tdir / sd / "input.npz").exists():
                    continue
                seen.add(key)
                states.append((task, key[1], tdir / sd))
    if LIMIT:
        states = states[:LIMIT]
    print(f"[km] states={len(states)} tasks={len({t for t,_,_ in states})}", flush=True)

    phys = int(os.environ.get("BENCH_PHYS_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "1")))
    pol = Pi0FastMatchedCDPolicy(gpu=phys)
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib

    model = pol.model
    rows = []
    for idx, (task, seed, sdir) in enumerate(states):
        meta = json.loads((sdir / "meta.json").read_text())
        npz = np.load(sdir / "input.npz")
        image = np.asarray(npz["image"])
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        prompt = meta["instruction"]
        raw = {
            "observation/primary_image": image,
            "observation/wrist_image": np.zeros_like(image),
            "observation/state": np.zeros((8,), dtype=np.float32),
            "prompt": prompt,
        }
        item = pol.policy._input_transform(dict(raw))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], item)
        obs = model_lib.Observation.from_dict(batch)
        obs = model_lib.preprocess_observation(
            jax.random.key(int(seed)), obs, train=False, image_keys=list(obs.images)
        )
        prefix, _, _ = model.embed_inputs(obs)
        visual = np.asarray(prefix[0, :N_VISUAL], dtype=np.float32)
        features = visual.astype(np.float64)
        entities = _extract_entities(prompt)

        t0 = time.perf_counter(); lab_s, in_s = scipy_kmeans(features); t_s = time.perf_counter() - t0
        t0 = time.perf_counter(); lab_k, in_k = sklearn_kmeans(features); t_k = time.perf_counter() - t0

        m_s, g_s, set_s = matched_set(lab_s, features, entities, pol)
        m_k, g_k, set_k = matched_set(lab_k, features, entities, pol)
        js = len(set(set_s) & set(set_k)) / max(1, len(set(set_s) | set(set_k)))
        top_s = _stable_top_m(np.ones(N_VISUAL, dtype=np.float32), min(m_s, N_VISUAL))
        top_k = _stable_top_m(np.ones(N_VISUAL, dtype=np.float32), min(m_k, N_VISUAL))
        row = {
            "task": task, "seed": int(seed), "instruction": prompt, "entities": entities,
            "scipy_s": t_s, "sklearn_s": t_k, "speedup": t_s / t_k if t_k else None,
            "scipy_inertia": in_s, "sklearn_inertia": in_k,
            "label_ari": ari(lab_s, lab_k),
            "m_scipy": int(m_s), "m_sklearn": int(m_k), "m_same": int(m_s) == int(m_k),
            "groups_scipy": g_s, "groups_sklearn": g_k,
            "matched_set_jaccard": js,
            "cluster_sizes_scipy": np.bincount(lab_s, minlength=KMEANS_GROUPS).tolist(),
            "cluster_sizes_sklearn": np.bincount(lab_k, minlength=KMEANS_GROUPS).tolist(),
        }
        rows.append(row)
        if idx % 5 == 0 or idx == len(states) - 1:
            print(f"[km] {idx+1}/{len(states)} {task.replace('google_robot_','').replace('widowx_','')} "
                  f"seed{seed}: scipy={t_s:.3f}s sklearn={t_k:.3f}s ari={row['label_ari']:.3f} "
                  f"m {m_s}->{m_k} jaccard={js:.3f}", flush=True)

    n = len(rows)
    summary = {
        "protocol": "PI0_FAST_KMEANS_BACKEND_V1",
        "states": n, "tasks": sorted({r["task"] for r in rows}),
        "scipy_mean_s": statistics.fmean(r["scipy_s"] for r in rows),
        "sklearn_mean_s": statistics.fmean(r["sklearn_s"] for r in rows),
        "speedup": statistics.fmean(r["scipy_s"] for r in rows) / statistics.fmean(r["sklearn_s"] for r in rows),
        "label_ari_mean": statistics.fmean(r["label_ari"] for r in rows),
        "label_ari_min": min(r["label_ari"] for r in rows),
        "m_same_fraction": sum(r["m_same"] for r in rows) / n,
        "matched_set_jaccard_mean": statistics.fmean(r["matched_set_jaccard"] for r in rows),
        "matched_set_jaccard_min": min(r["matched_set_jaccard"] for r in rows),
        "inertia_ratio_mean": statistics.fmean(
            r["sklearn_inertia"] / r["scipy_inertia"] for r in rows if r["scipy_inertia"]
        ),
        "rows": rows,
    }
    (OUT / "BACKEND_COMPARISON.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[km] states={n}  scipy={summary['scipy_mean_s']:.3f}s  sklearn={summary['sklearn_mean_s']:.3f}s  "
          f"speedup={summary['speedup']:.1f}x", flush=True)
    print(f"[km] label ARI mean={summary['label_ari_mean']:.3f} min={summary['label_ari_min']:.3f}", flush=True)
    print(f"[km] m_t identical in {summary['m_same_fraction']*100:.1f}% of states; "
          f"matched-set Jaccard mean={summary['matched_set_jaccard_mean']:.3f} "
          f"min={summary['matched_set_jaccard_min']:.3f}", flush=True)


if __name__ == "__main__":
    main()
