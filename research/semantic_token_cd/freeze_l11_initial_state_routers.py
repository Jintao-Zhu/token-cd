"""Fit and freeze four task-specific initial-state L11-vs-SHR routers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.semantic_token_cd.l11_state_probe_protocol import feature_path as discovery_feature_path
from research.semantic_token_cd.l11_initial_state_router_protocol import (
    BASELINE,
    FEATURE_NAMES,
    PCA_COMPONENTS,
    PROTOCOL,
    SHORT_TASKS,
    TASKS,
    THRESHOLD,
    TRAIN_SEEDS,
    atomic_json,
    load_outcomes,
    sha256,
)


def load_feature(path: Path) -> np.ndarray:
    with np.load(path) as arrays:
        vector = np.concatenate([
            np.asarray(arrays[name], dtype=np.float32).reshape(-1)
            for name in FEATURE_NAMES
        ])
    if not np.isfinite(vector).all():
        raise FloatingPointError(path)
    return vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--discovery-artifact", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    discovery = args.discovery_artifact.resolve()
    paired_results = args.paired_results.resolve()
    outcomes = load_outcomes(paired_results, TRAIN_SEEDS)
    model_dir = artifact / "frozen_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol_id": PROTOCOL,
        "freeze_stage": "completed before any test feature collection",
        "train_seeds": [min(TRAIN_SEEDS), max(TRAIN_SEEDS)],
        "test_seeds": [100, 199],
        "baseline": BASELINE,
        "features": list(FEATURE_NAMES),
        "threshold": THRESHOLD,
        "pipeline": {
            "variance_filter": "train std > 1e-6",
            "standard_scaler": "default sklearn StandardScaler",
            "pca_components": PCA_COMPONENTS,
            "pca_solver": "randomized",
            "logistic_regression_C": 1.0,
            "class_weight": "balanced",
        },
        "models": {},
    }
    for task in TASKS:
        rows = []
        labels = []
        seeds = []
        for seed in TRAIN_SEEDS:
            outcome = outcomes[(task, seed)]
            if outcome["l11_matched"] == outcome[BASELINE]:
                continue
            rows.append(load_feature(discovery_feature_path(discovery, task, seed)))
            labels.append(int(outcome["l11_matched"]))
            seeds.append(seed)
        features = np.stack(rows)
        target = np.asarray(labels, dtype=np.int64)
        varying = np.std(features, axis=0) > 1e-6
        filtered = features[:, varying]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(filtered)
        components = min(PCA_COMPONENTS, scaled.shape[0] - 2, scaled.shape[1])
        if components < 2 or len(np.unique(target)) != 2:
            raise RuntimeError(f"insufficient training data for {task}")
        pca = PCA(n_components=components, svd_solver="randomized", random_state=20260913)
        reduced = pca.fit_transform(scaled)
        classifier = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000, random_state=20260913
        )
        classifier.fit(reduced, target)
        payload = {
            "protocol_id": PROTOCOL,
            "task": task,
            "feature_names": FEATURE_NAMES,
            "threshold": THRESHOLD,
            "varying": varying,
            "scaler": scaler,
            "pca": pca,
            "classifier": classifier,
            "train_seeds": np.asarray(seeds, dtype=np.int16),
            "train_labels": target,
        }
        path = model_dir / f"{SHORT_TASKS[task]}.joblib"
        joblib.dump(payload, path, compress=3)
        manifest["models"][SHORT_TASKS[task]] = {
            "path": str(path),
            "sha256": sha256(path),
            "discordant_train_count": int(target.size),
            "prefer_l11": int(target.sum()),
            "prefer_shr": int(target.size - target.sum()),
            "pca_components_actual": int(components),
            "input_dimensions": int(features.shape[1]),
            "varying_dimensions": int(varying.sum()),
        }
    repo = Path(__file__).resolve().parents[2]
    manifest["code_sha256"] = {
        "protocol": sha256(repo / "research/semantic_token_cd/l11_initial_state_router_protocol.py"),
        "freeze": sha256(Path(__file__).resolve()),
        "collector": sha256(repo / "research/semantic_token_cd/collect_l11_initial_state_confirm_features.py"),
        "analyzer": sha256(repo / "research/semantic_token_cd/analyze_l11_initial_state_router_confirmation.py"),
    }
    manifest["paired_results_sha256"] = sha256(paired_results)
    atomic_json(artifact / "FROZEN_MODELS.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
