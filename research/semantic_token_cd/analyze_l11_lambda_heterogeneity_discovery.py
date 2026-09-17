"""Analyze fixed-lambda heterogeneity and grouped-OOF lambda-conditioned routing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from research.semantic_token_cd.prompt_attn_l11_lambda_heterogeneity_rollout import PROTOCOL, TASKS
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


LAMBDAS = np.asarray([0.0, 0.25, 0.5], dtype=np.float32)
SHORT = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}
FEATURE_SETS = {
    "task_only": ("task",),
    "prompt_context": ("prompt_hidden_mean",),
    "prompt_action": ("prompt_hidden_mean", "action_context_hidden"),
}
PRIMARY_FEATURE_SET = "prompt_action"
SWITCH_MARGIN = 0.05
FIXED_HALF_ROOTS = {
    "google_robot_open_drawer": Path("artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes"),
    "google_robot_close_drawer": Path("artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes"),
    "google_robot_pick_coke_can": Path("artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes"),
    "google_robot_move_near": Path("artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixed_half(path: Path) -> dict[tuple[str, int], bool]:
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]; seed = int(row["seed"])
            if task in TASKS and 0 <= seed <= 99:
                result[(task, seed)] = row["l11_matched"].lower() == "true"
    return result


def load_episode_records(artifact: Path, adaptive: Path, paired: Path, features: Path) -> list[dict]:
    fixed_half = load_fixed_half(paired)
    repo = Path(__file__).resolve().parents[2]
    records = []
    for task_index, task in enumerate(TASKS):
        for seed in range(100):
            zero_path = artifact / "episodes" / task / "l11_positive_only" / f"episode_{seed:03d}_summary.json"
            quarter_root = artifact if task == "google_robot_open_drawer" else adaptive
            quarter_path = quarter_root / "episodes" / task / "l11_fixed_025" / f"episode_{seed:03d}_summary.json"
            zero = json.loads(zero_path.read_text()); quarter = json.loads(quarter_path.read_text())
            half_path = repo / FIXED_HALF_ROOTS[task] / task / "prompt_single" / f"episode_{seed:03d}_summary.json"
            half = json.loads(half_path.read_text())
            if not zero.get("technical_pass") or not quarter.get("technical_pass"):
                raise RuntimeError(f"technical audit failure: {task} seed={seed}")
            if (zero["canonical_snapshot_sha256"], zero["initial_state_sha256"], zero["initial_rgb_sha256"]) != (
                quarter["canonical_snapshot_sha256"], quarter["initial_state_sha256"], quarter["initial_rgb_sha256"]
            ):
                raise RuntimeError(f"lambda arm pairing mismatch: {task} seed={seed}")
            hashes = (zero["canonical_snapshot_sha256"], zero["initial_state_sha256"], zero["initial_rgb_sha256"])
            if hashes != (half["canonical_snapshot_sha256"], half["initial_state_sha256"], half["initial_rgb_sha256"]):
                raise RuntimeError(f"lambda=.5 pairing mismatch: {task} seed={seed}")
            if bool(half["success"]) != fixed_half[(task, seed)]:
                raise RuntimeError(f"lambda=.5 success mismatch: {task} seed={seed}")
            feature_path = features / "features" / task / f"seed_{seed:03d}.npz"
            with np.load(feature_path) as arrays:
                feature_values = {
                    name: np.asarray(arrays[name], dtype=np.float32).reshape(-1)
                    for name in ("prompt_hidden_mean", "action_context_hidden")
                }
            outcomes = np.asarray([
                bool(zero["success"]), bool(quarter["success"]), bool(fixed_half[(task, seed)])
            ], dtype=np.int64)
            records.append({
                "task": task, "task_short": SHORT[task], "task_index": task_index,
                "seed": seed, "outcomes": outcomes, "features": feature_values,
            })
    return records


def folds(records: list[dict], count: int = 5) -> np.ndarray:
    assignment = np.empty(len(records), dtype=np.int64)
    rng = np.random.default_rng(20260913)
    for task in TASKS:
        indices = np.asarray([index for index, record in enumerate(records) if record["task"] == task])
        shuffled = indices.copy(); rng.shuffle(shuffled)
        assignment[shuffled] = np.arange(len(shuffled)) % count
    return assignment


def raw_features(records: list[dict], feature_set: str) -> np.ndarray:
    if feature_set == "task_only":
        encoder = OneHotEncoder(categories=[list(TASKS)], sparse_output=False, handle_unknown="ignore")
        return encoder.fit_transform(np.asarray([record["task"] for record in records], dtype=object)[:, None]).astype(np.float32)
    names = FEATURE_SETS[feature_set]
    return np.stack([
        np.concatenate([
            np.eye(len(TASKS), dtype=np.float32)[record["task_index"]],
            *[record["features"][name] for name in names],
        ])
        for record in records
    ]).astype(np.float32)


def fit_state_transform(train: np.ndarray, test: np.ndarray, feature_set: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if feature_set == "task_only":
        return train, test, {"kind": "identity"}
    varying = np.std(train, axis=0) > 1e-6
    scaler = StandardScaler(); train_scaled = scaler.fit_transform(train[:, varying]); test_scaled = scaler.transform(test[:, varying])
    components = min(32, train_scaled.shape[0] - 2, train_scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=20260913)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled), {
        "kind": "scaled_pca", "varying": varying, "scaler": scaler, "pca": pca,
    }


def expand(state: np.ndarray) -> np.ndarray:
    rows = []
    for vector in state:
        for lambda_index in range(len(LAMBDAS)):
            one_hot = np.eye(len(LAMBDAS), dtype=np.float32)[lambda_index]
            rows.append(np.concatenate([vector, one_hot, np.kron(one_hot, vector)]))
    return np.stack(rows)


def fit_classifier(state: np.ndarray, records: list[dict]) -> LogisticRegression:
    labels = np.concatenate([record["outcomes"] for record in records])
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=20260913)
    classifier.fit(expand(state), labels)
    return classifier


def predict_scores(classifier: LogisticRegression, state: np.ndarray) -> np.ndarray:
    return classifier.predict_proba(expand(state))[:, 1].reshape(len(state), len(LAMBDAS))


def conservative_choices(scores: np.ndarray) -> np.ndarray:
    """Keep lambda=.5 unless a weaker arm clears a fixed success margin."""
    weaker = np.argmax(scores[:, :2], axis=1)
    rows = np.arange(len(scores))
    switch = scores[rows, weaker] > scores[:, 2] + SWITCH_MARGIN
    return np.where(switch, weaker, 2)


def paired(outcomes: np.ndarray, choices: np.ndarray, baseline_index: int = 2) -> dict:
    selected = outcomes[np.arange(len(outcomes)), choices]
    baseline = outcomes[:, baseline_index]
    rescue = int(np.sum((selected == 1) & (baseline == 0)))
    harm = int(np.sum((selected == 0) & (baseline == 1)))
    return {
        "rescue": rescue, "harm": harm, "net": rescue - harm,
        "exact_p": float(binomtest(min(rescue, harm), rescue + harm, 0.5).pvalue) if rescue + harm else 1.0,
    }


def evaluate(records: list[dict], scores: np.ndarray) -> dict:
    outcomes = np.stack([record["outcomes"] for record in records])
    choices = conservative_choices(scores)
    selected = outcomes[np.arange(len(outcomes)), choices]
    fixed = outcomes[:, 2]
    oracle = outcomes.max(axis=1)
    oracle_gap = float(oracle.mean() - fixed.mean())
    model_gap = float(selected.mean() - fixed.mean())
    recovery = model_gap / oracle_gap if oracle_gap > 0 else 0.0
    by_task = {}
    for task in TASKS:
        indices = np.asarray([index for index, record in enumerate(records) if record["task"] == task])
        result = paired(outcomes[indices], choices[indices])
        result.update({
            "router_successes": int(selected[indices].sum()), "fixed_050_successes": int(fixed[indices].sum()),
            "oracle_successes": int(oracle[indices].sum()),
            "choice_counts": {str(float(LAMBDAS[index])): int(np.sum(choices[indices] == index)) for index in range(3)},
        })
        by_task[SHORT[task]] = result
    labels = outcomes.reshape(-1)
    return {
        "row_auc": float(roc_auc_score(labels, scores.reshape(-1))),
        "row_balanced_accuracy": float(balanced_accuracy_score(labels, scores.reshape(-1) >= 0.5)),
        "router_successes": int(selected.sum()), "fixed_050_successes": int(fixed.sum()),
        "oracle_successes": int(oracle.sum()), "oracle_gap": oracle_gap, "model_gap": model_gap,
        "oracle_recovery": recovery, "paired_vs_fixed_050": paired(outcomes, choices), "by_task": by_task,
    }


def grouped_oof(records: list[dict], feature_set: str) -> tuple[np.ndarray, dict]:
    assignment = folds(records); raw = raw_features(records, feature_set)
    scores = np.full((len(records), len(LAMBDAS)), np.nan, dtype=np.float64)
    for fold in range(5):
        train = np.flatnonzero(assignment != fold); test = np.flatnonzero(assignment == fold)
        train_state, test_state, _transform = fit_state_transform(raw[train], raw[test], feature_set)
        classifier = fit_classifier(train_state, [records[index] for index in train])
        scores[test] = predict_scores(classifier, test_state)
    if not np.isfinite(scores).all(): raise RuntimeError("incomplete OOF predictions")
    return scores, evaluate(records, scores)


def pattern_counts(records: list[dict]) -> dict:
    counts = Counter("".join(str(int(value)) for value in record["outcomes"]) for record in records)
    return {pattern: int(counts.get(pattern, 0)) for pattern in ("000", "100", "010", "001", "110", "101", "011", "111")}


def freeze_primary(records: list[dict], artifact: Path) -> dict:
    raw = raw_features(records, PRIMARY_FEATURE_SET)
    transformed, _, transform = fit_state_transform(raw, raw, PRIMARY_FEATURE_SET)
    classifier = fit_classifier(transformed, records)
    payload = {
        "protocol_id": PROTOCOL, "feature_set": PRIMARY_FEATURE_SET, "lambdas": LAMBDAS,
        "state_transform": transform, "classifier": classifier, "argmax_tie_break": "smallest lambda",
        "decision_rule": "fallback to lambda=.5 unless best weaker lambda exceeds q(.5) by switch_margin",
        "switch_margin": SWITCH_MARGIN,
        "training_groups": np.asarray([(record["task_index"], record["seed"]) for record in records], dtype=np.int16),
    }
    path = artifact / "frozen_model/lambda_conditioned_success_predictor.joblib"
    path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(payload, path, compress=3)
    return {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--adaptive-artifact", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--feature-artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    records = load_episode_records(artifact, args.adaptive_artifact.resolve(), args.paired_results.resolve(), args.feature_artifact.resolve())
    outcomes = np.stack([record["outcomes"] for record in records])
    patterns = pattern_counts(records)
    fixed = outcomes[:, 2]; oracle = outcomes.max(axis=1)
    feature_results = {}; prediction_rows = []
    for feature_set in FEATURE_SETS:
        scores, result = grouped_oof(records, feature_set); feature_results[feature_set] = result
        choices = conservative_choices(scores)
        for record, score, choice in zip(records, scores, choices):
            prediction_rows.append({
                "feature_set": feature_set, "task": record["task_short"], "seed": record["seed"],
                "q_0": score[0], "q_025": score[1], "q_050": score[2], "chosen_lambda": float(LAMBDAS[choice]),
                "selected_success": int(record["outcomes"][choice]),
            })
    primary = feature_results[PRIMARY_FEATURE_SET]
    go = {
        "oracle_gap_at_least_0_05": float(oracle.mean() - fixed.mean()) >= 0.05,
        "recovery_above_0_25": primary["oracle_recovery"] > 0.25,
        "router_beats_fixed_050": primary["router_successes"] > primary["fixed_050_successes"],
        "at_least_two_tasks_rescue_gt_harm": sum(row["rescue"] > row["harm"] for row in primary["by_task"].values()) >= 2,
    }
    go["passed"] = all(go.values())
    payload = {
        "protocol_id": PROTOCOL, "episodes": len(records), "lambdas": LAMBDAS.tolist(),
        "pattern_counts": patterns,
        "arm_successes": {str(float(LAMBDAS[index])): int(outcomes[:, index].sum()) for index in range(3)},
        "oracle_successes": int(oracle.sum()), "oracle_gap_vs_050": float(oracle.mean() - fixed.mean()),
        "primary_feature_set": PRIMARY_FEATURE_SET, "feature_results": feature_results,
        "go_no_go": go, "grouping": "all three lambda rows for one (task,seed) stay in one fold",
        "decision_rule": {"fallback_lambda": 0.5, "switch_margin": SWITCH_MARGIN},
    }
    if go["passed"]:
        payload["frozen_model"] = freeze_primary(records, artifact)
    output = artifact / "analysis"; atomic_json(output / "DISCOVERY_RESULTS.json", payload)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    lines = [
        "# L11 Fixed-Lambda Heterogeneity Discovery",
        "",
        "| Pattern (0,.25,.5) | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in patterns.items())
    lines.extend(["", f"- Fixed λ=.5: **{int(fixed.sum())}/400**", f"- Oracle: **{int(oracle.sum())}/400**", f"- Oracle gap: **{100*(oracle.mean()-fixed.mean()):.1f}pp**", "", "| Feature | OOF router | Recovery | Rescue/Harm | Row AUC |", "|---|---:|---:|---:|---:|"])
    for name, result in feature_results.items():
        pair = result["paired_vs_fixed_050"]
        lines.append(f"| {name} | {result['router_successes']}/400 | {100*result['oracle_recovery']:.1f}% | {pair['rescue']}/{pair['harm']} | {result['row_auc']:.3f} |")
    lines.extend(["", f"- Decision: **{'GO' if go['passed'] else 'STOP'}**"])
    (output / "DISCOVERY_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__": main()
