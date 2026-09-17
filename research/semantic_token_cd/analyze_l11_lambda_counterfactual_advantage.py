"""Grouped-OOF counterfactual advantage routing for fixed L11 lambdas."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from research.semantic_token_cd.analyze_l11_lambda_heterogeneity_discovery import (
    LAMBDAS,
    SHORT,
    TASKS,
    load_episode_records,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


PROTOCOL = "PROMPT_ATTN_L11_LAMBDA_COUNTERFACTUAL_ADVANTAGE_V1"
WEAK_INDICES = (0, 1)
BASELINE_INDEX = 2
FEATURE_SETS = {
    "task_only": (),
    "prompt_context": ("prompt_hidden_mean",),
    "prompt_action": ("prompt_hidden_mean", "action_context_hidden"),
}
PRIMARY_FEATURE_SET = "prompt_action"
PCA_COMPONENTS = 16
PREFERENCE_THRESHOLD = 0.65
RANDOM_SEED = 20260914


def fold_assignment(records: list[dict], count: int = 5) -> np.ndarray:
    assignment = np.empty(len(records), dtype=np.int64)
    rng = np.random.default_rng(RANDOM_SEED)
    for task in TASKS:
        indices = np.asarray([index for index, record in enumerate(records) if record["task"] == task])
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        assignment[shuffled] = np.arange(len(shuffled)) % count
    return assignment


def task_one_hot(records: list[dict]) -> np.ndarray:
    return np.eye(len(TASKS), dtype=np.float32)[[record["task_index"] for record in records]]


def raw_state(records: list[dict], feature_set: str) -> np.ndarray:
    names = FEATURE_SETS[feature_set]
    if not names:
        return np.zeros((len(records), 0), dtype=np.float32)
    return np.stack([
        np.concatenate([record["features"][name] for name in names])
        for record in records
    ]).astype(np.float32)


def fit_transform(
    train_records: list[dict],
    test_records: list[dict],
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_task = task_one_hot(train_records)
    test_task = task_one_hot(test_records)
    if feature_set == "task_only":
        return train_task, test_task
    train_raw = raw_state(train_records, feature_set)
    test_raw = raw_state(test_records, feature_set)
    varying = np.std(train_raw, axis=0) > 1e-6
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw[:, varying])
    test_scaled = scaler.transform(test_raw[:, varying])
    components = min(PCA_COMPONENTS, train_scaled.shape[0] - 2, train_scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=RANDOM_SEED)
    train_state = pca.fit_transform(train_scaled)
    test_state = pca.transform(test_scaled)
    train_heads = np.einsum("nt,nd->ntd", train_task, train_state).reshape(len(train_records), -1)
    test_heads = np.einsum("nt,nd->ntd", test_task, test_state).reshape(len(test_records), -1)
    return np.concatenate([train_task, train_heads], axis=1), np.concatenate([test_task, test_heads], axis=1)


def preference_labels(records: list[dict], weak_index: int) -> tuple[np.ndarray, np.ndarray]:
    outcomes = np.stack([record["outcomes"] for record in records])
    weak = outcomes[:, weak_index]
    baseline = outcomes[:, BASELINE_INDEX]
    discordant = weak != baseline
    labels = ((weak == 1) & (baseline == 0)).astype(np.int64)
    return discordant, labels


def fit_preference(features: np.ndarray, records: list[dict], weak_index: int) -> LogisticRegression:
    discordant, labels = preference_labels(records, weak_index)
    if labels[discordant].size < 8 or np.unique(labels[discordant]).size != 2:
        raise RuntimeError(f"insufficient discordant labels for lambda={LAMBDAS[weak_index]}")
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=4000,
        random_state=RANDOM_SEED,
    )
    classifier.fit(features[discordant], labels[discordant])
    return classifier


def conservative_choices(scores: np.ndarray) -> np.ndarray:
    best_weak = np.argmax(scores, axis=1)
    rows = np.arange(len(scores))
    return np.where(scores[rows, best_weak] >= PREFERENCE_THRESHOLD, best_weak, BASELINE_INDEX)


def paired(outcomes: np.ndarray, choices: np.ndarray) -> dict:
    selected = outcomes[np.arange(len(outcomes)), choices]
    baseline = outcomes[:, BASELINE_INDEX]
    rescue = int(np.sum((selected == 1) & (baseline == 0)))
    harm = int(np.sum((selected == 0) & (baseline == 1)))
    return {
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "exact_p": float(binomtest(min(rescue, harm), rescue + harm, 0.5).pvalue) if rescue + harm else 1.0,
    }


def preference_metrics(records: list[dict], scores: np.ndarray) -> dict:
    result = {}
    for column, weak_index in enumerate(WEAK_INDICES):
        discordant, labels = preference_labels(records, weak_index)
        y_true = labels[discordant]
        y_score = scores[discordant, column]
        result[str(float(LAMBDAS[weak_index]))] = {
            "discordant_count": int(discordant.sum()),
            "weak_wins": int(y_true.sum()),
            "baseline_wins": int(y_true.size - y_true.sum()),
            "auc": float(roc_auc_score(y_true, y_score)),
            "balanced_accuracy_at_threshold": float(
                balanced_accuracy_score(y_true, y_score >= PREFERENCE_THRESHOLD)
            ),
        }
    return result


def evaluate(records: list[dict], scores: np.ndarray) -> dict:
    outcomes = np.stack([record["outcomes"] for record in records])
    choices = conservative_choices(scores)
    selected = outcomes[np.arange(len(outcomes)), choices]
    baseline = outcomes[:, BASELINE_INDEX]
    oracle = outcomes.max(axis=1)
    pair = paired(outcomes, choices)
    by_task = {}
    for task in TASKS:
        indices = np.asarray([index for index, record in enumerate(records) if record["task"] == task])
        task_pair = paired(outcomes[indices], choices[indices])
        task_pair.update({
            "router_successes": int(selected[indices].sum()),
            "fixed_050_successes": int(baseline[indices].sum()),
            "oracle_successes": int(oracle[indices].sum()),
            "choice_counts": {
                str(float(LAMBDAS[index])): int(np.sum(choices[indices] == index))
                for index in range(3)
            },
        })
        by_task[SHORT[task]] = task_pair
    return {
        "router_successes": int(selected.sum()),
        "fixed_050_successes": int(baseline.sum()),
        "oracle_successes": int(oracle.sum()),
        "paired_vs_fixed_050": pair,
        "oracle_recovery": pair["net"] / max(1, int(oracle.sum() - baseline.sum())),
        "switch_count": int(np.sum(choices != BASELINE_INDEX)),
        "preference_metrics": preference_metrics(records, scores),
        "by_task": by_task,
    }


def grouped_oof(records: list[dict], feature_set: str) -> tuple[np.ndarray, dict]:
    assignment = fold_assignment(records)
    scores = np.full((len(records), len(WEAK_INDICES)), np.nan, dtype=np.float64)
    for fold in range(5):
        train_indices = np.flatnonzero(assignment != fold)
        test_indices = np.flatnonzero(assignment == fold)
        train_records = [records[index] for index in train_indices]
        test_records = [records[index] for index in test_indices]
        train_features, test_features = fit_transform(train_records, test_records, feature_set)
        for column, weak_index in enumerate(WEAK_INDICES):
            classifier = fit_preference(train_features, train_records, weak_index)
            scores[test_indices, column] = classifier.predict_proba(test_features)[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("incomplete grouped-OOF scores")
    return scores, evaluate(records, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--discovery-artifact", type=Path, required=True)
    parser.add_argument("--adaptive-artifact", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--feature-artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    records = load_episode_records(
        args.discovery_artifact.resolve(),
        args.adaptive_artifact.resolve(),
        args.paired_results.resolve(),
        args.feature_artifact.resolve(),
    )
    feature_results = {}
    prediction_rows = []
    for feature_set in FEATURE_SETS:
        scores, result = grouped_oof(records, feature_set)
        feature_results[feature_set] = result
        choices = conservative_choices(scores)
        for record, score, choice in zip(records, scores, choices):
            prediction_rows.append({
                "feature_set": feature_set,
                "task": record["task_short"],
                "seed": record["seed"],
                "preference_0_vs_050": score[0],
                "preference_025_vs_050": score[1],
                "chosen_lambda": float(LAMBDAS[choice]),
                "selected_success": int(record["outcomes"][choice]),
            })
    primary = feature_results[PRIMARY_FEATURE_SET]
    go = {
        "router_at_least_246_of_400": primary["router_successes"] >= 246,
        "rescue_gt_harm": primary["paired_vs_fixed_050"]["rescue"] > primary["paired_vs_fixed_050"]["harm"],
        "pick_net_nonnegative": primary["by_task"]["pick_coke_can"]["net"] >= 0,
        "at_least_two_tasks_net_positive": sum(result["net"] > 0 for result in primary["by_task"].values()) >= 2,
    }
    go["passed"] = all(go.values())
    payload = {
        "protocol_id": PROTOCOL,
        "episodes": len(records),
        "training_target": "pairwise weak-lambda preference on discordant outcomes only",
        "lambdas": LAMBDAS.tolist(),
        "baseline_lambda": 0.5,
        "preference_threshold": PREFERENCE_THRESHOLD,
        "pca_components": PCA_COMPONENTS,
        "primary_feature_set": PRIMARY_FEATURE_SET,
        "grouping": "one (task,seed) is assigned to exactly one OOF fold",
        "feature_results": feature_results,
        "go_no_go": go,
        "new_gpu_rollouts": 0,
    }
    output = artifact / "analysis"
    atomic_json(output / "COUNTERFACTUAL_ADVANTAGE_RESULTS.json", payload)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    lines = [
        "# L11 Lambda Counterfactual Advantage OOF", "",
        f"- Preference threshold: **{PREFERENCE_THRESHOLD:.2f}**",
        "- Default arm: **lambda=0.5**", "",
        "| Feature | Router | Rescue | Harm | Net | Oracle recovery | Switches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in feature_results.items():
        pair = result["paired_vs_fixed_050"]
        lines.append(
            f"| {name} | {result['router_successes']}/400 | {pair['rescue']} | {pair['harm']} | "
            f"{pair['net']:+d} | {100*result['oracle_recovery']:.1f}% | {result['switch_count']} |"
        )
    lines.extend(["", "## Primary task results", "", "| Task | Router | Fixed .5 | Oracle | Rescue | Harm | Net |", "|---|---:|---:|---:|---:|---:|---:|"])
    for task in TASKS:
        result = primary["by_task"][SHORT[task]]
        lines.append(
            f"| {SHORT[task]} | {result['router_successes']}/100 | {result['fixed_050_successes']}/100 | "
            f"{result['oracle_successes']}/100 | {result['rescue']} | {result['harm']} | {result['net']:+d} |"
        )
    lines.extend(["", f"- Decision: **{'GO' if go['passed'] else 'STOP'}**"])
    (output / "COUNTERFACTUAL_ADVANTAGE_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
