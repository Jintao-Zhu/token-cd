"""Apply frozen task routers once to held-out seeds 100-199."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from research.semantic_token_cd.l11_initial_state_router_protocol import (
    BASELINE,
    FEATURE_NAMES,
    PROTOCOL,
    SHORT_TASKS,
    TASKS,
    TEST_SEEDS,
    THRESHOLD,
    atomic_json,
    feature_path,
    load_outcomes,
    metadata_path,
    sha256,
)


def predict(model: dict, vector: np.ndarray) -> float:
    selected = vector[model["varying"]][None]
    scaled = model["scaler"].transform(selected)
    reduced = model["pca"].transform(scaled)
    return float(model["classifier"].predict_proba(reduced)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    outcomes = load_outcomes(args.paired_results.resolve(), TEST_SEEDS)
    frozen = json.loads((artifact / "FROZEN_MODELS.json").read_text())
    if frozen.get("protocol_id") != PROTOCOL or frozen.get("threshold") != THRESHOLD:
        raise RuntimeError("frozen manifest mismatch")

    task_results = {}
    rows = []
    all_labels = []
    all_predictions = []
    router_successes = l11_successes = shr_successes = 0
    for task in TASKS:
        short = SHORT_TASKS[task]
        model_info = frozen["models"][short]
        model_path = Path(model_info["path"])
        if sha256(model_path) != model_info["sha256"]:
            raise RuntimeError(f"frozen model hash mismatch: {model_path}")
        model = joblib.load(model_path)
        labels = []
        scores = []
        task_router = task_l11 = task_shr = 0
        for seed in TEST_SEEDS:
            path = feature_path(artifact, task, seed)
            metadata = json.loads(metadata_path(artifact, task, seed).read_text())
            if metadata.get("protocol_id") != PROTOCOL:
                raise RuntimeError("held-out feature protocol mismatch")
            with np.load(path) as arrays:
                vector = np.concatenate([
                    np.asarray(arrays[name], dtype=np.float32).reshape(-1)
                    for name in FEATURE_NAMES
                ])
            score = predict(model, vector)
            outcome = outcomes[(task, seed)]
            use_l11 = score >= THRESHOLD
            chosen_success = outcome["l11_matched"] if use_l11 else outcome[BASELINE]
            task_router += int(chosen_success)
            task_l11 += int(outcome["l11_matched"])
            task_shr += int(outcome[BASELINE])
            if outcome["l11_matched"] != outcome[BASELINE]:
                label = int(outcome["l11_matched"])
                labels.append(label)
                scores.append(score)
                all_labels.append(label)
                all_predictions.append(score)
                rows.append({
                    "task": short, "seed": seed, "label_prefer_l11": label,
                    "score_prefer_l11": score, "prediction_prefer_l11": int(use_l11),
                })
        labels_array = np.asarray(labels, dtype=np.int64)
        scores_array = np.asarray(scores, dtype=np.float64)
        task_results[short] = {
            "discordant_count": int(labels_array.size),
            "prefer_l11": int(labels_array.sum()),
            "prefer_shr": int(labels_array.size - labels_array.sum()),
            "auc": float(roc_auc_score(labels_array, scores_array)),
            "accuracy": float(accuracy_score(labels_array, scores_array >= THRESHOLD)),
            "balanced_accuracy": float(balanced_accuracy_score(labels_array, scores_array >= THRESHOLD)),
            "router_successes": task_router,
            "l11_fixed_successes": task_l11,
            "shr_fixed_successes": task_shr,
        }
        router_successes += task_router
        l11_successes += task_l11
        shr_successes += task_shr

    valid_core = [task_results[name]["auc"] for name in ("open_drawer", "close_drawer", "move_near")]
    macro_balanced = float(np.mean([result["balanced_accuracy"] for result in task_results.values()]))
    go = {
        "at_least_two_core_tasks_auc_ge_0_70": sum(value >= 0.70 for value in valid_core) >= 2,
        "macro_balanced_accuracy_ge_0_65": macro_balanced >= 0.65,
    }
    go["passed"] = all(go.values())
    payload = {
        "protocol_id": PROTOCOL,
        "test_seeds": [min(TEST_SEEDS), max(TEST_SEEDS)],
        "feature_set": list(FEATURE_NAMES),
        "threshold": THRESHOLD,
        "task_results": task_results,
        "macro_balanced_accuracy": macro_balanced,
        "pooled_discordant_auc": float(roc_auc_score(all_labels, all_predictions)),
        "router_successes": router_successes,
        "l11_fixed_successes": l11_successes,
        "shr_fixed_successes": shr_successes,
        "go_no_go": go,
        "retraining_or_calibration_on_test": False,
    }
    output = artifact / "confirmation"
    atomic_json(output / "CONFIRMATION_RESULTS.json", payload)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "discordant_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Frozen Initial-State Router Confirmation",
        "",
        "Models, preprocessing, feature set, and threshold were frozen on seeds 0-99 before held-out feature collection.",
        "",
        "| Task | Discordant | AUC | Accuracy | Balanced accuracy | Router | L11 | SHR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        result = task_results[SHORT_TASKS[task]]
        lines.append(
            f"| {SHORT_TASKS[task]} | {result['discordant_count']} | {result['auc']:.3f} | "
            f"{result['accuracy']:.3f} | {result['balanced_accuracy']:.3f} | "
            f"{result['router_successes']}/100 | {result['l11_fixed_successes']}/100 | {result['shr_fixed_successes']}/100 |"
        )
    lines.extend([
        "",
        f"- Macro balanced accuracy: **{macro_balanced:.3f}**",
        f"- Router total: **{router_successes}/400**; L11: **{l11_successes}/400**; SHR: **{shr_successes}/400**",
        f"- Decision: **{'GO' if go['passed'] else 'STOP'}**",
    ])
    (output / "CONFIRMATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
