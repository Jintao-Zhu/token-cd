from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.token_pcd_residual_anatomy.analyze import (cosine, downstream_logits, logit_direction,
                                                          probability_direction)
from research.token_pcd_stage_a.core import append_jsonl, read_jsonl
from research.token_pcd_stage_a.run import load_model


PRIMARY_RANK = 32
RANKS = (8, 16, 32, 64)
OBJECT_THRESHOLD = 0.25
HIDDEN = 256
EPOCHS = 250
LR = 1e-3
WEIGHT_DECAY = 1e-4
RIDGE_ALPHA = 1.0
BOOTSTRAP_DRAWS = 20000


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


class CoefficientMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, output_dim))

    def forward(self, value):
        return self.network(value)


def standardize(train, test, epsilon=1e-6):
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(epsilon)
    return (train - mean) / std, (test - mean) / std, mean, std


def train_mlp(train_x, train_y, test_x, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x_train, x_test, x_mean, x_std = standardize(train_x, test_x)
    y_train, _, y_mean, y_std = standardize(train_y, train_y)
    model = CoefficientMLP(x_train.shape[1], train_y.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x_train, x_test, y_train = x_train.cuda(), x_test.cuda(), y_train.cuda()
    model.train()
    loss = None
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x_train) - y_train).square())
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x_test).cpu() * y_std + y_mean
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    metadata = {"seed": seed, "final_train_loss": float(loss.detach().cpu()),
                "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
                "model_state": state}
    del model, x_train, x_test, y_train
    torch.cuda.empty_cache()
    return prediction, metadata


def ridge_predict(train_x, train_y, test_x):
    x_train, x_test, x_mean, x_std = standardize(train_x, test_x)
    y_train, _, y_mean, y_std = standardize(train_y, train_y)
    scale = math.sqrt(x_train.shape[1])
    x_train, x_test, y_train = (x_train / scale).cuda(), (x_test / scale).cuda(), y_train.cuda()
    kernel = x_train @ x_train.T
    weights = torch.linalg.solve(kernel + RIDGE_ALPHA * torch.eye(len(x_train), device=kernel.device), y_train)
    prediction = (x_test @ x_train.T @ weights).cpu() * y_std + y_mean
    del x_train, x_test, y_train, kernel, weights
    torch.cuda.empty_cache()
    return prediction, {"alpha": RIDGE_ALPHA, "x_mean": x_mean, "x_std": x_std,
                        "y_mean": y_mean, "y_std": y_std}


def pooled_inputs(clean, overlaps):
    global_pool = clean.mean(axis=1)
    object_pool = np.empty_like(global_pool)
    for index in range(len(clean)):
        selected = overlaps[index] >= OBJECT_THRESHOLD
        if not selected.any():
            raise RuntimeError(f"No object tokens for input row {index}")
        object_pool[index] = clean[index, selected].mean(axis=0)
    area = overlaps.mean(axis=1, keepdims=True)
    return {"primary": np.concatenate((object_pool, global_pool, area), axis=1),
            "object_only": np.concatenate((object_pool, area), axis=1),
            "global_only": np.concatenate((global_pool, area), axis=1)}


def project_coefficients(delta, basis):
    delta_gpu, basis_gpu = delta.cuda(), basis.cuda()
    result = (delta_gpu.reshape(-1, delta_gpu.shape[-1]) @ basis_gpu).reshape(delta.shape[0], delta.shape[1], -1).cpu()
    del delta_gpu, basis_gpu
    torch.cuda.empty_cache()
    return result


def cluster_bootstrap(rows, value, seed, paired_right=None):
    tasks = sorted(set(row["task"] for row in rows))
    grouped = {task: [row for row in rows if row["task"] == task] for task in tasks}
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = []
        for task_index in rng.integers(0, len(tasks), len(tasks)):
            task_rows = grouped[tasks[task_index]]
            sampled.extend(task_rows[index] for index in rng.integers(0, len(task_rows), len(task_rows)))
        left = np.asarray([row[value] for row in sampled], dtype=float)
        if paired_right is None:
            estimates[draw] = np.median(left)
        else:
            estimates[draw] = np.median(left - np.asarray([row[paired_right] for row in sampled], dtype=float))
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)), "mean": float(np.mean(values)),
            "iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))]}


def summarize(primary_rows, coefficient_rows):
    predicted = [row["predicted_cosine"] for row in primary_rows]
    norm = [row["predicted_norm_ratio"] for row in primary_rows]
    diff_mean = [row["predicted_cosine"] - row["mean_cosine"] for row in primary_rows]
    diff_spatial = [row["predicted_cosine"] - row["spatial_cosine"] for row in primary_rows]
    per_task = {}
    for task in sorted(set(row["task"] for row in primary_rows)):
        task_rows = [row for row in primary_rows if row["task"] == task]
        per_task[task] = {name: float(np.median([row[name] for row in task_rows]))
                          for name in ("predicted_cosine", "oracle_cosine", "mean_cosine", "ridge_cosine",
                                       "random_cosine", "spatial_cosine", "clean_pca_cosine")}
        per_task[task]["predicted_better_than_mean"] = bool(per_task[task]["predicted_cosine"] > per_task[task]["mean_cosine"])
    tasks_better = sum(row["predicted_better_than_mean"] for row in per_task.values())
    primary_median = float(np.median(predicted)); mean_improvement = float(np.median(diff_mean))
    spatial_improvement = float(np.median(diff_spatial))
    mean_ci = cluster_bootstrap(primary_rows, "predicted_cosine", 202608152, "mean_cosine")
    spatial_ci = cluster_bootstrap(primary_rows, "predicted_cosine", 202608153, "spatial_cosine")
    checks = {"predicted_cosine_ge_0_65": bool(primary_median >= .65),
              "improvement_over_mean_ge_0_15": bool(mean_improvement >= .15),
              "improvement_over_spatial_ge_0_10": bool(spatial_improvement >= .10),
              "both_improvement_ci_lower_gt_zero": bool(mean_ci[0] > 0 and spatial_ci[0] > 0),
              "effect_norm_ratio_in_0_7_1_3": bool(.7 <= np.median(norm) <= 1.3),
              "at_least_8_tasks_better_than_mean": bool(tasks_better >= 8),
              "no_task_collapse": bool(min(row["predicted_cosine"] for row in per_task.values()) >= 0)}
    if all(checks.values()):
        decision = "LATENT_PCD_COEFFICIENT_PREDICTION_GO"
        status = "GO"
    elif .50 <= primary_median < .65 and mean_improvement > 0 and spatial_improvement > 0:
        decision = "INCONCLUSIVE"
        status = "INCONCLUSIVE"
    else:
        decision = "NO_GO"
        status = "NO_GO"
    methods = ("predicted", "oracle", "mean", "ridge", "random", "spatial", "clean_pca")
    method_stats = {method: distribution([row[f"{method}_cosine"] for row in primary_rows]) for method in methods}
    method_stats["predicted"]["cluster_bootstrap_95_ci"] = cluster_bootstrap(primary_rows, "predicted_cosine", 202608151)
    true = np.concatenate([row["true"].reshape(-1, PRIMARY_RANK) for row in coefficient_rows], axis=0)
    pred = np.concatenate([row["predicted"].reshape(-1, PRIMARY_RANK) for row in coefficient_rows], axis=0)
    baseline = np.concatenate([row["train_mean"].reshape(-1, PRIMARY_RANK) for row in coefficient_rows], axis=0)
    r2 = 1.0 - float(np.square(pred - true).sum() / np.square(true - baseline).sum())
    correlations = [float(np.corrcoef(true[:, index], pred[:, index])[0, 1]) for index in range(PRIMARY_RANK)]
    return {"status": status, "decision": decision, "states": len(primary_rows), "tasks": len(per_task),
            "primary": {"methods": method_stats, "paired_median_improvement_over_mean": mean_improvement,
                        "paired_median_improvement_over_spatial": spatial_improvement,
                        "improvement_over_mean_cluster_bootstrap_95_ci": mean_ci,
                        "improvement_over_spatial_cluster_bootstrap_95_ci": spatial_ci,
                        "predicted_effect_norm_ratio": distribution(norm), "per_task": per_task,
                        "tasks_predicted_better_than_mean": int(tasks_better)},
            "coefficient_quality": {"median_state_cosine": float(np.median([row["cosine"] for row in coefficient_rows])),
                                    "global_r2_vs_training_mean": r2,
                                    "per_basis_dimension_correlation": correlations},
            "checks": checks, "rollout_performed": False, "next_protocol_ready": bool(status == "GO")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    pcd_root, anatomy, artifact = args.pcd_root.resolve(), args.anatomy.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True); (artifact / "coefficients").mkdir(exist_ok=True)
    rows = read_jsonl(anatomy / "extraction.jsonl")
    if len(rows) != 270 or len(set(row["task"] for row in rows)) != 9:
        raise RuntimeError("Expected complete 270-state, nine-task anatomy artifact")
    bases = torch.load(anatomy / "loto_bases.pt", map_location="cpu", weights_only=True)["bases"]
    prior = read_jsonl(anatomy / "reconstruction.jsonl")
    prior_lookup = {(row["state_id"], row["method"], row["rank"]): row for row in prior}
    clean = np.empty((270, 256, 4096), dtype=np.float32)
    delta = np.empty_like(clean); overlaps = np.empty((270, 256), dtype=np.float32)
    clean_logits, pixel_logits = [], []
    for index, row in enumerate(rows):
        with np.load(anatomy / row["state_file"]) as payload:
            clean[index] = payload["projector_clean"]; delta[index] = payload["projector_delta"]
            overlaps[index] = payload["overlaps"]; clean_logits.append(payload["clean_logits"].copy())
            pixel_logits.append(payload["pixel_logits"].copy())
    inputs = pooled_inputs(clean, overlaps)
    tasks = sorted(set(row["task"] for row in rows)); fold_predictions = {}; model_artifacts = {}
    coefficient_rows = []
    for fold, heldout in enumerate(tasks):
        train_idx = np.asarray([i for i, row in enumerate(rows) if row["task"] != heldout])
        test_idx = np.asarray([i for i, row in enumerate(rows) if row["task"] == heldout])
        basis64 = bases[heldout]["residual_svd"][:, :64]
        coefficients = project_coefficients(torch.from_numpy(delta), basis64).numpy()
        train64 = torch.from_numpy(coefficients[train_idx].reshape(len(train_idx), -1))
        test64 = torch.from_numpy(coefficients[test_idx].reshape(len(test_idx), -1))
        train32, test32 = train64.reshape(len(train_idx), 256, 64)[:, :, :32].reshape(len(train_idx), -1), test64.reshape(len(test_idx), 256, 64)[:, :, :32].reshape(len(test_idx), -1)
        x_train = torch.from_numpy(inputs["primary"][train_idx]); x_test = torch.from_numpy(inputs["primary"][test_idx])
        pred32, meta32 = train_mlp(x_train, train32, x_test, stable_seed(heldout + "|primary32"))
        if fold == 0:
            repeat32, _ = train_mlp(x_train, train32, x_test, stable_seed(heldout + "|primary32"))
            deterministic_error = float((pred32 - repeat32).abs().max())
            if deterministic_error != 0:
                raise RuntimeError(f"MLP deterministic rerun failed: {deterministic_error}")
        pred64, meta64 = train_mlp(x_train, train64, x_test, stable_seed(heldout + "|secondary64"))
        ridge32, ridge_meta = ridge_predict(x_train, train32, x_test)
        object_pred, object_meta = train_mlp(torch.from_numpy(inputs["object_only"][train_idx]), train32,
                                             torch.from_numpy(inputs["object_only"][test_idx]), stable_seed(heldout + "|object_only"))
        global_pred, global_meta = train_mlp(torch.from_numpy(inputs["global_only"][train_idx]), train32,
                                             torch.from_numpy(inputs["global_only"][test_idx]), stable_seed(heldout + "|global_only"))
        mean32 = train32.mean(dim=0)
        fold_predictions[heldout] = {"test_idx": test_idx, "true64": test64.reshape(len(test_idx), 256, 64),
                                     "pred32": pred32.reshape(len(test_idx), 256, 32),
                                     "pred64": pred64.reshape(len(test_idx), 256, 64),
                                     "ridge32": ridge32.reshape(len(test_idx), 256, 32),
                                     "object32": object_pred.reshape(len(test_idx), 256, 32),
                                     "global32": global_pred.reshape(len(test_idx), 256, 32),
                                     "mean32": mean32.reshape(256, 32), "train32": train32.reshape(len(train_idx), 256, 32),
                                     "train_idx": train_idx}
        model_artifacts[heldout] = {"train_tasks": sorted(set(rows[i]["task"] for i in train_idx)),
                                    "test_task": heldout, "primary32": meta32, "secondary64": meta64,
                                    "ridge32": ridge_meta, "object_only32": object_meta, "global_only32": global_meta}
        print(json.dumps({"phase": "train", "fold": fold + 1, "heldout": heldout,
                          "primary_loss": meta32["final_train_loss"]}), flush=True)
    torch.save(model_artifacts, artifact / "predictors.pt")
    del delta
    checkpoint = pcd_root / "source/PCD/pretrained/openvla-7b"; model, _ = load_model(checkpoint)
    results = artifact / "results.jsonl"; primary_output = artifact / "primary_state_results.jsonl"
    for path in (results, primary_output):
        if path.exists(): path.unlink()
    primary_rows = []; rank_rows = []
    for ordinal, row in enumerate(rows, 1):
        fold_data = fold_predictions[row["task"]]; local = int(np.flatnonzero(fold_data["test_idx"] == ordinal - 1)[0])
        basis = bases[row["task"]]["residual_svd"].to(model.device)
        clean_h = torch.from_numpy(clean[ordinal - 1]).to(model.device)
        true64 = fold_data["true64"][local]
        predicted32 = fold_data["pred32"][local]; predicted64 = fold_data["pred64"][local]
        ridge32 = fold_data["ridge32"][local]; mean32 = fold_data["mean32"]
        random_index = stable_seed(row["state_id"] + "|random_coeff") % len(fold_data["train32"])
        random32 = fold_data["train32"][random_index]
        fields = {"oracle32": true64[:, :32], "predicted32": predicted32, "mean32": mean32,
                  "zero32": torch.zeros_like(predicted32), "random32": random32, "ridge32": ridge32,
                  "object_only32": fold_data["object32"][local], "global_only32": fold_data["global32"][local],
                  "oracle8": true64[:, :8], "oracle16": true64[:, :16], "oracle64": true64,
                  "predicted8": predicted32[:, :8], "predicted16": predicted32[:, :16], "predicted64": predicted64}
        logits_by_method = {}
        for name, coefficient in fields.items():
            rank = coefficient.shape[-1]
            residual = coefficient.to(model.device) @ basis[:, :rank].T
            synthetic = (clean_h - residual).to(torch.bfloat16)
            if synthetic.shape != clean_h.shape or not torch.isfinite(synthetic).all():
                raise RuntimeError(f"Invalid synthetic projector for {row['state_id']} {name}")
            logits_by_method[name] = downstream_logits(model, row["teacher_ids"], synthetic.unsqueeze(0))[0]
        clean_l, pixel_l = torch.from_numpy(clean_logits[ordinal - 1]), torch.from_numpy(pixel_logits[ordinal - 1])
        pixel_direction = logit_direction(clean_l, pixel_l); pixel_probability = probability_direction(clean_l, pixel_l)
        pixel_norm = torch.linalg.vector_norm(pixel_direction.double())
        state_metrics = {}
        for name, logits in logits_by_method.items():
            direction = logit_direction(clean_l, logits); probability = probability_direction(clean_l, logits)
            state_metrics[name] = {"logit_cosine": cosine(direction, pixel_direction) if name != "zero32" else None,
                                   "probability_cosine": cosine(probability, pixel_probability) if name != "zero32" else None,
                                   "effect_norm_ratio": float(torch.linalg.vector_norm(direction.double()) / pixel_norm)}
        spatial = prior_lookup[(row["state_id"], "spatial_only", None)]
        clean_pca = prior_lookup[(row["state_id"], "clean_svd", 32)]
        primary = {"state_id": row["state_id"], "task": row["task"],
                   "predicted_cosine": state_metrics["predicted32"]["logit_cosine"],
                   "predicted_norm_ratio": state_metrics["predicted32"]["effect_norm_ratio"],
                   "oracle_cosine": state_metrics["oracle32"]["logit_cosine"],
                   "mean_cosine": state_metrics["mean32"]["logit_cosine"],
                   "ridge_cosine": state_metrics["ridge32"]["logit_cosine"],
                   "random_cosine": state_metrics["random32"]["logit_cosine"],
                   "spatial_cosine": spatial["logit_cosine"], "clean_pca_cosine": clean_pca["logit_cosine"]}
        append_jsonl(primary_output, primary); primary_rows.append(primary)
        for name, metrics in state_metrics.items():
            method = name.rstrip("0123456789"); rank = int(name[len(method):])
            record = {"state_id": row["state_id"], "task": row["task"], "method": method, "rank": rank, **metrics}
            append_jsonl(results, record); rank_rows.append(record)
        true32_np, pred32_np = true64[:, :32].numpy(), predicted32.numpy()
        train_mean_np = mean32.numpy()
        coeff_cos = cosine(torch.from_numpy(pred32_np), torch.from_numpy(true32_np))
        coeff_record = {"state_id": row["state_id"], "task": row["task"], "cosine": coeff_cos,
                        "true": true32_np, "predicted": pred32_np,
                        "train_mean": train_mean_np}
        coefficient_rows.append(coeff_record)
        np.savez_compressed(artifact / "coefficients" / f"{row['state_id']}.npz",
                            true=true32_np.astype(np.float16), predicted=pred32_np.astype(np.float16),
                            ridge=fold_data["ridge32"][local].numpy().astype(np.float16),
                            train_mean=train_mean_np.astype(np.float16))
        print(json.dumps({"phase": "replay", "ordinal": ordinal, "total": len(rows), "state_id": row["state_id"]}), flush=True)
    summary = summarize(primary_rows, coefficient_rows)
    rank_curve = {}
    for rank in RANKS:
        rank_curve[str(rank)] = {}
        for method in ("oracle", "predicted"):
            selected = [row for row in rank_rows if row["method"] == method and row["rank"] == rank]
            rank_curve[str(rank)][method] = distribution([row["logit_cosine"] for row in selected])
    for method in ("object_only", "global_only"):
        selected = [row for row in rank_rows if row["method"] == method and row["rank"] == 32]
        summary.setdefault("input_ablations", {})[method] = distribution([row["logit_cosine"] for row in selected])
    summary["rank_curve"] = rank_curve
    summary["integrity"] = {"states_complete": 270, "task_split_leakage": False,
                            "basis_heldout_leakage": False, "predictor_pixel_inputs": False,
                            "basis_orthonormal_max_error": max(float((value["residual_svd"][:, :64].T @ value["residual_svd"][:, :64] - torch.eye(64)).abs().max()) for value in bases.values()),
                            "deterministic_rerun_max_abs_error": deterministic_error,
                            "all_finite": True, "synthetic_shape": [256, 4096],
                            "protected_teacher_prefix": True, "rollout_count": 0}
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    decision = {"decision": summary["decision"], "checks": summary["checks"],
                "rollout_performed": False,
                "next_step": "WAIT_FOR_HUMAN_AUTHORIZATION" if summary["status"] == "GO" else "STOP_AND_REPORT"}
    (artifact / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
