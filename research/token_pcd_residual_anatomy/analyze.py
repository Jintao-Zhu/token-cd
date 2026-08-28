from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research.token_pcd_stage_a.core import action_token_slice, append_jsonl, read_jsonl
from research.token_pcd_stage_a.run import load_model


RANKS = (8, 16, 32, 64)
SVD_Q = 80
SVD_NITER = 4
VISUAL_TOKENS = 256


def cosine(left, right):
    left, right = left.flatten().double(), right.flatten().double()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator) if denominator > 0 else float("nan")


def logit_direction(clean, branch):
    delta = clean.float() - branch.float()
    return delta - delta.mean(dim=-1, keepdim=True)


def probability_direction(clean, branch):
    return torch.softmax(clean.float(), dim=-1) - torch.softmax(branch.float(), dim=-1)


def bootstrap_median(values, seed, draws=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[samples], axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def bootstrap_median_advantage(left, right, seed, draws=20000):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(left), size=(draws, len(left)))
    values = np.median(left[samples], axis=1) - np.median(right[samples], axis=1)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def fit_basis(rows, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    matrix = torch.from_numpy(np.ascontiguousarray(rows)).cuda()
    _, _, basis = torch.pca_lowrank(matrix, q=SVD_Q, center=False, niter=SVD_NITER)
    basis = basis[:, :max(RANKS)].detach().float().cpu().contiguous()
    del matrix
    torch.cuda.empty_cache()
    return basis


def random_basis(dim, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dim, max(RANKS), generator=generator, dtype=torch.float32)
    return torch.linalg.qr(matrix, mode="reduced").Q.contiguous()


def project(delta, basis, rank):
    vectors = basis[:, :rank]
    return (delta @ vectors) @ vectors.T


@torch.inference_mode()
def downstream_logits(model, teacher_ids, projectors, clean_action_count=7):
    batch = projectors.shape[0]
    ids = torch.tensor([teacher_ids], device=model.device, dtype=torch.long)
    text = model.get_input_embeddings()(ids).expand(batch, -1, -1)
    multimodal = torch.cat((text[:, :1], projectors, text[:, 1:]), dim=1)
    attention_mask = torch.ones(multimodal.shape[:2], device=model.device, dtype=torch.long)
    output = model.language_model(input_ids=None, attention_mask=attention_mask, inputs_embeds=multimodal,
                                  use_cache=False, output_hidden_states=False, return_dict=True)
    base_length = len(teacher_ids) - (clean_action_count - 1)
    query = [VISUAL_TOKENS + base_length - 1 + offset for offset in range(clean_action_count)]
    return output.logits[:, query, action_token_slice(model)].detach().float().cpu()


def replay_sentinel(model, extraction, row):
    with np.load(extraction / row["state_file"]) as payload:
        clean_projector = torch.from_numpy(payload["projector_clean"].copy()).to(model.device, dtype=torch.float32)
        delta = torch.from_numpy(payload["projector_delta"].copy()).to(model.device, dtype=torch.float32)
        expected_clean = torch.from_numpy(payload["clean_logits"].copy())
        expected_pixel = torch.from_numpy(payload["pixel_logits"].copy())
    actual_clean = downstream_logits(model, row["teacher_ids"], clean_projector.to(torch.bfloat16).unsqueeze(0))[0]
    actual_pixel = downstream_logits(model, row["teacher_ids"], (clean_projector - delta).to(torch.bfloat16).unsqueeze(0))[0]
    clean_error = float((actual_clean - expected_clean).abs().max())
    pixel_error = float((actual_pixel - expected_pixel).abs().max())
    return {"state_id": row["state_id"], "clean_replay_max_abs_error": clean_error,
            "pixel_replay_max_abs_error": pixel_error,
            "exact_parity": bool(clean_error == 0 and pixel_error == 0)}


def aggregate_summary(extraction_rows, reconstruction_rows):
    locations = list(extraction_rows[0]["locations"])
    anatomy = {}
    for location in locations:
        object_energy = sum(row["locations"][location]["object_energy"] for row in extraction_rows)
        total_energy = sum(row["locations"][location]["total_energy"] for row in extraction_rows)
        ratios = [row["locations"][location]["object_energy_ratio"] for row in extraction_rows]
        anatomy[location] = {"global_object_energy_ratio": object_energy / total_energy,
                             "median_state_object_energy_ratio": float(np.median(ratios))}
    projector_by_task = {}
    for task in sorted(set(row["task"] for row in extraction_rows)):
        task_rows = [row for row in extraction_rows if row["task"] == task]
        object_energy = sum(row["locations"]["projector"]["object_energy"] for row in task_rows)
        total_energy = sum(row["locations"]["projector"]["total_energy"] for row in task_rows)
        projector_by_task[task] = object_energy / total_energy
    anatomy["projector"]["global_object_energy_ratio_by_task"] = projector_by_task
    curves = {}
    for method in ("residual_svd", "random", "clean_svd"):
        curves[method] = {}
        for rank in RANKS:
            rows = [row for row in reconstruction_rows if row["method"] == method and row["rank"] == rank]
            curves[method][str(rank)] = {
                "median_logit_cosine": float(np.median([row["logit_cosine"] for row in rows])),
                "median_probability_cosine": float(np.median([row["probability_cosine"] for row in rows])),
                "median_effect_norm_ratio": float(np.median([row["effect_norm_ratio"] for row in rows])),
                "global_explained_residual_energy": sum(row["explained_energy_numerator"] for row in rows) /
                                                    sum(row["residual_energy_denominator"] for row in rows),
            }
    spatial = [row for row in reconstruction_rows if row["method"] == "spatial_only"]
    spatial_summary = {"median_logit_cosine": float(np.median([row["logit_cosine"] for row in spatial])),
                       "median_probability_cosine": float(np.median([row["probability_cosine"] for row in spatial])),
                       "median_effect_norm_ratio": float(np.median([row["effect_norm_ratio"] for row in spatial]))}
    learned32 = [row for row in reconstruction_rows if row["method"] == "residual_svd" and row["rank"] == 32]
    random32 = [row for row in reconstruction_rows if row["method"] == "random" and row["rank"] == 32]
    clean32 = [row for row in reconstruction_rows if row["method"] == "clean_svd" and row["rank"] == 32]
    learned_median = curves["residual_svd"]["32"]["median_logit_cosine"]
    random_median = curves["random"]["32"]["median_logit_cosine"]
    task_comparisons = {}
    for task in sorted(set(row["task"] for row in learned32)):
        left = float(np.median([row["logit_cosine"] for row in learned32 if row["task"] == task]))
        right = float(np.median([row["logit_cosine"] for row in random32 if row["task"] == task]))
        task_comparisons[task] = {"residual_svd": left, "random": right, "better": bool(left > right)}
    tasks_better = sum(item["better"] for item in task_comparisons.values())
    learned32_summary = curves["residual_svd"]["32"]
    checks = {
        "projector_object_energy_lt_0_50": bool(anatomy["projector"]["global_object_energy_ratio"] < .50),
        "top32_explained_energy_ge_0_70": bool(learned32_summary["global_explained_residual_energy"] >= .70),
        "top32_logit_cosine_ge_0_70": bool(learned_median >= .70),
        "advantage_over_random_ge_0_30": bool(learned_median - random_median >= .30),
        "advantage_over_spatial_ge_0_15": bool(learned_median - spatial_summary["median_logit_cosine"] >= .15),
        "effect_norm_ratio_in_0_7_1_3": bool(.7 <= learned32_summary["median_effect_norm_ratio"] <= 1.3),
        "at_least_8_tasks_better_than_random": bool(tasks_better >= 8),
    }
    passed = all(checks.values())
    learned_values = [row["logit_cosine"] for row in learned32]
    random_values = [row["logit_cosine"] for row in random32]
    clean_values = [row["logit_cosine"] for row in clean32]
    spatial_values = [row["logit_cosine"] for row in spatial]
    return {"status": "MECHANISM_GO" if passed else "NO_GO", "states": len(extraction_rows),
            "split": "leave-one-task-out", "anatomy": anatomy, "compression_curves": curves,
            "spatial_only": spatial_summary, "top32_logit_cosine_advantage_over_random": learned_median - random_median,
            "top32_logit_cosine_advantage_over_spatial": learned_median - spatial_summary["median_logit_cosine"],
            "tasks_better_than_random": int(tasks_better), "task_comparisons": task_comparisons,
            "top32_bootstrap_95_ci": {
                "residual_svd_median": bootstrap_median(learned_values, 202608151),
                "advantage_over_random": bootstrap_median_advantage(learned_values, random_values, 202608152),
                "advantage_over_clean_svd": bootstrap_median_advantage(learned_values, clean_values, 202608153),
                "advantage_over_spatial": bootstrap_median_advantage(learned_values, spatial_values, 202608154)
            },
            "top32_logit_cosine_advantage_over_clean_svd": learned_median - curves["clean_svd"]["32"]["median_logit_cosine"],
            "checks": checks, "rollout_performed": False, "rollout_authorized": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--replay-sentinel", action="store_true")
    args = parser.parse_args()
    pcd_root, extraction, artifact = args.pcd_root.resolve(), args.extraction.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(extraction / "extraction.jsonl")
    checkpoint = pcd_root / "source/PCD/pretrained/openvla-7b"
    if args.replay_sentinel:
        model, _ = load_model(checkpoint)
        report = replay_sentinel(model, extraction, rows[0])
        (artifact / "replay_sentinel.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if not report["exact_parity"]:
            raise RuntimeError(report)
        print(json.dumps(report)); return
    if len(rows) != 270 or len(set(row["task"] for row in rows)) != 9:
        raise RuntimeError("Formal analysis requires all 270 states and nine tasks")
    delta = np.empty((len(rows), VISUAL_TOKENS, 4096), dtype=np.float32)
    clean = np.empty_like(delta)
    for index, row in enumerate(rows):
        with np.load(extraction / row["state_file"]) as payload:
            delta[index] = payload["projector_delta"]
            clean[index] = payload["projector_clean"]
    tasks = sorted(set(row["task"] for row in rows))
    bases = {}
    for fold, heldout in enumerate(tasks):
        train = np.asarray([i for i, row in enumerate(rows) if row["task"] != heldout])
        print(json.dumps({"phase": "fit_basis", "fold": fold + 1, "heldout": heldout, "kind": "residual"}), flush=True)
        residual_basis = fit_basis(delta[train].reshape(-1, 4096), stable_seed(heldout + "|residual"))
        print(json.dumps({"phase": "fit_basis", "fold": fold + 1, "heldout": heldout, "kind": "clean"}), flush=True)
        clean_basis = fit_basis(clean[train].reshape(-1, 4096), stable_seed(heldout + "|clean"))
        random = random_basis(4096, stable_seed(heldout + "|random"))
        bases[heldout] = {"residual_svd": residual_basis, "clean_svd": clean_basis, "random": random}
    torch.save({"ranks": RANKS, "q": SVD_Q, "niter": SVD_NITER, "bases": bases}, artifact / "loto_bases.pt")
    del clean
    torch.cuda.empty_cache()
    model, _ = load_model(checkpoint)
    replay = replay_sentinel(model, extraction, rows[0])
    (artifact / "replay_sentinel.json").write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    if not replay["exact_parity"]:
        raise RuntimeError(replay)
    output = artifact / "reconstruction.jsonl"
    if output.exists():
        output.unlink()
    reconstruction = []
    for ordinal, row in enumerate(rows, 1):
        with np.load(extraction / row["state_file"]) as payload:
            clean_projector = torch.from_numpy(payload["projector_clean"].copy()).to(model.device, dtype=torch.float32)
            state_delta = torch.from_numpy(payload["projector_delta"].copy()).to(model.device, dtype=torch.float32)
            clean_logits = torch.from_numpy(payload["clean_logits"].copy())
            pixel_logits = torch.from_numpy(payload["pixel_logits"].copy())
            overlaps = torch.from_numpy(payload["overlaps"].copy()).to(model.device)
        fold_bases = {name: value.to(model.device) for name, value in bases[row["task"]].items()}
        labels, synthetic, projections = [], [], []
        for method in ("residual_svd", "random", "clean_svd"):
            for rank in RANKS:
                projected = project(state_delta, fold_bases[method], rank)
                labels.append((method, rank)); projections.append(projected)
                synthetic.append(clean_projector - projected)
        spatial = state_delta * (overlaps >= .25).unsqueeze(-1)
        labels.append(("spatial_only", None)); projections.append(spatial); synthetic.append(clean_projector - spatial)
        logits = torch.stack([downstream_logits(model, row["teacher_ids"], value.to(torch.bfloat16).unsqueeze(0))[0]
                              for value in synthetic])
        pixel_direction = logit_direction(clean_logits, pixel_logits)
        pixel_probability = probability_direction(clean_logits, pixel_logits)
        pixel_norm = torch.linalg.vector_norm(pixel_direction.double())
        for index, ((method, rank), projected) in enumerate(zip(labels, projections)):
            direction = logit_direction(clean_logits, logits[index])
            probability = probability_direction(clean_logits, logits[index])
            record = {"state_id": row["state_id"], "task": row["task"], "method": method, "rank": rank,
                      "logit_cosine": cosine(direction, pixel_direction),
                      "probability_cosine": cosine(probability, pixel_probability),
                      "effect_norm_ratio": float(torch.linalg.vector_norm(direction.double()) / pixel_norm) if pixel_norm > 0 else float("nan"),
                      "explained_energy_numerator": float(projected.double().square().sum()),
                      "residual_energy_denominator": float(state_delta.double().square().sum())}
            append_jsonl(output, record); reconstruction.append(record)
        print(json.dumps({"phase": "reconstruct", "ordinal": ordinal, "total": len(rows),
                          "state_id": row["state_id"]}), flush=True)
    summary = aggregate_summary(rows, reconstruction)
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
