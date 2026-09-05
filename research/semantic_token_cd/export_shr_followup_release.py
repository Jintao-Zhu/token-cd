"""Export compact, deterministic releases for the SHR follow-up experiments.

Raw rollout traces and NPZ arrays stay in ignored artifact directories.  Each
release contains the configuration, aggregate report, and one compact JSONL row
per episode with the pairing hashes and the diagnostics needed for re-analysis.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "github_results"

COMMON_KEYS = {
    "protocol_id", "task", "seed", "episode_id", "evaluation_seed", "arm",
    "success", "reference_arm", "reference_success", "failure_reason",
    "control_steps", "trajectory_length", "technical_pass", "environment_id",
    "canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256",
    "kmeans_K", "kmeans_seed", "lambda", "beta", "eta", "method",
    "runtime_seconds", "action_jitter_index", "action_jerk",
    "initial_lambda", "final_lambda", "adapt_trigger", "shift_ratio",
    "trigger_fraction", "triggered_control_steps", "step_adapt_triggers",
    "mean_num_components", "mean_selected_num_components",
    "mean_filtered_num_components", "mean_mask_tokens_before",
    "mean_mask_tokens_after", "deleted_token_fraction",
    "mean_residual_norm", "mean_selected_tokens", "replans", "act_steps",
    "shared_action_state_between_branches", "mean_lambda_eff",
    "mean_orthogonal_norm", "mean_parallel_ratio", "mean_projection_alpha",
    "mean_residual_positive_cosine", "projected_changes_original_shr_count",
    "mean_preserved_semantic_tokens", "mean_reconstructed_tokens",
    "mean_semantic_tokens",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def success(row: dict) -> bool:
    if "success" in row:
        return bool(row["success"])
    return bool(row.get("result", {}).get("success", False))


def compact(row: dict) -> dict:
    result = {key: row[key] for key in COMMON_KEYS if key in row}
    result["success"] = success(row)
    # Preserve every rollout integrity flag without retaining large traces.
    result.update({key: value for key, value in row.items() if key.startswith("all_")})
    return dict(sorted(result.items()))


def episode_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/episode_*_summary.json"))


def export_jsonl(source: Path, destination: Path) -> list[dict]:
    rows = [compact(read(path)) for path in episode_files(source)]
    rows.sort(key=lambda row: (
        str(row.get("task", "")), str(row.get("arm", "")),
        int(row.get("seed", row.get("evaluation_seed", row.get("episode_id", -1)))),
    ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def load_reference() -> dict[tuple[str, int], bool]:
    root = OUT / "vanilla_recon_shr_canonical_0_299_v2" / "episodes"
    values = {}
    for path in root.glob("*/shr_harmonic/episode_*_summary.json"):
        row = read(path)
        values[(row["task"], int(row["seed"]))] = bool(row["success"])
    if len(values) != 2700:
        raise RuntimeError(f"expected 2700 canonical SHR references, found {len(values)}")
    return values


def aggregate(rows: list[dict], reference: dict[tuple[str, int], bool] | None = None) -> dict:
    grouped: dict[tuple[str, str], dict[int, bool]] = defaultdict(dict)
    for row in rows:
        seed = int(row.get("seed", row.get("evaluation_seed", row.get("episode_id"))))
        grouped[(str(row["task"]), str(row["arm"]))][seed] = bool(row["success"])
    tasks = sorted({task for task, _ in grouped})
    arms = sorted({arm for _, arm in grouped})
    by_task = {}
    for task in tasks:
        task_result = {"arms": {}, "paired_vs_canonical_shr": {}}
        for arm in arms:
            values = grouped.get((task, arm), {})
            if not values:
                continue
            wins = sum(values.values())
            task_result["arms"][arm] = {
                "episodes": len(values), "success": wins,
                "success_rate": wins / len(values),
            }
            if reference is not None:
                seeds = sorted(seed for seed in values if (task, seed) in reference)
                rescue = sum(not reference[(task, seed)] and values[seed] for seed in seeds)
                harm = sum(reference[(task, seed)] and not values[seed] for seed in seeds)
                task_result["paired_vs_canonical_shr"][arm] = {
                    "pairs": len(seeds), "reference_success": sum(reference[(task, seed)] for seed in seeds),
                    "arm_success": sum(values[seed] for seed in seeds),
                    "rescue": rescue, "harm": harm, "net": rescue - harm,
                }
        # Pair arms inside the experiment whenever they share a seed.
        task_result["within_experiment_pairs"] = {}
        for left_index, left in enumerate(arms):
            for right in arms[left_index + 1:]:
                lv, rv = grouped.get((task, left), {}), grouped.get((task, right), {})
                seeds = sorted(set(lv) & set(rv))
                if not seeds:
                    continue
                rescue = sum(not lv[seed] and rv[seed] for seed in seeds)
                harm = sum(lv[seed] and not rv[seed] for seed in seeds)
                task_result["within_experiment_pairs"][f"{right}_vs_{left}"] = {
                    "pairs": len(seeds), "rescue": rescue, "harm": harm,
                    "net": rescue - harm,
                }
        by_task[task] = task_result

    overall = {"arms": {}, "paired_vs_canonical_shr": {}, "within_experiment_pairs": {}}
    for arm in arms:
        values = {
            (task, seed): outcome
            for (task, grouped_arm), seed_values in grouped.items()
            if grouped_arm == arm
            for seed, outcome in seed_values.items()
        }
        if not values:
            continue
        wins = sum(values.values())
        overall["arms"][arm] = {
            "episodes": len(values), "success": wins,
            "success_rate": wins / len(values),
        }
        if reference is not None:
            keys = sorted(key for key in values if key in reference)
            rescue = sum(not reference[key] and values[key] for key in keys)
            harm = sum(reference[key] and not values[key] for key in keys)
            overall["paired_vs_canonical_shr"][arm] = {
                "pairs": len(keys), "reference_success": sum(reference[key] for key in keys),
                "arm_success": sum(values[key] for key in keys),
                "rescue": rescue, "harm": harm, "net": rescue - harm,
            }
    for left_index, left in enumerate(arms):
        for right in arms[left_index + 1:]:
            left_values = {
                (task, seed): outcome
                for (task, grouped_arm), seed_values in grouped.items()
                if grouped_arm == left
                for seed, outcome in seed_values.items()
            }
            right_values = {
                (task, seed): outcome
                for (task, grouped_arm), seed_values in grouped.items()
                if grouped_arm == right
                for seed, outcome in seed_values.items()
            }
            keys = sorted(set(left_values) & set(right_values))
            if not keys:
                continue
            rescue = sum(not left_values[key] and right_values[key] for key in keys)
            harm = sum(left_values[key] and not right_values[key] for key in keys)
            overall["within_experiment_pairs"][f"{right}_vs_{left}"] = {
                "pairs": len(keys), "rescue": rescue, "harm": harm,
                "net": rescue - harm,
            }
    return {
        "episodes": len(rows), "tasks": tasks, "arms": arms, "by_task": by_task,
        "overall": overall,
        "technical_pass": sum(bool(row.get("technical_pass", True)) for row in rows),
    }


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_csv(source: Path, destination: Path) -> None:
    """Copy a CSV using repository-standard LF newlines."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text().replace("\r\n", "\n"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def export_experiment(name: str, source: Path, *, reference=None, config: Path | None = None) -> None:
    destination = OUT / name
    rows = export_jsonl(source, destination / "EPISODES.jsonl")
    write_json(destination / "FINAL_RESULTS.json", aggregate(rows, reference))
    if config is not None:
        copy(config, destination / "CONFIG_LOCK.json")


def main() -> None:
    reference = load_reference()

    adaptive_out = OUT / "adaptive_shr_experiment"
    adaptive_rows = export_jsonl(
        REPO / "adaptive_shr_experiment" / "episode_json",
        adaptive_out / "EPISODES.jsonl",
    )
    if len(adaptive_rows) != 2700:
        raise RuntimeError(f"Adaptive-SHR expected 2700 episodes, found {len(adaptive_rows)}")
    copy(REPO / "adaptive_shr_experiment" / "report.md", adaptive_out / "report.md")
    copy(REPO / "adaptive_shr_experiment" / "statistics" / "summary.json", adaptive_out / "FINAL_RESULTS.json")
    copy(REPO / "adaptive_shr_experiment" / "CONFIG_LOCK.json", adaptive_out / "CONFIG_LOCK.json")
    copy_csv(REPO / "adaptive_shr_experiment" / "paired_results" / "paired_episodes.csv",
             adaptive_out / "paired_episodes.csv")

    ic_out = OUT / "instruction_component_shr_v1"
    ic_rows = export_jsonl(
        REPO / "artifacts" / "instruction_component_shr_v1" / "episode_summary",
        ic_out / "EPISODES.jsonl",
    )
    if len(ic_rows) != 1200:
        raise RuntimeError(f"IC-SHR expected 1200 episodes, found {len(ic_rows)}")
    copy(REPO / "artifacts" / "instruction_component_shr_v1" / "report.md", ic_out / "report.md")
    copy(REPO / "artifacts" / "instruction_component_shr_v1" / "component_stats" / "summary.json",
         ic_out / "FINAL_RESULTS.json")
    copy_csv(REPO / "artifacts" / "instruction_component_shr_v1" / "paired_results" / "paired_episodes.csv",
             ic_out / "paired_episodes.csv")
    for image in sorted((REPO / "artifacts" / "instruction_component_shr_v1" / "visualizations").glob("*.png")):
        copy(image, ic_out / "visualizations" / image.name)

    export_experiment(
        "projected_shr_5task_0_299_v1",
        REPO / "artifacts" / "projected_shr_5task_0_299_v1" / "episodes",
        reference=reference,
        config=REPO / "artifacts" / "projected_shr_5task_0_299_v1" / "CONFIG_LOCK.json",
    )
    export_experiment(
        "sp_shr_boundary_partial_3task_0_99_v1",
        REPO / "artifacts" / "sp_shr_boundary_partial_3task_0_99_v1" / "episodes",
        reference=reference,
        config=REPO / "artifacts" / "sp_shr_boundary_partial_3task_0_99_v1" / "CONFIG_LOCK.json",
    )
    export_experiment(
        "pi0_shr_5task_0_299_v1",
        REPO / "artifacts" / "pi0_shr_5task_0_299_v1" / "episodes",
        config=REPO / "artifacts" / "pi0_shr_5task_0_299_v1" / "CONFIG_LOCK.json",
    )

    psc_source = REPO / "artifacts" / "positive_support_constrained_shr_v1" / "offline"
    psc_out = OUT / "positive_support_constrained_shr_v1" / "offline"
    copy(psc_source / "report.md", psc_out / "report.md")
    copy(psc_source / "summary.json", psc_out / "summary.json")

    print("Exported compact SHR follow-up releases to", OUT)


if __name__ == "__main__":
    main()
