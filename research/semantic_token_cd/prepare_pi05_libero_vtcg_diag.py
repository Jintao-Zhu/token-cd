from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE = Path(
    "/home/leju-suzhou/zjt_ws/token-cd/artifacts/pi05_libero_pcd_full_v1_20261024_01"
)
PLUS_CLASSIFICATION = Path(
    "/home/leju-suzhou/zjt_ws/libero_plus_src/libero/libero/benchmark/task_classification.json"
)
REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
TRACE_SOURCES = (
    "research/semantic_token_cd/pi05_matched_flow_cd.py",
    "research/semantic_token_cd/pi05_matched_flow_server.py",
    "research/semantic_token_cd/pi05_libero_plus_paired_rollout.py",
    "research/semantic_token_cd/replay_pi05_vtcg_trace.py",
    "research/semantic_token_cd/prepare_pi05_libero_vtcg_diag.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(seed: int, plan: str, bucket: str, task_index: int) -> str:
    value = f"20261025|{seed}|{plan}|{bucket}|{task_index}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_plan_rows(root: Path, plan_name: str) -> dict[tuple[int, str], dict]:
    path = root / plan_name / "episodes.jsonl"
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(int(row["task_index"]), str(row["arm"]))] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/leju-suzhou/.cache/openpi/openpi-assets/checkpoints/pi05_libero"),
    )
    args = parser.parse_args()

    source = SOURCE.resolve()
    out = args.out.resolve()
    if out == source or source in out.parents:
        raise ValueError("diagnostic output cannot be inside the source artifact")
    if out.exists():
        raise FileExistsError(out)

    task_manifest_path = source / "task_manifest.json"
    task_manifest = json.loads(task_manifest_path.read_text())
    classification = json.loads(PLUS_CLASSIFICATION.read_text())["libero_10"]
    category_by_name = {row["name"]: row["category"] for row in classification}
    plans = {row["name"]: row for row in task_manifest["plans"]}
    chosen = []

    plus_name = "libero_plus_perturb"
    plus_plan = plans[plus_name]
    plus_rows = _load_plan_rows(source, plus_name)
    by_category: dict[str, list[int]] = {}
    for task_index in plus_plan["task_indices"]:
        vanilla = plus_rows[(int(task_index), "vanilla")]
        category = category_by_name.get(str(vanilla["task_name"]))
        if category is None:
            raise ValueError(f"missing perturbation category for {vanilla['task_name']}")
        by_category.setdefault(category, []).append(int(task_index))

    for category, task_indices in sorted(by_category.items()):
        task_index = min(
            task_indices,
            key=lambda value: _rank(
                int(task_manifest["seed"]), plus_name, category, value
            ),
        )
        chosen.append((plus_name, category, task_index, plus_rows))

    for plan_name, plan in plans.items():
        if plan_name == plus_name:
            continue
        rows = _load_plan_rows(source, plan_name)
        task_index = min(
            (int(value) for value in plan["task_indices"]),
            key=lambda value: _rank(
                int(task_manifest["seed"]), plan_name, plan["suite"], value
            ),
        )
        chosen.append((plan_name, plan["suite"], task_index, rows))

    tasks = []
    for plan_name, stratum, task_index, rows in chosen:
        vanilla = rows[(task_index, "vanilla")]
        vtcg = rows[(task_index, "pcd")]
        if vanilla.get("error") or vtcg.get("error"):
            raise ValueError(f"selected source episode has an error: {plan_name}/{task_index}")
        if vanilla["initial_state_sha256"] != vtcg["initial_state_sha256"]:
            raise ValueError(f"source pair state mismatch: {plan_name}/{task_index}")
        plan = plans[plan_name]
        tasks.append(
            {
                "plan": plan_name,
                "suite": plan["suite"],
                "stratum": stratum,
                "task_index": task_index,
                "task_name": vanilla["task_name"],
                "trial": int(vanilla["trial"]),
                "source_initial_state_sha256": vanilla["initial_state_sha256"],
                "source_vanilla_action_sha256": vanilla["action_sha256"],
                "source_vtcg_action_sha256": vtcg["action_sha256"],
                "source_vanilla_success": bool(vanilla["success"]),
                "source_vtcg_success": bool(vtcg["success"]),
            }
        )

    source_paths = [task_manifest_path]
    source_paths.extend(source / plan_name / "episodes.jsonl" for plan_name in plans)
    source_hashes = {
        str(path.relative_to(source)): _sha256(path) for path in source_paths
    }
    source_code_hashes = {
        path: _sha256(REPO / path) for path in TRACE_SOURCES
    }
    checkpoint = args.checkpoint.resolve()
    checkpoint_inventory = []
    if checkpoint.is_dir():
        for path in sorted(checkpoint.rglob("*")):
            if path.is_file():
                stat = path.stat()
                checkpoint_inventory.append(
                    {
                        "path": str(path.relative_to(checkpoint)),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )

    manifest = {
        "protocol": "PI05_LIBERO_VTCG_TRACE_DIAG_V1",
        "source_artifact": str(source),
        "source_hashes_sha256": source_hashes,
        "source_code_hashes_sha256": source_code_hashes,
        "selection": {
            "rule": "minimum SHA256 rank per stratum; outcome-blind",
            "selection_salt": "20261025",
            "source_manifest_seed": int(task_manifest["seed"]),
            "pairs": len(tasks),
            "rollouts": 2 * len(tasks),
            "strata": tasks,
        },
        "replay_config": {
            "trial": int(task_manifest["trial"]),
            "env_seed": 7,
            "base_noise_seed": 20261021,
            "settle_steps": 10,
            "replan_steps": 5,
            "num_flow_steps": 10,
            "candidates": 8,
            "alpha": 0.5,
            "bandwidth_scale": 0.5,
            "checkpoint": str(checkpoint),
            "checkpoint_file_inventory": checkpoint_inventory,
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "diagnostic_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"artifact": str(out), "pairs": len(tasks), "tasks": tasks}, indent=2))


if __name__ == "__main__":
    main()
