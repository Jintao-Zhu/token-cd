"""Analyze L11-Matched on the full canonical 9-task, 0-299 evaluation."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/prompt_attn_l11_matched_full_9task_0_299_v1"
RUN_ROOT = ROOT / "run"
CANONICAL = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
OLD = REPO / "artifacts/prompt_attn_layer_selection_v1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
FIRST3 = {
    "google_robot_open_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
}
HASH_KEYS = ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def l11_path(task: str, seed: int) -> Path:
    if seed <= 99:
        base = OLD / ("closed_loop" if task in FIRST3 else "closed_loop_remaining6")
        return base / "episodes" / task / "prompt_single" / f"episode_{seed:03d}_summary.json"
    return RUN_ROOT / "episodes" / task / "prompt_single" / f"episode_{seed:03d}_summary.json"


def exact_mcnemar_p(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    k = min(rescue, harm)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired(rows: list[dict], baseline: str, candidate: str = "l11_matched") -> dict:
    rescue = sum((not row[baseline]) and row[candidate] for row in rows)
    harm = sum(row[baseline] and (not row[candidate]) for row in rows)
    return {
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "exact_mcnemar_p": exact_mcnemar_p(rescue, harm),
    }


def main() -> None:
    rows: list[dict] = []
    technical_failures: list[str] = []
    hash_mismatches: list[dict] = []
    for task in TASKS:
        for seed in range(300):
            paths = {
                "vanilla": CANONICAL / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json",
                "shr": CANONICAL / "episodes" / task / "shr_harmonic" / f"episode_{seed:03d}_summary.json",
                "l11_matched": l11_path(task, seed),
            }
            missing = [name for name, path in paths.items() if not path.exists()]
            if missing:
                raise FileNotFoundError({"task": task, "seed": seed, "missing": missing})
            data = {name: load(path) for name, path in paths.items()}
            hashes = {name: tuple(data[name].get(key) for key in HASH_KEYS) for name in data}
            if len(set(hashes.values())) != 1:
                hash_mismatches.append({"task": task, "seed": seed, "hashes": hashes})
            if data["l11_matched"].get("technical_pass") is not True:
                technical_failures.append(str(paths["l11_matched"]))
            rows.append({
                "task": task,
                "seed": seed,
                **{name: bool(data[name]["success"]) for name in data},
            })

    if hash_mismatches or technical_failures:
        payload = {
            "complete": False,
            "hash_mismatch_count": len(hash_mismatches),
            "technical_failure_count": len(technical_failures),
            "hash_mismatch_examples": hash_mismatches[:10],
            "technical_failure_examples": technical_failures[:10],
        }
        (ROOT / "FULL_VALIDATION_FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise RuntimeError(payload)

    summaries: dict[str, dict] = {}
    for key in (*TASKS, "overall"):
        subset = rows if key == "overall" else [row for row in rows if row["task"] == key]
        summaries[key] = {
            "n": len(subset),
            "successes": {
                arm: sum(row[arm] for row in subset)
                for arm in ("vanilla", "shr", "l11_matched")
            },
            "success_rate": {
                arm: sum(row[arm] for row in subset) / len(subset)
                for arm in ("vanilla", "shr", "l11_matched")
            },
            "paired_vs_vanilla": paired(subset, "vanilla"),
            "paired_vs_shr": paired(subset, "shr"),
        }

    old_rows = [row for row in rows if row["seed"] < 100]
    old_summary = {
        "n": len(old_rows),
        "successes": {
            arm: sum(row[arm] for row in old_rows)
            for arm in ("vanilla", "shr", "l11_matched")
        },
    }
    result = {
        "protocol_id": "PROMPT_ATTN_L11_MATCHED_FULL_9TASK_0_299_V1",
        "complete": True,
        "tasks": list(TASKS),
        "seeds_per_task": 300,
        "episodes": 2700,
        "identity_pass": True,
        "technical_pass": True,
        "l11_source": {
            "seeds_0_99": str(OLD),
            "seeds_100_299": str(RUN_ROOT),
        },
        "summaries": summaries,
        "old_0_99": old_summary,
    }
    (ROOT / "FINAL_RESULTS_0_299.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "paired_results_0_299.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "seed", "vanilla", "shr", "l11_matched"])
        writer.writeheader()
        writer.writerows(rows)

    def rate(values: dict, arm: str) -> str:
        return f'{values["successes"][arm]}/{values["n"]} ({100 * values["success_rate"][arm]:.1f}%)'

    lines = [
        "# L11-Matched full canonical evaluation: 9 tasks x 300 seeds",
        "",
        "All 2,700 episodes use the same canonical snapshot and pass the canonical/state/RGB hash checks.",
        "",
        "| Task | Vanilla | SHR | L11-Matched | L11 vs Vanilla net | L11 vs SHR net |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in (*TASKS, "overall"):
        values = summaries[task]
        lines.append(
            f'| {task} | {rate(values, "vanilla")} | {rate(values, "shr")} | '
            f'{rate(values, "l11_matched")} | '
            f'{values["paired_vs_vanilla"]["net"]:+d} | '
            f'{values["paired_vs_shr"]["net"]:+d} |'
        )
    lines += [
        "",
        "## Paired overall comparisons",
        "",
        "| Comparison | Rescue | Harm | Net | exact p |",
        "|---|---:|---:|---:|---:|",
        f'| L11 vs Vanilla | {summaries["overall"]["paired_vs_vanilla"]["rescue"]} | '
        f'{summaries["overall"]["paired_vs_vanilla"]["harm"]} | '
        f'{summaries["overall"]["paired_vs_vanilla"]["net"]:+d} | '
        f'{summaries["overall"]["paired_vs_vanilla"]["exact_mcnemar_p"]:.6g} |',
        f'| L11 vs SHR | {summaries["overall"]["paired_vs_shr"]["rescue"]} | '
        f'{summaries["overall"]["paired_vs_shr"]["harm"]} | '
        f'{summaries["overall"]["paired_vs_shr"]["net"]:+d} | '
        f'{summaries["overall"]["paired_vs_shr"]["exact_mcnemar_p"]:.6g} |',
    ]
    (ROOT / "FULL_REPORT_0_299.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "overall": summaries["overall"]}, indent=2))


if __name__ == "__main__":
    main()
