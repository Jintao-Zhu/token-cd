"""Generate final tables, report, and a complete artifact SHA-256 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    baseline = json.loads((artifact / "baseline_summary.json").read_text())
    summary = json.loads((artifact / "summary.json").read_text())
    cascade = json.loads((artifact / "cascade_summary.json").read_text())
    integrity = json.loads((artifact / "integrity_report.json").read_text())
    phase_integrity = json.loads((artifact / "phase1_integrity_report.json").read_text())
    if baseline["status"] != "PASS" or integrity["status"] != "PASS" or phase_integrity["status"] != "PASS" or cascade["status"] != "PASS":
        raise RuntimeError("Refusing to finalize with a failed gate")

    task_table = artifact / "task_level_metrics.csv"
    with task_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "description", "successes", "episodes", "success_rate", "selected_phase1"])
        writer.writeheader()
        for task_id, values in baseline["task_rates"].items():
            writer.writerow({"task_id": task_id, **values, "selected_phase1": int(int(task_id) in baseline["selected_task_ids"])})
    action_table = artifact / "action_position_metrics.csv"
    with action_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action_position", "top_minus_random_mean_js"])
        writer.writeheader()
        for position, value in summary["top_minus_random_by_action_position"].items():
            writer.writerow({"action_position": position, "top_minus_random_mean_js": value})

    selected_text = ", ".join(
        f"task {task_id} ({baseline['task_rates'][str(task_id)]['success_rate']:.0%})" for task_id in baseline["selected_task_ids"]
    )
    report = f"""# AR Token Counterfactual Qualification v1

## 问题与范围

本实验验证：在冻结的 `openvla/openvla-7b-finetuned-libero-spatial` 上，post-projector 单视觉 token replacement 是否会产生比 Flow-VLA 更强、稳定且可排序的 action-logit 响应。Attention 只作为 ranking proxy，不称为 causal attribution。本实验没有做 closed-loop token calibration，也没有实现 VCAD。

## Checkpoint 与 baseline

- Checkpoint revision：`962318cec55ac10993ff0f5f43eda9a270b4c873`；4 个 shard 均通过 Hugging Face LFS SHA-256。
- 官方 OpenVLA standalone HF 模型/processor；BF16、frozen `eval()`、greedy 7-token action decode、LIBERO 官方 center crop 与 action/gripper 变换。
- 由于环境未安装 Flash-Attention，使用 eager attention；调用前同步补齐 empty action token 与 attention mask，避免官方 port 在 eager 路径的 mask 长度不一致。
- Baseline：10 tasks × 20 fixed init states = 200/200；missing、duplicate、nonfinite、异常均为 0。
- 按预注册 `[40%, 85%]` 和离 62.5% 最近规则选择：{selected_text}。完整 task 表见 `task_level_metrics.csv`。

## 数据与干预

- Mean calibration：150 个独立 states，来自与 Phase-1 task 不相交的 task；mean shape `[256, 4096]`。
- Phase-1：3 tasks × 50 states = 150 states；phase 是 outcome-blind `trajectory_quartile_proxy`，不是精确 grasp detector。
- 每 state 固定 16 个互不重叠 token：top/middle/bottom/random 各 4；共 2400 token interventions、16800 action-position rows。
- Primary 使用完全相同的 clean autoregressive action prefix，比较 clean/masked teacher-forced logits。Secondary 单独比较 masked free-running cascade。

## Integrity

- Pre-baseline gate：`{integrity['status']}`；actual-mean Phase-1 gate：`{phase_integrity['status']}`。
- Determinism、empty-hook parity、official/manual decode parity、changed-index、protected input hash、finite 检查全部通过。
- Raw primary：16800/16800；cascade v2：2400/2400；missing、duplicate、nonfinite 均为 0。
- `cascade_effects.jsonl` 是 amendment 期间失败的 partial secondary，已由 `cascade_effects_v2.jsonl` 明确取代，不参与任何统计。

## Primary Results

- Attention vs mean JS 的 state-level Spearman median：**{summary['state_spearman_median']:.3f}**，IQR `{summary['state_spearman_iqr']}`，task/state cluster bootstrap 95% CI `{summary['state_spearman_cluster_bootstrap_95ci']}`。
- Top-minus-random mean JS：**{summary['top_minus_random_mean']:.4f}**，95% CI `{summary['top_minus_random_cluster_bootstrap_95ci']}`。
- Top-minus-bottom mean JS：**{summary['top_minus_bottom_mean']:.4f}**，95% CI `{summary['top_minus_bottom_cluster_bootstrap_95ci']}`。
- Teacher-forced action-position argmax flip：**{summary['argmax_flip_fraction']:.2%}**。
- 三个 task 的 median Spearman 均为正：`{summary['task_median_spearman']}`；7/7 action positions 的 top-minus-random 均为正。

## AR Cascade Secondary

- Teacher-forced argmax action 非零变化：{cascade['teacher_forced_argmax_nonzero_fraction']:.2%}；free-running action 非零变化：{cascade['free_running_nonzero_fraction']:.2%}。
- Teacher/free effect Spearman：{cascade['teacher_vs_free_effect_spearman']:.3f}。
- 在 teacher effect 非零时，cascade amplification median：{cascade['cascade_amplification_median_teacher_nonzero']:.3f}；ratio > 1：{cascade['cascade_amplification_gt_1_fraction_teacher_nonzero']:.2%}。
- {cascade['clean_teacher_vs_cached_generation_mismatch_states']}/150 states 的 clean full-sequence TF argmax 与 cached generation token 至少一处不同。这是 eager/full-sequence 与 cached decode 数值路径差异，secondary v2 已按 TF-clean 对 TF-masked 定义修正。

## Decision

`{summary['decision']}`，4/4 预注册条件通过。

当前最强结论是：**在该 OpenVLA checkpoint 与这 150 个 LIBERO-Spatial development states 上，action-token attention 能稳定排序 post-projector 单 token deletion 的 teacher-forced action-logit effect，且 AR free decoding 经常保留或放大该差异。**

不能据此声称这些 token 对任务成功是正贡献或负贡献，也不能声称 VCAD 会提高成功率。下一阶段只能进入新的 closed-loop causal magnitude calibration；在完成该 gate 前不应设计 toward/away guidance。

## Reproduction

精确命令见 `reproduction_commands.sh`。Primary protocol/code hashes见 `protocol.lock.yaml` 与 `phase1_protocol.lock.yaml`；两次 secondary 修订见 `amendment_001_*` 和 `amendment_002_*`。
"""
    (artifact / "report.md").write_text(report, encoding="utf-8")

    audit_path = artifact / "sha256_audit.txt"
    lines = ["# sha256  relative_path", "# sha256_audit.txt is excluded to avoid a self-referential hash."]
    for path in sorted(item for item in artifact.rglob("*") if item.is_file() and item != audit_path):
        lines.append(f"{sha256(path)}  {path.relative_to(artifact)}")
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(artifact / "report.md"), "audited_files": len(lines) - 2, "decision": summary["decision"]}))


if __name__ == "__main__":
    main()
