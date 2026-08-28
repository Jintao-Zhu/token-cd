#!/usr/bin/env python3
"""Audit whether v8 token effects can be joined to simulator segmentation.

This intentionally does not fabricate region labels.  It creates a provenance
artifact and returns an inconclusive decision when the source-state join key is
absent from the effects artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--region-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    effects = args.effects.resolve()
    region = args.region_artifact.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    rows = list(load_jsonl(effects))
    visual = [r for r in rows if r.get("modality") == "visual"]
    groups = {(r.get("suite"), r.get("task_id"), r.get("state_index"), r.get("group_id")) for r in visual}
    raw_episode_ids = {str(r.get("episode_id")) for r in visual}
    region_records = []
    for path in sorted((region / "episodes").glob("*/episode.json")):
        try:
            record = json.loads(path.read_text())
        except Exception as exc:
            region_records.append({"path": str(path), "parse_error": str(exc)})
            continue
        region_records.append({
            "path": str(path), "suite": record.get("suite"),
            "task_id": record.get("task_id"), "init_state_id": record.get("init_state_id"),
            "episode_id": record.get("episode_id"),
            "has_region_mapping": bool(record.get("region_mapping")),
            "has_initial_state_hash": bool(record.get("initial_sim_state_sha256")),
        })

    region_episode_ids = {str(r.get("episode_id")) for r in region_records if r.get("episode_id")}
    # Episode IDs are the only row-level identity in v8; frame/state hashes and
    # segmentation are not present, so a matching task name is insufficient.
    exact_episode_overlap = sorted(raw_episode_ids & region_episode_ids)
    effect_fields = sorted(rows[0]) if rows else []
    required_fields = {
        "observation_hash", "initial_sim_state_sha256", "segmentation_sha256",
        "segmentation_camera1_sha256", "segmentation_camera2_sha256",
        "region_mapping_sha256", "rgb_camera1_sha256", "rgb_camera2_sha256",
    }
    missing_join_fields = sorted(required_fields - set(effect_fields))
    report = {
        "artifact": "coreact_region_sign_probe_v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INCONCLUSIVE_DATA_OR_RESOURCES",
        "source_effects": {"path": str(effects), "sha256": sha256(effects), "rows": len(rows), "visual_rows": len(visual), "groups": len(groups), "fields": effect_fields},
        "region_source": {"path": str(region), "episode_json_count": len(region_records), "records": region_records, "exact_episode_overlap": exact_episode_overlap},
        "join_audit": {
            "required_row_identity_fields": sorted(required_fields),
            "missing_from_effects": missing_join_fields,
            "task_only_join_rejected": True,
            "reason": "v8 effects contain episode_id/frame_id but no observation or segmentation hash; region artifact contains only task-7 rollout records with different episode IDs.",
        },
        "qualification": {"region_features_available_for_all_effect_groups": False, "no_fabricated_labels": True, "classifier_run": False},
        "next_required_input": "rerun offline probe while persisting raw observation/segmentation hashes and per-camera token-region mapping for every state",
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    protocol = {
        "experiment_name": "coreact_region_sign_probe_v3",
        "purpose": "region-aware anchor/nuisance classification qualification",
        "label_source": "frozen-model Q_rel from v8 only; no closed-loop outcome labels",
        "features": ["target_object_fraction", "goal_container_fraction", "protected_fraction", "background_fraction", "ambiguous_fraction", "attention", "embedding_norm", "I magnitude"],
        "split": "leave-one-task-out",
        "gate": {"nuisance_precision_min": 0.2, "average_precision_min": 0.2},
        "status": "BLOCKED_UNTIL_ROW_LEVEL_SEGMENTATION_JOIN",
    }
    (out / "protocol.yaml").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (out / "decision.json").write_text(json.dumps({"status": report["status"], "reason": report["join_audit"]["reason"]}, indent=2) + "\n")
    report_md = f"""# Region-aware sign probe v3 audit

状态：**{report['status']}**

## 已核对

- v8 effects：{len(rows)} 行、{len(groups)} 个 visual groups、{len(raw_episode_ids)} 个 episode id。
- 区域诊断：{len(region_records)} 个 episode JSON，均为已有 region rollout；与 v8 episode id 精确重合：{len(exact_episode_overlap)}。
- 必需的 observation/segmentation/hash 字段缺失：`{', '.join(missing_join_fields)}`。

## 为什么不能继续分类

区域标签必须来自同一 simulator observation 的 instance segmentation。仅凭 suite/task/frame 顺序或 token index 不能证明对应关系；这样会把猜测当作监督信号。因此本次没有运行 v3 classifier，也没有运行 selective closed-loop。

## 下一步

重新抽取 development states，并在每条 effect row 保存 observation hash、两路 segmentation hash、initial simulator-state hash 和完整 per-camera token-region mapping；随后再按 leave-one-task-out 重跑分类器。
"""
    (out / "report.md").write_text(report_md)
    manifest = {"source_effects_sha256": sha256(effects), "region_artifact": str(region), "status": report["status"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "status": report["status"], "rows": len(rows), "groups": len(groups), "missing_join_fields": missing_join_fields}, indent=2))


if __name__ == "__main__":
    main()
