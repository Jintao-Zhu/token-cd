from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--workspace",type=Path,required=True); p.add_argument("--phase0-artifact",type=Path,required=True); p.add_argument("--source-manifest-artifact",type=Path,required=True); p.add_argument("--artifact",type=Path,required=True); args=p.parse_args()
    workspace,phase0,source,artifact=args.workspace.resolve(),args.phase0_artifact.resolve(),args.source_manifest_artifact.resolve(),args.artifact.resolve()
    if artifact.exists(): raise FileExistsError(artifact)
    phase_decision=json.loads((phase0/"decision.json").read_text())
    if phase_decision.get("decision")!="PHASE0_CALIBRATED_READY_FOR_ROLLOUT_PROTOCOL": raise RuntimeError("phase0 did not pass")
    for d in ("episodes","corrections","logs","status"): (artifact/d).mkdir(parents=True)
    for name in ("protocol.yaml","protocol.lock.yaml","phase0_calibration.json","phase0_state_manifest.jsonl","task_manifest.json","environment.json"):
        shutil.copy2(phase0/name,artifact/name)
    protocol=yaml.safe_load((artifact/"protocol.lock.yaml").read_text()); protocol["experiment_name"]="coreact_timestep_self_guidance_closed_loop_v1"; protocol["stage"]="mechanism_development_not_confirmation"; protocol["rollout_started"] = True; protocol["rollout_episodes"] = 600; protocol["source_phase0_artifact"] = str(phase0); protocol["source_phase0_calibration_sha256"] = sha256(phase0/"phase0_calibration.json"); protocol["source_outcome_artifact"] = str(source); protocol["old_episode_results_reused"] = False
    protocol["hashes"]["code"]={str(path.relative_to(workspace)):sha256(path) for path in [workspace/"research/coreact_closed_loop/guidance.py",workspace/"research/coreact_closed_loop/runtime.py",workspace/"research/coreact_self_guidance/sampler.py",workspace/"research/coreact_self_guidance/timestep_sampler.py",workspace/"research/coreact_self_guidance/prepare_timestep_rollout.py",workspace/"research/coreact_self_guidance/integrity.py",workspace/"research/coreact_self_guidance/run.py",workspace/"research/coreact_self_guidance/analyze.py"]}
    (artifact/"protocol.lock.yaml").write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8")
    base_rows=[json.loads(x) for x in (source/"episode_manifest.jsonl").read_text().splitlines()]; shift=float(protocol["locked_shift"]); rows=[]
    for row in base_rows:
        row={k:v for k,v in row.items() if k!="delta"}; row["shift"]=shift; rows.append(row)
    if len(rows)!=600: raise RuntimeError("manifest must contain 600 rows")
    with (artifact/"episode_manifest.jsonl").open("x") as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps({"artifact":str(artifact),"episodes":len(rows),"locked_shift":shift},indent=2))


if __name__=="__main__":main()
