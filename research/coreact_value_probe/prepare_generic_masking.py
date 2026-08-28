#!/usr/bin/env python3
"""Lock the new-suite generic masking development replication."""
from __future__ import annotations
import argparse,json,platform
from pathlib import Path
import yaml

TASKS=(0,1,2)
CONDITIONS=("vanilla","attention_top8","random8","bottom8")
LANGUAGES=("open the middle drawer of the cabinet","put the bowl on the stove","put the wine bottle on top of the cabinet")
def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();o.mkdir(parents=True,exist_ok=False);(o/"episodes").mkdir();(o/"logs").mkdir()
 protocol={"experiment_name":"coreact_generic_masking_replication_v1","stage":"exploratory_development_not_confirmation","suite":"libero_goal","task_ids":list(TASKS),"task_names":list(LANGUAGES),"init_states":"0-49 per task","episodes":600,"paired_states":150,"conditions":list(CONDITIONS),"mask_protocol":"first 3 replans only, then vanilla","group_count":8,"top_selector":"late-half action-to-context attention at tau=1 over native prefix","random_selector":"deterministic modality-masked random visual token selection","bottom_selector":"late-half attention bottom-8 visual tokens","replacement":"v8 position-conditioned visual mean","flow_steps":10,"chunk_size":50,"executed_action_prefix":10,"max_control_steps":280,"shared":"init state, reset seed, per-replan Gaussian noise, preprocessing, solver","primary_question":"generic early masking benefit versus selector identity","secondary_metrics":["matched-noise action mask-clean norm","action TV","chunk discontinuity","episode steps","gripper timing","eef displacement","token overlap"],"decision_gate":"three mask conditions all above vanilla means generic effect; top-only above vanilla supports selector effect; all near vanilla rejects task-8 signal","task_qualification":"new suite not represented in prior CoreAct masking artifacts; development only due finite canonical states"}
 (o/"protocol.lock.yaml").write_text(yaml.safe_dump(protocol,sort_keys=False))
 rows=[]
 for tid,lang in zip(TASKS,LANGUAGES):
  for init in range(50):
   base=99000000+tid*100000+init*100;pair=f"generic_mask__libero_goal__task{tid:02d}__init{init:02d}"
   for c in CONDITIONS:rows.append({"episode_id":f"{pair}__{c}","pair_id":pair,"suite":"libero_goal","task_id":tid,"init_state_id":init,"condition":c,"language":lang,"reset_seed":base+1,"action_noise_seed":base+2,"selection_seed":base+3})
 with (o/"episode_manifest.jsonl").open("x") as f:
  for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
 (o/"environment.json").write_text(json.dumps({"host":platform.node(),"checkpoint_repo":"lerobot/smolvla_libero","checkpoint_revision":"31d453f7edd78c839a8bbc39744a292686daf0de","frozen_eval":True,"manifest_rows":len(rows)},indent=2)+"\n")
 print(json.dumps({"artifact":str(o),"episodes":len(rows),"paired_states":150,"tasks":TASKS}))
if __name__=="__main__":main()
