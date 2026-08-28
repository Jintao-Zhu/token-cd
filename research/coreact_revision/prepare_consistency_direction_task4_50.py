#!/usr/bin/env python3
"""Create the locked 50-state, four-condition task-4 manifest."""
from __future__ import annotations
import argparse,json
from pathlib import Path
LANGUAGE="pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
CONDITIONS=("vanilla","attention_toward","consistency_mask_only","consistency_toward")
def main():
 p=argparse.ArgumentParser();p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();out=a.artifact/"episode_manifest.jsonl"
 if out.exists():raise FileExistsError(out)
 rows=[]
 for init in range(50):
  for condition in CONDITIONS:rows.append({"action_noise_seed":66100000+init,"condition":condition,"episode_id":f"task04__init{init:02d}__{condition}","init_state_id":init,"language":LANGUAGE,"pair_id":f"task04__init{init:02d}","reset_seed":66000000+init,"split":"development","suite":"libero_spatial","task_id":4})
 with out.open("x") as f:
  for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
 assert len(rows)==200 and len({r["episode_id"] for r in rows})==200
 for init in range(50):
  pair=[r for r in rows if r["init_state_id"]==init];assert len(pair)==4 and len({r["reset_seed"] for r in pair})==len({r["action_noise_seed"] for r in pair})==1
 print(json.dumps({"episodes":200,"paired_states":50,"conditions":CONDITIONS}))
if __name__=="__main__":main()
