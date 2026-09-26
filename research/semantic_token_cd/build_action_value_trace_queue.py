#!/usr/bin/env python3
"""Build a predeclared 40-pair trace-faithful action-value pilot queue."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
OUT=ROOT/'artifacts/libero_action_value_trace_pilot_v2'
TASKS={
 3:'KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it',
 10:'KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet',
 49:'LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket',
 72:'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate',
 73:'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy',
}

def main():
  (OUT/'cases').mkdir(parents=True,exist_ok=True)
  (OUT/'episodes').mkdir(parents=True,exist_ok=True)
  cases=[]
  for tid,name in TASKS.items():
    for init_id in range(8):
      case={'protocol_id':'LIBERO90_TRACE_FAITHFUL_ACTION_VALUE_PILOT_V2','case_id':f'task{tid:02d}__init{init_id:03d}',
            'task_id':tid,'task_name':name,'init_state_id':init_id,'case_seed':20261001+tid*100000+init_id,
            'env_seed':0,'settle_steps':10,'max_policy_steps':400,'arms':['vanilla','matched']}
      cases.append(case)
  manifest=OUT/'cases_manifest.json'
  manifest.write_text(json.dumps({'protocol_id':'LIBERO90_TRACE_FAITHFUL_ACTION_VALUE_PILOT_V2',
                                  'pair_count':len(cases),'sampling':'5 predeclared tasks x init_state_id 0..7',
                                  'tasks':TASKS,'case_seed_rule':'20261001 + task_id*100000 + init_state_id',
                                  'lambda':0.5,'attention_layers':[11],'entity_mode':'source_target_libero90',
                                  'query_mode':'instruction_only','cases':cases},indent=2)+'\n')
  # One exclusive queue per GPU; every case appears once.
  for gpu in [1,2,3,6,7]:
    assigned=cases[gpu-1::5] if gpu in (1,2,3) else cases[(3 if gpu==6 else 4)::5]
    (OUT/'cases'/f'gpu{gpu}.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in assigned))
  print(json.dumps({'pair_count':len(cases),'per_gpu':{str(g):len((OUT/'cases'/f'gpu{g}.jsonl').read_text().splitlines()) for g in [1,2,3,6,7]},'manifest':str(manifest)},indent=2))
if __name__=='__main__': main()
