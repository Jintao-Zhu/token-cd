#!/usr/bin/env python3
"""Build the preregistered 30-pair OSMesa positive-G gate pilot queue."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
OUT=ROOT/'artifacts/libero_action_value_gate_osmesa_pilot_v1_20260925'
TASKS={
 3:'KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it',
 10:'KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet',
 49:'LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket',
 72:'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate',
 73:'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy',
}
GPUS=(1,2,3,6,7)

def main():
    (OUT/'cases').mkdir(parents=True,exist_ok=True)
    cases=[]
    for tid,name in TASKS.items():
        for init_id in range(16,22):
            cases.append({'protocol_id':'LIBERO90_ACTION_VALUE_GPOSITIVE_GATE_PILOT_V1',
              'case_id':f'task{tid:02d}__init{init_id:03d}','task_id':tid,'task_name':name,
              'init_state_id':init_id,'case_seed':20261001+tid*100000+init_id,'env_seed':0,
              'settle_steps':10,'max_policy_steps':400,
              'arms':['vanilla','matched','positive_gated'],'renderer_backend':'osmesa'})
    manifest={'protocol_id':'LIBERO90_ACTION_VALUE_GPOSITIVE_GATE_PILOT_V1',
      'parent_heldout':'libero_action_value_trace_pilot_v2_osmesa_ext40_init8_15_20260925',
      'pair_count':len(cases),'sampling':'5 tasks x init_state_id 16..21',
      'tasks':TASKS,'case_seed_rule':'20261001 + task_id*100000 + init_state_id',
      'arms':['vanilla','matched','positive_gated'],'lambda':0.5,'gate':'median current-step G over action dimensions 0..5 > 0',
      'attention_layers':[11],'entity_mode':'source_target_libero90','query_mode':'instruction_only',
      'negative':'canonical_harmonic_beta0','renderer_backend':'osmesa','cases':cases}
    (OUT/'cases_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for i,gpu in enumerate(GPUS):
        assigned=cases[i::len(GPUS)]
        (OUT/'cases'/f'gpu{gpu}.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in assigned))
    print(json.dumps({'manifest':str(OUT/'cases_manifest.json'),'cases':len(cases),
      'per_gpu':{str(g):len((OUT/'cases'/f'gpu{g}.jsonl').read_text().splitlines()) for g in GPUS}},indent=2))
if __name__=='__main__': main()
