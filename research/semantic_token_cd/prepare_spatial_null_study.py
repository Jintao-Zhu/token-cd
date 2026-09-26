#!/usr/bin/env python3
"""Freeze manifest and deterministic queues for the Raw-L11 vs Spatial-Null study."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
ART=ROOT/'artifacts/libero_raw_l11_vs_spatial_null_5task_n40_v1_20260926'
PRIOR_SOURCE=ROOT/'artifacts/libero_position_prior_offline_v1_20260926/POSITION_PRIOR_ANALYSIS.json'
TASKS={
 3:'KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it',
 10:'KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet',
 49:'LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket',
 72:'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate',
 73:'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy',
}
INIT_IDS=list(range(0,8))+list(range(16,48))
GPUS=(1,2,3,4,5); SLOTS=3
PROTOCOL='LIBERO90_RAW_L11_VS_SPATIAL_NULL_V1'

def main():
    ART.mkdir(parents=True,exist_ok=False)
    prior_doc=json.loads(PRIOR_SOURCE.read_text())
    frozen={
      'source_analysis':str(PRIOR_SOURCE),
      'source_sha256':hashlib.sha256(PRIOR_SOURCE.read_bytes()).hexdigest(),
      'development_source':prior_doc['development_source'],
      'position_prior_definition':'per-task mean normalized attention over 400 OSMesa sampled states, init_state_ids 8..15, five tasks; each 16x16 map renormalized by the offline analysis',
      'position_priors':prior_doc['results']['task_specific']['prior_16x16'],
    }
    (ART/'position_prior_frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    prior_hash=hashlib.sha256((ART/'position_prior_frozen.json').read_bytes()).hexdigest()
    cases=[]
    for tid,name in TASKS.items():
      for iid in INIT_IDS:
        cases.append({
          'protocol_id':PROTOCOL,'case_id':f'task{tid:02d}__init{iid:03d}',
          'task_id':tid,'task_name':name,'init_state_id':iid,
          'case_seed':20261001+tid*100000+iid,'env_seed':0,'settle_steps':10,
          'max_policy_steps':400,'arms':['vanilla','matched','matched_spatial_null'],
          'renderer_backend':'egl','render_gpu':7,'position_prior_sha256':prior_hash,
        })
    if len(cases)!=200 or len({c['case_id'] for c in cases})!=200:
      raise RuntimeError('bad case manifest')
    manifest={
      'protocol_id':PROTOCOL,'status':'FROZEN_BEFORE_ROLLOUTS',
      'pair_count':len(cases),'planned_arm_episodes':len(cases)*3,
      'tasks':{str(k):{'task_name':v,'pair_count':40} for k,v in TASKS.items()},
      'init_state_ids_per_task':INIT_IDS,
      'seed_rule':'case_seed = 20261001 + task_id*100000 + init_state_id; env_seed=0',
      'seed_selection_note':'The fixed evaluation list excludes init_state_ids 8..15, which were used to fit the frozen position priors. It includes the pre-existing 0..7 and 16..21 diagnostic IDs, plus 22..47; no ID was selected or removed using this study outcome.',
      'arms':['vanilla','matched','matched_spatial_null'],
      'primary_comparison':'matched (Raw L11-Matched) vs matched_spatial_null',
      'primary_metric':'paired binary episode success; report Raw-only success and Null-only success, exact two-sided McNemar, task-wise paired net, and five-task macro paired effect',
      'lambda':0.5,'attention_layers':[11],'entity_mode':'source_target_libero90',
      'query_mode':'instruction_only','negative':'canonical harmonic reconstruction beta=0',
      'budget':'each intervention arm computes current-state canonical matched m_t; Null draws exactly m_t tokens from frozen task-specific position prior',
      'spatial_null':'one N1 realization per case and policy step; weighted sampling without replacement with p proportional to frozen task prior; overlap with Raw L11 is allowed',
      'spatial_null_seed_rule':'uint64(first 8 bytes SHA256(task_id:case_seed:policy_step:N1))',
      'checkpoint':'/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90',
      'renderer_backend':'MuJoCo EGL','render_gpu':7,'inference_gpus':list(GPUS),
      'workers_per_gpu':SLOTS,'frozen_position_prior_file':'position_prior_frozen.json',
      'position_prior_sha256':prior_hash,'episodes':cases,
    }
    (ART/'cases_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ART/'cases').mkdir(); (ART/'logs').mkdir()
    queues={f'gpu{g}_slot{s}':[] for g in GPUS for s in range(1,SLOTS+1)}
    workers=list(queues)
    for i,c in enumerate(cases): queues[workers[i%len(workers)]].append(c)
    for key,rows in queues.items():
      (ART/'cases'/f'{key}.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
    manifest['worker_assignment']={k:{'case_count':len(v),'case_ids':[x['case_id'] for x in v]} for k,v in queues.items()}
    (ART/'cases_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=f'''# Preregistration: Raw L11 vs Spatial Null, LIBERO five-task N=200\n\nStatus: **FROZEN BEFORE ROLLOUTS**.\n\nPrimary question: does current-state Raw L11-Matched outperform a selector that knows only the task-specific frozen spatial prior?\n\n- Five tasks: IDs 3, 10, 49, 72, 73; 40 paired initial states per task; 200 pairs total.\n- Arms: Vanilla, canonical Raw L11-Matched, and Spatial-Null-Matched. Exactly 600 closed-loop episodes.\n- Primary contrast: Raw L11-Matched vs Spatial Null. Vanilla provides Rescue/Harm context.\n- Spatial Null prior: frozen task-specific mean attention maps fit on OSMesa init IDs 8..15, attached in `position_prior_frozen.json`. It does not read current-step L11 scores when choosing the mask.\n- Every intervention step computes canonical `m_t`; Raw selects canonical Top-m, Null samples exactly `m_t` tokens without replacement with probabilities proportional to the frozen task prior. Canonical overlap is allowed. Harmonic reconstruction and lambda 0.5 are unchanged.\n- One null realization N1 per case/step. Seed is SHA256-derived from task ID, case seed, policy step, and N1.\n- Init IDs are frozen as 0..7 and 16..47 for each task, excluding 8..15 used to estimate the prior. Some included starts appeared in earlier diagnostic runs; they are retained by this predeclared ID rule and will not be filtered by outcome.\n- EGL is routed to GPU7; model inference uses three workers each on GPUs 1..5. Other tmux sessions are left untouched.\n- Report arm success, Raw-only/Null-only wins, exact two-sided McNemar for the primary contrast, task-wise paired net, and five-task macro paired effect. No seed/task substitution or efficacy-based interim stopping. Stop only for technical integrity or device errors.\n\nFull case list and queues: `cases_manifest.json`, `cases/`. Frozen-prior SHA256: `{prior_hash}`.\n'''
    (ART/'PREREGISTRATION.md').write_text(report)
    print(json.dumps({'artifact':str(ART),'pairs':len(cases),'episodes':len(cases)*3,'queues':{k:len(v) for k,v in queues.items()},'prior_sha256':prior_hash},indent=2))

if __name__=='__main__': main()
