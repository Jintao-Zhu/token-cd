#!/usr/bin/env python3
"""Select a few text-to-visual attention heads on independent dev states."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.libero_runtime import load_policy, prepare_agentview, set_determinism
from research.semantic_token_cd.libero_attention_role_diagnostics import (
    SWITCH_TEMPLATES,
    TASK_REFERENCE,
    find_phrase_span,
    forward_attention,
    mask_tokens,
    normalize_map,
    topk_indices,
)
from research.semantic_token_cd.libero_matched_rollout import source_relation_text

N_VISUAL=256


def parse_ints(value):
    out=[]
    for p in value.split(','):
        p=p.strip()
        if not p: continue
        if '-' in p:
            lo,hi=map(int,p.split('-',1)); out.extend(range(lo,hi+1))
        else: out.append(int(p))
    return sorted(set(out))


def target_m_from_matched(task, state):
    p=Path('artifacts/libero_official_matched_500_v1')/task/f'episode_{state:03d}.json'
    if not p.exists(): return None
    d=json.loads(p.read_text())
    return int(round(float(d['selected_token_count_mean'])))


def head_metric(attn, head, qidx_or_indices, target_tokens, distractor_tokens, m):
    qidx=qidx_or_indices if isinstance(qidx_or_indices,list) else [qidx_or_indices]
    vals=attn[0, head, [N_VISUAL+i for i in qidx], 1:1+N_VISUAL].detach().float().cpu().numpy()
    score=vals.mean(axis=0)
    score=normalize_map(score)
    visual_mass=float(vals.sum(axis=-1).mean())
    sel=topk_indices(score,m)
    tc=len(sel&target_tokens)/max(1,len(target_tokens))
    dc=len(sel&distractor_tokens)/max(1,len(distractor_tokens))
    tm=float(score[list(target_tokens)].sum())/max(1,len(target_tokens)) if target_tokens else 0.0
    dm=float(score[list(distractor_tokens)].sum())/max(1,len(distractor_tokens)) if distractor_tokens else 0.0
    return visual_mass,tc,dc,tm,dm


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--tasks',default='0-9')
    p.add_argument('--selection-states',default='0-9')
    p.add_argument('--validation-states',default='10-19')
    p.add_argument('--top-n',default='3,5')
    a=p.parse_args()
    os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'; os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    task_ids=parse_ints(a.tasks); sel_states=parse_ints(a.selection_states); val_states=parse_ints(a.validation_states)
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    set_determinism(7)
    model,processor=load_policy(a.checkpoint,Path('third_party/openvla/prismatic/extern/hf'),device=f'cuda:{a.gpu}')
    rows=[]
    for split,states in [('selection',sel_states),('validation',val_states)]:
        for tid in task_ids:
            task=suite.get_task(tid); bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
            env=SegmentationRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
            init_states=suite.get_task_init_states(tid)
            try:
                for sidx in states:
                    if sidx>=len(init_states): continue
                    env.seed(0); env.reset(); obs=env.set_init_state(init_states[sidx])
                    for _ in range(10): obs,_,_,_=env.step([0,0,0,0,0,0,-1])
                    seg=np.asarray(obs['agentview_segmentation_instance']); _,image=prepare_agentview(obs)
                    inputs,atts=forward_attention(model,processor,image,task.language)
                    ids=inputs['input_ids'][0].detach().cpu().tolist()
                    qspan=find_phrase_span(ids,processor.tokenizer,source_relation_text(task.language))
                    qidx=qspan[1]-1
                    m=target_m_from_matched(task.name,sidx)
                    if m is None: continue
                    tgt=mask_tokens(seg,env.instance_to_id,'akita_black_bowl_1')
                    dis=mask_tokens(seg,env.instance_to_id,'akita_black_bowl_2')
                    ref=set()
                    for name in TASK_REFERENCE[tid]: ref |= mask_tokens(seg,env.instance_to_id,name)
                    sw=SWITCH_TEMPLATES.get(tid)
                    sw_alt=set(); sw_orig=ref
                    sw_atts=None; sw_ids=None; sw_qidx=None
                    if sw is not None:
                        sw_instr,sw_ref=sw
                        sw_inputs,sw_atts=forward_attention(model,processor,image,sw_instr)
                        sw_ids=sw_inputs['input_ids'][0].detach().cpu().tolist()
                        sw_qspan=find_phrase_span(sw_ids,processor.tokenizer,source_relation_text(sw_instr))
                        sw_qidx=sw_qspan[1]-1
                        sw_alt=mask_tokens(seg,env.instance_to_id,sw_ref)
                    for layer in range(len(atts)):
                        for head in range(atts[layer].shape[1]):
                            vm,tc,dc,tm,dm=head_metric(atts[layer],head,qidx,tgt,dis,m)
                            sm=sc=0.0
                            if sw_atts is not None:
                                svm,stc,sdc,stm,sdm=head_metric(sw_atts[layer],head,sw_qidx,tgt,dis,m)
                                # switch response: alt-reference preference minus original-reference preference
                                sm=float(sw_atts[layer][0,head,N_VISUAL+sw_qidx,1:1+N_VISUAL].sum()==0) # placeholder, replaced below
                                # explicit maps
                                qa=atts[layer][0,head,[N_VISUAL+qidx],1:1+N_VISUAL].mean(0).detach().float().cpu().numpy(); qa=normalize_map(qa)
                                qs=sw_atts[layer][0,head,[N_VISUAL+sw_qidx],1:1+N_VISUAL].mean(0).detach().float().cpu().numpy(); qs=normalize_map(qs)
                                sm=float(qs[list(sw_alt)].sum())/max(1,len(sw_alt))-float(qa[list(sw_orig)].sum())/max(1,len(sw_orig))
                                sc=len(topk_indices(qs,m)&sw_alt)/max(1,len(sw_alt))-len(topk_indices(qa,m)&sw_orig)/max(1,len(sw_orig))
                            rows.append({'split':split,'task_id':tid,'task':task.name,'init_state':sidx,'layer':layer,'head':head,'visual_mass':vm,'target_coverage':tc,'distractor_coverage':dc,'target_discrimination':tc-dc,'target_mass_diff':tm-dm,'switch_alt_mass_diff':sm,'switch_alt_coverage_diff':sc,'m':m})
                    print(json.dumps({'split':split,'task':task.name,'init_state':sidx,'done':True}),flush=True)
            finally: env.close()
    out=a.artifact.resolve(); out.mkdir(parents=True,exist_ok=True)
    csv_path=out/'HEAD_METRICS.csv'
    keys=list(rows[0].keys())
    with csv_path.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(rows)
    # Aggregate selection metrics and choose a few heads.
    agg=defaultdict(list)
    for r in rows:
        if r['split']=='selection': agg[(r['layer'],r['head'])].append(r)
    scored=[]
    for (layer,head),g in agg.items():
        vm=np.mean([r['visual_mass'] for r in g]); td=np.mean([r['target_discrimination'] for r in g]); sw=np.mean([r['switch_alt_coverage_diff'] for r in g])
        scored.append({'layer':layer,'head':head,'visual_mass':vm,'target_discrimination':td,'switch_alt_coverage_diff':sw,'n':len(g)})
    # Filter by visual mass above median, then rank by target discrimination + switch response.
    med=np.median([x['visual_mass'] for x in scored]) if scored else 0
    filtered=[x for x in scored if x['visual_mass']>=med]
    td_rank={x['layer']*1000+x['head']:i for i,x in enumerate(sorted(filtered,key=lambda x:x['target_discrimination'],reverse=True))}
    sw_rank={x['layer']*1000+x['head']:i for i,x in enumerate(sorted(filtered,key=lambda x:x['switch_alt_coverage_diff'],reverse=True))}
    for x in filtered:
        key=x['layer']*1000+x['head']; x['rank_score']=td_rank[key]+sw_rank[key]
    selected={}
    for n in parse_ints(a.top_n):
        chosen=[]; used_layers=set()
        for x in sorted(filtered,key=lambda x:(x['rank_score'],-x['target_discrimination'])):
            if x['layer'] in used_layers: continue
            chosen.append({'layer':int(x['layer']),'head':int(x['head']),'metrics':x}); used_layers.add(x['layer'])
            if len(chosen)>=n: break
        selected[str(n)]=chosen
    (out/'SELECTED_HEADS.json').write_text(json.dumps(selected,indent=2)+"\n")
    print(json.dumps({'rows':len(rows),'csv':str(csv_path),'selected':selected}),flush=True)

if __name__=='__main__': main()
