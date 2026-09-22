#!/usr/bin/env python3
"""Offline comparison of instruction-query grouping schemes for LIBERO attention."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.libero_runtime import load_policy, prepare_agentview, set_determinism
from research.semantic_token_cd.libero_attention_role_diagnostics import (
    SWITCH_TEMPLATES,
    TASK_REFERENCE,
    attention_map,
    find_phrase_span,
    forward_attention,
    mask_tokens,
    normalize_map,
    role_texts,
    span_indices,
    topk_indices,
)

N_VISUAL = 256


def parse_ints(value: str) -> list[int]:
    out=[]
    for part in value.split(','):
        part=part.strip()
        if not part: continue
        if '-' in part:
            lo,hi=map(int,part.split('-',1)); out.extend(range(lo,hi+1))
        else: out.append(int(part))
    return sorted(set(out))


def parse_layer_groups(value: str):
    groups=[]
    for part in value.split(','):
        part=part.strip()
        if not part: continue
        if '-' in part:
            lo,hi=map(int,part.split('-',1)); groups.append((f'mean{lo}-{hi}',tuple(range(lo,hi+1))))
        else: groups.append((str(int(part)),(int(part),)))
    return groups


def source_relation_text(instruction: str) -> str:
    target, relation, reference = role_texts(instruction)
    return f"{target} {relation} {reference}"


def destination_text(instruction: str) -> str:
    text=instruction.lower()
    m=re.search(r"\b(?:place|put|move)\b.*?\b(?:on|in|onto|into|near|to|next\s+to)\b\s+(.*)$",text)
    if not m:
        raise RuntimeError(f'cannot parse destination from {instruction!r}')
    phrase=m.group(1).strip()
    phrase=re.sub(r"^(?:the|a|an)\s+",'',phrase)
    phrase=re.sub(r"\b(?:it|them|this|that)\b",'',phrase)
    return ' '.join(re.findall(r'[a-z0-9]+',phrase))


def group_maps(attentions, layers, ids, tokenizer, instruction):
    full_span=find_phrase_span(ids,tokenizer,instruction)
    target,relation,reference=role_texts(instruction)
    source_rel=source_relation_text(instruction)
    destination=destination_text(instruction)
    source_span=find_phrase_span(ids,tokenizer,source_rel)
    destination_clause=re.split(r"\band\s+(?:place|put|move)\b",instruction.lower(),maxsplit=1)[1].strip()
    destination_clause_span=find_phrase_span(ids,tokenizer,destination_clause)
    destination_span=find_phrase_span(ids,tokenizer,destination,*destination_clause_span)
    # Restrict role/reference spans to source clause.
    src_clause=re.split(r"\band\s+(?:place|put|move)\b",instruction.lower(),maxsplit=1)[0]
    src_span=find_phrase_span(ids,tokenizer,src_clause)
    role_spans={
        'target':span_indices(ids,tokenizer,(target,),*src_span),
        'relation':span_indices(ids,tokenizer,(relation,),*src_span),
        'reference':span_indices(ids,tokenizer,(reference,),*src_span),
    }
    out={name:[] for name in ['full','role','sr','sr_d_1.0','sr_d_0.5','sr_d_0.25']}
    for layer in layers:
        att=attentions[layer]
        full=normalize_map(attention_map(att,list(range(full_span[0],full_span[1]))))
        role=normalize_map(np.mean([
            normalize_map(attention_map(att,role_spans['target'])),
            normalize_map(attention_map(att,role_spans['relation'])),
            normalize_map(attention_map(att,role_spans['reference'])),
        ],axis=0))
        sr=normalize_map(attention_map(att,list(range(source_span[0],source_span[1]))))
        dst=normalize_map(attention_map(att,list(range(destination_span[0],destination_span[1]))))
        out['full'].append(full); out['role'].append(role); out['sr'].append(sr)
        out['sr_d_1.0'].append(normalize_map(sr+1.0*dst))
        out['sr_d_0.5'].append(normalize_map(sr+0.5*dst))
        out['sr_d_0.25'].append(normalize_map(sr+0.25*dst))
    return {k:normalize_map(np.mean(v,axis=0)) for k,v in out.items()}, source_span, destination_span


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--tasks',default='0-9')
    p.add_argument('--states',default='0-19')
    p.add_argument('--layer-groups',default='11,23-25')
    p.add_argument('--topk',default='16,32,50')
    a=p.parse_args()
    os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'; os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    task_ids=parse_ints(a.tasks); states=parse_ints(a.states); groups=parse_layer_groups(a.layer_groups); topks=parse_ints(a.topk)
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    set_determinism(7)
    model,processor=load_policy(a.checkpoint,Path('third_party/openvla/prismatic/extern/hf'),device=f'cuda:{a.gpu}')
    rows=[]
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
                target_tokens=mask_tokens(seg,env.instance_to_id,'akita_black_bowl_1')
                distractor_tokens=mask_tokens(seg,env.instance_to_id,'akita_black_bowl_2')
                ref_tokens=set()
                for name in TASK_REFERENCE[tid]: ref_tokens |= mask_tokens(seg,env.instance_to_id,name)
                switch=SWITCH_TEMPLATES.get(tid)
                switch_data=None
                if switch is not None:
                    switch_instr,switch_ref=switch
                    switch_inputs,switch_atts=forward_attention(model,processor,image,switch_instr)
                    switch_ids=switch_inputs['input_ids'][0].detach().cpu().tolist()
                    switch_data=(switch_instr,switch_ref,switch_atts,switch_ids)
                for group,layers in groups:
                    maps,source_span,dest_span=group_maps(atts,layers,ids,processor.tokenizer,task.language)
                    switch_maps=None
                    if switch_data is not None:
                        switch_maps,_,_=group_maps(switch_data[2],layers,switch_data[3],processor.tokenizer,switch_data[0])
                    for mode,values in maps.items():
                        for k in topks:
                            sel=topk_indices(values,k)
                            tc=len(sel&target_tokens)/max(1,len(target_tokens)); dc=len(sel&distractor_tokens)/max(1,len(distractor_tokens))
                            tm=float(values[list(target_tokens)].sum())/max(1,len(target_tokens)) if target_tokens else 0.0
                            dm=float(values[list(distractor_tokens)].sum())/max(1,len(distractor_tokens)) if distractor_tokens else 0.0
                            rm=float(values[list(ref_tokens)].sum())/max(1,len(ref_tokens)) if ref_tokens else 0.0
                            rows.append({'task_id':tid,'task':task.name,'init_state':sidx,'layer_group':group,'query_mode':mode,'topk':k,'target_tokens':len(target_tokens),'distractor_tokens':len(distractor_tokens),'target_discrimination':tc-dc,'target_mass_diff':tm-dm,'reference_mass':rm})
                    if switch_data is not None:
                        switch_instr,switch_ref,_,_=switch_data
                        alt_tokens=mask_tokens(seg,env.instance_to_id,switch_ref)
                        for mode,values in maps.items():
                            sm=switch_maps[mode]
                            am=float(sm[list(alt_tokens)].sum())/max(1,len(alt_tokens)) if alt_tokens else 0.0
                            om=float(values[list(ref_tokens)].sum())/max(1,len(ref_tokens)) if ref_tokens else 0.0
                            ac=len(topk_indices(sm,max(topks))&alt_tokens)/max(1,len(alt_tokens))
                            oc=len(topk_indices(values,max(topks))&ref_tokens)/max(1,len(ref_tokens))
                            rows.append({'task_id':tid,'task':task.name,'init_state':sidx,'layer_group':group,'query_mode':mode+'_switch','topk':max(topks),'target_tokens':len(target_tokens),'distractor_tokens':len(distractor_tokens),'target_discrimination':None,'target_mass_diff':None,'reference_mass':None,'switch_alt_mass_diff':am-om,'switch_alt_coverage_diff':ac-oc})
                print(json.dumps({'task':task.name,'init_state':sidx,'done':True}),flush=True)
        finally: env.close()
    out=a.artifact.resolve(); out.mkdir(parents=True,exist_ok=True)
    path=out/'GROUP_DIAGNOSTICS.csv'
    if rows:
        keys=list(rows[0].keys())
        # union keys for switch rows
        for r in rows: keys += [k for k in r if k not in keys]
        with path.open('w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(json.dumps({'rows':len(rows),'csv':str(path)}),flush=True)

if __name__=='__main__': main()
