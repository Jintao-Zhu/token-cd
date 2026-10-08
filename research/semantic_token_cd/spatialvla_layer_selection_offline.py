"""Offline all-layer prompt-attention screening for SpatialVLA on SIMPLER snapshots."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

WS=Path(__file__).resolve().parents[3]
REPO=WS/'token-cd'
OUT=REPO/'artifacts/spatialvla_layer_selection_v1b'
TASK_GROUPS={
 'fractal':('google_robot_open_drawer','google_robot_close_drawer','google_robot_pick_coke_can','google_robot_move_near','google_robot_place_apple_in_closed_top_drawer'),
 'bridge':('widowx_carrot_on_plate','widowx_put_eggplant_in_basket','widowx_spoon_on_towel','widowx_stack_cube'),
}
SEEDS=(100,110,120,130,150,160)
SYNONYMS={
 'google_robot_open_drawer':'pull the top drawer open',
 'google_robot_close_drawer':'shut the top drawer',
 'google_robot_pick_coke_can':'grasp the coke can',
 'google_robot_move_near':'move the same object close to the same reference object', # replaced dynamically
 'google_robot_place_apple_in_closed_top_drawer':'pull the top drawer open',
 'widowx_carrot_on_plate':'place the carrot onto the plate',
 'widowx_put_eggplant_in_basket':'place the eggplant inside the yellow basket',
 'widowx_spoon_on_towel':'place the spoon onto the towel',
 'widowx_stack_cube':'put the green block on top of the yellow block',
}

def get_controls(task, instruction, meta):
    names={a['name'].lower():int(a['id']) for a in meta.get('actors',[])}
    links={l['name'].lower():int(l['id']) for a in meta.get('articulations',[]) for l in a.get('links',[])}
    if task in ('google_robot_open_drawer','google_robot_close_drawer','google_robot_place_apple_in_closed_top_drawer'):
        old='top_drawer'; new='bottom_drawer'
        visible=set(meta.get('visible_actor_ids',[]))
        # One close-drawer seed occludes the bottom drawer completely. Use the
        # visible middle drawer for that locked state rather than scoring an
        # empty target mask.
        if links[new] not in visible:
            new='middle_drawer'
        new_inst=instruction.replace('top',new.split('_')[0])
        return new_inst,links[old],links[new]
    if task=='google_robot_pick_coke_can':
        old=next(k for k in names if 'coke_can' in k)
        pepsi=next(k for k in names if 'pepsi_can' in k)
        visible=set(meta.get('visible_actor_ids',[]))
        if names[pepsi] in visible:
            new=pepsi; phrase='pepsi can'
        else:
            # Some canonical seeds fully occlude the Pepsi distractor. Use a
            # visible segmented distractor in that state rather than an empty
            # target region; deterministic preference order is frozen here.
            options=[('apple','apple'),('sponge','sponge'),('orange','orange')]
            phrase_key=next(k for k,_ in options if names[k] in visible)
            phrase=dict(options)[phrase_key]
            new=phrase_key
        return f'pick {phrase}',names[old],names[new]
    if task=='google_robot_move_near':
        prefix='move '; sep=' near '
        if not (instruction.startswith(prefix) and sep in instruction): raise ValueError(instruction)
        source,target=instruction[len(prefix):].split(sep,1)
        new_inst=f'move {target} near {source}'
        def find_id(phrase):
            toks=[x for x in phrase.lower().replace('_',' ').split() if x not in {'the','a','an','plastic','baked','opened','v2','can'}]
            candidates=[]
            for n,i in names.items():
                score=sum(tok in n for tok in toks)
                if score: candidates.append((score,n,i))
            if not candidates: raise ValueError(f'cannot map {phrase!r} to {names}')
            return max(candidates)[2]
        return new_inst,find_id(target),find_id(source)
    if task=='widowx_carrot_on_plate':
        return 'put the plate on the carrot',next(v for k,v in names.items() if 'plate' in k),next(v for k,v in names.items() if 'carrot' in k)
    if task=='widowx_put_eggplant_in_basket':
        # SAPIEN asset encodes the benchmark basket goal as a target plane inside the sink.
        return 'put the yellow basket over the eggplant',next(v for k,v in names.items() if 'dummy_sink_target_plane' in k),next(v for k,v in names.items() if 'eggplant' in k)
    if task=='widowx_spoon_on_towel':
        return 'put the towel under the spoon',next(v for k,v in names.items() if 'table_cloth' in k),next(v for k,v in names.items() if 'spoon' in k)
    if task=='widowx_stack_cube':
        return 'stack the yellow block on the green block',next(v for k,v in names.items() if 'yellow_cube' in k),next(v for k,v in names.items() if 'green_cube' in k)
    raise ValueError(task)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--group',choices=tuple(TASK_GROUPS),required=True); p.add_argument('--gpu',type=int,choices=(1,2,3),default=1); p.add_argument('--limit',type=int,default=0,help='debug limit, 0 means all locked states'); a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu); os.environ['TOKENIZERS_PARALLELISM']='false'; os.environ['HF_HUB_OFFLINE']='1'
    for q in (WS/'spatialvla-runtime-overlay-20260928',WS/'SpatialVLA-SimplerEnv',WS/'SpatialVLA-SimplerEnv/ManiSkill2_real2sim',REPO/'task1/shim_site',REPO,WS/'pcd_openvla_simpler_box_31b027e/source'):
        if str(q) not in sys.path: sys.path.insert(0,str(q))
    from research.semantic_token_cd import spatialvla_l11_matched_remaining_rollout as runtime
    runtime.bootstrap()
    spatial_env=str(WS/'SpatialVLA-SimplerEnv')
    if spatial_env in sys.path: sys.path.remove(spatial_env)
    sys.path.insert(0,spatial_env)
    import torch
    from transformers.cache_utils import HybridCache
    from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
    torch.set_num_threads(1)
    model_path=WS/f"checkpoints/spatialvla-4b-224-sft-{a.group}"
    cfg=json.loads((model_path/'config.json').read_text()); n_layers=int(cfg['text_config']['num_hidden_layers'])
    if n_layers!=26: raise RuntimeError(f'expected audited 26 layers, found {n_layers}')
    base=SpatialVLAInference(saved_model_path=str(model_path),policy_setup='google_robot' if a.group=='fractal' else 'widowx_bridge',action_ensemble_temp=-0.8)
    model=base.vla; proc=base.processor; model.eval()
    def get_scores(image,instruction):
        base.reset(instruction)
        image=base._resize_image(image); base._add_image_to_history(image); images=base._obtain_image_history()
        inputs=proc(images=images,text=instruction,unnorm_key=base.unnorm_key,return_tensors='pt',do_normalize=False).to(torch.bfloat16).to(model.device)
        with torch.inference_mode():
            cache=HybridCache(config=model.config.text_config,batch_size=1,max_cache_len=inputs['input_ids'].shape[1]+256,device=model.device,dtype=model.dtype)
            out=model(**inputs,past_key_values=cache,use_cache=True,output_attentions=True,return_dict=True)
        ids=inputs['input_ids'][0]; ip=int(model.config.image_token_index)
        vpos=torch.nonzero(ids==ip,as_tuple=False).flatten()
        if vpos.numel()!=256: raise RuntimeError(f'image positions={vpos.numel()}')
        qids=proc.tokenizer(instruction,add_special_tokens=False)['input_ids']; seq=ids.detach().cpu().tolist()
        start=next((i for i in range(len(seq)-len(qids)+1) if seq[i:i+len(qids)]==qids),None)
        if start is None: raise RuntimeError(f'instruction token sequence not found: {instruction}')
        qpos=list(range(start,start+len(qids)))
        if len(out.attentions)!=n_layers: raise RuntimeError(f'attention layers={len(out.attentions)} expected {n_layers}')
        scores=[]
        for layer in range(n_layers):
            attn=out.attentions[layer][0,:,qpos,:][:,:,vpos]
            scores.append(attn.float().mean(dim=(0,1)).cpu().numpy())
        return np.asarray(scores,dtype=np.float32),{'query_positions':qpos,'visual_positions':vpos.cpu().tolist(),'attention_mass_by_layer':[float(x.sum()) for x in scores]}
    processed=0
    for task in TASK_GROUPS[a.group]:
        for seed in SEEDS:
            d=OUT/'states'/task/f'seed_{seed:03d}'; meta=json.loads((d/'meta.json').read_text()); data=np.load(d/'input.npz')
            image=data['image']; seg=data['seg_actor']; instruction=meta['instruction']; newinst,old_id,new_id=get_controls(task,instruction,meta)
            synonym=SYNONYMS[task]
            if task=='google_robot_move_near':
                src,tgt=instruction[len('move '):].split(' near ',1); synonym=f'move {src} close to {tgt}'
            arrays=[]; metas=[]
            for text in (instruction,synonym,newinst):
                score,sm=get_scores(image,text); arrays.append(score); metas.append(sm)
            import cv2
            def tokmask(actor_id):
                x=(seg==actor_id).astype(np.uint8); x=cv2.resize(x,(224,224),interpolation=cv2.INTER_NEAREST)
                x=x.reshape(16,14,16,14).any(axis=(1,3)).reshape(-1)
                return x.astype(np.uint8)
            oldmask=tokmask(old_id); newmask=tokmask(new_id)
            if not oldmask.any() or not newmask.any(): raise RuntimeError(f'empty target mask {task} seed {seed}: {old_id} {new_id}')
            summroot=REPO/'artifacts/spatialvla_simpler_l11matched_4task_n100' if task in TASK_GROUPS['fractal'][:4] else REPO/'artifacts/spatialvla_simpler_l11matched_remaining5_n100'
            summary=json.loads((summroot/'episodes'/task/'l11_matched'/f'episode_{seed}_summary.json').read_text())
            m=int(summary['l11_trace'][0]['m_t'])
            np.savez_compressed(d/'layers.npz',original=arrays[0],synonym=arrays[1],target_switch=arrays[2],old_target_mask=oldmask,new_target_mask=newmask)
            result={'protocol_id':'SPATIALVLA_LAYER_SELECTION_OFFLINE_V1','task':task,'seed':seed,'checkpoint_group':a.group,'layers':n_layers,'m':m,'instruction':instruction,'synonym_instruction':synonym,'target_switch_instruction':newinst,'old_target_actor_id':old_id,'new_target_actor_id':new_id,'old_target_tokens':int(oldmask.sum()),'new_target_tokens':int(newmask.sum()),'attention_meta':metas,'arrays_file':'layers.npz'}
            (d/'layers.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
            print(json.dumps({'task':task,'seed':seed,'m':m,'old_tokens':int(oldmask.sum()),'new_tokens':int(newmask.sum()),'done':True}),flush=True)
            torch.cuda.empty_cache()
            processed+=1
            if a.limit and processed>=a.limit:
                print('DEBUG_LIMIT_REACHED',flush=True)
                return
    print('EXTRACTION_COMPLETE',a.group,flush=True)
if __name__=='__main__': main()
