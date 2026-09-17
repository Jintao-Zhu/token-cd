"""Paper-based DTP calibration, not an author-code reproduction.

Only reads existing observations; no simulator rollout or outcome-based tuning.
Run from repository root with python -m research.semantic_token_cd.dtp_paper_calibration.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

ROOT = Path('artifacts/dtp_openvla_calibration_v1')
SEEDS = {'open_drawer':[0,11,32,95,97], 'close_drawer':[0,1,9,21,49],
         'pick_coke_can':[1,3,30,59,95], 'move_near':[0,1,2,31,73]}
KS = (64,109,154)
TAUS = (.5,1.,1.5)

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)

def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sources():
    rows=[]
    for task,seeds in SEEDS.items():
        prefix=Path('artifacts/prompt_attn_l11_budget_diagnostic_v1' if task=='close_drawer'
                    else 'artifacts/prompt_attn_layer_selection_v1')/'states'/('google_robot_'+task)
        for seed in seeds:
            files=sorted((prefix/f'seed_{seed:03d}').glob('step_*.npz'))
            assert len(files)==3,(task,seed)
            for file in files:
                m=json.loads(file.with_suffix('.json').read_text())
                with np.load(file) as a:
                    sha=hashlib.sha256(a['image'].tobytes()).hexdigest()
                    assert a['original'].shape==(32,256)
                rows.append(dict(task=task,seed=seed,phase=m['phase'],source=str(file.resolve()),
                    id=f'{task}__{seed:03d}__{file.stem}',rgb_sha256=sha,instruction=m['instruction']))
    assert len(rows)==60
    return rows

def top(scores,k):
    return np.lexsort((np.arange(256),-np.asarray(scores,dtype=np.float64)))[:k]

def spatial(score,mode):
    x=np.asarray(score,dtype=np.float64).copy()
    if mode in ('corners','both'): x[[0,15,240,255]]=0
    if mode in ('gaussian','both'): x=gaussian_filter(x.reshape(16,16),.65,mode='reflect',truncate=4).ravel()
    return x

def action_pattern(per_layer):
    """[32,7,256], weights = each layer visual mass / sum of masses.

    Using raw softmax visual rows preserves the paper's visual-mass weighting;
    do not renormalize each visual row before multiplying by its weight.
    """
    x=np.asarray(per_layer,dtype=np.float64)
    assert x.shape==(32,7,256) and np.isfinite(x).all() and (x>=0).all()
    mass=x.sum(-1); denom=mass.sum(0,keepdims=True)
    assert (denom>0).all()
    weights=mass/denom
    return (weights[:,:,None]*x).sum(0),weights

def prune(pattern,protect,tau):
    mask=pattern>tau*pattern[:,protect].max(1)[:,None]
    mask[:,protect]=False
    return mask

def prepare():
    rows=sources();save(ROOT/'SAMPLES.json',rows)
    save(ROOT/'IMPLEMENTATION_CHOICES.json',dict(
        origin='paper-based reimplementation; author repository expired',
        prompt='all non-special actual instruction tokens; all heads mean; layers calibrated',
        action='all 32 layers; seven separate predictive action queries; raw post-softmax weights',
        layer_weights='per-query layer visual mass divided by sum of layer visual masses',
        spatial='zero exactly four corner scores then Gaussian sigma=.65 patch units, reflect padding, truncate=4',
        unspecified=['corner strength/extent','Gaussian boundary convention','exact weight normalization','cache implementation'],
        pruning='masked attention, not harmonic reconstruction or contrastive logits',
        prefix='offline fixed clean prefix only; future closed loop must use refined autoregressive prefix',
        candidate_k=list(KS),candidate_tau=list(TAUS),
        stage_gate='layer choice before k/tau scan; no formal400 automatic launch',
        source='https://arxiv.org/html/2601.16065v1'))
    metrics=[]; correlations=[]
    for row in rows:
        with np.load(row['source']) as a:
            r=a['original'].astype(float)
            correlations.append(spearmanr(r,axis=1).statistic)
            meta=json.loads(Path(row['source']).with_suffix('.json').read_text())
            ctrl=meta.get('target_control')
            for l in range(32):
                z=dict(id=row['id'],task=row['task'],seed=row['seed'],phase=row['phase'],layer=l,
                       visual_mass=float(r[l].sum()),corner_mass=float(r[l,[0,15,240,255]].sum()/max(r[l].sum(),1e-30)))
                for name in ('synonym','target_switch'):
                    if name in a:
                        s=a[name][l].astype(float); st=set(top(s,109)); orig=set(top(r[l],109))
                        z[name+'_jaccard']=len(st&orig)/len(st|orig)
                        if name=='target_switch' and ctrl and 'old_target' in ctrl:
                            p=r[l]/max(r[l].sum(),1e-30);q=s/max(s.sum(),1e-30)
                            old,new=ctrl['old_target'],ctrl['new_target']
                            z['target_response']=float((q[new].mean()-q[old].mean())-(p[new].mean()-p[old].mean()))
                metrics.append(z)
    keys=sorted(set().union(*(x.keys() for x in metrics)))
    table(ROOT/'layer_per_state.csv',[{k:x.get(k,'') for k in keys} for x in metrics])
    np.save(ROOT/'layer_rank_correlation.npy',np.nanmean(correlations,axis=0))
    # Rank only on controls with explicit target references; synonym response penalizes instability.
    summary=[]
    for l in range(32):
        z=[x for x in metrics if x['layer']==l and 'target_response' in x]
        taskmeans=[np.mean([x['target_response'] for x in z if x['task']==t])
                   for t in SEEDS if any(x['task']==t for x in z)]
        summary.append(dict(layer=l,controls=len(z),tasks=len(taskmeans),
            mean_target_response=float(np.mean(taskmeans)) if taskmeans else 0,
            positive_response_fraction=float(np.mean([x['target_response']>0 for x in z])) if z else 0,
            synonym_jaccard=float(np.mean([x['synonym_jaccard'] for x in metrics if x['layer']==l and 'synonym_jaccard' in x]))))
    table(ROOT/'layer_summary.csv',summary)
    ranked=sorted(summary,key=lambda x:(-x['mean_target_response'],-x['synonym_jaccard'],x['layer']))
    eligible=[x for x in ranked if x['mean_target_response']>0 and x['positive_response_fraction']>=.6 and x['synonym_jaccard']>=.5]
    chosen=[x['layer'] for x in eligible[:2]]
    save(ROOT/'LAYER_CANDIDATES.json',dict(layers=chosen,eligible=len(eligible),
        selection='positive target response; >=60% control response positive; synonym Jaccard >=.5; highest task-balanced response',
        boundary='exploratory, approximate reference boxes; close drawer may lack target controls; no success labels used'))
    print(json.dumps({'states':len(rows),'layers':chosen}),flush=True)

def extract(gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.prompt_attn_shr_rollout import build_policies
    from research.ar_token_counterfactual.intervention import ensure_empty_action_token,projector_intervention
    from research.semantic_token_cd.prompt_attn_shr_policy import extract_prompt_attention_per_layer
    rows=json.loads((ROOT/'SAMPLES.json').read_text())
    base=OpenVLAInference(**get_policy_config('openvla',str(PCD_SOURCE/'pretrained/openvla-7b'),'google_robot_open_drawer',{},False))
    p=build_policies(base,'google_robot_open_drawer')['prompt_attn_shr']
    for row in rows:
        out=ROOT/'action_attention'/row['id']
        if out.with_suffix('.json').exists():continue
        with np.load(row['source']) as a:
            image=a['image'].copy()
            expected=a['clean_positive'].astype(float) if 'clean_positive' in a else None
            expected_prompt=a['original'].astype(float)
        instruction=row['instruction'];p.reset(instruction,seed=row['seed'])
        inp=p.process_inputs(image,task_description=instruction)
        with torch.inference_mode():
            with projector_intervention(p.vla) as tr: clean=p._forward_scores(inp,p.unnorm_key,do_sample=False)
            visual=tr.before
            prompt,pmeta=extract_prompt_attention_per_layer(p,inp,instruction,visual)
            prompt_error=float(np.abs(prompt-expected_prompt).max())
            if prompt_error>2e-6:raise RuntimeError(f'cached prompt mismatch: {prompt_error}')
            ids,mask=ensure_empty_action_token(inp['input_ids'],inp['attention_mask'])
            captured=[[] for _ in range(32)]
            def hook_for(layer):
                def hook(module,args,output):
                    att=output[1]
                    if att is None:raise RuntimeError('missing generation attention')
                    captured[layer].append(att[0,:,-1,1:257].float().mean(0).detach().cpu())
                return hook
            handles=[layer.self_attn.register_forward_hook(hook_for(i))
                     for i,layer in enumerate(p.vla.language_model.model.layers)]
            with projector_intervention(p.vla) as tr:
                try: recorded=p._forward_scores(inp,p.unnorm_key,do_sample=False,output_attentions=True)
                finally:
                    for h in handles:h.remove()
            assert torch.equal(tr.before,visual)
            qs=[256+ids.shape[1]-1+q for q in range(7)]
            assert all(len(x)==7 for x in captured)
            per=torch.stack([torch.stack(x) for x in captured]).numpy()
            start=int(p.vla.vocab_size)-256
            ref=clean[:,start:start+256].float().cpu().numpy()
            actual=recorded[:,start:start+256].float().cpu().numpy()
            equality=bool(np.array_equal(ref.argmax(-1),actual.argmax(-1)))
            source_equal=bool(np.array_equal(ref.argmax(-1),expected.argmax(-1))) if expected is not None else None
            meta=dict(**row,generation_attention_clean_greedy_equal=equality,source_clean_greedy_equal=source_equal,
                generation_max_abs=float(np.abs(ref-actual).max()),source_max_abs=float(np.abs(ref-expected).max()) if expected is not None else None,
                cached_prompt_max_abs=prompt_error,
                action_query_positions=qs,prompt_audit=pmeta)
            if not equality or source_equal is False:
                save(ROOT/'AUDIT_FAILURE.json',meta);raise RuntimeError('clean action audit failed')
            pattern,w=action_pattern(per)
            out.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out.with_suffix('.npz'),per_layer_action=per,weighted_action=pattern,
                                layer_weights=w,positive=ref,prompt=prompt)
            save(out.with_suffix('.json'),meta)
            del recorded,clean,visual
        print(json.dumps({'extracted':row['id']}),flush=True)

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((ROOT/'SAMPLES.json').read_text())
    layers=json.loads((ROOT/'LAYER_CANDIDATES.json').read_text())['layers']
    if not layers:raise RuntimeError('no layer passes exploratory response screen')
    stats=[]; spatial_rows=[]
    for row in rows:
        with np.load(ROOT/'action_attention'/f"{row['id']}.npz") as a: pattern=a['weighted_action']
        with np.load(row['source']) as a:r=a['original'];image=a['image']
        for l in layers:
            for mode in ('raw','corners','gaussian','both'):
                score=spatial(r[l],mode)
                for k in KS:
                    protect=top(score,k);orig=set(top(r[l],k))
                    spatial_rows.append(dict(id=row['id'],layer=l,mode=mode,k=k,replacements=k-len(orig&set(protect))))
                    for tau in TAUS:
                        masks=prune(pattern,protect,tau)
                        for q in range(7):
                            stats.append(dict(id=row['id'],task=row['task'],seed=row['seed'],phase=row['phase'],
                                layer=l,spatial=mode,k=k,tau=tau,dimension=q,pruned=int(masks[q].sum()),
                                protected_tokens=json.dumps(protect.tolist()),pruned_tokens=json.dumps(np.flatnonzero(masks[q]).tolist())))
            fig,axes=plt.subplots(2,3,figsize=(12,7))
            for ax in axes.ravel():ax.axis('off');ax.imshow(image)
            axes[0,0].set_title(row['id'])
            for ax,s,title in [(axes[0,1],r[l],'Raw prompt'),(axes[0,2],spatial(r[l],'both'),'Corners + Gaussian')]:
                ax.imshow(s.reshape(16,16),alpha=.6,extent=(0,image.shape[1],image.shape[0],0));ax.set_title(title)
            for ax,k in zip(axes[1],KS):
                protected=top(spatial(r[l],'both'),k);mask=prune(pattern,protected,1.)
                rgba=np.zeros((256,4));rgba[protected]=[0,1,0,.4];rgba[mask.any(0)]=[1,0,0,.65]
                ax.imshow(rgba.reshape(16,16,4),extent=(0,image.shape[1],image.shape[0],0))
                ax.set_title(f'k={k}, tau=1; red=any query pruned, green=protected')
            (ROOT/'figures').mkdir(exist_ok=True)
            fig.tight_layout();fig.savefig(ROOT/'figures'/f"{row['id']}__L{l}.png",dpi=100);plt.close(fig)
    table(ROOT/'pruning_per_dimension.csv',stats);table(ROOT/'spatial_changes.csv',spatial_rows)
    summary=[]
    for l in layers:
        for k in KS:
            for tau in TAUS:
                z=[x for x in stats if x['layer']==l and x['k']==k and x['tau']==tau and x['spatial']=='both']
                counts=np.array([x['pruned'] for x in z])
                empty_states=sum(all(x['pruned']==0 for x in z if x['id']==s['id']) for s in rows)
                summary.append(dict(layer=l,k=k,tau=tau,mean_pruned=float(counts.mean()),max_pruned=int(counts.max()),
                    zero_query_fraction=float((counts==0).mean()),zero_state_fraction=empty_states/60))
    table(ROOT/'pruning_summary.csv',summary)
    save(ROOT/'OFFLINE_RESULTS.json',dict(states=60,layers=layers,candidates=summary,
         status='offline attention and pruning calibration only; masked regeneration and closed loop pending',
         author_code_reproduced=False))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    lr=list(csv.DictReader((ROOT/'layer_summary.csv').open()))
    axes[0].plot([int(x['layer']) for x in lr],[float(x['mean_target_response']) for x in lr],marker='.')
    axes[0].axhline(0,color='gray');axes[0].set(xlabel='Layer (0-based)',ylabel='Target-switch response')
    axes[1].imshow(np.load(ROOT/'layer_rank_correlation.npy'),vmin=-1,vmax=1,cmap='coolwarm')
    axes[1].set(xlabel='Layer',ylabel='Layer',title='Mean within-state rank correlation')
    fig.tight_layout();fig.savefig(ROOT/'layer_diagnostics.png',dpi=160);plt.close(fig)
    print(json.dumps(summary),flush=True)

def preflight(gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.prompt_attn_shr_rollout import build_policies
    from research.semantic_token_cd.dtp_openvla_policy import decode_dtp
    base=OpenVLAInference(**get_policy_config('openvla',str(PCD_SOURCE/'pretrained/openvla-7b'),'google_robot_open_drawer',{},False))
    p=build_policies(base,'google_robot_open_drawer')['prompt_attn_shr']
    rows=json.loads((ROOT/'SAMPLES.json').read_text());results=[]
    layers=json.loads((ROOT/'LAYER_CANDIDATES.json').read_text())['layers']
    for task in SEEDS:
        def severity(row):
            with np.load(ROOT/'action_attention'/f"{row['id']}.npz") as a:pattern=a['weighted_action'];prompt=a['prompt']
            return int(prune(pattern,top(spatial(prompt[11],'both'),64),.5).sum())
        row=max((x for x in rows if x['task']==task),key=severity)
        with np.load(row['source']) as a:image=a['image'];prompt=a['original']
        p.reset(row['instruction'],seed=row['seed']);inp=p.process_inputs(image,task_description=row['instruction'])
        with torch.inference_mode():clean=p._forward_scores(inp,p.unnorm_key,do_sample=False)
        plain=decode_dtp(p,inp,prompt,enabled=False)
        eq=plain['tokens']==clean.argmax(-1).tolist()
        start=int(p.vla.vocab_size)-256
        error=float(np.abs(plain['final_logits']-clean[:,start:start+256].float().cpu().numpy()).max())
        results.append(dict(id=row['id'],variant='no_pruning',clean_greedy_equal=eq,max_abs=error))
        if not eq:
            save(ROOT/'PREFLIGHT.json',dict(passed=False,results=results))
            raise RuntimeError('autoregressive adapter changes clean action')
        for layer in layers:
            result=decode_dtp(p,inp,prompt,layer=layer,k=64,tau=.5)
            finite=bool(np.isfinite(result['final_logits']).all())
            trace=result.pop('trace')
            out=ROOT/'preflight'/f"{row['id']}__L{layer}"
            out.parent.mkdir(exist_ok=True)
            np.savez_compressed(out.with_suffix('.npz'),**result)
            save(out.with_suffix('.json'),dict(**row,layer=layer,k=64,tau=.5,trace=trace))
            results.append(dict(id=row['id'],variant=f'L{layer}',finite=finite,
                token_flips=sum(a!=b for a,b in zip(plain['tokens'],result['tokens'])),
                pruned_counts=[len(x['pruned']) for x in trace]))
            assert finite
        print(json.dumps({'preflight_task':task}),flush=True)
    save(ROOT/'PREFLIGHT.json',dict(passed=True,states=4,results=results,
         boundary='numerical/mask audit; does not establish success rate'))

def scene_split():
    import pickle
    canonical=Path('artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots')
    result={}
    for task in SEEDS:
        formal=[];novel=[];rejected=[];all_selected=[]
        for seed in range(300):
            file=canonical/('google_robot_'+task)/f'seed_{seed:03d}.pkl'
            snap=pickle.loads(file.read_bytes());state=np.asarray(snap['sim_state'],dtype=float)
            inst=snap['instruction']
            # Compare every formal state, not only representatives: avoid leaking
            # near duplicates that lie on opposite sides of a rounding boundary.
            comparisons=formal if seed<100 else formal+all_selected
            match=next((s for s,i,a in comparisons if i==inst and a.shape==state.shape
                        and np.allclose(a,state,atol=1e-6,rtol=0)),None)
            if seed<100:formal.append((seed,inst,state))
            elif match is None:
                novel.append(seed);all_selected.append((seed,inst,state))
            else:rejected.append(dict(seed=seed,duplicate_of=match))
        result[task]=dict(eligible_seeds=novel,eligible_count=len(novel),rejected=rejected)
    save(ROOT/'CALIBRATION_SCENE_AUDIT.json',dict(tasks=result,
        rule='exclude all first100 same-instruction sim states within absolute1e-6, rtol0; dedup remainder the same way',
        boundary='conservative physical-scene filter; agent/controller and renderer metadata still require per-episode restoration audit',
        calibration_ready=all(x['eligible_count']>=5 for x in result.values())))
    print(json.dumps({t:x['eligible_count'] for t,x in result.items()}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract','report','preflight','scene_split']);ap.add_argument('--gpu',type=int,default=2)
    args=ap.parse_args()
    {'prepare':prepare,'extract':lambda:extract(args.gpu),'report':report,'preflight':lambda:preflight(args.gpu),'scene_split':scene_split}[args.mode]()
