from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
from pathlib import Path
import h5py, numpy as np, torch
WS=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(WS/'LIBERO'),str(WS/'third_party/openvla'),str(WS)]
from research.ar_token_counterfactual.intervention import action_logit_metrics, masked_action_token_ids, teacher_forced_forward, tensor_sha256
from research.ar_token_counterfactual.libero_runtime import build_prompt, load_policy, prepare_agentview, set_determinism
from libero.libero import benchmark

def write_json(p,x):
 t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); t.replace(p)
def expert_ids(model, action):
 bins=np.linspace(-1,1,256); disc=np.digitize(np.clip(action,-1,1),bins); return torch.tensor([[int(model.vocab_size-int(x)) for x in disc]],device=model.device,dtype=torch.long)
def logprob(logits, ids):
 lp=torch.log_softmax(logits,dim=-1); return float(lp[0,torch.arange(ids.shape[1]),ids[0].to('cpu')].sum())
def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--task',type=int,default=None); p.add_argument('--max-states',type=int); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shard-count',type=int,default=1); a=p.parse_args(); w=a.workspace.resolve(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True); (out/'states').mkdir(exist_ok=True); os.environ.setdefault('HF_HOME',str(w/'task1/.hf-cache')); os.environ.setdefault('TRANSFORMERS_CACHE',str(w/'task1/.hf-cache/hub'))
 tasks=[a.task] if a.task is not None else list(range(10)); suite=benchmark.get_benchmark_dict()['libero_spatial'](); mean=torch.load(w/'artifacts/ar_token_counterfactual_qualification_v1_20260808_231044/position_conditioned_visual_mean.pt',map_location='cpu',weights_only=True)['mean']; ck=w/'checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873'; model,proc=load_policy(ck,w/'third_party/openvla/prismatic/extern/hf'); set_determinism(7)
 plans=[]
 for task in tasks:
  lang=suite.get_task(task).language.lower().replace(' ','_').replace('.','').replace(',',''); files=glob.glob(str(w/'LIBERO/libero/datasets/libero_spatial'/f'{lang}_demo.hdf5'))
  if len(files)!=1: raise RuntimeError(f'demo file not found for task {task}: {lang}')
  with h5py.File(files[0],'r') as h:
   for demo in sorted(h['data'].keys(),key=lambda x:int(x.split('_')[-1]))[:50]:
    n=len(h['data'][demo]['actions']); plans.append((task,Path(files[0]),demo,max(0,min(n-1,n//2))))
 if a.max_states is not None: plans=plans[:a.max_states]
 plans=[x for i,x in enumerate(plans) if i%a.shard_count==a.shard_index]
 for task,path,demo,frame in plans:
  sid=f'task{task:02d}__{demo}__f{frame:04d}'; dest=out/'states'/f'{sid}.json'
  if dest.exists():continue
  with h5py.File(path,'r') as h: image=np.asarray(h['data'][demo]['obs']['agentview_rgb'][frame]); action=np.asarray(h['data'][demo]['actions'][frame],dtype=np.float32)
  _, pil=prepare_agentview({'agentview_image':image}); inputs=proc(build_prompt(suite.get_task(task).language),pil).to(model.device,dtype=torch.bfloat16); ids=expert_ids(model,action); clean=teacher_forced_forward(model,inputs,ids,record_attention=True); scores=clean.attention_scores.numpy(); top=np.argsort(-scores,kind='stable')[:16].tolist(); bottom=np.argsort(scores,kind='stable')[:8].tolist(); remain=[i for i in range(256) if i not in set(top)|set(bottom)]; rng=np.random.default_rng(20260811+task*100+int(demo.split('_')[-1])); random=rng.choice(remain,8,replace=False).tolist(); candidates=top+bottom+random; clean_lp=logprob(clean.logits,ids); clean_generated,_=masked_action_token_ids(model,inputs,[],mean); expert_action=action
  results=[]
  for token in candidates:
   masked=teacher_forced_forward(model,inputs,ids,[token],mean); cf_lp=logprob(masked.logits,ids); masked_ids,_=masked_action_token_ids(model,inputs,[token],mean); clean_action_ids,_=masked_action_token_ids(model,inputs,[],mean); clean_dec=__import__('research.ar_token_counterfactual.intervention',fromlist=['decode_action_ids']).decode_action_ids(model,clean_action_ids); masked_dec=__import__('research.ar_token_counterfactual.intervention',fromlist=['decode_action_ids']).decode_action_ids(model,masked_ids); results.append({'token_index':int(token),'category':'attention_top16' if token in top else ('attention_bottom8' if token in bottom else 'random8'),'U_logprob':clean_lp-cf_lp,'clean_logprob':clean_lp,'cf_logprob':cf_lp,'teacher_forced_action_l2':float(np.linalg.norm(clean_dec-expert_action)-np.linalg.norm(masked_dec-expert_action)),'free_running_action_l2':float(np.linalg.norm(masked_dec-expert_action)),'free_running_clean_l2':float(np.linalg.norm(clean_dec-expert_action)),'masked_action_token_ids':masked_ids[0].cpu().tolist()})
  write_json(dest,{'snapshot_id':sid,'task_id':task,'demo':demo,'frame':frame,'expert_action':action.tolist(),'expert_action_token_ids':ids[0].cpu().tolist(),'selected':{'attention_top16':top,'attention_bottom8':bottom,'random8':random},'clean_logprob':clean_lp,'clean_free_running_action_l2':float(np.linalg.norm(clean_dec-expert_action)),'candidates':results,'finite':True})
  print(json.dumps({'task':task,'snapshot_id':sid,'complete':True}),flush=True)
if __name__=='__main__':main()
