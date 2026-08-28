from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from research.coreact_closed_loop.guidance import _full_velocity,_prefix_cache,replace_visual_tokens,select_visual_tokens,tensor_sha256
from research.coreact_exploration.instrumentation import attention_ranking_scores,build_prefix_span_map
from research.coreact_region.segmented_runtime import batched_observation

TAUS=tuple(round(1.-.1*i,1) for i in range(10));CAMERAS=('camera1','camera2')
def array_sha(value):
 value=np.ascontiguousarray(value);h=hashlib.sha256();h.update(str(value.dtype).encode());h.update(json.dumps(list(value.shape)).encode());h.update(value.tobytes());return h.hexdigest()
def action_chunk(actions,index,size=50):
 valid=min(size,len(actions)-index)
 if valid<=0:raise ValueError('state has no demonstration action')
 values=list(actions[index:index+valid]);values.extend([values[-1]]*(size-valid));return torch.tensor(np.asarray(values),dtype=torch.float32),torch.tensor([False]*valid+[True]*(size-valid),dtype=torch.bool)
def rewrite_demo_xml(xml,workspace):return xml.replace('/Users/yifengz/workspace/libero-dev/chiliocosm/assets',str(workspace/'LIBERO/libero/libero/assets'))
def prepare_demo_state(policy,preprocessor,env_preprocessor,env,raw,language,actions,frame):
 observation=env._format_raw_obs(raw);chunk,pad=action_chunk(actions,frame);batch=preprocess_observation(batched_observation(observation));batch['task']=[language];batch['action']=chunk.unsqueeze(0);batch['action_is_pad']=pad.unsqueeze(0);batch=preprocessor(env_preprocessor(batch));images,image_masks=policy.prepare_images(batch);return {'images':images,'image_masks':image_masks,'state':policy.prepare_state(batch),'actions':policy.prepare_action(batch),'action_is_pad':batch['action_is_pad'],'lang_tokens':batch['observation.language.tokens'],'lang_masks':batch['observation.language.attention_mask'],'preprocessing_sha256':{f'image{i}':tensor_sha256(x) for i,x in enumerate(images)}|{'state':tensor_sha256(policy.prepare_state(batch)),'action':tensor_sha256(policy.prepare_action(batch)),'action_is_pad':tensor_sha256(batch['action_is_pad']),'lang_tokens':tensor_sha256(batch['observation.language.tokens']),'lang_masks':tensor_sha256(batch['observation.language.attention_mask'])}}
def vector_metrics(clean,counterfactual,target,valid,guidance_scale=.5,kappa=.25):
 clean=clean[...,:valid.shape[-1]];counterfactual=counterfactual[...,:valid.shape[-1]];target=target[...,:valid.shape[-1]];toward=counterfactual-clean;expert=target-clean;flat=valid
 a=toward[flat];b=expert[flat];cos=float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)+1e-12));raw=clean-counterfactual;scale=torch.clamp(kappa*torch.linalg.vector_norm(clean)/(torch.linalg.vector_norm(raw)+1e-12),max=1.);applied=guidance_scale*toward*scale;guided_toward=clean+applied;guided_away=clean-applied;clean_loss=torch.mean((clean[flat]-target[flat])**2);toward_loss=torch.mean((guided_toward[flat]-target[flat])**2);away_loss=torch.mean((guided_away[flat]-target[flat])**2)
 return {'cos_toward_expert':cos,'clean_mse':float(clean_loss),'toward_mse':float(toward_loss),'away_mse':float(away_loss),'delta_toward_mse':float(toward_loss-clean_loss),'delta_away_mse':float(away_loss-clean_loss),'raw_correction_norm':float(torch.linalg.vector_norm(toward)),'applied_correction_norm':float(torch.linalg.vector_norm(applied)),'trust_clip_scale':float(scale),'toward_positive_cos':cos>0,'toward_improves_mse':bool(toward_loss<clean_loss),'away_improves_mse':bool(away_loss<clean_loss)}
@torch.no_grad()
def evaluate_unit(model,batch,means,noise_seed,random_seed):
 actions=batch['actions'];g=torch.Generator(device=actions.device).manual_seed(noise_seed);noise=torch.randn(actions.shape,generator=g,device=actions.device,dtype=actions.dtype);prefix,pad,att=model.embed_prefix(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],state=batch['state']);span=build_prefix_span_map(model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],pad,camera_ids=CAMERAS)[0]
 native,traces=_full_velocity(model,prefix,pad,att,noise,torch.ones(1,device=actions.device),record_attention=True);score=attention_ranking_scores(traces,prefix.shape[1])['late_half_action_to_context_attention'];selected={'attention':select_visual_tokens(span,score,method='top',count=8,random_seed=random_seed),'random':select_visual_tokens(span,score,method='random',count=8,random_seed=random_seed)}
 replacements={}
 for token in span:
  if token.modality=='visual' and token.intervention_allowed:replacements[token.index]=means['visual_position_mean'][CAMERAS.index(token.camera_id),token.visual_token_index]
 masked={name:replace_visual_tokens(prefix,span,indices,means['visual_position_mean'],CAMERAS) for name,indices in selected.items()};prefix3=torch.cat([prefix,masked['attention'],masked['random']],dim=0);pad3=pad.expand(3,-1);att3=att.expand(3,-1);cache=_prefix_cache(model,prefix3,pad3,att3);valid=(~batch['action_is_pad']).unsqueeze(-1).expand(-1,-1,actions.shape[-1]);target=noise-actions;per_tau=[];cached_tau1=None
 for tau_value in TAUS:
  tau=torch.tensor([tau_value],device=actions.device,dtype=actions.dtype);x=tau[:,None,None]*noise+(1-tau[:,None,None])*actions;x3=x.expand(3,-1,-1);velocity=model.denoise_step(pad3,cache,x3,tau.expand(3));clean=velocity[:1];cached_tau1=clean if tau_value==1. else cached_tau1;per_tau.append({'tau':tau_value,'attention':vector_metrics(clean,velocity[1:2],target,valid),'random':vector_metrics(clean,velocity[2:3],target,valid)})
 changed={name:(prefix!=value).any(dim=-1).nonzero(as_tuple=False)[:,1].tolist() for name,value in masked.items()}
 return {'noise_seed':noise_seed,'noise_sha256':tensor_sha256(noise),'selected_indices':selected,'changed_indices':changed,'prefix_sha256':tensor_sha256(prefix),'masked_prefix_sha256':{name:tensor_sha256(value) for name,value in masked.items()},'eligible_visual_count':sum(x.modality=='visual' and x.intervention_allowed for x in span),'native_attention_layers':len(traces),'cached_full_tau1_max_abs':float((cached_tau1-native).abs().max()),'per_tau':per_tau,'finite':all(np.isfinite(v) for row in per_tau for name in ('attention','random') for v in row[name].values() if isinstance(v,(int,float)))}
