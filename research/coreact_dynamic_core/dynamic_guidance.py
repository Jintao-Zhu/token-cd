"""Dynamic visual-token selectors with the locked mean-replacement toward operator."""
from __future__ import annotations
from dataclasses import asdict
from typing import Literal,Sequence
import torch
from research.coreact_closed_loop.guidance import _full_velocity,_prefix_cache,replace_visual_tokens,select_visual_tokens,tensor_sha256
from research.coreact_exploration.instrumentation import attention_ranking_scores,build_prefix_span_map,validate_intervention_group

Selector=Literal['random','attention','instability','combined']
COMMON_SELECTORS=('random','attention','instability')

def _capture_denoise_attention(model,pad,cache,x,timestep):
 traces=[];owner=model.vlm_with_expert;previous=getattr(owner,'attention_trace_callback',None);owner.attention_trace_callback=traces.append
 try:v=model.denoise_step(pad,cache,x,timestep)
 finally:owner.attention_trace_callback=previous
 return v,traces

def _visual_distribution(score,candidates):
 x=score.detach().float().cpu()[candidates].clamp_min(0);return x/(x.sum()+1e-12)

def temporal_instability(distributions,beta=.9):
 if len(distributions)<2:raise ValueError('at least two flow steps required')
 ema=torch.zeros_like(distributions[0]);eps=1e-12
 for a,b in zip(distributions[:-1],distributions[1:]):
  m=.5*(a+b);js=.5*(a*(a.add(eps).log()-m.add(eps).log())+b*(b.add(eps).log()-m.add(eps).log()));ema=beta*ema+(1-beta)*js
 return ema

def _candidate_rank_score(values,prefix_len,candidates):
 order=torch.argsort(values,stable=True);ranks=torch.empty_like(values);ranks[order]=torch.arange(1,len(values)+1,dtype=values.dtype);out=torch.zeros(prefix_len);out[candidates]=ranks/len(values);return out

def common_min_scales(norms,cap,eps=1e-12):
 common=min(float(cap),*(float(norms[name]) for name in COMMON_SELECTORS))
 return common,{name:common/(float(norms[name])+eps) for name in COMMON_SELECTORS}

@torch.no_grad()
def sample_common_min_actions(model,images:Sequence[torch.Tensor],image_masks:Sequence[torch.Tensor],lang_tokens,lang_masks,state,noise,visual_position_mean,*,selection_seed:int,magnitude_cap:float,camera_ids=('camera1','camera2'),group_count=8,trust_region_kappa=.25,action_dim=7,ema_beta=.9):
 if model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError('frozen eval model required')
 prefix,pad,att=model.embed_prefix(images,image_masks,lang_tokens,lang_masks,state=state);span=build_prefix_span_map(model,images,image_masks,lang_tokens,lang_masks,pad,camera_ids=camera_ids)[0];candidates=[x.index for x in span if x.modality=='visual' and x.intervention_allowed]
 clean_cache=_prefix_cache(model,prefix,pad,att);dt=-1/model.config.num_steps;x_clean=noise.clone();flow_scores=[];flow_layer_counts=[]
 for step in range(model.config.num_steps):
  tau=torch.full((1,),1+step*dt,dtype=torch.float32,device=state.device);v,traces=_capture_denoise_attention(model,pad,clean_cache,x_clean,tau);flow_scores.append(attention_ranking_scores(traces,prefix.shape[1])['late_half_action_to_context_attention']);flow_layer_counts.append(len(traces));x_clean=x_clean+dt*v
 distributions=[_visual_distribution(s,candidates) for s in flow_scores];instability_values=temporal_instability(distributions,beta=ema_beta);instability_score=torch.zeros(prefix.shape[1]);instability_score[candidates]=instability_values
 _,native_traces=_full_velocity(model,prefix,pad,att,noise,torch.ones(1,device=state.device),record_attention=True);attention_score=attention_ranking_scores(native_traces,prefix.shape[1])['late_half_action_to_context_attention'].cpu()
 selected={'random':select_visual_tokens(span,attention_score,method='random',count=group_count,random_seed=selection_seed),'attention':select_visual_tokens(span,attention_score,method='top',count=group_count,random_seed=selection_seed),'instability':select_visual_tokens(span,instability_score,method='top',count=group_count,random_seed=selection_seed)}
 negative_caches={};negative_prefix_hashes={};changed={}
 for name in COMMON_SELECTORS:
  masked=replace_visual_tokens(prefix,span,selected[name],visual_position_mean,camera_ids);changed[name]=(prefix!=masked).any(dim=-1).nonzero(as_tuple=False)[:,1].tolist()
  if sorted(changed[name])!=sorted(selected[name]):raise RuntimeError(f'{name} changed indices mismatch')
  validate_intervention_group(span,selected[name]);negative_caches[name]=_prefix_cache(model,masked,pad,att);negative_prefix_hashes[name]=tensor_sha256(masked)
 xs={name:noise.clone() for name in COMMON_SELECTORS};steps={name:[] for name in COMMON_SELECTORS}
 for step in range(model.config.num_steps):
  tau=torch.full((1,),1+step*dt,dtype=torch.float32,device=state.device);clean={};base={};norms={};raw_norms={};clip_scales={}
  for name in COMMON_SELECTORS:
   clean[name]=model.denoise_step(pad,clean_cache,xs[name],tau);negative=model.denoise_step(pad,negative_caches[name],xs[name],tau);raw=clean[name]-negative;raw[...,action_dim:]=0;clean_norm=torch.linalg.vector_norm(clean[name][...,:action_dim]);raw_norm=torch.linalg.vector_norm(raw[...,:action_dim]);clip_scale=float(torch.clamp(trust_region_kappa*clean_norm/(raw_norm+1e-12),max=1.));base[name]=-raw*clip_scale;norms[name]=torch.linalg.vector_norm(base[name][...,:action_dim]);raw_norms[name]=raw_norm;clip_scales[name]=clip_scale
  common,alphas=common_min_scales(norms,magnitude_cap)
  for name in COMMON_SELECTORS:
   applied=base[name]*alphas[name];guided=clean[name]+applied
   if not torch.isfinite(guided).all():raise RuntimeError(f'nonfinite {name} guided velocity')
   xs[name]=xs[name]+dt*guided;steps[name].append({'step':step,'raw_delta_norm':float(raw_norms[name]),'postclip_base_norm':float(norms[name]),'common_norm':common,'common_over_attention':common/(float(norms['attention'])+1e-12),'applied_guidance_norm':float(torch.linalg.vector_norm(applied[...,:action_dim])),'alpha':alphas[name],'clip_scale':clip_scales[name],'clipped':bool(clip_scales[name]<1),'extrapolated':bool(alphas[name]>1+1e-8),'clean_velocity_sha256':tensor_sha256(clean[name])})
 traces={name:{'selector':name,'selected_indices':selected[name],'changed_indices':changed[name],'attention_top_indices':selected['attention'],'instability_top_indices':selected['instability'],'eligible_visual_count':len(candidates),'ema_beta':ema_beta,'native_attention_layer_count':len(native_traces),'flow_attention_layer_counts':flow_layer_counts,'clean_flow_final_sha256':tensor_sha256(x_clean),'prefix_sha256':tensor_sha256(prefix),'negative_prefix_sha256':negative_prefix_hashes[name],'selected_tokens':[asdict(span[i]) for i in selected[name]],'step_traces':steps[name],'all_output_finite':bool(torch.isfinite(xs[name]).all()),'protected_tokens_untouched':all(span[i].modality=='visual' and span[i].intervention_allowed for i in changed[name])} for name in COMMON_SELECTORS}
 return xs,traces

@torch.no_grad()
def sample_dynamic_core_actions(model,images:Sequence[torch.Tensor],image_masks:Sequence[torch.Tensor],lang_tokens,lang_masks,state,noise,visual_position_mean,*,selector:Selector,selection_seed:int,camera_ids=('camera1','camera2'),group_count=8,guidance_scale=.5,trust_region_kappa=.25,action_dim=7,ema_beta=.9,norm_target_rms=None):
 if model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError('frozen eval model required')
 if selector not in ('random','attention','instability','combined'):raise ValueError(selector)
 prefix,pad,att=model.embed_prefix(images,image_masks,lang_tokens,lang_masks,state=state);span=build_prefix_span_map(model,images,image_masks,lang_tokens,lang_masks,pad,camera_ids=camera_ids)[0];candidates=[x.index for x in span if x.modality=='visual' and x.intervention_allowed]
 clean_cache=_prefix_cache(model,prefix,pad,att);dt=-1/model.config.num_steps;x_clean=noise.clone();flow_scores=[];flow_layer_counts=[]
 for step in range(model.config.num_steps):
  tau=torch.full((1,),1+step*dt,dtype=torch.float32,device=state.device);v,traces=_capture_denoise_attention(model,pad,clean_cache,x_clean,tau);score=attention_ranking_scores(traces,prefix.shape[1])['late_half_action_to_context_attention'];flow_scores.append(score);flow_layer_counts.append(len(traces));x_clean=x_clean+dt*v
 distributions=[_visual_distribution(s,candidates) for s in flow_scores];instability_values=temporal_instability(distributions,beta=ema_beta);instability_score=torch.zeros(prefix.shape[1]);instability_score[candidates]=instability_values
 # Preserve the published Attn8 baseline exactly: full native prefix at tau=1.
 _,native_traces=_full_velocity(model,prefix,pad,att,noise,torch.ones(1,device=state.device),record_attention=True);attention_score=attention_ranking_scores(native_traces,prefix.shape[1])['late_half_action_to_context_attention'].cpu()
 combined_score=_candidate_rank_score(attention_score[candidates],prefix.shape[1],candidates)+_candidate_rank_score(instability_values,prefix.shape[1],candidates)
 scores={'attention':attention_score,'instability':instability_score,'combined':combined_score}
 if selector=='random':selected=select_visual_tokens(span,attention_score,method='random',count=group_count,random_seed=selection_seed)
 else:selected=select_visual_tokens(span,scores[selector],method='top',count=group_count,random_seed=selection_seed)
 attention_top=select_visual_tokens(span,attention_score,method='top',count=group_count,random_seed=selection_seed);instability_top=select_visual_tokens(span,instability_score,method='top',count=group_count,random_seed=selection_seed);combined_top=select_visual_tokens(span,combined_score,method='top',count=group_count,random_seed=selection_seed)
 masked=replace_visual_tokens(prefix,span,selected,visual_position_mean,camera_ids);changed=(prefix!=masked).any(dim=-1).nonzero(as_tuple=False)[:,1].tolist()
 if sorted(changed)!=sorted(selected):raise RuntimeError('changed indices mismatch')
 negative_cache=_prefix_cache(model,masked,pad,att);x=noise.clone();steps=[]
 for step in range(model.config.num_steps):
  tau=torch.full((1,),1+step*dt,dtype=torch.float32,device=state.device);clean=model.denoise_step(pad,clean_cache,x,tau);negative=model.denoise_step(pad,negative_cache,x,tau);raw=clean-negative;raw[...,action_dim:]=0;clean_norm=torch.linalg.vector_norm(clean[...,:action_dim]);raw_norm=torch.linalg.vector_norm(raw[...,:action_dim]);clip_scale=float(torch.clamp(trust_region_kappa*clean_norm/(raw_norm+1e-12),max=1.));base=-raw*clip_scale;base_norm=torch.linalg.vector_norm(base[...,:action_dim]);
  if norm_target_rms is None: alpha=float(guidance_scale);applied=base*guidance_scale
  else: alpha=min(1.,float(norm_target_rms)/(float(base_norm)+1e-12));applied=base*alpha
  scale=clip_scale;guided=clean+applied
  if not torch.isfinite(guided).all():raise RuntimeError('nonfinite guided velocity')
  x=x+dt*guided;steps.append({'step':step,'raw_delta_norm':float(raw_norm),'postclip_base_norm':float(base_norm),'applied_guidance_norm':float(torch.linalg.vector_norm(applied[...,:action_dim])),'clip_scale':float(scale),'alpha':alpha,'under_matched':bool(norm_target_rms is not None and alpha>=1.),'clipped':bool(scale<1),'clean_velocity_sha256':tensor_sha256(clean)})
 validate_intervention_group(span,selected)
 trace={'selector':selector,'selected_indices':selected,'changed_indices':changed,'attention_top_indices':attention_top,'instability_top_indices':instability_top,'combined_top_indices':combined_top,'attn_instability_overlap':len(set(attention_top)&set(instability_top)),'attn_combined_overlap':len(set(attention_top)&set(combined_top)),'instability_combined_overlap':len(set(instability_top)&set(combined_top)),'eligible_visual_count':len(candidates),'ema_beta':ema_beta,'native_attention_layer_count':len(native_traces),'flow_attention_layer_counts':flow_layer_counts,'clean_flow_final_sha256':tensor_sha256(x_clean),'prefix_sha256':tensor_sha256(prefix),'negative_prefix_sha256':tensor_sha256(masked),'selected_tokens':[asdict(span[i]) for i in selected],'step_traces':steps,'all_output_finite':bool(torch.isfinite(x).all()),'protected_tokens_untouched':all(span[i].modality=='visual' and span[i].intervention_allowed for i in changed)}
 return x,trace
