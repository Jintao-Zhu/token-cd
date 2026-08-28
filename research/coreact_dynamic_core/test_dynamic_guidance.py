import torch
from research.coreact_dynamic_core.dynamic_guidance import temporal_instability,_candidate_rank_score,common_min_scales

def test_temporal_instability_is_zero_for_constant_distribution():
 p=torch.tensor([.2,.3,.5]);out=temporal_instability([p,p,p]);assert torch.equal(out,torch.zeros_like(p))

def test_temporal_instability_identifies_changing_token_support():
 a=torch.tensor([.8,.1,.1]);b=torch.tensor([.1,.8,.1]);out=temporal_instability([a,b],beta=.9);assert out[0]>out[2] and out[1]>out[2]

def test_rank_score_assigns_larger_value_to_larger_input():
 out=_candidate_rank_score(torch.tensor([.2,.9,.1]),8,[1,3,6]);assert out[3]>out[1]>out[6] and torch.count_nonzero(out)==3

def test_common_min_scales_are_interpolation_only_and_equalize_norms():
 norms={'random':2.5,'attention':6.,'instability':4.};common,scales=common_min_scales(norms,3.)
 assert common==2.5 and all(0<=x<=1 for x in scales.values())
 assert all(abs(scales[name]*norms[name]-common)<1e-9 for name in norms)
