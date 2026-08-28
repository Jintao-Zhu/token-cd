import torch
from research.coreact_expert_direction.direction_audit import action_chunk,vector_metrics
def test_action_chunk_repeats_endpoint_and_marks_padding():
 x=torch.arange(21,dtype=torch.float32).reshape(3,7).numpy();a,p=action_chunk(x,1,4);assert a.shape==(4,7) and p.tolist()==[False,False,True,True] and torch.equal(a[-1],a[1])
def test_vector_metrics_detects_toward_and_away():
 clean=torch.full((1,2,1),.5);target=torch.ones_like(clean);valid=torch.ones_like(clean,dtype=torch.bool);good=vector_metrics(clean,torch.ones_like(clean),target,valid);bad=vector_metrics(clean,torch.zeros_like(clean),target,valid);assert good['cos_toward_expert']>0 and good['delta_toward_mse']<0;assert bad['cos_toward_expert']<0 and bad['delta_away_mse']<0
