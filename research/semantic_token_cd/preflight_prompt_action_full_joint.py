"""One-state fail-closed audit for PA-Full-Joint-Top2K."""
from __future__ import annotations
import json, os
import numpy as np
from research.semantic_token_cd.prompt_action_complement_protocol import STATE_SOURCE
from research.semantic_token_cd.prompt_action_full_joint_protocol import ARTIFACT,PCD_SOURCE,atomic_json
from research.semantic_token_cd.prompt_action_full_joint_rollout import build_policy
from research.semantic_token_cd.prompt_action_rerank_rollout import build_policy as build_old
from research.semantic_token_cd.prompt_attn_shr_policy import construct_prompt_action_full_joint

TASK="google_robot_open_drawer"
def run(p,image,instruction):
    p._episode_trace=[]; p._episode_logits=[]; p._episode_seed=0; p._selector_step=0; p.reset(instruction,seed=0)
    p.step(image,None,instruction,proprio=np.zeros(8,dtype=np.float64)); return p._episode_trace[-1],p._episode_logits[-1]
def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=os.environ.get("PA_FULL_JOINT_PREFLIGHT_GPU","2"); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    image_path=STATE_SOURCE/"runs/emitted_states/google_robot_open_drawer/seed_000/step_0013.npz"
    if not image_path.exists(): image_path=sorted((STATE_SOURCE/"runs/emitted_states/google_robot_open_drawer").glob("seed_*/*.npz"))[0]
    meta=STATE_SOURCE/"runs/offline/google_robot_open_drawer"/image_path.parent.name/image_path.name
    instruction=json.loads(meta.with_suffix(".json").read_text())["instruction"]; image=np.asarray(np.load(image_path)["image"],dtype=np.uint8)
    base=OpenVLAInference(**get_policy_config("openvla",str(PCD_SOURCE/"pretrained/openvla-7b"),TASK,{},False))
    baseline=build_old(base,TASK); baseline.prompt_action_rerank_multiplier=None
    bt,br=run(baseline,image,instruction); jt,jr=run(build_policy(base,TASK),image,instruction); m=bt["m_t"]
    emulated=construct_prompt_action_full_joint(jr["prompt_attention"],jr["prompt_attention"],m)
    checks={"same_m":jt["m_t"]==m,"exact_k":len(set(jt["selected_token_ids"]))==m,
            "inside_top2k":jt["full_joint_subset_of_candidate_pool"] is True,
            "candidate_size":jt["candidate_pool_size"]==min(256,2*m),
            "clean_logits_exact":np.array_equal(br["positive"],jr["positive"]),
            "clean_action_exact":bt["clean_action"]==jt["clean_action"],
            "equal_scores_recover_l11":emulated["selected"]==bt["selected_token_ids"],
            "locked_layers":jt["attention_layers"]==[11] and jt["action_attention_layers"]==list(range(16,32)),
            "locked_downstream":jt["beta"]==0.0 and jt["lambda"]==0.5 and jt["guided_prefix"] is True and jt["non_target_bit_identical"] is True}
    checks["technical_pass"]=all(checks.values()); report={"task":TASK,"instruction":instruction,"m":m,"candidate_pool":jt["candidate_pool_size"],"overlap":jt["full_joint_overlap_ratio"],"checks":checks}
    atomic_json(ARTIFACT/"preflight/report.json",report)
    if not checks["technical_pass"]: raise RuntimeError(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
