"""Where does a vanilla OpenVLA step actually spend its time?"""
from __future__ import annotations
import statistics, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
CANON = REPO_ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots"
for p in (str(REPO_ROOT), str(PCD_SOURCE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def t(fn, n=20):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); a = time.perf_counter()
        r = fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - a) * 1000)
    return statistics.median(ts), r


def main():
    from research.semantic_token_cd.vla_pruner_formal_rollout import load_base_policy
    from parallel_inference import get_image_from_maniskill2_obs_dict
    sys.argv = [sys.argv[0]]
    pol = load_base_policy("google_robot_pick_coke_can")
    vla = pol.vla
    img = np.zeros((512, 640, 3), dtype=np.uint8)
    pil = Image.fromarray(img)
    inputs = pol.processor("pick coke can", pil).to("cuda:0", dtype=torch.bfloat16)
    input_ids, pixel_values = inputs["input_ids"], inputs["pixel_values"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)

    print("input_ids", tuple(input_ids.shape), "pixel_values", tuple(pixel_values.shape))
    prep, _ = t(lambda: pol._resize_image(img) if hasattr(pol, "_resize_image") else img, 20)
    print(f"preprocess/resize        : {prep:7.2f} ms")
    proc, ins = t(lambda: pol.processor("pick coke can", pil).to("cuda:0", dtype=torch.bfloat16), 20)
    print(f"processor (PIL->tensor)  : {proc:7.2f} ms")
    vis, feats = t(lambda: vla.vision_backbone(pixel_values), 20)
    print(f"vision backbone (SigLIP) : {vis:7.2f} ms   out={tuple(feats.shape)}")
    proj, emb = t(lambda: vla.projector(feats), 20)
    print(f"projector                : {proj:7.2f} ms   out={tuple(emb.shape)}")
    full, _ = t(lambda: vla.generate(input_ids=input_ids, pixel_values=pixel_values,
                                    max_new_tokens=7, output_scores=True,
                                    return_dict_in_generate=True), 20)
    print(f"full generate (7 tokens) : {full:7.2f} ms")
    print(f"  = vision+proj+LLM      : {vis+proj:.2f} + {full-vis-proj:.2f} ms")
    print(f"  LLM share              : {100*(full-vis-proj)/full:.0f}%   vision share: {100*(vis+proj)/full:.0f}%")


if __name__ == "__main__":
    main()
