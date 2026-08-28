#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii")); digest.update(str(tuple(value.shape)).encode("ascii")); digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def velocity(model, images, masks, tokens, token_masks, state, x_t, timestep):
    prefix, prefix_pad, prefix_att = model.embed_prefix(images, masks, tokens, token_masks, state=state)
    suffix, suffix_pad, suffix_att = model.embed_suffix(x_t, timestep)
    pads, atts = torch.cat([prefix_pad, suffix_pad], 1), torch.cat([prefix_att, suffix_att], 1)
    attention = make_att_2d_masks(pads, atts)
    positions = torch.cumsum(pads, dim=1) - 1
    (_, output), _ = model.vlm_with_expert.forward(attention_mask=attention, position_ids=positions, past_key_values=None, inputs_embeds=[prefix, suffix], use_cache=False, fill_kv_cache=False)
    return model.action_out_proj(output[:, -model.config.chunk_size :].float())


def inputs(model, device):
    generator = torch.Generator(device=device).manual_seed(20260816)
    images = [torch.randn((1, 3, 512, 512), generator=generator, device=device) * 0.1 for _ in range(2)]
    masks = [torch.ones((1,), dtype=torch.bool, device=device) for _ in images]
    encoded = model.vlm_with_expert.processor.tokenizer("pick up the object\n", return_tensors="pt", padding="max_length", max_length=48, truncation=True)
    tokens, token_masks = encoded["input_ids"].to(device), encoded["attention_mask"].bool().to(device)
    state = torch.randn((1, 32), generator=generator, device=device)
    x_t = torch.randn((1, 50, 32), generator=generator, device=device)
    timestep = torch.tensor([0.6], device=device)
    return images, masks, tokens, token_masks, state, x_t, timestep


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True); parser.add_argument("--phase", choices=("before", "after"), required=True); args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    checkpoint = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model"
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True); config.device="cuda"; config.compile_model=False
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, local_files_only=True).eval()
    values = inputs(policy.model, torch.device("cuda")); output = velocity(policy.model, *values)
    payload = {"phase": args.phase, "checkpoint": str(checkpoint), "input_hashes": [tensor_hash(item) for group in values for item in (group if isinstance(group, list) else [group])], "output_hash": tensor_hash(output), "output_shape": list(output.shape), "finite": bool(torch.isfinite(output).all())}
    target = artifact / "implementation_gate"; target.mkdir(exist_ok=True)
    torch.save(output.detach().cpu(), target / f"16l_{args.phase}_output.pt")
    if args.phase == "after":
        before = torch.load(target / "16l_before_output.pt", weights_only=True)
        payload["max_abs_vs_before"] = float((before - output.cpu()).abs().max())
        payload["regression_pass"] = payload["max_abs_vs_before"] < 1e-6
        if not payload["regression_pass"]: raise RuntimeError(payload)
    (target / f"16l_{args.phase}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__": main()
