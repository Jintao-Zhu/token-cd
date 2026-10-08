"""Offline prompt-to-image attention layer scan for standard Bridge-Beta Pi0.

Uses six fixed WidowX carrot observations with original, synonym, and
target-switched prompts. No simulator is launched. The scan is candidate
selection only; it does not claim closed-loop optimality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
OPENPI = ROOT / "open-pi-zero"
CHECKPOINT = ROOT / "checkpoints/pi0/bridge_beta_step19296_2024-12-26_22-30_42.pt"
SOURCE = REPO / "artifacts/spatialvla_layer_selection_v1b/states"
OUT = REPO / "artifacts/pi0_bridge_carrot_layer_selection_v1"
TASK = "widowx_carrot_on_plate"
SEEDS = (100, 110, 120, 130, 150, 160)
N_LAYERS = 18
K = 32
SYNONYM = "place the carrot onto the plate"


def find_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("empty prompt token sequence")
    for i in range(len(values) - len(needle) + 1):
        if values[i:i + len(needle)] == needle:
            return i
    raise RuntimeError("could not locate prompt tokens in prepared PiZero input")


def patch_mask(segmentation: np.ndarray, actor_id: int) -> np.ndarray:
    resized = cv2.resize((segmentation == actor_id).astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
    return resized.reshape(16, 14, 16, 14).any(axis=(1, 3)).reshape(256).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--limit", type=int, default=0, help="debug state limit; 0 scans all 6 states")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path[:0] = [str(OPENPI), str(REPO / "task1/shim_site"), str(REPO)]
    os.chdir(OPENPI)

    import torch
    import hydra
    from omegaconf import OmegaConf
    from src.model.vla.pizero import PiZeroInference
    from src.model.vla import joint_model as joint_model_module
    from research.semantic_token_cd.spatialvla_layer_selection_offline import get_controls

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cfg = OmegaConf.load("config/eval/bridge.yaml")
    cfg.use_torch_compile = False
    cfg.use_bf16 = False
    cfg.checkpoint_path = str(CHECKPOINT)
    cfg.gpu_id = 0
    cfg.env.adapter.dataset_statistics_path = str(OPENPI / "config/bridge_statistics.json")
    cfg.env.adapter.pretrained_model_path = str(ROOT / "checkpoints/pi0/paligemma-3b-pt-224")
    model = PiZeroInference(cfg, use_ddp=False)
    raw = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    weights = {k.replace("_orig_mod.", ""): v for k, v in raw["model"].items()}
    model.load_state_dict(weights, strict=True)
    del raw, weights
    model.freeze_all_weights()
    model.to(torch.float32).to(torch.device("cuda:0")).eval()
    adapter = hydra.utils.instantiate(cfg.env.adapter)
    if int(model.joint_model.num_hidden_layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} language layers, got {model.joint_model.num_hidden_layers}")
    device = torch.device("cuda:0")
    image_token_index = int(model.image_token_index)
    stats = adapter.dataset_statistics["proprio"]
    raw_zero = np.zeros((7,), dtype=np.float32)
    neutral_proprio = adapter.normalize_bound(raw_zero, np.asarray(stats["p01"]), np.asarray(stats["p99"]), clip_min=-1, clip_max=1)

    def attention_for(image: np.ndarray, prompt: str) -> tuple[np.ndarray, dict]:
        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        image_tensor = torch.as_tensor(resized, dtype=torch.uint8).permute(2, 0, 1)[None]
        encoded = adapter.processor(text=[prompt], images=image_tensor)
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in encoded.items()}
        inputs["proprios"] = torch.as_tensor(neutral_proprio, dtype=torch.float32, device=device)[None, None]
        ids = inputs["input_ids"]
        image_positions = torch.nonzero(ids[0] == image_token_index, as_tuple=False).flatten()
        if image_positions.numel() != 256:
            raise RuntimeError(f"expected 256 image tokens, got {image_positions.numel()}")
        phrase_ids = adapter.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        qstart = find_subsequence(ids[0].detach().cpu().tolist(), list(phrase_ids))
        qpos = list(range(qstart, qstart + len(phrase_ids)))
        prefix = model._forward_siglip_and_text_embedding(ids, inputs["pixel_values"])
        causal, vlm_pos, prop_pos, _ = model.build_causal_mask_and_position_ids(inputs["attention_mask"], dtype=inputs["pixel_values"].dtype)
        causal = causal.to(device)
        vlm_pos, prop_pos = vlm_pos.to(device), prop_pos.to(device)
        vlm_prop_mask, _ = model.split_full_mask_into_submasks(causal)
        proprio = model.proprio_encoder(inputs["proprios"])
        captured: dict[int, np.ndarray] = {}
        previous = joint_model_module.ATTENTION_CAPTURE_CALLBACK

        def capture(layer_idx, active_names, q_lens, weights):
            qidx = torch.as_tensor(qpos, device=weights.device, dtype=torch.long)
            captured[int(layer_idx)] = weights[0, :, qidx, :256].float().mean(dim=(0, 1)).detach().cpu().numpy()

        joint_model_module.ATTENTION_CAPTURE_CALLBACK = capture
        try:
            with torch.inference_mode():
                caches = model.joint_model.build_mixture_caches()
                model.joint_model(attention_mask=vlm_prop_mask,
                                  position_ids_all={"vlm": vlm_pos, "proprio": prop_pos},
                                  embeds_all={"vlm": prefix, "proprio": proprio},
                                  kv_caches=caches, return_caches=True)
        finally:
            joint_model_module.ATTENTION_CAPTURE_CALLBACK = previous
        if sorted(captured) != list(range(N_LAYERS)):
            raise RuntimeError(f"captured layers {sorted(captured)}")
        scores = np.stack([captured[i] for i in range(N_LAYERS)]).astype(np.float32)
        if scores.shape != (N_LAYERS, 256) or not np.isfinite(scores).all():
            raise RuntimeError(f"invalid attention array {scores.shape}")
        return scores, {"query_token_ids": list(phrase_ids), "query_positions": qpos,
                        "visual_positions": image_positions.detach().cpu().tolist(),
                        "per_layer_visual_attention_mass": scores.sum(axis=1).astype(float).tolist(),
                        "preprocessing": "BridgeSimplerAdapter: RGB uint8 -> cv2 Lanczos resize 224x224 -> PiZero VLAProcessor",
                        "proprio": "raw zero vector normalized with Bridge proprio p01/p99; fixed across prompt variants"}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SCAN_CONFIG.json").write_text(json.dumps({
        "protocol_id": "PI0_BRIDGE_CARROT_PROMPT_ATTENTION_LAYER_SCAN_V1",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "model": "open-pi-zero PyTorch PiZeroInference, standard Bridge-Beta fine-tune",
        "candidate_layers": "0-based PiZero/PaliGemma language layers L0-L17; action expert excluded",
        "task": TASK, "seeds": SEEDS, "states": len(SEEDS), "rank_budget": K,
        "prompts": "original / frozen synonym / frozen target-switched prompt",
        "source_states": str(SOURCE), "renderer_note": "No renderer invoked; exact saved RGB+actor segmentation used",
        "selection_note": "Exploratory single-task candidate selection, not closed-loop validation",
        "gpus_visible_to_process": os.environ.get("CUDA_VISIBLE_DEVICES")
    }, indent=2) + "\n")

    processed = 0
    for seed in SEEDS:
        state_dir = SOURCE / TASK / f"seed_{seed:03d}"
        meta = json.loads((state_dir / "meta.json").read_text())
        data = np.load(state_dir / "input.npz")
        image, segmentation = np.asarray(data["image"]), np.asarray(data["seg_actor"])
        instruction = meta["instruction"]
        switched, old_actor, new_actor = get_controls(TASK, instruction, meta)
        scores, capture_meta = [], []
        for prompt in (instruction, SYNONYM, switched):
            score, details = attention_for(image, prompt)
            scores.append(score)
            capture_meta.append(details)
        old_mask = patch_mask(segmentation, old_actor)
        new_mask = patch_mask(segmentation, new_actor)
        if not old_mask.any() or not new_mask.any():
            raise RuntimeError(f"empty actor patch mask for seed {seed}: {old_actor}/{new_actor}")
        out_dir = OUT / "states" / TASK / f"seed_{seed:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "layers.npz", original=scores[0], synonym=scores[1],
                            target_switch=scores[2], old_target_mask=old_mask, new_target_mask=new_mask)
        result = {"protocol_id": "PI0_BRIDGE_CARROT_PROMPT_ATTENTION_LAYER_SCAN_V1", "task": TASK,
                  "seed": seed, "instruction": instruction, "synonym_instruction": SYNONYM,
                  "target_switch_instruction": switched, "old_target_actor_id": int(old_actor),
                  "new_target_actor_id": int(new_actor), "old_target_patches": int(old_mask.sum()),
                  "new_target_patches": int(new_mask.sum()), "source_rgb_sha256": meta["rgb_sha256"],
                  "capture": capture_meta, "arrays_file": "layers.npz"}
        (out_dir / "meta.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"task": TASK, "seed": seed, "old_patches": int(old_mask.sum()),
                          "new_patches": int(new_mask.sum()), "done": True}), flush=True)
        processed += 1
        if args.limit and processed >= args.limit:
            print("DEBUG_LIMIT_REACHED", flush=True)
            return
    print("EXTRACTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
