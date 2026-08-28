#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


SOURCE_NAME = "coreact_trained_weak_local_finetune_v1_20260812_121522"
WEAKS = {"weak_a_8l": 8, "weak_b_4l": 4}
SEED = 1729


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensors_sha256(items) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_audit(policy):
    categories = {name: {"total": 0, "trainable": 0} for name in ("vision", "vlm_text_connector", "action_expert", "action_interface", "all")}
    for name, parameter in policy.named_parameters():
        count = parameter.numel()
        if ".vlm_with_expert.lm_expert." in name:
            category = "action_expert"
        elif ".vlm_with_expert.vlm.model.vision_model." in name:
            category = "vision"
        elif ".vlm_with_expert.vlm." in name:
            category = "vlm_text_connector"
        else:
            category = "action_interface"
        for key in (category, "all"):
            categories[key]["total"] += count
            categories[key]["trainable"] += count if parameter.requires_grad else 0
    return categories


def architecture_audit(policy):
    combined = policy.model.vlm_with_expert
    model_layers = combined.get_model_layers([combined.get_vlm_model().text_model, combined.lm_expert])
    mapping = [index for index, layer in enumerate(model_layers[1]) if layer is not None]
    expert_layer = combined.lm_expert.layers[0]
    return {
        "vlm_layers": combined.num_vlm_layers,
        "expert_layers": combined.num_expert_layers,
        "expert_to_vlm_stage_mapping": mapping,
        "vlm_hidden_size": combined.config.text_config.hidden_size,
        "expert_hidden_size": combined.expert_hidden_size,
        "expert_intermediate_size": expert_layer.mlp.gate_proj.out_features,
        "expert_attention_heads": expert_layer.self_attn.config.num_attention_heads,
        "expert_key_value_heads": expert_layer.self_attn.config.num_key_value_heads,
        "head_dim": expert_layer.self_attn.head_dim,
        "vision_hidden_size": combined.config.vision_config.hidden_size,
        "vision_attention_heads": combined.config.vision_config.num_attention_heads,
        "chunk_size": policy.config.chunk_size,
        "action_dimension": policy.config.output_features["action"].shape[0],
        "attention_mode": policy.config.attention_mode,
        "expert_width_multiplier": policy.config.expert_width_multiplier,
    }


def copy_processors(source: Path, destination: Path):
    copied = []
    for path in sorted(source.glob("policy_*processor*")):
        target = destination / path.name
        shutil.copy2(path, target)
        copied.append({"name": path.name, "sha256": file_sha256(target)})
    if not copied:
        raise RuntimeError("no processor artifacts copied")
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = (args.output or workspace / "artifacts" / f"coreact_capacity_weak_v1_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if artifact.exists():
        raise FileExistsError(artifact)
    for directory in ("initializations", "training", "logs", "status", "selection", "confirmation", "figures"):
        (artifact / directory).mkdir(parents=True, exist_ok=True)

    source = workspace / "artifacts" / SOURCE_NAME
    strong_checkpoint = source / "training_run/trajectory/checkpoints/015000/pretrained_model"
    strong_config_json = json.loads((strong_checkpoint / "config.json").read_text())
    if strong_config_json["num_expert_layers"] > 0 or strong_config_json["num_vlm_layers"] != 16:
        raise RuntimeError("Strong is not the locked 16-layer action expert")

    audits = {}
    vlm_hashes = {}
    for weak_name, depth in WEAKS.items():
        if 16 % depth:
            raise RuntimeError(f"depth {depth} is not natively supported")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        config = PreTrainedConfig.from_pretrained(strong_checkpoint, local_files_only=True)
        config.pretrained_path = None
        config.num_expert_layers = depth
        config.device = "cuda"
        config.compile_model = False
        policy = SmolVLAPolicy(config)
        initialization = artifact / "initializations" / weak_name
        policy.save_pretrained(initialization)
        processors = copy_processors(strong_checkpoint, initialization)
        combined = policy.model.vlm_with_expert
        audit = {
            "name": weak_name,
            "initialization_seed": SEED,
            "architecture": architecture_audit(policy),
            "parameters": parameter_audit(policy),
            "processor_files": processors,
            "model_sha256": file_sha256(initialization / "model.safetensors"),
            "vlm_initialization_sha256": tensors_sha256(combined.vlm.state_dict().items()),
            "action_expert_initialization_sha256": tensors_sha256(combined.lm_expert.state_dict().items()),
        }
        expected_mapping = list(range(0, 16, 16 // depth))
        if audit["architecture"]["expert_to_vlm_stage_mapping"] != expected_mapping:
            raise RuntimeError(f"native mapping mismatch for {weak_name}: {audit['architecture']['expert_to_vlm_stage_mapping']}")
        audits[weak_name] = audit
        vlm_hashes[weak_name] = audit["vlm_initialization_sha256"]
        del policy
        torch.cuda.empty_cache()
    if len(set(vlm_hashes.values())) != 1:
        raise RuntimeError("Weak-A and Weak-B VLM initializations differ")

    protocol = {
        "experiment_name": "coreact_capacity_degraded_weak_compatibility_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "TRAINING_AUTHORIZED_SELECTION_LOCKED_CONFIRMATION_SEALED_NO_ROLLOUT",
        "strong": {"checkpoint": str(strong_checkpoint), "training_step": 15000, "expert_layers": 16, "immutable": True},
        "weak_candidates": {
            "weak_a_8l": {"expert_layers": 8, "mapping": [0, 2, 4, 6, 8, 10, 12, 14], "primary_checkpoint": 15000},
            "weak_b_4l": {"expert_layers": 4, "mapping": [0, 4, 8, 12], "primary_checkpoint": 15000},
        },
        "saved_checkpoints": [5000, 10000, 15000],
        "primary_checkpoint_only": 15000,
        "initialization": {"seed": SEED, "same_seed_for_capacity_isolation": True, "vlm": "same HuggingFaceTB/SmolVLM2-500M-Video-Instruct pretrained weights", "action_expert": "fresh random initialization from native SmolVLA constructor", "vlm_hash_equal_across_weaks": True},
        "training": {
            "dataset_repo_id": "lerobot/libero", "dataset_root": str(source / "dataset_cache/lerobot/libero"), "dataset_revision": "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
            "steps": 15000, "save_freq": 5000, "batch_size": 32, "num_workers": 4, "seed": SEED,
            "optimizer": "AdamW", "lr": 0.0001, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 1e-10, "grad_clip_norm": 10.0,
            "scheduler_warmup_steps": 1000, "scheduler_decay_steps": 25000, "scheduler_decay_lr": 2.5e-6,
            "freeze_vision_encoder": False, "train_expert_only": False, "train_state_proj": True,
            "flow_target": "epsilon-action", "flow_time_distribution": "Beta(1.5,1.0)*0.999+0.001", "chunk": [50, 7], "normalization": "MEAN_STD checkpoint processors copied byte-identically from Strong",
            "rename_map": {"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"},
        },
        "offline": {
            "suite": "libero_spatial", "states": 500, "states_per_task": 50, "selection_states_per_task": 25, "confirmation_states_per_task": 25,
            "noise_seeds_per_state": 3, "flow_timesteps": 10, "lambda": 0.5, "trust_region_kappa": 0.25,
            "selection_gate": {"weak_error_gt_strong": True, "p_g_positive_min": 0.60, "ci_lower_min": 0.50, "tasks_above_half_min": 7, "p_r_gt_1_min": 0.55, "p_applied_gain_min": 0.55},
            "candidate_tie_rule": "higher P(G>0); if difference <3pp choose Weak-A 8L",
            "confirmation_gate": {"p_g_positive_min": 0.60, "ci_lower_min": 0.50, "tasks_above_half_min": 7, "p_r_gt_1_min": 0.55, "p_applied_gain_min": 0.55},
        },
        "prohibited": ["5k/10k candidate selection", "timestep window selection", "lambda tuning", "condition perturbation", "confirmation fallback candidate", "automatic rollout"],
        "source": {"strong_config_sha256": file_sha256(strong_checkpoint / "config.json"), "strong_model_sha256": file_sha256(strong_checkpoint / "model.safetensors"), "training_contract_sha256": file_sha256(source / "resolved_training_contract.yaml")},
    }
    (artifact / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (artifact / "architecture_audit.json").write_text(json.dumps(audits, indent=2, sort_keys=True) + "\n")
    (artifact / "status/current.json").write_text(json.dumps({"stage": "INITIALIZED", "training": "pending"}, indent=2) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"decision": "PENDING_TRAINING"}, indent=2) + "\n")
    print(artifact)


if __name__ == "__main__":
    main()
