from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from research.token_pcd_stage_a.core import TASKS, file_sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--protocol-id", default="TOKEN_PCD_BRIDGE_CLOSED_LOOP_PILOT_R1")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--scope", default="ONE_BRIDGE_PILOT_NO_ADAPTATION")
    args = parser.parse_args(); workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=False); frozen = artifact / "frozen"; frozen.mkdir()
    stage_a = workspace / "artifacts/token_pcd_openvla_simpler_stage_a_v1_20260814"
    sources = {
        "position_conditioned_visual_mean.pt": stage_a / "position_conditioned_visual_mean.pt",
        "stage_a_decision.json": stage_a / "stage_a_decision.json",
        "stage_a_final_manifest.json": stage_a / "final_manifest.json",
    }
    for name, source in sources.items(): shutil.copy2(source, frozen / name)
    early_means = workspace / "artifacts/token_pcd_mechanism_early_vs_late_v1_20260815/early_patch_means.pt"
    if early_means.is_file(): shutil.copy2(early_means, frozen / "early_patch_means.pt")
    protocol = {
        "protocol_id": args.protocol_id,
        "hypothesis": "Frozen object-specific latent counterfactual has closed-loop utility despite Stage A non-equivalence",
        "stage_a_status_immutable": "INCONCLUSIVE", "scope": args.scope,
        "tasks": list(TASKS), "seeds": list(range(args.seed_start, args.seed_start + args.seed_count)),
        "arms": ["vanilla", "pixel_pcd", "object_token_pcd", "random_token_pcd"],
        "episodes": len(TASKS) * args.seed_count * 4, "pairing": "same canonical simulator/controller/RNG snapshot across four arms",
        "frozen_method": {"overlap_threshold": 0.25, "replacement": "Stage A position-conditioned mean",
                          "alpha": 0.8, "visual_tokens": 256, "pcd_reranked_positions": [0,1,2,3,4,5]},
        "random": "same token count, non-object only, deterministic by task/seed/timestep/clean RGB hash",
        "strong_go": {"object_minus_vanilla_pp_min": 5.0, "object_minus_random_pp_min": 3.0,
                      "object_minus_vanilla_paired_ci_lower_gt": 0.0, "tasks_object_nonnegative_min": 7,
                      "catastrophic_harm_pp_at_least": 10.0, "catastrophic_harm_paired_ci_upper_lt": 0.0,
                      "pixel_positive_control_direction_normal": True},
        "no_go": "Object-Vanilla < +3pp OR Object-Random < +3pp; stop Token-PCD without tuning",
        "inconclusive_reference": "If Pixel-PCD positive control is not directionally positive, reference failure makes bridge utility inconclusive",
        "forbidden": ["threshold tuning", "replacement tuning", "alpha tuning", "layer tuning", "token-count tuning", "Stage A relabeling"],
    }
    write_json(artifact / "protocol.lock.json", protocol)
    provenance = {"protocol_sha256": file_sha256(artifact / "protocol.lock.json"),
                  "frozen_sha256": {name: file_sha256(frozen / name) for name in sources},
                  "code_sha256": {str(path.relative_to(workspace)): file_sha256(path)
                                  for path in sorted((workspace / "research/token_pcd_bridge").glob("*.py"))},
                  "status": "LOCKED_BEFORE_SENTINEL"}
    write_json(artifact / "provenance.lock.json", provenance)


if __name__ == "__main__": main()
