"""CPU component gates for the VLA-Pruner 4.40 port.

These gates compare the ported selector against a literal reference of the
upstream 84d4b71 implementation and verify the compressed attention geometry.
They do not replace the later full-model active-pruning gold test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research.semantic_token_cd.vla_pruner_llama import (
    _redundancy_minimization,
    _select_keep_indices,
)
from research.semantic_token_cd.vla_pruner_policy import (
    PAPER_HISTORY_LAYERS,
    aggregate_action_history,
    extract_action_vision_attentions,
)

OUT = ROOT / "artifacts/vla_pruner_openvla_reproduction/paper_v5_gates"


def upstream_selector_reference(attn, embeds, cfg, seq_len):
    """Literal executable reference of upstream modeling_llama.py:1077-1137."""
    start, length = int(cfg["image_token_start_index"]), int(cfg["image_token_length"])
    avg = torch.mean(attn, dim=0)
    prefill = avg.mean(dim=0)
    current = prefill if cfg["use_prefil_attention"] else avg[-1]
    image_vec = current[start : start + length]
    num_keep = round(length * (1 - float(cfg["fastv_r"])))
    hist = cfg.get("historical_attention")
    if cfg["use_temporal"] and hist is not None:
        guide_topk = hist.topk(num_keep).indices
        current_topk = image_vec.topk(num_keep).indices
        unique_indices = torch.unique(torch.cat([guide_topk, current_topk]))
        if len(unique_indices) > num_keep:
            visual = embeds[0, start : start + length]
            chosen = _redundancy_minimization(visual[unique_indices], num_keep)
            final_topk = unique_indices[chosen]
        else:
            final_topk = unique_indices
        ranked = final_topk + start
    else:
        ranked = image_vec.topk(num_keep).indices + start
    image_end = min(start + length, seq_len)
    keep = torch.cat((torch.arange(start), ranked, torch.arange(image_end, seq_len))).sort().values
    pruned = torch.arange(seq_len)[~torch.isin(torch.arange(seq_len), keep)]
    return keep, pruned


def selector_gate():
    rows = []
    for seed in range(20):
        torch.manual_seed(seed)
        seq_len, heads, dim = 264, 4, 32
        attn = torch.rand(heads, seq_len, seq_len, dtype=torch.float32)
        embeds = torch.randn(1, seq_len, dim, dtype=torch.float32)
        hist = torch.rand(256, dtype=torch.float32)
        for ratio in (0.0, 0.5, 0.75):
            cfg = {
                "fastv_k": 3, "fastv_r": ratio,
                "image_token_start_index": 1, "image_token_length": 256,
                "historical_attention": hist, "use_temporal": True,
                "use_text_vision_selection": False, "use_prefil_attention": True,
                "SparseVLM": False,
            }
            got = _select_keep_indices(attn, embeds, cfg, seq_len)
            ref_keep, ref_pruned = upstream_selector_reference(attn, embeds, cfg, seq_len)
            rows.append({
                "seed": seed, "fastv_r": ratio,
                "keep_exact": bool(torch.equal(got["kept_indices"].cpu(), ref_keep)),
                "pruned_exact": bool(torch.equal(got["pruned_indices"].cpu(), ref_pruned)),
                "visual_kept": int(got["num_keep"]),
            })
    return {"cases": len(rows), "passed": all(r["keep_exact"] and r["pruned_exact"] for r in rows), "rows": rows}


def attention_geometry_gate():
    torch.manual_seed(7)
    layers, heads, actions, original_seq, prune_layer = 6, 2, 7, 264, 3
    visual_keep = torch.tensor(sorted(torch.randperm(256)[:64].tolist()))
    keep = torch.cat([torch.tensor([0]), visual_keep + 1, torch.arange(257, original_seq)])
    kept_seq = int(keep.numel())
    attentions = []
    for action_idx in range(actions):
        per_layer = []
        for layer in range(layers):
            if action_idx == 0:
                q = original_seq if layer < prune_layer else kept_seq
                kv = q
            else:
                q = 1
                kv = original_seq + action_idx if layer < prune_layer else kept_seq + action_idx
            per_layer.append(torch.rand(1, heads, q, kv))
        attentions.append(tuple(per_layer))
    pi = {"kept_indices": keep, "pruning_layer": prune_layer}
    restored = extract_action_vision_attentions(attentions, pi)
    checks = []
    pruned_visual = torch.tensor([i for i in range(256) if i not in set(visual_keep.tolist())])
    for action_idx in range(actions):
        for layer in range(layers):
            src = attentions[action_idx][layer][0]
            row = src[:, -1, :] if action_idx == 0 else src[:, 0, :]
            got = restored[layer, :, action_idx]
            if layer < prune_layer:
                expected = row[:, 1:257]
                ok = torch.equal(got, expected)
            else:
                ok = torch.equal(got[:, visual_keep], row[:, 1:65])
                ok = ok and bool(torch.count_nonzero(got[:, pruned_visual]) == 0)
            checks.append(bool(ok))
    return {"cases": len(checks), "passed": all(checks), "kept_visual": 64}


def main():
    result = {
        "selector": selector_gate(),
        "attention_geometry": attention_geometry_gate(),
    }
    torch.manual_seed(11)
    av = torch.rand(32, 4, 7, 256, dtype=torch.bfloat16)
    result["history_aggregation"] = {
        "code_l15_exact": bool(torch.equal(
            aggregate_action_history(av, (15,)), av[15].float().mean(dim=0).mean(dim=0)
        )),
        "paper_l16_31_exact": bool(torch.equal(
            aggregate_action_history(av, PAPER_HISTORY_LAYERS),
            av[16:32].float().mean(dim=(0, 1, 2)),
        )),
    }
    result["all_passed"] = bool(
        result["selector"]["passed"]
        and result["attention_geometry"]["passed"]
        and all(result["history_aggregation"].values())
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "COMPONENT_GATES.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v if k == "all_passed" else v.get("passed", v)
                      for k, v in result.items()}, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
