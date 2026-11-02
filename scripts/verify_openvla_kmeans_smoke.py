#!/usr/bin/env python3
"""Fail-closed same-process-prefix K8 reproduction and K16/K32 smoke gate."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.semantic_token_cd.openvla_kmeans_budget_rollout import ARM_K, k_checks
TASKS = ['google_robot_close_drawer', 'google_robot_pick_coke_can', 'widowx_carrot_on_plate']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    args = ap.parse_args()
    rows, errors = [], []
    reference = ROOT / 'artifacts/openvla_lambda1_extension_200_299_v1'
    for task in TASKS:
        refpath = reference / 'episodes' / task / 'lambda_100/episode_200_summary.json'
        ref = json.loads(refpath.read_text())
        for arm, k in ARM_K.items():
            p = args.artifact / 'episodes' / task / arm / 'episode_200_summary.json'
            try:
                d = json.loads(p.read_text())
                checks = k_checks(d['selector_trace'], arm)
                checks.update({
                    'technical_pass': d.get('technical_pass') is True,
                    'success_boolean': type(d.get('success')) is bool,
                    'identity': d.get('task') == task and d.get('arm') == arm and d.get('seed') == 200,
                    'K_top_level': d.get('kmeans_K') == k,
                    'process_prefix': d.get('process_seed_block') == [200],
                    'trace_length': len(d['selector_trace']) == d['steps'] and d['steps'] > 0,
                    'reference_identity': all(d.get(key) == ref.get(key) for key in (
                        'canonical_snapshot_sha256', 'initial_state_sha256', 'initial_rgb_sha256',
                        'instruction', 'rendering', 'lambda', 'attention_layers',
                        'counterfactual_constructor', 'guidance_execution')),
                })
                with np.load(p.parent / d['arrays_file']) as arrays:
                    checks['finite_arrays'] = all(np.isfinite(arrays[key]).all() for key in arrays.files)
                    checks['action_length'] = arrays['executed_actions'].shape[0] == d['steps']
                    if k == 8:
                        checks['K8_outcome_exact'] = all(d.get(key) == ref.get(key) for key in ('success','steps','failure_reason'))
                        keys = ('selected_token_ids','m_t','positive_token_ids','negative_token_ids','final_token_ids')
                        checks['K8_trace_exact'] = len(d['selector_trace']) == len(ref['selector_trace']) and all(
                            all(a.get(key) == b.get(key) for key in keys)
                            for a,b in zip(d['selector_trace'],ref['selector_trace']))
                        with np.load(refpath.parent / ref['arrays_file']) as old:
                            checks['K8_actions_bit_exact'] = np.array_equal(arrays['executed_actions'], old['executed_actions'])
                            # Diagnostics only: logits may differ in inconsequential last bits.
                            logit_diffs = {}
                            for key in ('positive_logits','negative_logits'):
                                if key in old and key in arrays and old[key].shape == arrays[key].shape:
                                    logit_diffs[key] = float(np.max(np.abs(old[key]-arrays[key])))
                                else:
                                    logit_diffs[key] = None
                passed = all(bool(v) for v in checks.values())
                row = {'task':task,'arm':arm,'success':d['success'],'steps':d['steps'],
                       'mean_m_t':d['mean_selected_token_count'],'checks':checks,'technical_pass':passed}
                if k == 8:
                    row['logit_max_abs_difference'] = logit_diffs
                rows.append(row)
                if not passed:
                    errors.append({'task':task,'arm':arm,'failed':[key for key,val in checks.items() if not val]})
            except Exception as exc:
                errors.append({'task':task,'arm':arm,'error':repr(exc)})
    result = {'technical_pass':len(rows)==9 and not errors,'episodes':len(rows),'errors':errors,'rows':rows,
              'scope':'Three tasks, seed200 fresh process, K8 reproducibility and K16/K32 audit; not an efficacy test.'}
    out = args.artifact / 'SMOKE_VALIDATION.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['technical_pass'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
