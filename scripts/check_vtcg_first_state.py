"""Incremental first-state parity check, independent of episode outcomes.

Checks, on the SAME restored initial observation only:
 - equal budget counts between the attention and KMeans ranking rows,
 - identical clean attention ranking,
 - identical selected mask for the norm-noise control vs the matched arm it mirrors,
 - float32 norm matching of the noise constructor,
 - every non-vanilla arm's mask is exactly the top-m of its own full ranking,
 - tokens outside the selected set are bit-identical after the projector merge.

Later closed-loop states are NOT expected to match across methods. This script never
reads success labels. A pending case is reported as pending, never as a pass.
"""
import argparse, json
from pathlib import Path

BUDGETS = ('matched', 'fixed', 'topp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--phase', choices=('engineering', 'formal'), default='engineering')
    a = ap.parse_args()
    c = json.loads((a.artifact / 'CONFIG_LOCK.json').read_text())
    seeds = c[a.phase + '_seeds']
    checks, pending, failures = [], [], []
    for task in c['tasks']:
        for seed in seeds:
            rows = {}
            for arm in c['arms']:
                root = a.artifact / 'episodes' / a.phase / task / arm
                if (root / f'episode_{seed:03d}_summary.json').exists() and \
                   (root / f'episode_{seed:03d}_trace.json').exists():
                    rows[arm] = json.loads((root / f'episode_{seed:03d}_trace.json').read_text())[0]
            for budget in BUDGETS:
                aa, kk = 'attn_' + budget, 'kmeans_' + budget
                if aa not in rows or kk not in rows:
                    pending.append([task, seed, aa, kk])
                    continue
                if rows[aa]['num_tokens'] != rows[kk]['num_tokens']:
                    failures.append(f'same-state budget differs {task}/{seed}/{budget}')
                if rows[aa]['attention_full_ranking_token_ids'] != rows[kk]['attention_full_ranking_token_ids']:
                    failures.append(f'same-state clean attention differs {task}/{seed}/{budget}')
                checks.append([task, seed, budget, 'count_attention_parity'])
            if 'norm_noise' in rows and 'attn_matched' in rows:
                if rows['norm_noise']['selected_token_ids'] != rows['attn_matched']['selected_token_ids']:
                    failures.append(f'noise mask differs {task}/{seed}')
                if not rows['norm_noise']['constructor_audit']['float32_norms_match']:
                    failures.append(f'noise float32 norm mismatch {task}/{seed}')
                checks.append([task, seed, 'norm_noise', 'mask_norm_parity'])
            for arm, row in rows.items():
                if arm == 'vanilla':
                    continue
                ids, m = row['full_ranking_token_ids'], row['num_tokens']
                if sorted(ids) != list(range(256)):
                    failures.append(f'invalid ranking {task}/{seed}/{arm}')
                if sorted(ids[:m]) != sorted(row['selected_token_ids']):
                    failures.append(f'mask not ranking top m {task}/{seed}/{arm}')
                if not row['constructor_audit']['applied_non_target_bit_identical']:
                    failures.append(f'outside-selected token changed {task}/{seed}/{arm}')
                if row['count_rule'] == 'matched' and (
                        row['num_tokens'] != row['matched_count']
                        or row['matched_count'] != len(row['reference_shr_token_ids'])):
                    failures.append(f'matched budget is not the canonical entity union {task}/{seed}/{arm}')
            for budget in ('topp',):
                aa, kk = 'attn_' + budget, 'kmeans_' + budget
                if aa in rows and kk in rows and rows[aa]['top_p_raw_count'] != rows[kk]['top_p_raw_count']:
                    failures.append(f'TopP raw count differs between ranking rows {task}/{seed}')
            if rows:
                checks.append([task, seed, 'all_arms', 'union_matched_and_topp_source'])
    complete = len(checks) == len(c['tasks']) * len(seeds) * 5 and not pending
    out = {'phase': a.phase, 'checked': checks, 'pending': pending, 'failures': failures,
           'complete': complete, 'expected_checks': len(c['tasks']) * len(seeds) * 5,
           'scope': 'initial state only; no success labels inspected'}
    (a.artifact / f'{a.phase.upper()}_FIRST_STATE_AUDIT.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f"checks {len(checks)}/{out['expected_checks']} pending {len(pending)} failures {len(failures)} complete {complete}")
    if failures:
        print('\n'.join(failures[:10]))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
