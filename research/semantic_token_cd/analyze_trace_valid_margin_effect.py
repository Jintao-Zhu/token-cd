#!/usr/bin/env python3
"""Offline action-margin decomposition on replay states with historical mask match."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path('/home/leju-suzhou/zjt_ws/token-cd')
DATA = REPO / 'artifacts/libero_transfer_diagnostics/libero_mild_view_logits'
OUT = REPO / 'artifacts/libero_transfer_diagnostics'
BOOT = 20000
SEED = 20260926
EPS = 1e-6


def auc(rescue: np.ndarray, harm: np.ndarray) -> float:
    if len(rescue) == 0 or len(harm) == 0:
        return float('nan')
    d = rescue[:, None] - harm[None, :]
    return float((np.sum(d > 0) + .5 * np.sum(d == 0)) / d.size)


def episode_summary(case_dir: Path, require_trace_match: bool) -> dict:
    meta = json.loads((case_dir / 'metadata.json').read_text())
    arrays = np.load(case_dir / meta['arrays'])
    pos = arrays['positive'].astype(np.float32)
    neg = arrays['negative'].astype(np.float32)
    state_rows = []
    for i, frame in enumerate(meta['frames']):
        view0 = frame['views'][0]
        exact = bool(view0['identity_trace_m_match'] and view0['identity_trace_selected_exact'])
        if require_trace_match and not exact:
            continue
        z = pos[i, 0, :6]
        r = z - neg[i, 0, :6]
        clean_margin, guidance_margin, guided_margin = [], [], []
        leverage, flip = [], []
        for q in range(6):
            order = np.argsort(z[q])[-2:][::-1]
            win, runner = int(order[0]), int(order[1])
            cm = float(z[q, win] - z[q, runner])
            gm = float(r[q, win] - r[q, runner])
            fm = cm + .5 * gm
            clean_margin.append(cm)
            guidance_margin.append(gm)
            guided_margin.append(fm)
            leverage.append(.5 * gm / max(cm, EPS))
            flip.append(float(fm < 0.0))
        if not clean_margin:
            continue
        step = int(frame['step'])
        denom = max(1, int(meta['trajectory_steps']) - 1)
        state_rows.append({
            'step': step,
            'phase': step / denom,
            'm': int(view0['m_current']),
            'clean_margin': float(np.mean(clean_margin)),
            'guidance_margin': float(np.mean(guidance_margin)),
            'guided_margin': float(np.mean(guided_margin)),
            'relative_guidance_leverage': float(np.mean(leverage)),
            'clean_winner_flip_fraction': float(np.mean(flip)),
        })
    if not state_rows:
        return {
            'case_id': meta['case_id'], 'task_id': meta['task_id'], 'task': meta['task'],
            'outcome': meta['outcome'], 'states': 0, 'metrics': {},
        }
    early = [x for x in state_rows if x['phase'] < .5]
    late = [x for x in state_rows if x['phase'] >= .5]

    def mean(rows: list[dict], key: str):
        vals = [x[key] for x in rows]
        return float(np.mean(vals)) if vals else None

    return {
        'case_id': meta['case_id'], 'task_id': meta['task_id'], 'task': meta['task'],
        'outcome': meta['outcome'], 'states': len(state_rows),
        'metrics': {k: mean(state_rows, k) for k in (
            'clean_margin', 'guidance_margin', 'guided_margin',
            'relative_guidance_leverage', 'clean_winner_flip_fraction')},
        'early': {k: mean(early, k) for k in (
            'clean_margin', 'guidance_margin', 'guided_margin',
            'relative_guidance_leverage', 'clean_winner_flip_fraction')},
        'late': {k: mean(late, k) for k in (
            'clean_margin', 'guidance_margin', 'guided_margin',
            'relative_guidance_leverage', 'clean_winner_flip_fraction')},
    }


def summarize(rows: list[dict], metric: str, phase: str | None = None) -> dict:
    get = lambda x: x['metrics'][metric] if phase is None else x[phase][metric]
    valid = [x for x in rows if x['outcome'] in ('rescue', 'harm') and x['states'] > 0 and get(x) is not None]
    rescue = np.asarray([get(x) for x in valid if x['outcome'] == 'rescue'], dtype=float)
    harm = np.asarray([get(x) for x in valid if x['outcome'] == 'harm'], dtype=float)
    rng = np.random.default_rng(SEED + sum(ord(c) for c in metric + str(phase)))
    auc_draws, diff_draws = [], []
    for _ in range(BOOT):
        rb = rng.choice(rescue, len(rescue), replace=True)
        hb = rng.choice(harm, len(harm), replace=True)
        auc_draws.append(auc(rb, hb))
        diff_draws.append(float(np.mean(rb) - np.mean(hb)))
    tasks = []
    for task_id in sorted({x['task_id'] for x in valid}):
        tr = np.asarray([get(x) for x in valid if x['task_id'] == task_id and x['outcome'] == 'rescue'])
        th = np.asarray([get(x) for x in valid if x['task_id'] == task_id and x['outcome'] == 'harm'])
        if len(tr) and len(th):
            tasks.append({'task_id': int(task_id), 'task': next(x['task'] for x in valid if x['task_id'] == task_id),
                          'rescue_n': len(tr), 'harm_n': len(th), 'auc_rescue_higher': auc(tr, th),
                          'rescue_minus_harm_mean': float(np.mean(tr) - np.mean(th))})
    macro = float(np.mean([x['auc_rescue_higher'] for x in tasks])) if tasks else None
    macro_draws = []
    if tasks:
        for _ in range(BOOT):
            vals = []
            for task in tasks:
                tr = np.asarray([get(x) for x in valid if x['task_id'] == task['task_id'] and x['outcome'] == 'rescue'])
                th = np.asarray([get(x) for x in valid if x['task_id'] == task['task_id'] and x['outcome'] == 'harm'])
                vals.append(auc(rng.choice(tr, len(tr), replace=True), rng.choice(th, len(th), replace=True)))
            macro_draws.append(float(np.mean(vals)))
    return {
        'episode_n': len(valid), 'rescue_n': len(rescue), 'harm_n': len(harm),
        'rescue_mean': float(np.mean(rescue)), 'harm_mean': float(np.mean(harm)),
        'rescue_minus_harm_mean': float(np.mean(rescue) - np.mean(harm)),
        'mean_difference_episode_bootstrap_95pct_ci': [float(np.quantile(diff_draws, .025)), float(np.quantile(diff_draws, .975))],
        'auc_rescue_higher': auc(rescue, harm),
        'auc_episode_bootstrap_95pct_ci': [float(np.quantile(auc_draws, .025)), float(np.quantile(auc_draws, .975))],
        'within_task_auc': tasks,
        'macro_within_task_auc': macro,
        'macro_within_task_auc_episode_bootstrap_95pct_ci': ([float(np.quantile(macro_draws, .025)), float(np.quantile(macro_draws, .975))] if macro_draws else None),
    }


def main() -> None:
    all_rows, exact_rows = [], []
    for case_dir in sorted(DATA.glob('*/metadata.json')):
        case_dir = case_dir.parent
        all_rows.append(episode_summary(case_dir, False))
        exact_rows.append(episode_summary(case_dir, True))
    metrics = ('clean_margin', 'guidance_margin', 'guided_margin',
               'relative_guidance_leverage', 'clean_winner_flip_fraction')
    result = {
        'protocol': 'TRACE_VALID_HARMONIC_ACTION_MARGIN_DECOMPOSITION_V1',
        'statistical_unit': 'episode; bootstrap resamples whole episodes',
        'counterfactual': 'Mstar = Mclean + 0.5 * (r[winner] - r[runner_up]); winner/runner-up are fixed from identity clean logits',
        'trace_valid_definition': 'at least one sampled state has current identity-view m and selected IDs both exactly matching historical trace',
        'all_state_episode_summaries': all_rows,
        'trace_valid_episode_summaries': exact_rows,
        'all_state_metrics': {metric: summarize(all_rows, metric) for metric in metrics},
        'trace_valid_metrics': {metric: summarize(exact_rows, metric) for metric in metrics},
        'trace_valid_early_metrics': {metric: summarize(exact_rows, metric, 'early') for metric in metrics},
        'trace_valid_late_metrics': {metric: summarize(exact_rows, metric, 'late') for metric in metrics},
        'limits': [
            'This is an offline association of the identity-view counterfactual margin with whole-episode Rescue/Harm labels, not a causal intervention on the closed loop.',
            'Winner/runner-up is a continuous logit comparison, not proof that either action is physically correct.',
            'Early/late are normalized replay-time halves, not simulator event phases such as grasp or placement.',
            'Trace validity checks m and selected IDs, but the current replay RGB may differ from historical RGB.',
        ],
    }
    out_json = OUT / 'TRACE_VALID_MARGIN_EFFECT_ANALYSIS.json'
    out_json.write_text(json.dumps(result, indent=2) + '\n')
    lines = [
        '# Trace-valid counterfactual action-margin analysis', '',
        '## Question', '',
        'On states where current identity-view `m` and selected token IDs both match the historical trace, does the harmonic negative push the clean action winner/runner-up margin differently in Rescue and Harm episodes?', '',
        'For each continuous action dimension, `Mclean = z+(winner)-z+(runner_up)`, `Mr = r(winner)-r(runner_up)`, and `Mguided = Mclean + 0.5 Mr`. The winner/runner-up pair comes from identity clean logits. Episode means are the unit; bootstrap resamples whole episodes.', '',
        '## Trace-valid primary results', '',
        '| Episode summary | Rescue | Harm | Rescue−Harm (95% CI) | AUC Rescue higher (95% CI) |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, key in [('Clean margin', 'clean_margin'), ('Guidance margin contribution', 'guidance_margin'),
                      ('Guided margin', 'guided_margin'), ('Relative guidance leverage', 'relative_guidance_leverage'),
                      ('Clean-pair winner flip fraction', 'clean_winner_flip_fraction')]:
        x = result['trace_valid_metrics'][key]
        diff = x['mean_difference_episode_bootstrap_95pct_ci']
        ci = x['auc_episode_bootstrap_95pct_ci']
        lines.append(f"| {name} | {x['rescue_mean']:.4f} | {x['harm_mean']:.4f} | {x['rescue_minus_harm_mean']:.4f} [{diff[0]:.4f}, {diff[1]:.4f}] | {x['auc_rescue_higher']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines += ['', f"Trace-valid episodes: {result['trace_valid_metrics']['clean_margin']['rescue_n']} Rescue / {result['trace_valid_metrics']['clean_margin']['harm_n']} Harm.", '',
              '## Task-stratified results', '',
              'AUCs by task are in the JSON record for each metric. Task 10 has no trace-valid Rescue episode and is omitted from macro AUC.', '',
              '## Interpretation limits', '',
              *[f'- {x}' for x in result['limits']], '',
              'This analysis can locate an association in the margin decomposition. It cannot prove that the negative branch caused the final episode outcome; changing the executed action would be needed for that.', '']
    (OUT / 'TRACE_VALID_MARGIN_EFFECT_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'trace_valid_counts': {
        'rescue': result['trace_valid_metrics']['clean_margin']['rescue_n'],
        'harm': result['trace_valid_metrics']['clean_margin']['harm_n']},
        'trace_valid': result['trace_valid_metrics'],
        'trace_valid_early': result['trace_valid_early_metrics'],
        'trace_valid_late': result['trace_valid_late_metrics']}, indent=2))


if __name__ == '__main__':
    main()
