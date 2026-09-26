#!/usr/bin/env python3
"""Episode-level SIMPLER/LIBERO mild-view L11 guidance-stability analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path('/home/leju-suzhou/zjt_ws/token-cd')
SIMPLER = REPO / 'artifacts/rt_appearance_factor_pilot/analysis/intervention_stability/view_logits'
LIBERO = REPO / 'artifacts/libero_transfer_diagnostics/libero_mild_view_logits'
OUT = REPO / 'artifacts/libero_transfer_diagnostics'
BOOT = 20000
SEED = 20260925
EPS = 1e-7


def auc(rescue: np.ndarray, harm: np.ndarray) -> float:
    if not len(rescue) or not len(harm):
        return float('nan')
    d = rescue[:, None] - harm[None, :]
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size)


def state_metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Scores one state; inputs are [view, action_dim, 256]."""
    effect = pos - neg
    q_sign, q_strict, q_mag, q_pair = [], [], [], []
    m_signs = []
    for q in range(6):
        identity_order = np.argsort(pos[0, q])[-2:][::-1]
        winner, runner = int(identity_order[0]), int(identity_order[1])
        margins = effect[:, q, winner] - effect[:, q, runner]
        if abs(float(margins[0])) <= EPS:
            continue
        same_sign = np.sign(margins[1:]) == np.sign(margins[0])
        q_sign.append(float(np.mean(same_sign)))
        q_strict.append(float(np.all(same_sign)))
        clean_margin = float(pos[0, q, winner] - pos[0, q, runner])
        scale = max(abs(float(margins[0])), 0.1 * abs(clean_margin), 1e-4)
        q_mag.append(float(np.mean(np.exp(-np.abs(margins[1:] - margins[0]) / scale))))
        pair_same = []
        for view in range(pos.shape[0]):
            order = np.argsort(pos[view, q])[-2:][::-1]
            pair_same.append(int(order[0]) == winner and int(order[1]) == runner)
        q_pair.append(float(np.mean(pair_same[1:])))
        m_signs.append(np.sign(margins[1:]).tolist())
    return {
        'valid_dimensions': len(q_sign),
        'guidance_sign_consistency': float(np.mean(q_sign)) if q_sign else None,
        'strict_all_views_sign_consistency': float(np.mean(q_strict)) if q_strict else None,
        'guidance_magnitude_similarity': float(np.mean(q_mag)) if q_mag else None,
        'clean_pair_consistency': float(np.mean(q_pair)) if q_pair else None,
    }


def load_dataset(root: Path, benchmark: str) -> list[dict]:
    rows = []
    for meta_path in sorted(root.glob('*/metadata.json')):
        meta = json.loads(meta_path.read_text())
        arrays = np.load(meta_path.parent / meta['arrays'])
        pos = arrays['positive'].astype(np.float32)
        neg = arrays['negative'].astype(np.float32)
        if pos.shape != neg.shape or pos.ndim != 4 or pos.shape[1:] != (7, 7, 256):
            raise RuntimeError(f'bad logits shape in {meta_path}: {pos.shape}/{neg.shape}')
        if len(meta['frames']) != len(pos):
            raise RuntimeError(f'frame/logit count mismatch in {meta_path}')
        state_rows = []
        for i, frame in enumerate(meta['frames']):
            metric = state_metrics(pos[i], neg[i])
            step = int(frame['step'])
            if benchmark == 'simpler':
                historical_match = bool(frame.get('historical_identity_trace_exact', False))
                view_rows = frame.get('views', [])
                budgets = [int(v.get('m_t', -1)) for v in view_rows]
                ids = [set(v.get('selected_token_ids', [])) for v in view_rows]
            else:
                views = frame['views']
                historical_match = bool(views[0]['identity_trace_m_match'] and
                                        views[0]['identity_trace_selected_exact'])
                budgets = [int(v['m_current']) for v in views]
                ids = [set(v['selected_ids_current']) for v in views]
            jaccards = []
            for selected in ids[1:]:
                union = ids[0] | selected
                jaccards.append(len(ids[0] & selected) / len(union) if union else 1.0)
            metric.update({
                'step': step,
                'phase': step / max(1, int(meta.get('trajectory_steps', 1)) - 1),
                'historical_identity_trace_exact': historical_match,
                'mild_view_budget_range': int(max(budgets) - min(budgets)) if budgets else None,
                'mild_view_mask_jaccard_mean': float(np.mean(jaccards)) if jaccards else None,
            })
            state_rows.append(metric)

        def mean_metric(key: str, subset: list[dict]) -> float | None:
            vals = [x[key] for x in subset if x.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        valid = [x for x in state_rows if x['guidance_sign_consistency'] is not None]
        late = [x for x in valid if x['phase'] >= 0.5]
        exact = [x for x in valid if x['historical_identity_trace_exact']]
        exact_late = [x for x in exact if x['phase'] >= 0.5]
        outcome = meta['outcome']
        rows.append({
            'benchmark': benchmark,
            'case_id': meta.get('case_id', str(meta.get('seed'))),
            'task_id': meta.get('task_id'), 'task': meta.get('task', meta.get('task_name')),
            'outcome': outcome, 'state_count': len(state_rows),
            'trace_exact_state_count': len(exact),
            'initial_state_hash_matches_history': meta.get('replay_validation', {}).get('initial_state_matches_history'),
            'replay_success_matches_history': meta.get('replay_validation', {}).get('success_matches_history',
                                               meta.get('replay_outcome_match')),
            'mean_sign_stability': mean_metric('guidance_sign_consistency', valid),
            'late_sign_stability': mean_metric('guidance_sign_consistency', late),
            'strict_mean_stability': mean_metric('strict_all_views_sign_consistency', valid),
            'strict_late_stability': mean_metric('strict_all_views_sign_consistency', late),
            'exact_trace_sign_stability': mean_metric('guidance_sign_consistency', exact),
            'exact_late_sign_stability': mean_metric('guidance_sign_consistency', exact_late),
            'mean_pair_consistency': mean_metric('clean_pair_consistency', valid),
            'mean_magnitude_similarity': mean_metric('guidance_magnitude_similarity', valid),
            'mean_budget_range': mean_metric('mild_view_budget_range', valid),
            'mean_mask_jaccard': mean_metric('mild_view_mask_jaccard_mean', valid),
            'state_rows': state_rows,
        })
    return rows


def bootstrap_auc(r: np.ndarray, h: np.ndarray, rng: np.random.Generator) -> list[float]:
    return [auc(rng.choice(r, len(r), replace=True), rng.choice(h, len(h), replace=True))
            for _ in range(BOOT)]


def auc_summary(rows: list[dict], score: str, task_stratified: bool = False) -> dict:
    rng = np.random.default_rng(SEED)
    valid = [x for x in rows if x['outcome'] in ('rescue', 'harm') and x.get(score) is not None]
    r = np.asarray([x[score] for x in valid if x['outcome'] == 'rescue'], dtype=float)
    h = np.asarray([x[score] for x in valid if x['outcome'] == 'harm'], dtype=float)
    result = {'rescue_n': len(r), 'harm_n': len(h), 'auc_rescue_higher': auc(r, h)}
    if len(r) and len(h):
        bs = np.asarray(bootstrap_auc(r, h, rng))
        result['stratified_episode_bootstrap_95pct_ci'] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        result['rescue_mean'] = float(np.mean(r)); result['harm_mean'] = float(np.mean(h))
        result['rescue_minus_harm_mean'] = float(np.mean(r) - np.mean(h))

    task_ids = sorted({x['task_id'] for x in valid if x['task_id'] is not None})
    tasks = []
    for task_id in task_ids:
        subset = [x for x in valid if x['task_id'] == task_id]
        tr = np.asarray([x[score] for x in subset if x['outcome'] == 'rescue'], dtype=float)
        th = np.asarray([x[score] for x in subset if x['outcome'] == 'harm'], dtype=float)
        if len(tr) and len(th):
            tasks.append({'task_id': task_id, 'task': subset[0]['task'], 'rescue_n': len(tr),
                          'harm_n': len(th), 'auc': auc(tr, th),
                          'difference': float(np.mean(tr) - np.mean(th))})
    result['task_auc'] = tasks
    if task_stratified and tasks:
        task_lookup = {x['task_id']: [y for y in valid if y['task_id'] == x['task_id']] for x in tasks}
        macro_point = float(np.mean([x['auc'] for x in tasks]))
        macro_boot, diff_boot = [], []
        for _ in range(BOOT):
            aucs, diffs = [], []
            for task in tasks:
                subset = task_lookup[task['task_id']]
                tr = np.asarray([x[score] for x in subset if x['outcome'] == 'rescue'], dtype=float)
                th = np.asarray([x[score] for x in subset if x['outcome'] == 'harm'], dtype=float)
                rb = rng.choice(tr, len(tr), replace=True); hb = rng.choice(th, len(th), replace=True)
                aucs.append(auc(rb, hb)); diffs.append(float(np.mean(rb) - np.mean(hb)))
            macro_boot.append(float(np.mean(aucs))); diff_boot.append(float(np.mean(diffs)))
        result['macro_within_task_auc'] = macro_point
        result['macro_within_task_auc_bootstrap_95pct_ci'] = [float(np.quantile(macro_boot,.025)), float(np.quantile(macro_boot,.975))]
        result['macro_within_task_difference'] = float(np.mean([x['difference'] for x in tasks]))
        result['macro_within_task_difference_bootstrap_95pct_ci'] = [float(np.quantile(diff_boot,.025)), float(np.quantile(diff_boot,.975))]
    return result


def main() -> None:
    simpler = load_dataset(SIMPLER, 'simpler')
    libero = load_dataset(LIBERO, 'libero')
    if not simpler or not libero:
        raise RuntimeError(f'incomplete inputs: SIMPLER={len(simpler)} LIBERO={len(libero)}')
    metrics = ('mean_sign_stability', 'late_sign_stability', 'strict_mean_stability',
               'strict_late_stability', 'exact_trace_sign_stability',
               'exact_late_sign_stability')
    aucs = {}
    for benchmark, rows in (('simpler', simpler), ('libero', libero)):
        aucs[benchmark] = {key: auc_summary(rows, key, task_stratified=(benchmark == 'libero'))
                           for key in metrics}

    # Descriptive cross-benchmark contrast. Bootstrap whole episodes within
    # outcome class, keeping the different benchmark/task/checkpoint caveat.
    rng = np.random.default_rng(SEED + 1)
    cross = {}
    for key in metrics:
        cross[key] = {}
        for label in ('all_discordant', 'rescue', 'harm'):
            a = [x[key] for x in simpler if x['outcome'] in ('rescue','harm') and
                 (label == 'all_discordant' or x['outcome'] == label) and x[key] is not None]
            b = [x[key] for x in libero if x['outcome'] in ('rescue','harm') and
                 (label == 'all_discordant' or x['outcome'] == label) and x[key] is not None]
            if not a or not b:
                continue
            draws = []
            for _ in range(BOOT):
                draws.append(float(np.mean(rng.choice(b, len(b), replace=True)) -
                                    np.mean(rng.choice(a, len(a), replace=True))))
            cross[key][label] = {
                'simpler_n': len(a), 'libero_n': len(b),
                'simpler_mean': float(np.mean(a)), 'libero_mean': float(np.mean(b)),
                'libero_minus_simpler': float(np.mean(b) - np.mean(a)),
                'episode_bootstrap_95pct_ci': [float(np.quantile(draws,.025)), float(np.quantile(draws,.975))],
            }

    result = {
        'protocol': 'SIMPLER_LIBERO_MILD_VIEW_GUIDANCE_DIRECTION_TRANSFER_V1',
        'episode_unit': True,
        'transformations': ['brightness x0.95/1.05','contrast x0.95/1.05','gamma 0.95/1.05'],
        'primary_metric': 'identity clean positive-logit top-two fixed; guidance margin sign agreement across each of six mild views, averaged across six continuous action dimensions, sampled states and episodes',
        'simpler_episode_count': len(simpler), 'libero_episode_count': len(libero),
        'simpler_outcome_counts': {k: sum(x['outcome']==k for x in simpler) for k in ('rescue','harm')},
        'libero_outcome_counts': {k: sum(x['outcome']==k for x in libero) for k in ('rescue','harm')},
        'libero_replay_validation': {
            'initial_state_hash_matches': sum(bool(x['initial_state_hash_matches_history']) for x in libero),
            'final_success_matches': sum(bool(x['replay_success_matches_history']) for x in libero),
            'identity_trace_exact_states': sum(x['trace_exact_state_count'] for x in libero),
            'sampled_states': sum(x['state_count'] for x in libero),
        },
        'auc_rescue_higher': aucs,
        'cross_benchmark_descriptive_contrast_libero_minus_simpler': cross,
        'episodes': [{k:v for k,v in x.items() if k!='state_rows'} for x in simpler + libero],
        'notes': [
            'SIMPLER uses one Full-RT pick-coke task and five Rescue / twelve Harm; LIBERO uses the frozen five-task evaluation and the LIBERO-finetuned checkpoint. Benchmark, task set, renderer and checkpoint differ together, so cross-benchmark score differences are descriptive rather than causal.',
            'LIBERO mild views are generated from one live replay RGB per physical state; each episode was replayed from the recorded init state and matched-arm actions. Initial state hash and final paired outcome are retained in per-episode metadata.',
            'The current identity selector/mask is compared to historical traces. Low exact-match coverage limits attribution to the historical intervention, even when physical replay outcome matches.',
            'Do not train or tune a reliability gate from these exploratory data.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'mild_view_transfer_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = [
        '# Mild-view counterfactual direction stability: SIMPLER vs LIBERO', '',
        '## Main result', '',
        'Unit is the episode. At each sampled state the identity-view clean positive-logit winner/runner-up is fixed, and the sign of the L11 guidance margin effect is checked across six mild photometric views. The primary score is the mean sign agreement across valid continuous action dimensions and sampled states.', '',
        '| Benchmark | episodes | Rescue | Harm | all-state AUC (higher stability predicts Rescue) | late-state AUC | strict all-view late AUC |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, rows in (('SIMPLER Full RT pick-coke', simpler), ('LIBERO five-task subset', libero)):
        nres=sum(x['outcome']=='rescue' for x in rows); nharm=sum(x['outcome']=='harm' for x in rows)
        vals=[aucs['simpler' if name.startswith('SIMPLER') else 'libero'][key] for key in ('mean_sign_stability','late_sign_stability','strict_late_stability')]
        lines.append(f"| {name} | {len(rows)} | {nres} | {nharm} | {vals[0].get('auc_rescue_higher',float('nan')):.3f} | {vals[1].get('auc_rescue_higher',float('nan')):.3f} | {vals[2].get('auc_rescue_higher',float('nan')):.3f} |")
    lines += ['', '## LIBERO replay and historical trace fidelity', '',
              f"Matched-arm action replays reproduced final paired outcomes in {result['libero_replay_validation']['final_success_matches']}/{len(libero)} episodes. Initial state hashes and detailed per-case status are in the input metadata. Identity `m` and selected-token agreement was exact in {result['libero_replay_validation']['identity_trace_exact_states']}/{result['libero_replay_validation']['sampled_states']} sampled states.", '',
              '## Per-task Rescue/Harm discrimination', '',
              'The primary task-stratified summary is the equal-weight macro AUC across LIBERO tasks that have both Rescue and Harm episodes. The task-level counts, AUCs, macro estimates, and episode-bootstrap intervals are listed below.', '',
              '| Task id | Task | Rescue | Harm | AUC |', '|---:|---|---:|---:|---:|']
    task_items=aucs['libero']['mean_sign_stability'].get('task_auc',[])
    for x in task_items:
        lines.append(f"| {x['task_id']} | {x['task']} | {x['rescue_n']} | {x['harm_n']} | {x['auc']:.3f} |")
    macro=aucs['libero']['mean_sign_stability']
    lines += ['', f"Macro within-task AUC: {macro.get('macro_within_task_auc',float('nan')):.3f}, 95% episode bootstrap interval {macro.get('macro_within_task_auc_bootstrap_95pct_ci')}.", '',
              '## Interpretation limits', '',
              'The SIMPLER set is one task (5 Rescue, 12 Harm); LIBERO uses a different task set and a LIBERO-finetuned checkpoint. Thus even a benchmark gap cannot be attributed to benchmark alone. LIBERO replay outcomes match history, but the current identity selector/mask often does not; treat Rescue/Harm associations as evidence about the current replayed observation pipeline, not exact historical logits. This analysis is exploratory and is not a validated reliability gate.', '',
              'Machine-readable full episode results and bootstrap details: `mild_view_transfer_analysis.json`.']
    (OUT / 'MILD_VIEW_TRANSFER_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'episodes': [len(simpler),len(libero)], 'outcomes': [result['simpler_outcome_counts'],result['libero_outcome_counts']], 'primary_auc': {'simpler':aucs['simpler']['mean_sign_stability'], 'libero':aucs['libero']['mean_sign_stability']}, 'macro_libero_auc':macro}, indent=2))


if __name__ == '__main__':
    main()
