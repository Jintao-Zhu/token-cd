"""Freeze a manifest of pipeline sources + key artifacts for auditability."""
from pathlib import Path
import argparse, hashlib, json, time

SCRIPTS = ['vtcg_factorial_queue.py', 'run_vtcg_factorial_queue.sh', 'watch_vtcg_formal.sh',
           'finish_vtcg_factorial.sh', 'check_vtcg_first_state.py', 'check_vtcg_model_parity.py',
           'audit_vtcg_media_fast.py', 'audit_vtcg_factorial_outputs.py', 'analyze_vtcg_factorial.py',
           'summarize_vtcg_events.py', 'classify_vtcg_failure_stage.py',
           'prepare_vtcg_failure_annotations.py', 'prepare_vtcg_panels.py',
           'build_vtcg_report.py', 'monitor_vtcg_run.py', 'telemetry_vtcg_run.py',
           'release_vtcg_formal.py', 'profile_vtcg_factorial.py',
           'profile_vtcg_online_paired_state.py', 'export_vtcg_profile_states.py',
           'test_vtcg_factorial_math.py', 'test_vtcg_factorial_policy.py',
           'extract_vtcg_review_frames.py']
SOURCES = ['research/semantic_token_cd/vtcg_factorial_policy.py',
           'research/semantic_token_cd/vtcg_factorial_math.py',
           'research/semantic_token_cd/vtcg_factorial_episode.py',
           'research/semantic_token_cd/prompt_attn_shr_policy.py',
           'research/semantic_token_cd/prompt_attn_l11_budget_causal_rollout.py',
           'research/semantic_token_cd/distractor_rollout.py',
           'research/semantic_token_cd/global_merge_policy.py']
ARTIFACTS = ['CONFIG_LOCK.json', 'ENGINEERING_AUDIT_PASS.json', 'FORMAL_MEMBERSHIP_AUDIT.json',
             'FORMAL_ANALYSIS.json', 'FORMAL_MEDIA_AUDIT.json', 'FORMAL_MODEL_PARITY.json',
             'FORMAL_FIRST_STATE_AUDIT.json', 'ANALYSIS_PLAN.md', 'FAILURE_ANNOTATION_PROTOCOL.md',
             'PROFILING_PLAN.md', 'REPORT.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = {'generated_at': time.time(), 'protocol': 'VTCG_FACTORIAL_CONTROLS_V1',
           'scripts': {}, 'research_sources': {}, 'artifacts': {}, 'missing': []}
    for group, base, names in (('scripts', root / 'scripts', SCRIPTS),
                               ('research_sources', root, SOURCES),
                               ('artifacts', a.artifact, ARTIFACTS)):
        for name in names:
            p = base / name
            if p.is_file():
                out[group][name] = sha(p)
            else:
                out['missing'].append(f'{group}/{name}')
    (a.artifact / 'PIPELINE_MANIFEST.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'scripts': len(out['scripts']), 'research_sources': len(out['research_sources']),
                      'artifacts': len(out['artifacts']), 'missing': out['missing']}, indent=2))


if __name__ == '__main__':
    main()
