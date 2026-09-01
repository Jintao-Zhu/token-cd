import glob, json, os
from pathlib import Path

out = Path('github_results')
out.mkdir(exist_ok=True)
for src in glob.glob('artifacts/**/episode_*_summary.json', recursive=True):
    try: d=json.load(open(src))
    except Exception: continue
    keep={k:d.get(k) for k in ('protocol_id','task','seed','episode_id','arm','beta','success','failure_reason','control_steps','technical_pass','canonical_snapshot_sha256','initial_state_sha256','initial_rgb_sha256','environment_id') if k in d}
    rel=Path(src).relative_to('artifacts')
    dst=out/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(keep,sort_keys=True)+'\n')
for src in glob.glob('artifacts/**/*.md',recursive=True)+glob.glob('artifacts/**/analysis.json',recursive=True)+glob.glob('artifacts/**/decision.json',recursive=True)+glob.glob('artifacts/**/CONFIG_LOCK.json',recursive=True)+glob.glob('artifacts/**/pairing_manifest*.json',recursive=True):
    rel=Path(src).relative_to('artifacts'); dst=out/rel; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(Path(src).read_bytes())
