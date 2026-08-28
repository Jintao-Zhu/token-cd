from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();audits=[json.loads(x.read_text()) for x in o.joinpath('audits').glob('*.json')];valid=len(audits)==100 and all(x['valid'] for x in audits) and len(list(o.joinpath('snapshots').glob('*.pt')))==100;decision='LOCAL_SUCCESS_SNAPSHOT_GATE_PASS' if valid else 'LOCAL_SUCCESS_SNAPSHOT_GATE_FAIL';(o/'snapshot_gate.json').write_text(json.dumps({'decision':decision,'valid':sum(x['valid'] for x in audits),'expected':100,'trajectory_successes':sum(x['trajectory_success'] for x in audits)//2},indent=2)+'\n');(o/'status.json').write_text(json.dumps({'status':decision,'snapshots_complete':len(audits),'snapshots_planned':100},indent=2)+'\n');print(decision)
 if not valid:raise SystemExit(1)
if __name__=='__main__':main()
