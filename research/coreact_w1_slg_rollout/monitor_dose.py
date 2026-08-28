from __future__ import annotations
import argparse, json, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--artifact',type=Path,required=True); a=p.parse_args(); out=a.artifact.resolve()
    while True:
        n=len(list((out/'episodes').glob('*.json'))); invalid=len(list((out/'invalid_pairs').glob('*.json'))); selected=(out/'selected_dose.lock.json').exists(); decision=(out/'decision.json').exists()
        status='complete' if decision else ('rollout_failed' if invalid else ('confirmation_running' if selected and n>1250 else ('selection_running' if n<1250 else 'selection_analysis_pending')))
        (out/'status.json').write_text(json.dumps({'status':status,'episodes_complete':n,'episodes_planned':1750,'invalid_pairs':invalid,'selected_dose_locked':selected},indent=2)+'\n')
        if decision or invalid: return
        time.sleep(60)
if __name__=='__main__': main()
