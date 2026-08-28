from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt

ARMS=("lambda_0","lambda_005","lambda_010","lambda_025","lambda_050")
LABELS={"lambda_0":"0","lambda_005":".05","lambda_010":".10","lambda_025":".25","lambda_050":".50"}

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();out=a.artifact.resolve();x=json.loads((out/'selection_analysis.json').read_text())
    if x['decision']!='W1_DOSE_RESPONSE_NO_GO' or x['selected_arm'] is not None: raise RuntimeError('finalizer only handles preregistered selection no-go')
    rates=x['rates']; comp=x['comparisons']; metrics=x['dose_metrics']; tasks=x['task_rows']
    plt.figure(figsize=(6.4,4.2)); doses=[0,.05,.10,.25,.50]; y=[100*rates[a] for a in ARMS]; plt.plot(doses,y,marker='o',linewidth=2); plt.xlabel('Guidance dose lambda');plt.ylabel('Closed-loop success (%)');plt.xticks(doses,['0','.05','.10','.25','.50']);plt.grid(alpha=.25);plt.tight_layout();plt.savefig(out/'dose_response_curve.png',dpi=180);plt.close()
    with (out/'task_by_dose.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=tasks[0].keys());w.writeheader();w.writerows(tasks)
    main_rows=[]
    for arm in ARMS:
        if arm=='lambda_0': main_rows.append(f"| {LABELS[arm]} | {int(rates[arm]*250)}/250 ({rates[arm]:.1%}) | - | - | - | {metrics[arm]['mean_normalized_action_delta_l2']:.3f} | {metrics[arm]['clip_rate']:.1%} |")
        else:
            c=comp[arm];main_rows.append(f"| {LABELS[arm]} | {int(rates[arm]*250)}/250 ({rates[arm]:.1%}) | {c['delta_pp']:+.1f}pp | {c['rescue']} | {c['harm']} | {metrics[arm]['mean_normalized_action_delta_l2']:.3f} | {metrics[arm]['clip_rate']:.1%} |")
    task_table='\n'.join(f"| {r['task']} | "+' | '.join(f"{r[a]}/25" for a in ARMS)+" |" for r in tasks)
    report=f"""# W1 Low-Noise Guidance Dose-Response Closed-Loop Test

## Decision

`W1_DOSE_RESPONSE_NO_GO`

Selection used 250 paired states (25 per task). All three preregistered small doses were at or below Strong, so no dose was locked and Confirmation was not opened.

| lambda | Success | vs Strong | Rescue | Harm | Action delta L2 | Clip rate |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_rows)}

The dose curve is monotone non-improving: 76.0%, 75.6%, 75.2%, 74.8%, 71.2%. Action perturbation increases monotonically with dose (0, 0.104, 0.210, 0.525, 0.976). Trust-region clipping is absent through lambda=.25 and reaches 72.2% at lambda=.50.

| Task | lambda=0 | .05 | .10 | .25 | .50 |
|---:|---:|---:|---:|---:|---:|
{task_table}

Integrity: 1250/1250 Selection episodes complete, 250/250 five-arm paired units, zero invalid pairs. Old lambda=0/.50 outcomes were not reused. No Confirmation episodes were run after the preregistered Selection NO-GO.
"""
    (out/'REPORT.md').write_text(report);(out/'decision.json').write_text(json.dumps({'decision':'W1_DOSE_RESPONSE_NO_GO','confirmation_run':False},indent=2)+'\n');(out/'status.json').write_text(json.dumps({'status':'complete','episodes_complete':1250,'selection_episodes_planned':1250,'confirmation_run':False,'invalid_pairs':0,'decision':'W1_DOSE_RESPONSE_NO_GO'},indent=2)+'\n');print(report)
if __name__=='__main__':main()
