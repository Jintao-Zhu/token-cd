from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from research.ar_token_closed_loop.common import file_sha256,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();art=a.artifact.resolve(); rows=[json.loads(x.read_text()) for x in (art/'episodes').glob('*.json')]
 conds=('vanilla','top8_mask_only','away','toward'); by={}
 for r in rows: by.setdefault(r['pair_id'],{})[r['condition']]=r
 if len(rows)!=200 or len(by)!=50 or any(set(x)!=set(conds) for x in by.values()): raise RuntimeError('incomplete paired artifact')
 rates={c:float(np.mean([by[k][c]['success'] for k in by])) for c in conds}; comparisons={}
 base=np.array([by[k]['vanilla']['success'] for k in by],int)
 for c in conds[1:]:
  treated=np.array([by[k][c]['success'] for k in by],int); x=int(((base==1)&(treated==0)).sum());y=int(((base==0)&(treated==1)).sum()); comparisons[c]={'delta':rates[c]-rates['vanilla'],'vanilla_only':x,'condition_only':y,'discordant':x+y,'mcnemar_p':1.0 if x+y==0 else float(binomtest(min(x,y),n=x+y).pvalue),'mean_replans':float(np.mean([by[k][c]['replans'] for k in by])),'all_replans_masked':all(by[k][c]['all_replans_masked'] for k in by)}
 summary={'episodes':len(rows),'paired_states':len(by),'success_rates':rates,'comparisons':comparisons,'decision':'FLOW_DIRECT_DELETE_DIRECTION_REPLICATION_COMPLETE','scope':'task-4 development replication; direct prefix deletion, not embedding replacement'};write_json(art/'summary.json',summary);write_json(art/'decision.json',{'decision':summary['decision'],'integrity_pass':True});(art/'report.md').write_text('# Flow Direct-Delete Every-Replan Direction Replication\n\n'+json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8');write_json(art/'sha256_audit.json',{str(x.relative_to(art)):file_sha256(x) for x in sorted(art.rglob('*')) if x.is_file() and x.name!='sha256_audit.json'});print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
