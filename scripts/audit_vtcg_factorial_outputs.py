"""Strict membership/pair audit; never infer missing episodes as failures."""
import argparse, json, hashlib
from pathlib import Path

# 2026-10-18 审计口径更新（依据：AUDIT_UPDATE_20261018_SOURCE_VERSIONS.md）：
# 2026-10-17 的共享文件事故使正式 run 记录了两个已审计源码组合；两组合的
# 逐位等价证据在 SOURCE_PIN_UPDATE.json。只有这两个组合可以过审；除此以外
# 的文件集合必须唯一；任何第三个组合依旧 fail-fast。
AUDITED_FILES=('prompt_attn_shr_policy.py','vtcg_factorial_policy.py')
AUDITED_COMBOS={
 ('0ed313033aa2d3a07ce78c3219a3cec2ed84e15695b5bd39aea2912cb012338b',
  '30bebc7e74fdb491c51bf90dbeecf5b66eb81adac03469d4ea5dad0281e26d54'),  # 事故前（717 集）
 ('75eb975ea384f2fe4ad0d23e54a5a23cbe79d093c386ed2adea3c0fbc7799bde',
  'a9cde791c6879152e719450f7764f91953e989cb50da702c6428a130c99a356a'),  # 事故后（2883 集）
}
def _combo(src):
 n={Path(k).name:v for k,v in src.items()}; return tuple(n.get(f) for f in AUDITED_FILES)
def _other_sig(src):
 n={Path(k).name:v for k,v in src.items()}
 return json.dumps({k:v for k,v in n.items() if k not in AUDITED_FILES},sort_keys=True)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--artifact',type=Path,required=True); p.add_argument('--phase',choices=['engineering','formal'],required=True); a=p.parse_args()
 cfg=json.loads((a.artifact/'CONFIG_LOCK.json').read_text()); failures=[]; files=[]; pairs={}; other_sigs=set(); combos={}
 expected={(t,int(s),arm) for t in cfg['tasks'] for s in cfg[a.phase+'_seeds'] for arm in cfg['arms']}
 observed={}
 root=a.artifact/'episodes'/a.phase
 for f in root.glob('*/*/*_summary.json'):
  d=json.loads(f.read_text()); k=(d['task'],int(d['seed']),d['arm'])
  if k in observed: failures.append('duplicate:'+str(k))
  observed[k]=d
  if not d.get('technical_pass'): failures.append('technical:'+str(k))
  if d.get('preflightdebug'): failures.append('debug:'+str(k))
  if d.get('run_config_id')!=cfg['run_config_ids'][a.phase]: failures.append('configid:'+str(k))
  pairs.setdefault(k[:2],set()).add((d['initial_state_sha256'],d['initial_rgb_sha256'],d['canonical_snapshot_sha256']))
  other_sigs.add(_other_sig(d['source_sha256'])); ck=_combo(d['source_sha256']); combos[ck]=combos.get(ck,0)+1
  arrays=f.with_name(f.name.replace('_summary.json','_arrays.npz'))
  if not arrays.exists(): failures.append('arrays_missing:'+str(k))
  elif hashlib.sha256(arrays.read_bytes()).hexdigest()!=d['arrays_file_sha256']: failures.append('arrays_digest:'+str(k))
 for k,v in pairs.items():
  if len(v)!=1: failures.append('pair_hash:'+str(k))
 if len(other_sigs)>1: failures.append('multiple_source_versions')
 for ck in combos:
  if ck not in AUDITED_COMBOS: failures.append('unaudited_source_combo:'+','.join((x or 'missing')[:12] for x in ck))
 missing=expected-set(observed); extra=set(observed)-expected
 passed=not failures and not missing and not extra
 source_combos=[{'prompt_attn_shr_policy.py':ck[0],'vtcg_factorial_policy.py':ck[1],'episodes':n,'audited':ck in AUDITED_COMBOS} for ck,n in sorted(combos.items(),key=lambda kv:-kv[1])]
 report={'phase':a.phase,'passed':passed,'expected':len(expected),'observed':len(observed),'missing_count':len(missing),'missing':sorted(missing),'extra':sorted(extra),'failures':failures,'paired_cases':len(pairs),'source_versions':len(combos),'source_combos':source_combos,'source_version_note':'2026-10-17 incident split the run into two audited source combinations; bitwise equivalence evidence in SOURCE_PIN_UPDATE.json; any third combination fails this audit (see AUDIT_UPDATE_20261018_SOURCE_VERSIONS.md)','limitation':'initial hashes do not prove full trajectory equivalence; engineering additionally requires same-state policy and video review'}
 out=a.artifact/(a.phase.upper()+'_MEMBERSHIP_AUDIT.json'); out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['missing','extra','failures']},indent=2))
 if not passed: raise SystemExit(1)
if __name__=='__main__': main()
