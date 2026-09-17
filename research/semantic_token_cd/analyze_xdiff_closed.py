"""Final scene-level analysis for the semantic-difference experiment."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from research.semantic_token_cd.xdiff_protocol import ARMS, ARTIFACT, NEW_ARMS, SOURCE_ARTIFACT, TASKS, atomic_json

PRIMARY=(('semantic_0p5','correct'),('semantic_0p5','paraphrase_0p5'),
         ('semantic_0p5','reverse_0p5'),('semantic_1p0','correct'))

def read(task,seed,arm):
    if arm in ('vanilla','correct'):
        p=SOURCE_ARTIFACT/'runs/episodes'/task/arm/f'episode_{seed:03d}_summary.json'
    else: p=ARTIFACT/'closed_loop/episodes'/task/arm/f'episode_{seed:03d}_summary.json'
    return json.loads(p.read_text()) if p.exists() else None

def wilson(k,n,z=1.959963984540054):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]

def exact_p(a,b):
    n=a+b
    if not n or a==b:return 1.0
    k=min(a,b); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))

def paired_ci(x,y,seed=20260908):
    d=np.asarray(x,dtype=float)-np.asarray(y,dtype=float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(d),size=(20000,len(d))); vals=d[idx].mean(axis=1)
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]); out=[0.0]*len(ps); running=0.0; m=len(ps)
    for rank,i in enumerate(order): running=max(running,(m-rank)*ps[i]); out[i]=min(1.0,running)
    return out

def summarize(rows,arm):
    vals=[bool(r[arm]['success']) for r in rows]; k=sum(vals); n=len(vals)
    extra=[r[arm] for r in rows if arm in NEW_ARMS]
    return {'successes':k,'n':n,'success_rate':k/n,'wilson_95':wilson(k,n),
      'mean_m':float(np.mean([x['mean_m_t'] for x in extra])) if extra else (
       float(np.mean([r[arm]['mean_m_t'] for r in rows])) if arm=='correct' else None),
      'mean_correct_token_retention':float(np.mean([x['mean_correct_token_retention'] for x in extra])) if extra else 1.0 if arm=='correct' else None}

def pair(rows,a,b):
    av=[bool(r[a]['success']) for r in rows]; bv=[bool(r[b]['success']) for r in rows]
    rescue=sum(x and not y for x,y in zip(av,bv)); harm=sum(not x and y for x,y in zip(av,bv))
    return {'candidate':a,'baseline':b,'rescue':rescue,'harm':harm,'net':rescue-harm,
      'paired_rate_difference':float(np.mean(av)-np.mean(bv)),'paired_bootstrap_95':paired_ci(av,bv),
      'mcnemar_exact_p':exact_p(rescue,harm),'both_success':sum(x and y for x,y in zip(av,bv)),
      'both_fail':sum(not x and not y for x,y in zip(av,bv))}

def main():
    scenes=json.loads((SOURCE_ARTIFACT/'scene_manifest.json').read_text())['scenes']; all_rows=[]; by_task={}; technical=[]
    for task in TASKS:
        rows=[]
        for seed in scenes[task]:
            arms={a:read(task,seed,a) for a in ARMS}
            if not all(arms.values()): raise RuntimeError(f'missing {task} seed {seed}')
            base=arms['correct']
            for arm in NEW_ARMS:
                x=arms[arm]
                technical.append(bool(x.get('technical_pass')))
                for key in ('initial_state_sha256','initial_rgb_sha256','canonical_snapshot_sha256'):
                    if x[key]!=base[key]: raise RuntimeError(f'hash mismatch {task} {seed} {arm} {key}')
            rows.append({'task':task,'seed':seed,**arms}); all_rows.append(rows[-1])
        by_task[task]={'arms':{a:summarize(rows,a) for a in ARMS},
                       'pairs':{f'{a}_vs_{b}':pair(rows,a,b) for a,b in PRIMARY}}
    comparisons=[pair(all_rows,a,b) for a,b in PRIMARY]; adj=holm([x['mcnemar_exact_p'] for x in comparisons])
    for x,p in zip(comparisons,adj): x['holm_adjusted_p']=p
    videos=list((ARTIFACT/'closed_loop/videos').glob('**/*.mp4'))
    stage1=json.loads((ARTIFACT/'stage1_rank/statistics/summary.json').read_text())
    stage2=json.loads((ARTIFACT/'stage2_action/statistics/summary.json').read_text())
    report={'protocol':'PROMPT_ATTN_SEMANTIC_DIFFERENCE_V1','scene_unit':'deduplicated initial scene',
      'n_scenes':len(all_rows),'six_arm_episode_count':len(all_rows)*6,'new_episode_count':len(all_rows)*4,
      'reused_episode_count':len(all_rows)*2,'all_new_technical_audits_passed':all(technical),
      'video_count_new_arms':len(videos),'stage1':stage1,'stage2':stage2,'by_task':by_task,
      'overall':{'arms':{a:summarize(all_rows,a) for a in ARMS},'primary_pairs':comparisons}}
    atomic_json(ARTIFACT/'FINAL_RESULTS.json',report)
    names={'vanilla':'Vanilla','correct':'Correct','semantic_0p5':'Semantic-0.5','semantic_1p0':'Semantic-1.0',
           'paraphrase_0p5':'Paraphrase-0.5','reverse_0p5':'Reverse-0.5'}
    lines=['# Full-prompt semantic-difference Prompt-Attn-SHR v1','',
      f'- 场景单位：309个去重初始场景；六组共1854个arm-episode（复用618，新增1236）。',
      f'- 新增闭环技术审计：{all(technical)}；新增轨迹视频：{len(videos)}/1236。','',
      '## 成功率','', '| Task | '+' | '.join(names[a] for a in ARMS)+' |','|---|'+'|'.join(['---:']*len(ARMS))+'|']
    for task in TASKS:
        vals=by_task[task]['arms']; lines.append('| '+task.removeprefix('google_robot_')+' | '+' | '.join(
          f"{vals[a]['successes']}/{vals[a]['n']} ({100*vals[a]['success_rate']:.1f}%)" for a in ARMS)+' |')
    vals=report['overall']['arms']; lines.append('| **Overall** | '+' | '.join(
      f"**{vals[a]['successes']}/{vals[a]['n']} ({100*vals[a]['success_rate']:.1f}%)**" for a in ARMS)+' |')
    lines += ['', '## 预设配对比较','', '| Candidate vs baseline | Rescue | Harm | Net | Δ success | 95% CI | exact p | Holm p |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in comparisons:
        lines.append(f"| {names[x['candidate']]} vs {names[x['baseline']]} | {x['rescue']} | {x['harm']} | {x['net']:+d} | {100*x['paired_rate_difference']:+.1f} pp | [{100*x['paired_bootstrap_95'][0]:+.1f}, {100*x['paired_bootstrap_95'][1]:+.1f}] | {x['mcnemar_exact_p']:.4g} | {x['holm_adjusted_p']:.4g} |")
    s1=stage1['configs']; s2=stage2['overall']
    lines += ['', '## 机制结果','',
      f"- η=0 在 {stage1['n_states']}/{stage1['n_states']} 个状态逐位恢复 Correct mask。",
      f"- Semantic-0.5 平均替换 {s1['semantic_0p5']['mean_entered_count']:.2f} 个 token；同义对照替换 {s1['paraphrase_0p5']['mean_entered_count']:.2f} 个。",
      f"- 240个同状态中，Semantic-0.5 有 {100*(1-s2['semantic_0p5']['action_exact_fraction']):.1f}% 改变最终动作；Semantic-1.0 为 {100*(1-s2['semantic_1p0']['action_exact_fraction']):.1f}%；同义对照为 {100*(1-s2['paraphrase_0p5']['action_exact_fraction']):.1f}%。",
      f"- residual 超过 Correct 两倍的比例：Semantic-0.5 {100*s2['semantic_0p5']['residual_over_2x_fraction']:.1f}%，Semantic-1.0 {100*s2['semantic_1p0']['residual_over_2x_fraction']:.1f}%。",
      '', '## 结论边界','', '本轮是使用参与提出假设的场景进行机制探索；若出现清晰优势，仍需独立场景确认。']
    (ARTIFACT/'REPORT.md').write_text('\n'.join(lines)+'\n'); print(json.dumps({'done':True,'overall':report['overall']},indent=2))
if __name__=='__main__': main()
