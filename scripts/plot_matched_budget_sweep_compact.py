"""Compact descriptive budget sweep; values parsed from the existing evidence table."""
from pathlib import Path
import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

repo = Path(__file__).resolve().parents[1]
source = repo / 'docs/MATCHED_EVIDENCE_TABLE.md'
section = source.read_text().split('## G1.')[1].split('## G2.')[0]
rows = {}
for line in section.splitlines():
    cells = [s.strip() for s in line.strip().strip('|').split('|')]
    if len(cells) >= 3 and re.fullmatch(r'\d+/\d+', cells[1]):
        s, n = map(int, cells[1].split('/'))
        assert n == 400
        rows[cells[0]] = (s, n, 100*s/n)
x = [16,24,32,48,64]
fixed = [rows[f'fixed {k}'] for k in x]
m = rows['matched']
top_name, top = max(((k,v) for k,v in rows.items() if k.startswith('TopP')), key=lambda kv: kv[1][2])
assert top_name == 'TopP 0.85 (clipped)'
plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'], 'font.size':9,
                     'axes.labelsize':9, 'axes.titlesize':10, 'pdf.fonttype':42, 'ps.fonttype':42,
                     'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(4.7,3.65))
fig.subplots_adjust(left=.15,right=.97,bottom=.19,top=.68)
blue, gray, pink = '#315D92','#687B8C','#BE6883'
ax.plot(x,[v[2] for v in fixed],color=gray,lw=1.5,marker='o',ms=4.5,zorder=3)
ax.axhline(m[2],color=blue,lw=1.6,ls=(0,(5,2.5)),zorder=2)
ax.axhline(top[2],color=pink,lw=1.35,ls=(0,(1.5,2)),zorder=2)
for k,(s,n,r) in zip(x,fixed):
    dy = -12 if k in (32,48,64) else 9
    ax.annotate(f'{r:.2f}%\n{s}/{n}',(k,r),xytext=(0,dy),textcoords='offset points',
                ha='center',va='top' if dy<0 else 'bottom',fontsize=7.5,color='#26333F',linespacing=1.25)
ax.set(xlim=(12,68),ylim=(35,55.5),xticks=x,yticks=[35,40,45,50,55],
       xlabel='Fixed intervention budget $m$',ylabel='Success rate (%)')
ax.spines[['top','right']].set_visible(False)
for side in ['left','bottom']:
    ax.spines[side].set_color('#89919A'); ax.spines[side].set_linewidth(.65)
ax.tick_params(width=.65,color='#89919A',labelsize=8)
ax.grid(axis='y',color='#E5E8EC',lw=.6)
ax.set_axisbelow(True)
fig.text(.5,.97,'(b) Intervention budget sweep',ha='center',va='top',weight='bold',fontsize=10)
fig.text(.5,.91,r'4 Google tasks · seeds 100–199 · $\lambda=0.5$',ha='center',va='top',fontsize=8.3)
handles=[Line2D([0],[0],color=gray,marker='o',ms=4,lw=1.5),
         Line2D([0],[0],color=blue,lw=1.6,ls=(0,(5,2.5))),
         Line2D([0],[0],color=pink,lw=1.35,ls=(0,(1.5,2)))]
fig.legend(handles,['Fixed budget',f'Matched: {m[2]:.2f}% ({m[0]}/{m[1]})',
                    f'Best observed TopP*: {top[2]:.2f}% ({top[0]}/{top[1]})'],
           loc='upper left',bbox_to_anchor=(.16,.865),frameon=False,fontsize=8,
           handlelength=2.8,labelspacing=.4,borderaxespad=0)
fig.text(.15,.045,'* p = 0.85, clipped. Lines show observed success rates.',fontsize=7,color='#555555')
out=repo/'docs/figures'; out.mkdir(exist_ok=True)
for ext in ['png','pdf','svg']:
    fig.savefig(out/f'matched_budget_sweep_compact.{ext}',dpi=300,facecolor='white')
caption = ('Intervention budget sweep on four Google tasks (seeds 100–199, lambda=0.5; '
           '400 episodes per method). Fixed budgets range from 16 to 64 tokens; horizontal '
           'lines denote Matched and the best observed TopP variant in this sweep '
           '(p=0.85, clipped). Matched achieves the highest observed success rate '
           '(213/400, 53.25%), versus 207/400 (51.75%) for the best observed fixed budget '
           'and 206/400 (51.50%) for the best observed TopP variant. Values are descriptive '
           'point estimates, not evidence of significant superiority. These results have a '
           'different evaluation setting/reference from the 48.5% full-method component ablation.\n')
(out/'matched_budget_sweep_compact_caption.txt').write_text(caption)
(out/'matched_budget_sweep_compact_data.json').write_text(json.dumps({'source':str(source),
    'seeds':[100,199],'lambda':.5,'fixed':dict(zip(x,fixed)),'matched':m,
    'best_observed_topp':{'variant':top_name,'values':top}},indent=2)+'\n')
print('Verified denominators: 400; exported PNG, PDF, SVG, caption, and source data.')
