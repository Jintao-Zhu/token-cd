"""Reproduce user-supplied component ablations + verified budget sweep.
Panel a transcribes Table 7 in the supplied screenshot, not episode-level revalidation.
Panel b parses the existing evidence table. No invented error bars or significance marks.
"""
from pathlib import Path
import re,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
repo=Path(__file__).resolve().parents[1]
out=repo/'docs/figures'; out.mkdir(exist_ok=True)
section=(repo/'docs/MATCHED_EVIDENCE_TABLE.md').read_text().split('## G1.')[1].split('## G2.')[0]
rows={}
for line in section.splitlines():
    c=[v.strip() for v in line.strip().strip('|').split('|')]
    if len(c)>=3 and re.fullmatch(r'\d+/\d+',c[1]):
        s,n=map(int,c[1].split('/')); assert n==400
        rows[c[0]]=(s,n,100*s/n)
xs=[16,24,32,48,64]; fixed=[rows[f'fixed {k}'] for k in xs]
matched=rows['matched']
topname,top=max(((k,v) for k,v in rows.items() if k.startswith('TopP')),key=lambda kv:kv[1][2])
assert topname=='TopP 0.85 (clipped)'
groups=[('Ranking','#ECA4B8', [('Random',16.25),('KMeans',10.25),('Geometry',9.25)]),
        ('Budget','#8AB1D1',[(r'Shuffle-$m$',9.00),('Fixed-32',7.75)]),
        ('Counterfactual','#D77E9B',[('Local mean',9.00),('Zero',4.75),('Global mean',4.25)]),
        ('Intervention point','#5F83B3',[('L11',13.00),('L5',7.50)])]
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8.5,
                     'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(9.2,4.65))
a=fig.add_axes([.175,.19,.295,.61]); b=fig.add_axes([.615,.19,.365,.61])
for ax in [a,b]:
    ax.spines[['top','right']].set_visible(False)
    for side in ['left','bottom']:
        ax.spines[side].set_color('#929AA2'); ax.spines[side].set_linewidth(.6)
    ax.tick_params(color='#929AA2',width=.6,labelsize=8)
    ax.set_axisbelow(True)
# Original horizontal layout with reference-inspired pastel fills and thin outlines.
positions=[]; labels=[]; recorded=[]; y=0
palette=['#DDA3A4','#ABBAD1','#C0D4CF']
for name,color,variants in groups:
    a.text(-.49,y,name,transform=a.get_yaxis_transform(),fontsize=8.1,fontweight='bold',va='center',clip_on=False)
    y+=.9
    for j,(label,drop) in enumerate(variants):
        a.barh(y,drop,height=.56,color=palette[j],edgecolor='#687079',linewidth=.6,zorder=3)
        a.text(drop+.3,y,f'{drop:.2f}',va='center',fontsize=7.5,color='#3C4248')
        positions.append(y); labels.append(label)
        recorded.append({'component':name,'variant':label,'drop_pp':drop})
        y+=.92
    y+=.35
assert len(recorded)==10
a.set(yticks=positions,yticklabels=labels,xlim=(0,18.8),xticks=[0,5,10,15],ylim=(y-.2,-.8),
      xlabel='Success-rate drop vs. full (pp)')
a.tick_params(axis='y',length=0,pad=7,labelsize=8)
for spine in a.spines.values():
    spine.set_visible(True); spine.set_color('#565D64'); spine.set_linewidth(.65)
a.grid(axis='x',color='#E1E4E7',lw=.6)
fig.text(.285,.96,'(a) Component ablations',ha='center',va='top',weight='bold',fontsize=10.5)
fig.text(.285,.88,'Full method: 48.5%',ha='center',va='top',fontsize=8.2,color='#444A50')
blue,gray,pink='#45688E','#658F89','#BF828C'
# Subtle panel tint, pastel-filled markers, and distinct line styles; no uncertainty shading.
b.set_facecolor('#FAFBFC')
for spine in b.spines.values():
    spine.set_visible(True); spine.set_color('#687580'); spine.set_linewidth(.65)
b.plot(xs,[v[2] for v in fixed],color=gray,lw=2.1,zorder=4,solid_capstyle='round')
b.scatter(xs,[v[2] for v in fixed],s=53,facecolor='#C0D4CF',edgecolor='white',linewidth=2,zorder=5)
b.scatter(xs,[v[2] for v in fixed],s=53,facecolor='none',edgecolor=gray,linewidth=.85,zorder=6)
b.axhline(matched[2],color=blue,lw=1.8,ls=(0,(6,3)),zorder=3)
b.axhline(top[2],color=pink,lw=1.6,ls=(0,(1.4,2.3)),zorder=3)
offsets={16:(0,-13),24:(-9,12),32:(9,-14),48:(0,-15),64:(0,-14)}
for x,(s,n,r) in zip(xs,fixed):
    dx,dy=offsets[x]
    b.annotate(f'{r:.2f}%',(x,r),xytext=(dx,dy),textcoords='offset points',
               ha='center',va='top' if dy<0 else 'bottom',fontsize=7.8,color='#345B56',
               bbox=dict(boxstyle='round,pad=.22',facecolor='white',edgecolor='#D7E4E1',linewidth=.5),zorder=7)
b.set(xlim=(11,69),ylim=(35,55.5),xticks=xs,yticks=[35,40,45,50,55],
      xlabel='Fixed intervention budget $m$',ylabel='Success rate (%)')
b.grid(axis='y',color='#E7E9EC',lw=.55)
fig.text(.7975,.96,'(b) Intervention budget sweep',ha='center',va='top',weight='bold',fontsize=10.5)
# Evaluation configuration is retained in the caption rather than inside the figure.
handles=[Line2D([0],[0],color=gray,marker='o',markerfacecolor='#C0D4CF',markeredgecolor=gray,ms=5,lw=2.1),
         Line2D([0],[0],color=blue,lw=1.8,ls=(0,(6,3))),
         Line2D([0],[0],color=pink,lw=1.6,ls=(0,(1.4,2.3)))]
fig.legend(handles,['Fixed budget','Matched: 53.25%',
                     'Best observed TopP: 51.50%'],
           loc='upper left',bbox_to_anchor=(.615,.91),frameon=False,fontsize=7.6,
           handlelength=2.3,labelspacing=.25,borderaxespad=0)
# Detailed notes, denominator, and TopP configuration are in the companion caption.
for ext in ['png','pdf','svg']:
    fig.savefig(out/f'ablation_budget_combined.{ext}',dpi=350,facecolor='white',bbox_inches='tight',pad_inches=.08)
caption=('Figure X. Component ablations and intervention-budget comparison. '
'(a) Success-rate drops relative to the full method (48.5%) on 400 seed-matched episodes; '
'bars are grouped by ablated component, with pastel colours distinguishing variants within each group. Values are reproduced from the supplied ablation table; error bars are not shown because corresponding uncertainty estimates are unavailable. '
'(b) Fixed-budget sweep over 16–64 tokens on four Google tasks (seeds 100–199, lambda=0.5; '
'400 episodes per method). Horizontal lines show Matched and the best observed TopP variant '
'(p=0.85, clipped). Matched achieves the highest observed success rate in this sweep. '
'The two panels use separate evaluation settings and reference success rates. '
'Panel (b) reports descriptive point estimates, not significant superiority.\n')
(out/'ablation_budget_combined_caption.txt').write_text(caption)
(out/'ablation_budget_combined_data.json').write_text(json.dumps({'panel_a':{'source':'User-supplied screenshot, Table 7','full_success_percent':48.5,'n':400,'rows':recorded},'panel_b':{'source':'docs/MATCHED_EVIDENCE_TABLE.md, G1','fixed':dict(zip(xs,fixed)),'matched':matched,'best_observed_topp':{'name':topname,'values':top}}},indent=2)+'\n')
print('Exported combined PNG/PDF/SVG, caption, and provenance JSON. 10 ablations; 5 fixed budgets.')
