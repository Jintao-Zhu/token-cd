from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path('artifacts/matched_evidence_figures_v1')
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                     'figure.dpi': 150, 'savefig.dpi': 300, 'axes.grid': True,
                     'grid.alpha': 0.25, 'axes.axisbelow': True})

# 1. Overall paired controls (net = matched success - control success)
controls = [
    ('K16', 62, 1.6e-8), ('K24', 40, 8e-4), ('TopP75', 38, 0.0027),
    ('L14', 24, 0.034), ('+uniform16', 22, 0.043), ('Gaussian', 22, 0.043),
    ('TopP90', 22, 0.065), ('TopP80', 19, 0.099), ('K32', 17, 0.122),
    ('Scale x2.5', 17, 0.097), ('K64', 15, 0.176), ('Shuffle', 15, 0.221),
    ('Patch ablation*', 8, 0.039), ('TopP85', 7, 0.569), ('K48', 6, 0.640),
    ('Corners', -1, 1.0), ('Scale x1.5', -2, 0.86),
]
controls = sorted(controls, key=lambda x: x[1])
labels = [x[0] for x in controls]
nets = [x[1] for x in controls]
ps = [x[2] for x in controls]
colors = ['#2f8f5b' if p < 0.05 else '#e0a43a' if p < 0.10 else '#9aa0a6' for p in ps]
fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(labels))
ax.barh(y, nets, color=colors)
ax.axvline(0, color='black', lw=1)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel('Paired net episodes (Matched - control)')
ax.set_title('Matched versus controls: paired success net')
for i, (v, p) in enumerate(zip(nets, ps)):
    ax.text(v + (0.8 if v >= 0 else -0.8), i, f'{v:+d} (p={p:.3g})',
            va='center', ha='left' if v >= 0 else 'right', fontsize=8)
ax.legend(handles=[Patch(color='#2f8f5b', label='p < 0.05'),
                   Patch(color='#e0a43a', label='p < 0.10'),
                   Patch(color='#9aa0a6', label='p >= 0.10')], loc='lower right')
fig.tight_layout(); fig.savefig(OUT/'fig01_overall_controls_net.png'); plt.close(fig)

# 2. Key arms by task
tasks = ['open', 'close', 'pick', 'move']
arms = {
    'Matched': [44,76,33,60],
    'K32': [40,67,35,54],
    'Shuffle': [35,70,35,58],
    'Entity-TopP': [31,74,38,55],
    'TopP80': [32,68,34,60],
    'Random mask': [19,43,28,51],
}
x = np.arange(len(tasks)); width = 0.13
fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, vals) in enumerate(arms.items()):
    ax.bar(x + (i - (len(arms)-1)/2)*width, vals, width, label=name)
ax.set_xticks(x); ax.set_xticklabels(tasks)
ax.set_ylabel('Success count / 100')
ax.set_title('Key matched-count and position controls by task')
ax.set_ylim(0, 85); ax.legend(ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'fig02_key_arms_by_task.png'); plt.close(fig)

# 3. Fixed-m sweep and matched reference
m = [16,24,32,48,64]
rate = [37.8,43.2,49.0,51.7,49.5]
fig, ax = plt.subplots(figsize=(7,4.5))
ax.plot(m, rate, marker='o', lw=2, color='#2b6cb0', label='L11 fixed m')
ax.axhline(53.2, color='#c0392b', ls='--', lw=2, label='L11+Matched overall')
for xx, yy in zip(m, rate): ax.annotate(f'{yy:.1f}%', (xx,yy), textcoords='offset points', xytext=(0,7), ha='center')
ax.set_xlabel('Fixed mask count m'); ax.set_ylabel('Success rate (%)')
ax.set_title('Fixed-count response curve')
ax.set_ylim(32,56); ax.legend()
fig.tight_layout(); fig.savefig(OUT/'fig03_fixed_count_curve.png'); plt.close(fig)

# 4. Mechanism correlations heatmap
metrics = ['bbox area','mean spatial distance','density','cluster intra-dist','semantic centroid cos','cluster size rank']
data = np.array([
 [0.831,0.899,0.804,0.871],
 [0.856,0.861,0.677,0.780],
 [0.321,0.205,-0.308,-0.466],
 [0.551,0.775,0.398,0.183],
 [-0.499,-0.493,-0.328,-0.103],
 [-0.898,-0.940,-0.924,-0.705],
])
fig, ax = plt.subplots(figsize=(7.5,5))
im = ax.imshow(data, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(4)); ax.set_xticklabels(tasks)
ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics)
for i in range(data.shape[0]):
 for j in range(data.shape[1]):
  ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',fontsize=8)
ax.set_title('What matched m correlates with (Spearman)')
fig.colorbar(im, ax=ax, shrink=0.8, label='rho')
fig.tight_layout(); fig.savefig(OUT/'fig04_budget_mechanism_heatmap.png'); plt.close(fig)

# 5. Dose response by matched-m tercile
r = [0.5,0.75,1.0,1.25,1.5]
curves = {
 'D_feat': {'Small':[0.1483,0.1821,0.2116,0.2374,0.2616], 'Middle':[0.2190,0.2700,0.3157,0.3572,0.3946], 'Large':[0.2650,0.3301,0.3857,0.4352,0.4803]},
 'D_action': {'Small':[0.2531,0.2946,0.3212,0.3419,0.3522], 'Middle':[0.3195,0.3492,0.3715,0.3770,0.3798], 'Large':[0.3979,0.4134,0.4219,0.4251,0.4309]},
 'D_support': {'Small':[2.6250,3.2826,3.8267,4.3059,4.6461], 'Middle':[3.8503,4.6039,5.1860,5.5755,5.8166], 'Large':[5.7227,6.4919,6.7769,6.8940,7.0887]},
}
fig, axes = plt.subplots(1,3,figsize=(13,4))
for ax,(metric, groups) in zip(axes, curves.items()):
 for name, vals in groups.items(): ax.plot(r, vals, marker='o', label=name)
 ax.axvline(1.0, color='black', ls='--', alpha=.5)
 ax.set_title(metric); ax.set_xlabel('r = m / m_matched'); ax.set_ylabel(metric)
ax.legend(); fig.suptitle('Matched-relative intervention response by state size', y=1.03)
fig.tight_layout(); fig.savefig(OUT/'fig05_dose_response_by_size.png', bbox_inches='tight'); plt.close(fig)

# 6. Shuffle mismatch bins
bins = ['D < 20%','20-40%','D >= 40%']; n = [11,126,263]
matched_only=[2,25,46]; shuffle_only=[1,8,49]; net=[a-b for a,b in zip(matched_only,shuffle_only)]
fig, ax = plt.subplots(figsize=(7,4.5))
xx=np.arange(3); w=.35
ax.bar(xx-w/2, matched_only, w, label='Matched-only')
ax.bar(xx+w/2, shuffle_only, w, label='Shuffle-only')
for i,v in enumerate(net): ax.text(xx[i], max(matched_only[i],shuffle_only[i])+1, f'net {v:+d}', ha='center')
ax.set_xticks(xx); ax.set_xticklabels([f'{b}\n(n={c})' for b,c in zip(bins,n)])
ax.set_ylabel('Discordant episodes')
ax.set_title('Matched-shuffle outcome by budget mismatch')
ax.legend(); fig.tight_layout(); fig.savefig(OUT/'fig06_shuffle_mismatch_bins.png'); plt.close(fig)

# 7. Position/count factorial (full 400-episode arms)
arms7 = [
 ('L11+Matched',213),('L11+TopP85',206),('L11+K48',207),('L11+Shuffle',198),
 ('L11+Entity',198),('L11+K64',198),('L11+K32',196),('L11+TopP80',194),
 ('KMeans+Matched',193),('Random+Matched',141)
]
arms7=sorted(arms7,key=lambda x:x[1])
fig,ax=plt.subplots(figsize=(9,5))
names=[a[0] for a in arms7]; vals=[a[1] for a in arms7]
colors=['#2f8f5b' if 'L11+' in a[0] else '#e0a43a' if 'KMeans' in a[0] else '#c0392b' for a in arms7]
ax.barh(range(len(names)), vals, color=colors)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlabel('Success count / 400'); ax.set_xlim(0,230)
for i,v in enumerate(vals): ax.text(v+3,i,str(v),va='center')
ax.set_title('Position and count factorial: full paired arms')
ax.legend(handles=[Patch(color='#2f8f5b',label='L11 position'),Patch(color='#e0a43a',label='KMeans position'),Patch(color='#c0392b',label='Random position')],loc='lower right')
fig.tight_layout(); fig.savefig(OUT/'fig07_position_count_factorial.png'); plt.close(fig)

# 8. Phase A knee correlation
tasks4=['open','close','pick','move']; vals=[0.267,-0.225,0.128,0.002]
fig,ax=plt.subplots(figsize=(6,4))
colors=['#2f8f5b' if v>0 else '#c0392b' for v in vals]
ax.bar(tasks4,vals,color=colors); ax.axhline(0,color='black',lw=1)
for i,v in enumerate(vals): ax.text(i,v+(0.015 if v>=0 else -0.025),f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top')
ax.set_ylabel('Spearman rho'); ax.set_title('Matched m vs effective knee')
fig.tight_layout(); fig.savefig(OUT/'fig08_phaseA_knee.png'); plt.close(fig)

print(OUT)
