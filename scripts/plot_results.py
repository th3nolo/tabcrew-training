"""Generate standalone PNG/SVG figures and CSV curves using pinned Matplotlib."""
import csv,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path(__file__).resolve().parents[1];dest=root/'figures';dest.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for name,label,color in [('grpo','GRPO from original','#a55834'),('grpo-sft','GRPO after SFT','#246c4e')]:
    path=root/'results'/name/'learning-curve.jsonl'
    if not path.exists():continue
    rows=[json.loads(l) for l in path.read_text().splitlines() if l];rows=[r for r in rows if 'reward' in r]
    if not rows:continue
    fields=['step','reward','reward_std','frac_reward_zero_std','completions/clipped_ratio','kl']
    with (dest/f'{name}-curve.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r.get(k) for k in fields} for r in rows)
    axes[0].plot([r['step'] for r in rows],[r['reward'] for r in rows],marker='o',color=color,label=label)
    axes[1].plot([r['step'] for r in rows],[r['reward_std'] for r in rows],marker='s',color=color,label=label)
axes[0].set(xlabel='Optimizer step',ylabel='Mean sampled reward',ylim=(-1.05,1.05),title='On-policy reward (different prompts per step)')
axes[1].set(xlabel='Optimizer step',ylabel='Within-group reward standard deviation',title='Available learning signal')
for ax in axes:ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('TabCrew local GRPO pilots · Training curves are not held-out quality')
fig.savefig(dest/'grpo-learning-curves.png',dpi=180);fig.savefig(dest/'grpo-learning-curves.svg');plt.close(fig)
arms=[('final-before','Original'),('final-pure-grpo','GRPO only'),('final-sft','SFT only'),('final-sft-grpo','SFT + GRPO')]
available=[(label,json.loads((root/'results'/path/'summary.json').read_text())) for path,label in arms if (root/'results'/path/'summary.json').exists()]
if available:
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    for ax,field,title in zip(axes,['validity_rate','mean_ari_valid','exact_partition_rate'],['Valid assignments','ARI among valid outputs','Exact reference partition']):
        vals=[s['metrics'][field] or 0 for _,s in available]
        ax.bar(range(len(vals)),vals,color=['#b6c6bc','#a55834','#80a78c','#246c4e'][:len(vals)])
        ax.set_xticks(range(len(vals)),[label.replace(' ','\n') for label,_ in available]);ax.set(title=title,ylabel='Rate' if 'rate' in field else 'Mean ARI',ylim=(min(0,min(vals)-.05),1.08))
        for i,v in enumerate(vals):ax.text(i,v+.035,f'{v:.3f}',ha='center',fontsize=9)
    fig.suptitle('Frozen final synthetic pilot · Single training seed; no human-certified labels')
    fig.savefig(dest/'held-out-results.png',dpi=180);fig.savefig(dest/'held-out-results.svg');plt.close(fig)
print(json.dumps({'figures':[p.name for p in dest.iterdir()]}))
