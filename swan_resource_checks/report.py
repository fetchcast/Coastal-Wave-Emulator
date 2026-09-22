"""Export measured resource/sensitivity tables and per-seed comparison figures."""
import argparse,csv
from pathlib import Path
from common import read

def write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(root):
    rows=[]
    for p in sorted((root/'resources').glob('*/result.json')):
        r=read(p);rows.append({k:v for k,v in r.items() if not isinstance(v,(list,dict))})
    write(root/'resources.csv',rows)
    rr=[]
    for p in sorted((root/'robustness').glob('*/result.json')):rr.extend(read(p))
    write(root/'robustness.csv',rr)
    pilots=[]
    for p in sorted((root/'pilot_training').glob('*/run_summary.json')):
        r=read(p);a=r.get('repair_audit',{});j=r.get('repair_job',{})
        if not r.get('failed') and a.get('updates')==a.get('target_updates') and j:
            pilots.append(dict(model=r['model'],seed=j['seed'],lr=j['hyperparams']['max_lr'],updates=a['updates'],validation_hs_mae=a['best_val_hs_mae'],schedule='short OneCycle, exploratory'))
    write(root/'pilot_validation_only.csv',pilots)
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    cohorts={(r['gpu'],r['precision']) for r in rows}
    if len(cohorts)!=1:raise ValueError('Mixed device/precision cohorts; CSVs saved, compare cohorts separately')
    models=sorted(set(r['model'] for r in rows));colors=plt.get_cmap('tab10')
    for n,m in enumerate(models):
        rr=[r for r in rows if r['model']==m]
        for ax,key in zip(axes,('peak_allocated_GiB','forward_median_ms')):
            ax.scatter([r[key] for r in rr],[r['annual_hs_mae'] for r in rr],label=m,color=colors(n%10))
            ax.set_ylabel('2021 annual Hs MAE (m)');ax.grid(alpha=.2)
    axes[0].set_xlabel('Peak inference allocated memory (GiB)')
    axes[1].set_xlabel('Median forward time (ms/frame)');axes[1].set_xscale('log')
    axes[1].legend(fontsize=8)
    fig.suptitle('Frozen configurations; each point is one seed; batch 1 / FP32')
    fig.tight_layout();fig.savefig(root/'resource_accuracy.png',dpi=180);fig.savefig(root/'resource_accuracy.pdf');plt.close(fig)
    # Per-event clean high-wave error from the same sampled hours as robustness.
    rr=[]
    for p in (root/'robustness').glob('*/result.json'):rr.extend(read(p))
    lookup={(r['model'],r['seed']):r for r in rows}
    clean=[r for r in rr if r['condition']=='clean' and r['hs_ge5_mae'] is not None and (r['model'],r['seed']) in lookup]
    if clean:
        events=sorted({r['event'] for r in clean});fig,axes=plt.subplots(1,len(events),figsize=(5*len(events),4),squeeze=False)
        for ax,event in zip(axes[0],events):
            for n,m in enumerate(models):
                selected=[r for r in clean if r['event']==event and r['model']==m]
                ax.scatter([lookup[m,r['seed']]['peak_allocated_GiB'] for r in selected],[r['hs_ge5_mae'] for r in selected],label=m,color=colors(n%10))
            ax.set_title('Event '+event+' (sampled hours)');ax.set_xlabel('Peak inference allocated memory (GiB)');ax.set_ylabel('Hs MAE where truth >= 5 m (m)')
        axes[0,-1].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'resource_high_wave.png',dpi=180);plt.close(fig)
    if rr:
        for event in sorted({r['event'] for r in rr}):
            fig,axes=plt.subplots(1,4,figsize=(16,4))
            for ax,kind in zip(axes,('wind','boundary_hs','boundary_direction','boundary_delay')):
                for n,m in enumerate(models):
                    selected=[r for r in rr if r['event']==event and r['model']==m and r['condition']==kind]
                    levels=sorted({r['level'] for r in selected})
                    if not levels:continue
                    means=[sum(r['delta_mae'] for r in selected if r['level']==v)/sum(r['level']==v for r in selected) for v in levels]
                    ax.plot(levels,means,marker='o',label=m,color=colors(n%10))
                ax.axhline(0,color='gray',lw=.7);ax.set_title(kind);ax.set_ylabel('MAE increase vs clean (m)')
                ax.set_xlabel('Scale fraction' if kind in ('wind','boundary_hs') else 'Degrees' if kind=='boundary_direction' else 'Delay (hours)')
            handles,labels=axes[0].get_legend_handles_labels()
            if handles:fig.legend(handles,labels,fontsize=7,loc='upper right')
            fig.suptitle('Event '+event+': sampled hours; mean across available seeds')
            fig.tight_layout();fig.savefig(root/('robustness_'+event+'.png'),dpi=180);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
