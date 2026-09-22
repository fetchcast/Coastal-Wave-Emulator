"""Plot completed A configurations and seed-42 pilot anchors without test-based selection."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('csv_file',type=Path)
p.add_argument('--output',type=Path,default=Path('.'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with a.csv_file.open() as f:
    rows=[r for r in csv.DictReader(f) if r['seed']=='42' and ('_A_' in r['config_id'] or '_pilot_' in r['config_id'])]
if len(rows)!=32:
    raise ValueError('Expected 29 A configurations and three seed-42 pilot anchors')
colors={'fno':'#2672AD','ffno':'#009E73','tno':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(10.8,4.7),sharey=True)
for model in ('fno','ffno','tno'):
    data=[r for r in rows if r['model']==model]
    y=np.array([float(r['validation_hs_mae']) for r in data])
    best=int(np.argmin(y))
    for ax,key,scale in zip(axs,['parameters','wallclock_s'],[1e6,3600]):
        x=np.array([float(r[key])/scale for r in data])
        ax.scatter(x,y,s=32,color=colors[model],alpha=.65,label=model.upper(),edgecolors='white',linewidths=.4)
        ax.scatter(x[best],y[best],s=100,marker='*',color=colors[model],edgecolors='black',linewidths=.5,zorder=4)
        cfg=data[best]['config_id'].split('_w')[1].split('_')[0].replace('d','/').replace('m','/')
        offset={'fno':(9,6),'ffno':(-8,-25),'tno':(-8,-24)}[model]
        ax.annotate(f'{model.upper()} {cfg}',(x[best],y[best]),xytext=offset,textcoords='offset points',
                    ha='left' if model=='fno' else 'right',fontsize=9,color=colors[model])
for ax in axs:
    ax.set_xscale('log');ax.grid(alpha=.17,which='major');ax.set_axisbelow(True)
    ax.set_ylim(.050,.082);ax.margins(x=.17)
axs[0].set_xlabel('Parameters (millions; log scale)')
axs[1].set_xlabel('Recorded training wall time (hours; log scale)')
axs[0].set_ylabel('Best validation Hs MAE (m)')
axs[0].set_title('(a) Model size',loc='left')
axs[1].set_title('(b) Observed training cost',loc='left')
axs[0].legend(frameon=False,loc='upper left')
fig.text(.5,.025,'29 A configurations + 3 pilot anchors; seed 42 only. Stars: lowest validation MAE per family. B/C not included.',
         ha='center',fontsize=8)
fig.tight_layout(rect=[0,.06,1,1])
for ext in ('pdf','svg','png'):
    fig.savefig(a.output/f'cost_accuracy_A.{ext}',dpi=200)
plt.close(fig)
