#!/usr/bin/env python3
"""Summarize frozen held-out predictions without selecting new models."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from campaign import atomic, read


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def summarize(rows):
    if not rows:
        return dict(frames=0)
    result = dict(frames=len(rows))
    for v in ('hs','tm'):
        result[v+'_mae'] = float(np.mean([r[v+'_mae'] for r in rows]))
        result[v+'_pooled_rmse'] = float(np.sqrt(np.mean([r[v+'_mse'] for r in rows])))
        result[v+'_mean_frame_rmse'] = float(np.mean([r[v+'_rmse'] for r in rows]))
        result[v+'_bias'] = float(np.mean([r[v+'_bias'] for r in rows]))
    result['dir_circular_mae_deg'] = float(np.mean([r['dir_mae'] for r in rows]))
    result['dir_circular_rmse_deg'] = float(np.sqrt(np.mean([r['dir_mse'] for r in rows])))
    result['dir_radius_lt_0p1_fraction'] = float(np.mean([r['dir_radius_lt_0p1_fraction'] for r in rows]))
    for threshold in (3,5):
        tag = f'hs_ge{threshold}'
        n = sum(r[tag+'_count'] for r in rows)
        result[tag+'_wet_cell_hours'] = int(n)
        result[tag+'_mae'] = sum(r[tag+'_sae'] for r in rows)/n if n else None
        result[tag+'_rmse'] = math.sqrt(sum(r[tag+'_sse'] for r in rows)/n) if n else None
        result[tag+'_bias'] = sum(r[tag+'_se'] for r in rows)/n if n else None
    result['true_tm_max_s'] = max(r['true_tm_max'] for r in rows)
    result['true_tm_gt30_wet_cell_hours'] = int(sum(r['true_tm_gt30_count'] for r in rows))
    result['true_tm_negative_wet_cell_hours'] = int(sum(r['true_tm_negative_count'] for r in rows))
    result['pred_tm_negative_wet_cell_hours'] = int(sum(r['pred_tm_negative_count'] for r in rows))
    return result


def peak_metrics(rows):
    if not rows:
        return {}
    truth = max(rows, key=lambda r:r['true_hs_max'])
    pred = max(rows, key=lambda r:r['pred_hs_max'])
    return dict(true_domain_peak_hs_m=truth['true_hs_max'], pred_domain_peak_hs_m=pred['pred_hs_max'],
                domain_peak_bias_m=pred['pred_hs_max']-truth['true_hs_max'],
                domain_peak_time_error_h=float((np.datetime64(pred['time'])-np.datetime64(truth['time']))/np.timedelta64(1,'h')),
                true_peak_time=truth['time'], pred_peak_time=pred['time'],
                pred_at_true_peak_m=truth['pred_hs_at_true_max'],
                true_peak_location_time_bias_m=truth['pred_hs_at_true_max']-truth['true_hs_max'])


def seed_statistics(rows):
    result = []
    for scope in sorted({r['scope'] for r in rows}):
        for model in ('fno','tno','ffno'):
            group = [r for r in rows if r['scope']==scope and r['model']==model]
            if len(group)!=3 or {r['seed'] for r in group}!={42,43,44}:
                raise ValueError('Three seeds are required for each reported scope')
            out = dict(scope=scope, model=model, seeds=3)
            keys = sorted(set.intersection(*(set(r) for r in group))-{'scope','model','seed','frames','config_id'})
            for k in keys:
                vals = [r[k] for r in group]
                if all(isinstance(v,(int,float)) and np.isfinite(v) for v in vals):
                    out[k+'_mean'] = float(np.mean(vals))
                    out[k+'_seed_sd'] = float(np.std(vals,ddof=1))
            result.append(out)
    return result


def event_plot(root, event, series):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=True)
    colors = dict(fno='tab:blue',tno='tab:orange',ffno='tab:green')
    for model in ('fno','tno','ffno'):
        for seed in (42,43,44):
            rows = series[model,seed]
            rows = [r for r in rows if event['start']<=r['time'][:13]<=event['end']]
            times = np.array([r['time'] for r in rows],dtype='datetime64[ns]')
            label = model.upper() if seed==42 else None
            for ax,key in zip(axes,('pred_hs_max','hs_rmse')):
                ax.plot(times,[r[key] for r in rows],color=colors[model],alpha=.7 if seed==42 else .35,label=label)
            if model=='fno' and seed==42:
                axes[0].plot(times,[r['true_hs_max'] for r in rows],color='black',lw=1.8,label='SWAN')
    axes[0].set_ylabel('Domain maximum Hs (m)')
    axes[1].set_ylabel('Spatial Hs RMSE (m)')
    axes[1].set_xlabel('UTC')
    axes[0].set_title(f'{event["id"]} {event["name"]}: three seeds per model')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend()
    fig.autofmt_xdate();fig.tight_layout()
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'event_{event["id"]}_timeseries.png',dpi=160);plt.close(fig)


def spatial_plot(root, event, target_index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arrays={}
    for model in ('fno','tno','ffno'):
        with np.load(root/'models'/f'{model}_s42'/'snapshots'/f'{target_index}.npz') as z:
            arrays[model]={k:z[k] for k in ('pred','true','kcs','time')}
    for var,ch,unit in [('hs',0,'m'),('tm',1,'s'),('dir',None,'degrees')]:
        values={}
        for model,z in arrays.items():
            if ch is None:
                p=np.degrees(np.arctan2(z['pred'][2],z['pred'][3]))%360
                t=np.degrees(np.arctan2(z['true'][2],z['true'][3]))%360
                e=np.abs((p-t+180)%360-180)
            else:
                p,t=z['pred'][ch],z['true'][ch];e=np.abs(p-t)
            mask=z['kcs']>0
            values[model]=[np.ma.masked_where(~mask,v) for v in (t,p,e)]
        vmax=360 if ch is None else max(float(v.max()) for a in values.values() for v in a[:2])
        vmin=0 if ch is None else min(0,min(float(v.min()) for a in values.values() for v in a[:2]))
        emax=180 if ch is None else max(float(a[2].max()) for a in values.values())
        fig,axs=plt.subplots(3,3,figsize=(12,11),layout='constrained')
        for i,model in enumerate(('fno','tno','ffno')):
            for j,v in enumerate(values[model]):
                im=axs[i,j].imshow(v,origin='lower',interpolation='nearest',
                    cmap=('twilight' if ch is None else 'viridis') if j<2 else 'magma',
                    vmin=vmin if j<2 else 0,vmax=vmax if j<2 else max(emax,1e-12))
                axs[i,j].set_title(f'{model.upper()} '+('SWAN','Prediction','Absolute error')[j])
                axs[i,j].set_xlabel('Grid column');axs[i,j].set_ylabel('Grid row')
                fig.colorbar(im,ax=axs[i,j],shrink=.7,label=unit)
        fig.suptitle(f'{event["id"]} {event["name"]}: {var.upper()}, seed 42, {arrays["fno"]["time"]} UTC\nShared full-range scales; no outlier removal')
        fig.savefig(root/'figures'/f'event_{event["id"]}_{var}_seed42.png',dpi=130);plt.close(fig)


def report(root):
    root=Path(root)
    events=read(root/'events_2021.json')['events']
    selected=read(root/'selected_2021.json')
    rows_out=[];qc=[];series={};reference_truth=None
    for entry in selected:
        job=entry['job'];model,seed=job['model'],job['seed']
        path=root/'models'/f'{model}_s{seed}'/'hourly.csv'
        with path.open() as f:
            rows=[{k:(v if k=='time' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
        rows.sort(key=lambda r:r['target_index'])
        times=np.array([r['time'] for r in rows],dtype='datetime64[ns]')
        expected=np.arange(np.datetime64('2021-01-01T12','h'),np.datetime64('2022-01-01T00','h')).astype('datetime64[ns]')
        if not np.array_equal(times,expected):raise ValueError('Incomplete or duplicated hourly evaluation')
        truth=np.array([[r[k] for k in ('true_hs_max','true_hs_mean','true_tm_max')] for r in rows])
        if reference_truth is None:reference_truth=truth
        elif not np.allclose(truth,reference_truth,rtol=1e-6,atol=1e-6):raise ValueError('Truth differs across selected runs')
        union=np.zeros(len(rows),dtype=bool);ty_union=union.copy();masks=[]
        for e in events:
            mask=(times>=np.datetime64(e['start']))&(times<=np.datetime64(e['end']))
            union|=mask
            if e['lifetime_typhoon']:ty_union|=mask
            masks.append((e['id'],mask,True))
        masks=[('annual',np.ones(len(rows),bool),False),('named_TC_union',union,False),
               ('typhoon_lifetime_union',ty_union,False),('outside_TC_windows',~union,False)]+masks
        for scope,mask,is_event in masks:
            subset=[r for r,yes in zip(rows,mask) if yes]
            output=dict(scope=scope,model=model,seed=seed,config_id=job['config_id'],**summarize(subset))
            if is_event:output.update(peak_metrics(subset))
            rows_out.append(output)
        for row in rows:
            if row['true_tm_gt30_count'] or row['true_tm_negative_count'] or row['pred_tm_negative_count']:
                qc.append(dict(model=model,seed=seed,time=row['time'],true_tm_max_s=row['true_tm_max'],
                    true_tm_gt30_count=row['true_tm_gt30_count'],true_tm_negative_count=row['true_tm_negative_count'],
                    pred_tm_negative_count=row['pred_tm_negative_count']))
        series[model,seed]=rows
    write_csv(root/'event_metrics_by_seed.csv',rows_out)
    write_csv(root/'event_metrics_seed_summary.csv',seed_statistics(rows_out))
    if qc:write_csv(root/'tm_quality_flags.csv',qc)
    else: (root/'tm_quality_flags.csv').write_text('model,seed,time,true_tm_max_s,true_tm_gt30_count,true_tm_negative_count,pred_tm_negative_count\n')
    snapshots={v['event_id']:v['target_index'] for v in read(root/'snapshots.json')}
    for event in events:
        event_plot(root,event,series);spatial_plot(root,event,snapshots[event['id']])
    atomic(root/'analysis_completed.json',dict(events=len(events),runs=len(selected),
        notes=['Outside-event hours are not necessarily calm.',
               'Overlapping events are counted once in union metrics.',
               'High-wave metrics use truth thresholds at wet cell-hours, not a selected model.',
               'Peak timing refers to domain maximum Hs, which may change location.',
               'Tm flags are diagnostic; no labels or predictions were clipped or removed.',
               'Seed SD is not an event-sampling confidence interval.']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    report(p.parse_args().root)
