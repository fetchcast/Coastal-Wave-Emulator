"""Analyze physical-unit Hs arrays without treating cells as independent events."""
import csv
import json
from pathlib import Path
import numpy as np

BINS = np.array([-np.inf, 0, 1, 2, 3, 4, 5, 6, np.inf])
LABELS = ['negative', '0_to_1', '1_to_2', '2_to_3', '3_to_4', '4_to_5', '5_to_6', 'ge6']

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows: {path}')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)

def sufficient(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    d = pred - true
    return dict(count=d.size, sae=float(abs(d).sum()), sse=float((d*d).sum()), se=float(d.sum()))

def metrics(s):
    n = s['count']
    return dict(mae=s['sae']/n if n else None,
                rmse=np.sqrt(s['sse']/n) if n else None,
                bias=s['se']/n if n else None)

def detection(pred, true, threshold):
    p, t = pred >= threshold, true >= threshold
    tp, fp, fn = int((p&t).sum()), int((p&~t).sum()), int((~p&t).sum())
    tn = int((~p&~t).sum())
    return dict(threshold_m=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                false_alarm_ratio=fp/(tp+fp) if tp+fp else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None,
                csi=tp/(tp+fp+fn) if tp+fp+fn else None)

def analyze_event(folder, event, model, seed, make_plots=True):
    folder = Path(folder)
    manifest = json.loads((folder/'arrays_complete.json').read_text())
    ids = manifest['indices']
    mask = np.load(folder/'mask.npy').astype(bool)
    rows, bins = [], []
    total_bins = [dict(count=0, sae=0., sse=0., se=0.) for _ in LABELS]
    counters = {v:dict(tp=0, fp=0, fn=0, tn=0) for v in (3., 5.)}
    spatial_abs = np.zeros(mask.shape, dtype=np.float64)
    spatial_bias = np.zeros_like(spatial_abs)
    truth_max = -np.inf
    peak_idx, peak_cell = None, None
    for idx in ids:
        with np.load(folder/'frames'/f'{idx}.npz', allow_pickle=False) as z:
            p, t = z['pred_hs'].astype(float), z['true_hs'].astype(float)
            timestamp = str(z['time'])
        if p.shape != mask.shape or t.shape != mask.shape or not np.isfinite(p[mask]).all() or not np.isfinite(t[mask]).all():
            raise ValueError('Invalid spatial arrays')
        pp, tt = p[mask], t[mask]
        st = sufficient(pp, tt)
        loc = np.unravel_index(np.argmax(np.where(mask, t, -np.inf)), t.shape)
        if t[loc] > truth_max:
            truth_max, peak_idx, peak_cell = float(t[loc]), idx, loc
        r = dict(model=model, seed=seed, event=event['id'], target_index=idx,
                 time=timestamp, **st, **metrics(st), true_mean_hs=float(tt.mean()),
                 true_max_hs=float(tt.max()), pred_max_hs=float(pp.max()))
        rows.append(r)
        spatial_abs[mask] += abs(pp-tt)
        spatial_bias[mask] += pp-tt
        for b, label in enumerate(LABELS):
            select = (tt >= BINS[b]) & (tt < BINS[b+1])
            ss = sufficient(pp[select], tt[select])
            for k in ss:
                total_bins[b][k] += ss[k]
        for threshold in counters:
            dd = detection(pp, tt, threshold)
            for k in counters[threshold]:
                counters[threshold][k] += dd[k]
    peak_row = next(r for r in rows if r['target_index'] == peak_idx)
    peak_time = np.datetime64(peak_row['time'])
    # The peak window is fixed from truth, not selected from model errors.
    for r in rows:
        dt = float((np.datetime64(r['time'])-peak_time)/np.timedelta64(1, 'h'))
        r['hours_from_true_peak'] = dt
        r['phase'] = 'before_peak_window' if dt < -12 else ('after_peak_window' if dt > 12 else 'peak_window_pm12h')
        with np.load(folder/'frames'/f'{r["target_index"]}.npz') as z:
            r['true_at_peak_cell'] = float(z['true_hs'][peak_cell])
            r['pred_at_peak_cell'] = float(z['pred_hs'][peak_cell])
    total = sum(r['count'] for r in rows)
    for label, ss in zip(LABELS, total_bins):
        bins.append(dict(model=model, seed=seed, event=event['id'], bin=label,
                         **ss, **metrics(ss), sample_fraction=ss['count']/total,
                         contribution_to_event_mae=ss['sae']/total))
    all_s = {k:sum(r[k] for r in rows) for k in ('count','sae','sse','se')}
    peak_pred = max(rows, key=lambda r:r['pred_max_hs'])
    local_pred = max(rows, key=lambda r:r['pred_at_peak_cell'])
    summary = dict(model=model, seed=seed, event=event['id'], frames=len(rows), **metrics(all_s),
                   true_peak_hs=truth_max, true_peak_time=peak_row['time'], peak_row=int(peak_cell[0]), peak_col=int(peak_cell[1]),
                   pred_at_true_peak=peak_row['pred_at_peak_cell'],
                   bias_at_true_peak=peak_row['pred_at_peak_cell']-truth_max,
                   domain_peak_bias=peak_pred['pred_max_hs']-truth_max,
                   domain_peak_time_error_h=float((np.datetime64(peak_pred['time'])-peak_time)/np.timedelta64(1,'h')),
                   fixed_cell_peak_time_error_h=float((np.datetime64(local_pred['time'])-peak_time)/np.timedelta64(1,'h')))
    phases = []
    for phase in sorted({r['phase'] for r in rows}):
        rr = [r for r in rows if r['phase']==phase]
        ss = {k:sum(r[k] for r in rr) for k in all_s}
        phases.append(dict(model=model,seed=seed,event=event['id'],phase=phase,frames=len(rr),**ss,**metrics(ss)))
    detect_rows = []
    for threshold, dd in counters.items():
        tp,fp,fn,tn = (dd[k] for k in ('tp','fp','fn','tn'))
        detect_rows.append(dict(model=model,seed=seed,event=event['id'],threshold_m=threshold,**dd,
             recall=tp/(tp+fn) if tp+fn else None, precision=tp/(tp+fp) if tp+fp else None,
             false_alarm_ratio=fp/(tp+fp) if tp+fp else None,
             false_positive_rate=fp/(fp+tn) if fp+tn else None,
             csi=tp/(tp+fp+fn) if tp+fp+fn else None))
    write_csv(folder/'hourly_diagnostics.csv', rows)
    write_csv(folder/'wave_bins.csv', bins)
    write_csv(folder/'phases.csv', phases)
    write_csv(folder/'detection.csv', detect_rows)
    write_csv(folder/'event_metrics.csv', [summary])
    spatial_abs[~mask] = np.nan
    spatial_bias[~mask] = np.nan
    np.savez_compressed(folder/'spatial_errors.npz',mae=spatial_abs/len(rows),bias=spatial_bias/len(rows),mask=mask)
    if make_plots:
        plot_event(folder, rows, peak_idx, mask, spatial_abs/len(rows), model, seed, event)
    return summary

def plot_event(folder, rows, peak_idx, mask, error_map, model, seed, event):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = np.array([r['time'] for r in rows], dtype='datetime64[ns]')
    fig, axes = plt.subplots(3,1,figsize=(11,8),sharex=True)
    for key, label in [('true_at_peak_cell','SWAN'),('pred_at_peak_cell','Emulator')]:
        axes[0].plot(times,[r[key] for r in rows],label=label)
    axes[0].set_ylabel('Hs at true peak cell (m)'); axes[0].legend()
    for key,label in [('true_max_hs','SWAN'),('pred_max_hs','Emulator')]:
        axes[1].plot(times,[r[key] for r in rows],label=label)
    axes[1].set_ylabel('Domain maximum Hs (m)')
    axes[2].plot(times,[r['mae'] for r in rows]);axes[2].set_ylabel('Spatial Hs MAE (m)')
    axes[2].set_xlabel('Time (UTC)')
    fig.suptitle(f'{event["name"]}: {model}, seed {seed}')
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(folder/'time_series.png',dpi=180);plt.close(fig)
    with np.load(folder/'frames'/f'{peak_idx}.npz') as z:
        p,t = z['pred_hs'],z['true_hs']
    hi=max(float(p[mask].max()),float(t[mask].max()))
    diff=p-t;lim=max(float(abs(diff[mask]).max()),1e-6)
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,arr,title,lo,upper,cmap in zip(axes,[t,p,diff,error_map],
          ['SWAN at true peak time','Emulator at same time','Prediction minus SWAN','Event mean absolute error'],
          [0,0,-lim,0],[hi,hi,lim,float(np.nanmax(error_map))],['viridis','viridis','RdBu_r','magma']):
        im=ax.imshow(np.where(mask,arr,np.nan),origin='lower',vmin=lo,vmax=upper,cmap=cmap)
        ax.set_title(title);ax.set_xlabel('Grid column');ax.set_ylabel('Grid row')
        fig.colorbar(im,ax=ax,label='m',shrink=.7)
    fig.tight_layout();fig.savefig(folder/'spatial_comparison.png',dpi=180);plt.close(fig)

def aggregate(root):
    import pandas as pd
    root=Path(root)
    files=sorted(root.glob('models/*/events/*/event_metrics.csv'))
    if not files:
        raise ValueError('No completed event diagnostics')
    for filename,group in [('event_metrics.csv',['model','event']),('wave_bins.csv',['model','event','bin']),
                           ('phases.csv',['model','event','phase']),('detection.csv',['model','event','threshold_m'])]:
        df=pd.concat([pd.read_csv(f.parent/filename) for f in files],ignore_index=True)
        if df.duplicated(group+['seed']).any():raise ValueError('Duplicate seed rows')
        df.to_csv(root/('all_'+filename),index=False)
        cols=[c for c in df.select_dtypes('number').columns if c not in group+['seed']]
        out=df.groupby(group)[cols].agg(['mean','std'])
        out.columns=['_'.join(c) for c in out.columns]
        out['n_seeds']=df.groupby(group).seed.nunique()
        out.reset_index().to_csv(root/('summary_'+filename),index=False)
    # Use common bin weights only where every selected event has support.
    df=pd.read_csv(root/'all_wave_bins.csv')
    standardized=[]
    for (model,seed),g in df.groupby(['model','seed']):
        counts=g.pivot(index='event',columns='bin',values='count').reindex(columns=LABELS)
        maes=g.pivot(index='event',columns='bin',values='mae').reindex(columns=LABELS)
        common=(counts>0).all(axis=0)
        weights=counts.loc[:,common].sum(axis=0)
        weights=weights/weights.sum()
        for event in counts.index:
            standardized.append(dict(model=model,seed=int(seed),event=event,
                standardized_mae=float((maes.loc[event,common]*weights).sum()),
                retained_sample_fraction=float(counts.loc[event,common].sum()/counts.loc[event].sum()),
                common_bins=';'.join(counts.columns[common]),
                interpretation='Exploratory comparison on shared Hs-bin support; not a causal attribution'))
    write_csv(root/'common_bin_standardized_mae.csv',standardized)
