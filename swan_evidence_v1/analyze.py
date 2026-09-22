#!/usr/bin/env python3
"""Read completed v3 results; never train or select a model."""
import argparse, csv, hashlib, itertools, json
from pathlib import Path
import numpy as np
MODELS=('fno','ffno','tno'); SEEDS=(42,43,44)
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(p,rows):
    if not rows: return
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)
def exact_p(d):
    """Paired sign randomization, conditional on exchangeability."""
    d=np.asarray(d,float)
    if not 0<len(d)<=16 or not np.isfinite(d).all(): raise ValueError('Invalid pairs')
    values=[abs(np.mean(d*s)) for s in itertools.product((-1,1),repeat=len(d))]
    return float(np.mean(np.array(values)>=abs(d.mean())-1e-14))
def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return out
def seed_tests(rows):
    result=[]
    for scope in sorted({r['scope'] for r in rows}):
        for metric in ('hs_mae','hs_pooled_rmse','hs_ge3_mae','hs_ge5_mae'):
            for a,b in itertools.combinations(MODELS,2):
                groups=[]
                for m in (a,b):
                    rr=sorted([r for r in rows if r['scope']==scope and r['model']==m],key=lambda r:int(r['seed']))
                    if [int(r['seed']) for r in rr]!=list(SEEDS): raise ValueError(f'Missing/duplicate seeds: {scope}/{m}')
                    groups.append(rr)
                if any(r.get(metric,'')=='' for g in groups for r in g): continue
                x,y=[np.array([float(r[metric]) for r in g]) for g in groups]
                if not np.isfinite([x,y]).all(): raise ValueError('Nonfinite metric')
                d=x-y
                result.append(dict(scope=scope,metric=metric,model_a=a,model_b=b,n_seed_pairs=len(d),
                    a_mean=x.mean(),a_seed_sd=x.std(ddof=1),b_mean=y.mean(),b_seed_sd=y.std(ddof=1),
                    difference_a_minus_b=d.mean(),difference_seed_sd=d.std(ddof=1),p_exact=exact_p(d),
                    interpretation='Exploratory; negative difference favors A; nonsignificance is not equivalence'))
    for r,p in zip(result,holm([r['p_exact'] for r in result])): r['p_holm_all_report_tests']=p
    return result
def moving_block_ci(d,block,repetitions,rng):
    """Non-circular overlapping moving blocks of paired differences."""
    n=len(d)
    if block>n: raise ValueError('Block longer than series')
    means=np.empty(repetitions)
    for i in range(repetitions):
        starts=rng.integers(0,n-block+1,size=(n+block-1)//block)
        indices=(starts[:,None]+np.arange(block)).ravel()[:n]
        means[i]=d[indices].mean()
    return np.quantile(means,[.025,.975])
def load_hourly(root):
    selected=read(root/'selected_2021.json'); keys=[(v['job']['model'],int(v['job']['seed'])) for v in selected]
    if len(keys)!=9 or set(keys)!=set(itertools.product(MODELS,SEEDS)): raise ValueError('Expected nine frozen selections')
    expected=np.arange(np.datetime64('2021-01-01T12','h'),np.datetime64('2022-01-01','h'))
    series={}; truth=None
    for m,s in keys:
        with (root/'models'/f'{m}_s{s}'/'hourly.csv').open() as f: rr=list(csv.DictReader(f))
        times=np.array([r['time'] for r in rr],dtype='datetime64[ns]')
        if not np.array_equal(times,expected): raise ValueError(f'Incomplete/unsorted hourly data: {m}/{s}')
        t=np.array([[float(r[k]) for k in ('true_hs_mean','true_hs_max')] for r in rr])
        if not np.isfinite(t).all(): raise ValueError('Nonfinite truth')
        if truth is not None and not np.allclose(t,truth,rtol=1e-6,atol=1e-6): raise ValueError('Truth mismatch')
        truth=t; series[m,s]=np.array([float(r['hs_mae']) for r in rr])
        if not np.isfinite(series[m,s]).all(): raise ValueError('Nonfinite MAE')
    return series
def analyze(a):
    root=a.eval_root; out=a.output; out.mkdir(parents=True,exist_ok=False)
    marker=root/'analysis_completed.json'
    if not marker.exists():
        (out/'STATUS.json').write_text(json.dumps(dict(status='WAITING',reason='v3 held-out evaluation/report must finish first.'),indent=2))
        print('WAITING: 2021 evaluation incomplete. No significance results generated.'); return
    series=load_hourly(root)
    with (root/'event_metrics_by_seed.csv').open() as f: rows=list(csv.DictReader(f))
    expected_scopes={'annual','named_TC_union','typhoon_lifetime_union','outside_TC_windows'} | {e['id'] for e in read(root/'events_2021.json')['events']}
    if {r['scope'] for r in rows}!=expected_scopes: raise ValueError('Missing or unexpected event scopes')
    configs={(e['job']['model'],int(e['job']['seed'])):e['job']['config_id'] for e in read(root/'selected_2021.json')}
    for r in rows:
        if r['config_id']!=configs[r['model'],int(r['seed'])]: raise ValueError('Selected configuration mismatch')
        if r['scope']=='annual' and not np.isclose(float(r['hs_mae']),series[r['model'],int(r['seed'])].mean()): raise ValueError('Stale event metrics')
    write_csv(out/'seed_comparisons.csv',seed_tests(rows)); write_csv(out/'typhoon_and_annual_by_seed.csv',rows)
    summary=[]
    for scope in sorted({r['scope'] for r in rows}):
        for m in MODELS:
            group=[r for r in rows if r['scope']==scope and r['model']==m]
            for metric in ('hs_mae','hs_pooled_rmse','hs_ge3_mae','hs_ge5_mae','domain_peak_bias_m','domain_peak_time_error_h','true_peak_location_time_bias_m'):
                if not all(r.get(metric,'')!='' for r in group): continue
                vals=np.array([float(r[metric]) for r in group])
                summary.append(dict(scope=scope,model=m,metric=metric,mean=vals.mean(),seed_sd=vals.std(ddof=1),n_seeds=len(vals)))
    write_csv(out/'typhoon_and_annual_summary.csv',summary)
    block_rows=[]; rng=np.random.default_rng(20260917)
    for x,y in itertools.combinations(MODELS,2):
        d=np.mean([series[x,s]-series[y,s] for s in SEEDS],axis=0)
        for block in (24,72,168):
            lo,hi=moving_block_ci(d,block,a.bootstrap,rng)
            block_rows.append(dict(model_a=x,model_b=y,metric='annual_hs_mae',block_hours=block,
                difference_a_minus_b=d.mean(),conditional_ci95_low=lo,conditional_ci95_high=hi,
                interpretation='Exploratory; conditional on these trained models and year; not new-region evidence'))
    write_csv(out/'temporal_block_sensitivity.csv',block_rows)
    paths=[marker,root/'selected_2021.json',root/'events_2021.json',root/'event_metrics_by_seed.csv',*root.glob('models/*/hourly.csv')]
    (out/'provenance.json').write_text(json.dumps(dict(input_root=str(root.resolve()),sha256={str(p.relative_to(root)):sha(p) for p in paths},code_sha256=sha(__file__),bootstrap_repetitions=a.bootstrap),indent=2))
    (out/'STATUS.json').write_text(json.dumps(dict(status='COMPLETE',geographic_generalization='NOT_EVALUATED',inference_speed='RUN_SEPARATE_GPU_BENCHMARK',cautions=[
        'Three paired seeds have minimum two-sided exact sign-randomization p=0.25.',
        'Seed 42 participated in configuration selection; seed inference is exploratory.',
        'Matching seed labels alone do not establish exchangeability.',
        'Block intervals are sensitivity analyses: seasonal nonstationarity and long dependence may invalidate coverage.',
        'Overlapping event windows are not independent replicates; no event-hour t tests.',
        '2021 is temporal, not geographic, generalization.']),indent=2))
    print('Complete:',out)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-root',type=Path,default=Path('/home/jovyan/swan/runs/iclr_typhoon_2021_v3'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    if a.bootstrap<100:p.error('--bootstrap must be >=100')
    analyze(a)
