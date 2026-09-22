#!/usr/bin/env python3
"""Freeze twelve evaluations, then compare two- and three-year models on 2022."""
import csv,fcntl,hashlib,os,signal,subprocess,sys,time
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import campaign as c
from settings import load,paths,HERE,environment
from common import verified_entry

def compare(work):
    rows=[];lookup={};truth=None
    for model in ('fno','ffno'):
        for cohort in ('two_year','three_year'):
            root=work/'evaluation'/f'{cohort}_{model}'
            with (root/'event_metrics_by_seed.csv').open() as f:
                for row in csv.DictReader(f):
                    row=dict(cohort=cohort,**row);rows.append(row)
                    key=(cohort,model,int(row['seed']),row['scope'])
                    if key in lookup:raise ValueError('Duplicate comparison key')
                    lookup[key]=row
            for seed in (42,43,44):
                with (root/'models'/f'{model}_s{seed}'/'hourly.csv').open() as f:hours=list(csv.DictReader(f))
                values=np.array([[float(r[k]) for k in ('true_hs_max','true_hs_mean')] for r in hours])
                times=[r['time'] for r in hours]
                if truth is None:truth=(times,values)
                elif times!=truth[0] or not np.allclose(values,truth[1],atol=1e-5,rtol=1e-5):raise ValueError('Cohorts do not share physical evaluation truth/times')
    from report_events import write_csv
    write_csv(work/'comparison_by_seed.csv',rows)
    changes=[]
    for model in ('fno','ffno'):
        scopes=sorted({k[3] for k in lookup if k[1]==model})
        for scope in scopes:
            for metric in ('hs_mae','hs_pooled_rmse','hs_ge3_mae','hs_ge5_mae','domain_peak_bias_m','domain_peak_time_error_h'):
                pairs=[]
                for seed in (42,43,44):
                    old=lookup['two_year',model,seed,scope];new=lookup['three_year',model,seed,scope]
                    if old.get(metric,'')=='' or new.get(metric,'')=='':break
                    pairs.append((float(old[metric]),float(new[metric])))
                if len(pairs)!=3:continue
                v=np.array(pairs);d=v[:,1]-v[:,0]
                changes.append(dict(model=model,scope=scope,metric=metric,two_year_mean=v[:,0].mean(),two_year_seed_sd=v[:,0].std(ddof=1),
                    three_year_mean=v[:,1].mean(),three_year_seed_sd=v[:,1].std(ddof=1),difference_three_minus_two=d.mean(),difference_seed_sd=d.std(ddof=1),
                    note='Negative favors three-year for unsigned error metrics only; not a significance claim'))
    write_csv(work/'comparison_summary.csv',changes)

def main():
    a=load();server,work=paths(a)
    if not (work/'training_completed.json').exists():raise RuntimeError('Finish v5 training first')
    if a['require_v4_complete'] and not (server/'runs/iclr_parallel_v4/completed.json').exists():raise RuntimeError('v4 not complete')
    if not a['require_v4_complete'] and set(a['gpus'])&set(range(8)):raise ValueError('Concurrent mode requires GPUs >=8')
    environment(a);r=c.load_runner(HERE/'trainer');c.freeze(work/'config.json',a)
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    with ExitStack() as stack:
        lock=stack.enter_context((work/'v5.lock').open('a+'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a['require_v4_complete']:
            old=stack.enter_context((server/'runs/iclr_parallel_v4/controller.lock').open('a+'))
            fcntl.flock(old,fcntl.LOCK_EX|fcntl.LOCK_NB)
        records=c.read(work/'selected_6.json');queue=[]
        if len(records)!=6 or {(v['job']['model'],v['job']['seed']) for v in records}!={(m,s) for m in a['models'] for s in a['seeds']}:raise ValueError('Expected six v5 selections')
        # Freeze all cohort entries before opening any held-out data.
        for model in a['models']:
            control=c.read(work/'controls'/f'{model}.json')
            for cohort in ('two_year','three_year'):
                name=f'{cohort}_{model}';root=work/'evaluation'/name
                if cohort=='two_year':entries=control['entries'];package=control['package'];reference=a['source_2019_2020']
                else:
                    entries=[];package=str(HERE/'trainer');reference=str(work/'data/training_2019_2021.nc')
                    for record in records:
                        j=record['job']
                        if j['model']!=model:continue
                        s=r.checked_summary(work/'training',j)
                        if s is None or s['best_weight']!=record['checkpoint']:raise ValueError('v5 result invalid')
                        entries.append(verified_entry(s,r))
                c.freeze(root/'selected_2022.json',entries)
                c.freeze(root/'cohort.json',dict(package=package,reference=reference,cohort=cohort,model=model))
                if not (root/'completed.json').exists():queue.append(name)
        for p in (Path(a['source_2022']),Path(a['boundary_2022'])):
            if not p.exists():raise FileNotFoundError(f'2022 is not ready: {p}. Training results are preserved.')
        import events
        source=server/'swan_bc_typhoon_v3/jma_besttrack.txt'
        if source.exists() and not (HERE/'jma_besttrack.txt').exists():(HERE/'jma_besttrack.txt').write_bytes(source.read_bytes())
        events.build_events(Path(a['source_2022']),HERE,c.read(work/'evaluation_protocol.json'),work/'events_2022.json')
        active={}
        try:
            while queue or active:
                inv=r.gpu_inventory()
                for gpu in a['gpus']:
                    if not queue or len(active)>=a['evaluation_workers']:break
                    if gpu not in inv:raise ValueError('GPU absent')
                    if gpu in active or inv[gpu]['busy']:continue
                    name=queue.pop(0);path=work/'logs'/f'eval_{name}.log';path.parent.mkdir(exist_ok=True)
                    log=path.open('a');proc=subprocess.Popen([sys.executable,str(HERE/'worker_eval.py'),'--group',name,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
                    active[gpu]=(proc,name,log)
                for gpu,(proc,name,log) in list(active.items()):
                    rc=proc.poll()
                    if rc is None:continue
                    log.close();del active[gpu]
                    if rc or not (work/'evaluation'/name/'completed.json').exists():raise RuntimeError(f'Evaluation failed: {name}; see logs')
                c.atomic(work/'evaluation_status.json',dict(queued=queue,active={str(g):n for g,(p,n,l) in active.items()},updated=time.time()))
                if queue or active:time.sleep(5)
        finally:
            for proc,name,log in active.values():
                if proc.poll() is None:proc.terminate()
            for proc,name,log in active.values():proc.wait(timeout=45);log.close()
        compare(work);c.atomic(work/'completed.json',dict(year=2022,evaluated_models=12,cohorts=['two_year','three_year']))
        print('[COMPLETE]',work/'comparison_summary.csv')
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
