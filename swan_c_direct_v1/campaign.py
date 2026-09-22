#!/usr/bin/env python3
"""Run a frozen, validation-selected SWAN campaign using the existing worker."""
import argparse
import copy
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MODELS = ('fno', 'tno', 'ffno')

def read(p):
    return json.loads(Path(p).read_text())

def atomic(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(p)

def freeze(p, value):
    if Path(p).exists():
        if read(p) != value:
            raise ValueError(f'Frozen configuration differs: {p}')
    else:
        atomic(p, value)

def load_runner(package):
    sys.path.insert(0, str(package))
    r = importlib.import_module('run_repaired')
    if Path(r.__file__).resolve() != package / 'run_repaired.py':
        raise ValueError('Unexpected runner module')
    return r

def make_job(base, tag, w=None, d=None, m=None, seed=42, lr=None, fraction=1., cycles=30):
    j = copy.deepcopy(base)
    h = j['hyperparams']
    structural = any(v is not None for v in (w,d,m))
    w = w or h['fno_width']
    d = d or h['fno_depth']
    m = m or h['modes_x']
    h.update(fno_width=w, fno_depth=d, modes_x=m, modes_y=m)
    if lr is not None:
        h['max_lr'] = lr
    # Keep a nominal effective batch of four; record actual worker counts.
    micro = 1 if d >= 8 or w >= 256 or (j['model'] == 'tno' and (w >= 128 or m >= 48)) else (2 if m >= 48 else 4)
    if structural:
        h.update(batch_size=micro, acc_steps=4 // micro)
        if j['model'] == 'tno':
            h.update(width=w, depth=d, modes_t=4, use_checkpoint=True)
    j.update(config_id=f'{j["model"]}_{tag}_w{w}d{d}m{m}_s{seed}', stage='pilot',
             seed=seed, train_fraction=fraction, epochs=cycles, early_stop_patience=0)
    j.pop('max_updates', None)
    j.pop('eval_every_updates', None)
    return j

def architecture_jobs(bases):
    jobs = []
    for model in MODELS:
        configs = [(w,d,24) for w in (64,128,256) for d in (4,6,8)
                   if (w,d) != (64,4) and not (model == 'tno' and w == 256 and d > 4)]
        configs += [(w,4,48) for w in ((64,) if model == 'tno' else (64,128,256))]
        jobs += [make_job(bases[model], 'A', *c) for c in configs]
    return jobs

def validation(s):
    v = float(s['repair_audit']['best_val_hs_mae'])
    if not math.isfinite(v):
        raise ValueError('Nonfinite selection metric')
    return v

def choose(rows, count):
    # Never consult final/test metrics for selection.
    return sorted(rows, key=lambda s: (validation(s), s['config_id']))[:count]

def stage_b(rows):
    jobs = []
    for model in MODELS:
        for i, s in enumerate(choose([s for s in rows if s['model'] == model], 2)):
            for lr in (5e-5, 2e-4):
                jobs.append(make_job(s['repair_job'], f'B{i}_lr{lr:g}', lr=lr))
    return jobs

def tail(path, n=262144):
    try:
        with Path(path).open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell()-n))
            return f.read().decode(errors='replace')
    except OSError:
        return ''

def export(root, rows):
    fields = ['model','config_id','seed','fraction','cycles','validation_hs_mae',
              'hs_rmse','parameters','updates','wallclock_s','best_weight']
    tmp = root / 'results.csv.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in rows:
            j = s['repair_job']; a = s['repair_audit']
            writer.writerow(dict(model=s['model'],config_id=s['config_id'],seed=j['seed'],
                fraction=j['train_fraction'],cycles=j['epochs'],validation_hs_mae=validation(s),
                hs_rmse=s.get('legacy_metrics',{}).get('rmse_m'),parameters=a.get('n_parameters'),
                updates=a.get('updates'),wallclock_s=s.get('training_wallclock_s'),best_weight=s.get('best_weight')))
    tmp.replace(root / 'results.csv')

def worker(a):
    r = load_runner(a.package)
    j = read(a.job)
    # Each supervisor owns one worker group, so one OOM cannot kill other jobs.
    r.launch(a.root, [j], [a.gpu])

def run_stage(a, r, label, jobs, reference, probe=False):
    root = a.root / label
    root.mkdir(parents=True, exist_ok=True)
    freeze(root / 'plan.json', jobs)
    queue = []; rows = []; active = {}; skipped = []
    for j in jobs:
        p = root / '_jobs' / (j['config_id'] + '.json')
        freeze(p, j)
        s = r.checked_summary(root, j)
        if s:
            r.compare_protocols([reference,s]); rows.append(s)
        elif (root / (j['config_id']+'.resource_skip.json')).exists():
            skipped.append(j['config_id'])
        else:
            queue.append((j,p))
    try:
        while queue or active:
            inv = r.gpu_inventory()
            for gpu in a.gpus:
                if not queue:
                    break
                if gpu in active or inv[gpu]['busy']:
                    continue
                j,p = queue.pop(0)
                if r.parameter_bytes_floor(j) > inv[gpu]['mib']*.85:
                    atomic(root / (j['config_id']+'.resource_skip.json'), dict(reason='spectral_state_floor',job=j))
                    skipped.append(j['config_id']); continue
                log = open(root / (j['config_id']+'.supervisor.log'), 'a')
                cmd = [sys.executable,str(HERE/'campaign.py'),'--worker','--package',str(a.package),
                       '--root',str(root),'--job',str(p),'--gpu',str(gpu)]
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (proc,j,log)
                print(f'[START {label}] {j["config_id"]} GPU={gpu}', flush=True)
            for gpu,(proc,j,log) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close(); del active[gpu]
                s = r.checked_summary(root,j)
                if rc == 0 and s is not None:
                    r.compare_protocols([reference,s]); rows.append(s)
                    print(f'[DONE {label}] {j["config_id"]}',flush=True)
                else:
                    rd = root / r.run_name(j)
                    logs = sorted(rd.glob('attempt_*.log'), key=lambda p:p.stat().st_mtime)
                    message = tail(logs[-1]) if logs else ''
                    # Unknown failures remain fatal. Only explicit CUDA OOM is recoverable.
                    oom = 'CUDA out of memory' in message or 'torch.OutOfMemoryError' in message
                    if oom:
                        atomic(root / (j['config_id']+'.resource_skip.json'),dict(reason='CUDA_OOM',job=j,returncode=rc))
                        skipped.append(j['config_id'])
                        print(f'[RESOURCE SKIP] {j["config_id"]}',flush=True)
                    else:
                        raise RuntimeError(f'{j["config_id"]} failed; inspect {rd}. Restart after resolving the cause.')
            atomic(a.root/'status.json',dict(stage=label,completed=len(rows),queued=len(queue),
                active={str(g):j['config_id'] for g,(_,j,_) in active.items()},resource_skips=skipped,
                updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            if queue or active:
                time.sleep(5)
    finally:
        for proc,j,log in active.values():
            proc.terminate()
        for proc,j,log in active.values():
            # The existing supervisor terminates its torchrun process group on SIGTERM.
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f'[STOP] Supervisor {proc.pid} has not exited; inspect before restarting.',flush=True)
            log.close()
    freeze(root/'completed.json',dict(successful=sorted(s['config_id'] for s in rows),resource_skips=sorted(skipped)))
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'))
    p.add_argument('--package',type=Path)
    p.add_argument('--root',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--job',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--gpu',type=int,help=argparse.SUPPRESS)
    a = p.parse_args()
    a.server_root = a.server_root.resolve()
    a.package = (a.package or a.server_root/'swan_repaired_v1').resolve()
    a.root = (a.root or a.server_root/'runs/iclr_expanded_v1').resolve()
    if a.worker:
        return worker(a)
    a.data = (a.data or a.server_root/'wavm-Waves_2019_2020_v2.nc').resolve()
    a.gpus = [int(g) for g in a.gpus.split(',')]
    if not a.gpus or len(set(a.gpus)) != len(a.gpus) or min(a.gpus)<0:
        raise ValueError('Invalid GPU list')
    parents = [a.server_root/'runs/repaired_timegap_v1',a.server_root/'runs/repaired_timegap_extra_v1']
    for parent in parents+[a.package]:
        if a.root == parent or parent in a.root.parents or a.root in parent.parents:
            raise ValueError('Choose a separate result root')
    bases = {j['model']:j for j in read(parents[0]/'plan_pilot.json')}
    jobs = architecture_jobs(bases)
    if a.plan:
        for j in jobs:
            print(j['config_id'],j['hyperparams'])
        print('A=29 new full runs; B=12; C<=6; D=3; E=18. Total <=68 full runs plus 29 two-update preflights.')
        return
    os.environ.update(SWAN_SERVER_ROOT=str(a.server_root),SWAN_DATA_PATH=str(a.data))
    r = load_runner(a.package)
    expected = read(HERE/'expected_hashes.json')
    if r.repair.code_hashes(a.package) != expected:
        raise ValueError('Server training code differs from the reviewed files. No jobs started.')
    protocol = read(parents[0]/'protocol.json')
    current = dict(version=r.repair.VERSION,code_hashes=expected,
        asset_signature=r.repair.asset_signature(a.server_root,str(a.data)),direction_policy=protocol['direction_policy'])
    if current != protocol:
        raise ValueError('Parent protocol or assets differ')
    pilot_rows = []
    for root, plan in [(parents[0],list(bases.values())),(parents[1],read(parents[1]/'extra_plan.json')['jobs'])]:
        for j in plan:
            s = r.checked_summary(root,j)
            if not s:
                raise ValueError(f'Unverified completed pilot: {j["config_id"]}')
            pilot_rows.append(s)
    if len(pilot_rows)!=9 or {(s['model'],s['seed']) for s in pilot_rows}!={(m,s) for m in MODELS for s in (42,43,44)}:
        raise ValueError('Expected all nine parent pilots')
    r.compare_protocols(pilot_rows)
    reference = pilot_rows[0]
    a.root.mkdir(parents=True,exist_ok=True)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    with open(a.root/'campaign.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze(a.root/'protocol.json',current)
        freeze(a.root/'campaign_config.json',dict(code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            package=str(a.package),data=str(a.data),parents=[str(x) for x in parents]))
        # Copy verified source-check cache records into every stage before launching.
        for label in ('preflight','A','B','C','D','E'):
            for cache in (parents[0]/'_source_checks').glob('*.json'):
                value=read(cache)
                if value.get('checked') is True and value.get('signature')==current['asset_signature']:
                    freeze(a.root/label/'_source_checks'/cache.name,value)
        probes=[]
        for j in jobs:
            q=copy.deepcopy(j);q['config_id']='probe_'+j['config_id']
            q.update(max_updates=2,eval_every_updates=2)
            probes.append(q)
        probe_rows=run_stage(a,r,'preflight',probes,reference,True)
        allowed={s['config_id'].removeprefix('probe_') for s in probe_rows}
        a_rows=run_stage(a,r,'A',[j for j in jobs if j['config_id'] in allowed],reference)
        rows=pilot_rows+a_rows;export(a.root,rows)
        candidates=[s for s in pilot_rows if s['seed']==42]+a_rows
        b_rows=run_stage(a,r,'B',stage_b(candidates),reference)
        rows+=b_rows;export(a.root,rows)
        selected={m:choose([s for s in candidates+b_rows if s['model']==m],1)[0] for m in MODELS}
        freeze(a.root/'selection.json',{m:dict(config_id=s['config_id'],validation_hs_mae=validation(s),job=s['repair_job']) for m,s in selected.items()})
        c=[]
        for m,s in selected.items():
            if s['config_id']==bases[m]['config_id']:
                continue
            for seed in (43,44):
                c.append(make_job(s['repair_job'],'C',seed=seed))
        rows+=run_stage(a,r,'C',c,reference);export(a.root,rows)
        d=[make_job(s['repair_job'],'D60',cycles=60) for s in selected.values()]
        rows+=run_stage(a,r,'D',d,reference);export(a.root,rows)
        e=[make_job(s['repair_job'],f'E_f{f}',seed=seed,fraction=f) for s in selected.values()
           for f in (.25,.5) for seed in (42,43,44)]
        rows+=run_stage(a,r,'E',e,reference);export(a.root,rows)
        atomic(a.root/'campaign_completed.json',dict(results=len(rows),selection='validation_hs_mae_only',
            note='Inspect resource_skip files; completed does not imply every candidate fit in memory.'))
        print('[CAMPAIGN COMPLETE]',a.root,flush=True)

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
