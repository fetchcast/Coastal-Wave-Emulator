#!/usr/bin/env python3
"""Isolated, resumable launch of repaired SWAN smoke/pilot/follow-up jobs."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import repair_support as repair

PACKAGE=Path(__file__).resolve().parent


def batch_rule(model,w,d,m):
    if model=='fno':
        if w>=384 or d>=8 or m>=64:return 1,4
        if w>=256 or m>=48:return 2,2
        return 4,1
    if w>=512 or d>=8 or m>=64:return 1,4
    if w>=384 or m>=48:return 2,2
    return 4,1


def job(model,w,d,m,stage,tag,frac=1.,lr=1e-4,wd=1e-4,cycles=30,seed=42):
    b,a=batch_rule(model,w,d,m)
    j=dict(model=model,config_id=f'{model}_{tag}_w{w}d{d}m{m}_s{seed}',stage=stage,seed=seed,
        use_bnd='on',time_steps=17498,epochs=cycles,train_fraction=frac,early_stop_patience=0,
        protocol_version=repair.VERSION,hyperparams=dict(seq_length=12,hidden_dim=256,
            fno_width=w,fno_depth=d,modes_x=m,modes_y=m,modes_t=4,
            batch_size=b,acc_steps=a,max_lr=lr,weight_decay=wd))
    if model=='tno':j['hyperparams'].update(width=w,depth=d)
    if stage=='smoke':j.update(max_updates=2,eval_every_updates=2)
    return j


def plan(stage):
    if stage in ('smoke','pilot'):
        # Identical width, depth and spatial modes isolate the implementation check.
        return [job(m,64,4,24,stage,stage) for m in ('fno','tno','ffno')]
    control={'ffno':(128,4,48),'fno':(128,4,48)}
    large={'ffno':[(512,6,48),(384,6,48)],'fno':[(384,8,48)]}
    jobs=[]
    # Corrected full-data controls replace the incompatible old benchmark baselines.
    for model in control:
        for c in [control[model]]+large[model]:jobs.append(job(model,*c,stage,'base'))
        for c in large[model]:
            for wd in (1e-3,1e-2):jobs.append(job(model,*c,stage,f'R{wd:g}',wd=wd))
        jobs.append(job(model,*control[model],stage,'R0.01',wd=1e-2))
        for c in [control[model],large[model][0]]:
            for f in (.25,.5):jobs.append(job(model,*c,stage,f'D{f:g}',frac=f))
        for lr in (5e-5,2.5e-5):jobs.append(job(model,*large[model][0],stage,f'L{lr:g}',lr=lr))
        jobs.append(job(model,*large[model][0],stage,'B60',cycles=60))
    return jobs


def run_name(j):
    h=j['hyperparams']
    return f"{j['stage']}_{j['model']}_{j['config_id']}_seed{j['seed']}_seq{h['seq_length']}_lr{h['max_lr']:.0e}_wd{h['weight_decay']:.0e}"


def gpu_inventory():
    def query(args):
        return subprocess.run(['nvidia-smi',*args],check=True,text=True,capture_output=True,timeout=20).stdout
    rows=query(['--query-gpu=index,uuid,memory.total','--format=csv,noheader,nounits'])
    inventory={}
    for row in rows.strip().splitlines():
        i,u,mb=[v.strip() for v in row.split(',')];inventory[int(i)]=dict(uuid=u,mib=float(mb),busy=False)
    apps=query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
    occupied={r.split(',')[0].strip() for r in apps.strip().splitlines() if r.strip()}
    for v in inventory.values():v['busy']=v['uuid'] in occupied
    return inventory


def parameter_bytes_floor(j):
    """Lower bound: spectral weights, gradients, Adam states, EMA and EMA backup.

    Activations and other parameters are deliberately excluded. This is a
    rejection test for impossible jobs, not a guarantee that a job fits.
    """
    h=j['hyperparams'];w=h['fno_width'];d=h['fno_depth'];m=h['modes_x'];n=h['modes_y']
    if j['model']=='fno':p=2*2*w*w*m*n*d
    elif j['model']=='tno':p=4*2*w*w*m*n*h['modes_t']*d
    elif j['model']=='ffno':p=2*w*w*(m+n)*d
    else:return 0.
    return p*24/(1024**2)


def checked_summary(root,j):
    p=root/run_name(j)/'run_summary.json'
    if not p.exists():return None
    s=json.loads(p.read_text());a=s.get('repair_audit',{})
    if s.get('failed') or s.get('repair_job')!=j:return None
    if a.get('signature',{}).get('job')!=j or a.get('selection_metric')!='val_hs_mae_ema':return None
    if a.get('updates')!=a.get('target_updates') and not a.get('early_stop'):return None
    if a.get('completed_cycles',0)<1:return None
    rmse=s.get('legacy_metrics',{}).get('rmse_m')
    if rmse is None or not math.isfinite(float(rmse)):return None
    if not Path(s.get('best_weight','')).is_file():return None
    return s


def freeze(path,record):
    if path.exists():
        if json.loads(path.read_text())!=record:
            raise ValueError(f'Frozen plan differs: {path}. Use a new --root for a changed protocol.')
    else:repair.atomic_json(path,record)


def compare_protocols(summaries):
    seen=set()
    for s in summaries:
        a=s['repair_audit'];direction=a.get('direction') or {}
        seen.add((a['base_split_hash'],direction.get('chosen'),direction.get('calibration_hash')))
    if len(seen)>1:raise ValueError('Split or direction calibration differs between jobs')


def launch(root,jobs,gpus):
    pending=[]; summaries=[]; active={}; failed=[]
    root.mkdir(parents=True,exist_ok=True)
    for j in jobs:
        path=root/'_jobs'/(j['config_id']+'.json');freeze(path,j)
        s=checked_summary(root,j)
        if s:summaries.append(s); print('[SKIP]',j['config_id'],flush=True)
        else:
            rd=root/run_name(j)
            manifest=rd/'run_manifest.json'
            if manifest.exists() and json.loads(manifest.read_text()).get('job')!=j:
                raise ValueError(f'Existing run has different provenance: {rd}')
            if (rd/'run_summary.json').exists():
                # Keep the invalid summary for diagnosis and allow the worker to
                # resume its own checkpoint instead of triggering the legacy skip.
                (rd/'run_summary.json').rename(rd/f'run_summary.invalid.{time.time_ns()}.json')
            pending.append((j,path))
    def stop(*args):raise KeyboardInterrupt
    previous={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    last_message=0.
    try:
        while pending or active:
            if pending:
                inv=gpu_inventory()
                for gpu in gpus:
                    if not pending:break
                    if gpu not in inv:raise ValueError(f'GPU {gpu} does not exist')
                    if gpu in active or inv[gpu]['busy']:continue
                    j,p=pending.pop(0)
                    floor=parameter_bytes_floor(j)
                    if floor>inv[gpu]['mib']*.85:
                        raise ValueError(f'{j["config_id"]}: spectral-state floor {floor/1024:.1f} GiB exceeds the GPU budget. '
                            'Choose a smaller configuration explicitly in a new --plan-file; no silent downsizing is performed.')
                    rd=root/run_name(j);rd.mkdir(parents=True,exist_ok=True)
                    log=open(rd/f'attempt_{time.time_ns()}.log','w')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),SWAN_RESULTS_ROOT=str(root),
                             SWAN_REPAIR_JOB=json.dumps(j),OMP_NUM_THREADS='8',PYTHONUNBUFFERED='1',
                             SWAN_SYNTHETIC_BENCH='0' if j['stage']=='smoke' else '1',MPLBACKEND='Agg')
                    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','1',
                         str(PACKAGE/'train_repaired.py'),'--worker','--job_file',str(p)]
                    proc=subprocess.Popen(cmd,env=env,cwd=PACKAGE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[gpu]=(proc,j,log);print(f'[START] {j["config_id"]} GPU={gpu} PID={proc.pid}',flush=True)
            for gpu,(proc,j,log) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                log.close();del active[gpu]
                s=checked_summary(root,j)
                if rc or s is None:
                    failed.append(dict(job=j['config_id'],returncode=rc))
                    repair.atomic_json(root/'failures.json',failed)
                    raise RuntimeError(f'{j["config_id"]} failed (exit={rc}); inspect {root/run_name(j)}')
                summaries.append(s);compare_protocols(summaries)
                print('[DONE]',j['config_id'],flush=True)
            if pending and not active and time.time()-last_message>55:
                print('[WAIT] selected GPUs have running compute processes',flush=True);last_message=time.time()
            if pending or active:time.sleep(5)
    finally:
        for proc,j,log in active.values():
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.time()+5
        while active and time.time()<deadline and any(p.poll() is None for p,_,_ in active.values()):time.sleep(.2)
        for proc,j,log in active.values():
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait();log.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
    compare_protocols(summaries)
    return summaries


def report(root):
    for p in sorted(root.glob('*/run_summary.json')):
        s=json.loads(p.read_text());a=s.get('repair_audit',{})
        if not a:continue
        print(s['config_id'],'cycles=',a.get('completed_cycles'),'updates=',a.get('updates'),
              'skipped=',a.get('skipped_updates'),'best_update=',a.get('best_update'),
              'Hs_RMSE=',s.get('legacy_metrics',{}).get('rmse_m'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['smoke','pilot','followup'],default='pilot')
    ap.add_argument('--server-root',default='/home/jovyan/swan')
    ap.add_argument('--root',default=None)
    ap.add_argument('--data',default=None)
    ap.add_argument('--gpus',default='0,1,2')
    ap.add_argument('--bnd-transform',default='train_auto')
    ap.add_argument('--plan',action='store_true');ap.add_argument('--report',action='store_true')
    ap.add_argument('--plan-file',help='Explicit JSON list of jobs; use a new root after changing a plan')
    a=ap.parse_args()
    server=Path(a.server_root).resolve();root=Path(a.root or server/'runs/repaired_v1').resolve()
    if root==server/'runs/v2_focused_all' or root==server/'runs/v2_followup':raise ValueError('Choose a new result root')
    if a.report:report(root);return
    jobs=json.loads(Path(a.plan_file).read_text()) if a.plan_file else plan(a.stage)
    if len({run_name(j) for j in jobs})!=len(jobs):raise ValueError('Duplicate run names')
    if a.plan:
        for j in jobs:
            print(j['config_id'],f'fraction={j["train_fraction"]}',f'full-data cycles={j["epochs"]}',
                  f'spectral state lower bound={parameter_bytes_floor(j)/1024:.1f} GiB')
        return
    gpus=[int(v.strip()) for v in a.gpus.split(',') if v.strip()]
    if not gpus or len(gpus)!=len(set(gpus)) or min(gpus)<0:raise ValueError('GPU list must contain distinct nonnegative indices')
    data=str(Path(a.data or server/'wavm-Waves_2019_2020_v2.nc').resolve())
    os.environ.update(SWAN_SERVER_ROOT=str(server),SWAN_DATA_PATH=data,SWAN_RESULTS_ROOT=str(root))
    root.mkdir(parents=True,exist_ok=True)
    with open(root/'launcher.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        hashes=repair.code_hashes(PACKAGE);assets=repair.asset_signature(server,data)
        protocol=dict(version=repair.VERSION,code_hashes=hashes,asset_signature=assets,direction_policy=a.bnd_transform)
        freeze(root/'protocol.json',protocol)
        def prepare(js):
            for j in js:j.update(code_hashes=hashes,asset_signature=assets,bnd_dir_transform=a.bnd_transform)
            return js
        # Every expensive stage requires a successful three-model smoke test.
        smoke=prepare(plan('smoke'));freeze(root/'plan_smoke.json',smoke)
        launch(root,smoke,gpus)
        if a.stage=='smoke' and not a.plan_file:return
        # Follow-up is optional; it never starts automatically after the pilot.
        if a.stage=='followup':
            pilots=prepare(plan('pilot'))
            if not all(checked_summary(root,j) for j in pilots):
                raise ValueError('Complete and inspect the corrected pilot before starting follow-up')
        jobs=prepare(jobs);freeze(root/f'plan_{a.stage}.json',jobs)
        # Reject impossible jobs before consuming any additional training time.
        inv=gpu_inventory();largest=max(inv[g]['mib'] for g in gpus)
        impossible=[j['config_id'] for j in jobs if parameter_bytes_floor(j)>largest*.85]
        if impossible:raise ValueError('Requested configurations exceed the memory floor: '+', '.join(impossible))
        launch(root,jobs,gpus)
        repair.atomic_json(root/f'{a.stage}_completed.json',dict(protocol=protocol,jobs=[j['config_id'] for j in jobs]))
        report(root)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
