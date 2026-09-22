#!/usr/bin/env python3
"""Resume v1 training unchanged, then evaluate nine selected runs on 2021."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from campaign import atomic,freeze,read,load_runner,make_job

HERE=Path(__file__).resolve().parent

def training_busy(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'campaign.lock').open('a+') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False

def wait_child(cmd,env=None):
    proc=subprocess.Popen(cmd,env=env)
    try:
        return proc.wait()
    except BaseException:
        proc.terminate()
        try:proc.wait(timeout=45)
        except subprocess.TimeoutExpired:print(f'Process {proc.pid} still exiting; inspect before restart',flush=True)
        raise

def entries(a,r):
    selection=read(a.train_root/'selection.json')
    parent=a.server_root/'runs/repaired_timegap_v1';extra=a.server_root/'runs/repaired_timegap_extra_v1'
    bases={j['model']:j for j in read(parent/'plan_pilot.json')}
    extras={(j['model'],j['seed']):j for j in read(extra/'extra_plan.json')['jobs']}
    results=[];summaries=[]
    for model in ('fno','tno','ffno'):
        chosen=selection[model]['job']
        for seed in (42,43,44):
            if chosen['config_id']==bases[model]['config_id']:
                job=bases[model] if seed==42 else extras[model,seed]
                root=parent if seed==42 else extra
            elif seed==42:
                job=chosen
                stage='A' if '_A_' in chosen['config_id'] else 'B'
                root=a.train_root/stage
            else:
                job=make_job(chosen,'C',seed=seed);root=a.train_root/'C'
            s=r.checked_summary(root,job)
            if s is None:raise ValueError(f'Selected run not verified/completed: {job["config_id"]}')
            rd=root/r.run_name(job);norm=rd/'normalization.json';direction=rd/'direction_manifest.json'
            if not norm.is_file() or not direction.is_file():raise FileNotFoundError(f'Missing saved normalization/direction in {rd}')
            dr=read(direction)
            if dr!=s['repair_audit']['direction']:raise ValueError('Saved direction differs from training audit')
            results.append(dict(job=job,checkpoint=s['best_weight'],normalization=str(norm.resolve()),
                direction=dr['chosen'],source_summary=str((rd/'run_summary.json').resolve()),
                normalization_sha256=hashlib.sha256(norm.read_bytes()).hexdigest()))
            summaries.append(s)
    r.compare_protocols(summaries)
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'))
    p.add_argument('--package',type=Path)
    p.add_argument('--train-root',type=Path)
    p.add_argument('--eval-root',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--nc-2021',type=Path)
    p.add_argument('--bnd-2021',type=Path)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--eval-workers',type=int,default=2)
    p.add_argument('--max-bnd-gap-hours',type=float,default=6.)
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    a.server_root=a.server_root.resolve();a.package=(a.package or a.server_root/'swan_repaired_v1').resolve()
    a.train_root=(a.train_root or a.server_root/'runs/iclr_expanded_v1').resolve()
    a.eval_root=(a.eval_root or a.server_root/'runs/iclr_2021_v2').resolve()
    a.data=(a.data or a.server_root/'wavm-Waves_2019_2020_v2.nc').resolve()
    a.nc_2021=(a.nc_2021 or a.server_root/'swan_2021_nc_v2/wavm-Waves.nc').resolve()
    a.bnd_2021=(a.bnd_2021 or a.server_root/'bnd_2021_v2').resolve()
    gpus=[int(g) for g in a.gpus.split(',')]
    if not gpus or len(set(gpus))!=len(gpus) or min(gpus)<0 or not 1<=a.eval_workers<=len(gpus) or a.max_bnd_gap_hours<=0:
        raise ValueError('Invalid GPU/worker/gap settings')
    if a.train_root==a.eval_root or a.train_root in a.eval_root.parents or a.eval_root in a.train_root.parents:
        raise ValueError('Evaluation root must be separate from training')
    traincmd=[sys.executable,str(HERE/'campaign.py'),'--server-root',str(a.server_root),'--package',str(a.package),
        '--root',str(a.train_root),'--data',str(a.data),'--gpus',a.gpus]
    if a.plan:
        rc=wait_child(traincmd+['--plan'])
        print('Then: frozen selected models x 3 seeds = 9 held-out evaluations; no 2021 training.')
        return rc
    for path in (a.nc_2021,a.data):
        if not path.is_file():raise FileNotFoundError(path)
    if not a.bnd_2021.is_dir():raise FileNotFoundError(a.bnd_2021)
    a.eval_root.mkdir(parents=True,exist_ok=True)
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    with (a.eval_root/'v2.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not a.eval_only:
            while training_busy(a.train_root):
                print('[WAIT] Existing v1 campaign is running. It will not be stopped.',flush=True)
                atomic(a.eval_root/'status.json',dict(stage='waiting_for_v1',updated=time.time()))
                time.sleep(30)
            print('[TRAIN] Validate/resume the unchanged v1 campaign',flush=True)
            atomic(a.eval_root/'status.json',dict(stage='training_v1',updated=time.time()))
            if wait_child(traincmd):raise RuntimeError('Training campaign failed. Resolve its log before held-out evaluation.')
        os.environ.update(SWAN_SERVER_ROOT=str(a.server_root),SWAN_DATA_PATH=str(a.data))
        r=load_runner(a.package)
        if r.repair.code_hashes(a.package)!=read(HERE/'expected_hashes.json'):raise ValueError('Training code fingerprints differ')
        protocol=read(a.train_root/'protocol.json')
        if protocol['asset_signature']!=r.repair.asset_signature(a.server_root,str(a.data)):raise ValueError('Training assets changed')
        selected=entries(a,r)
        # Freeze the selection before any held-out field or performance is read.
        freeze(a.eval_root/'selected_2021.json',selected)
        queue=[]
        for entry in selected:
            name=f'{entry["job"]["model"]}_s{entry["job"]["seed"]}'
            ep=a.eval_root/'entries'/(name+'.json');freeze(ep,entry)
            prepared=a.eval_root/'prepared'/(name+'.json')
            cmd=[sys.executable,str(HERE/'evaluate_2021.py'),'prepare','--server-root',str(a.server_root),
                '--package',str(a.package),'--root',str(a.eval_root),'--entry',str(ep),'--result',str(prepared),
                '--nc',str(a.nc_2021),'--bnd',str(a.bnd_2021),'--reference',str(a.data),
                '--max-bnd-gap-hours',str(a.max_bnd_gap_hours)]
            print('[PREPARE]',name,flush=True)
            atomic(a.eval_root/'status.json',dict(stage='prepare_2021',model=name,updated=time.time()))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg',OMP_NUM_THREADS='4')
            if wait_child(cmd,env):raise RuntimeError(f'2021 preparation failed: {name}')
            result=a.eval_root/'models'/name/'result.json'
            queue.append((name,ep,prepared,result))
        active={}
        try:
            while queue or active:
                inv=r.gpu_inventory()
                for gpu in gpus:
                    if not queue or len(active)>=a.eval_workers:break
                    if gpu not in inv:raise ValueError(f'GPU missing: {gpu}')
                    if gpu in active or inv[gpu]['busy']:continue
                    name,ep,prepared,result=queue.pop(0);result.parent.mkdir(parents=True,exist_ok=True)
                    log=(result.parent/'evaluate.log').open('a')
                    cmd=[sys.executable,str(HERE/'evaluate_2021.py'),'evaluate','--server-root',str(a.server_root),
                         '--package',str(a.package),'--root',str(a.eval_root),'--entry',str(ep),'--prepared',str(prepared),'--result',str(result)]
                    proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MPLBACKEND='Agg',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                    active[gpu]=(proc,name,result,log);print('[EVAL START]',name,'GPU',gpu,flush=True)
                for gpu,(proc,name,result,log) in list(active.items()):
                    rc=proc.poll()
                    if rc is None:continue
                    log.close();del active[gpu]
                    if rc or not result.exists() or not read(result).get('complete'):raise RuntimeError(f'Evaluation failed: {name}; inspect {result.parent}/evaluate.log')
                    print('[EVAL DONE]',name,flush=True)
                atomic(a.eval_root/'status.json',dict(stage='evaluate_2021',queued=len(queue),active={str(g):n for g,(_,n,_,_) in active.items()},updated=time.time()))
                if queue or active:time.sleep(5)
        finally:
            for proc,name,result,log in active.values():proc.terminate()
            for proc,name,result,log in active.values():
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                log.close()
        rows=[read(a.eval_root/'models'/f'{e["job"]["model"]}_s{e["job"]["seed"]}'/'result.json') for e in selected]
        fields=['model','seed','frames']+list(rows[0]['metrics'])
        with (a.eval_root/'summary_2021.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for row in rows:w.writerow({**{k:row[k] for k in ('model','seed','frames')},**row['metrics']})
        atomic(a.eval_root/'completed.json',dict(models=9,year=2021,selection_source=str(a.train_root/'selection.json'),
            note='Evaluation only; 2021 not used for training or selection'))
        atomic(a.eval_root/'status.json',dict(stage='complete',updated=time.time()))
        print('[V2 COMPLETE]',a.eval_root/'summary_2021.csv',flush=True)
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
