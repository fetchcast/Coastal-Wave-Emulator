#!/usr/bin/env python3
"""Isolated, resumable event inference using frozen v4 evaluation entries."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from analysis import analyze_event, aggregate

HERE=Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def freeze(path,value):
    value=json.loads(json.dumps(value))
    if Path(path).exists():
        if read(path)!=value:raise ValueError(f'Conflicting provenance: {path}. Use a new output directory.')
    else:atomic(path,value)

def available_gpu(gpu):
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if any(r.split(',')[0].strip()==uuid for r in rows.splitlines()):
        raise RuntimeError(f'GPU {gpu} already has a compute process. No existing job was stopped.')

def selected_tasks(a):
    tasks=[]
    for model in a.models.split(','):
        family=a.eval_root/model
        if not (family/'completed.json').exists():
            raise ValueError(f'Family evaluation not complete: {family}')
        entries=read(family/'selected_2021.json')
        for seed in [int(s) for s in a.seeds.split(',')]:
            matches=[e for e in entries if e['job']['seed']==seed and e['job']['model']==model]
            if len(matches)!=1:raise ValueError(f'Expected one frozen entry: {model}, {seed}')
            tasks.append((model,seed))
    return tasks

def worker(a):
    family=a.eval_root/a.model
    entry=next(e for e in read(family/'selected_2021.json') if e['job']['seed']==a.seed)
    name=f'{a.model}_s{a.seed}'
    prepared=read(family/'prepared'/(name+'.json'))
    rd=family/'models'/name
    original=read(rd/'result.json')
    if not original.get('complete') or original['signature']['entry']!=entry or original['signature']['prepared']!=prepared:
        raise ValueError('Original evaluation provenance differs from the frozen entry')
    checkpoint=Path(entry['checkpoint'])
    st=checkpoint.stat();stamp=original['signature']['checkpoint']
    if st.st_size!=stamp['size'] or st.st_mtime_ns!=stamp['mtime_ns']:
        raise ValueError('Checkpoint changed after original evaluation')
    if sha(entry['normalization'])!=entry['normalization_sha256']:raise ValueError('Normalization changed')
    norm=read(entry['normalization']);cache=Path(prepared['cache'])
    if read(cache/'signature.json')!=prepared['signature']:raise ValueError('Cache signature mismatch')
    complete=read(cache/'complete.json')
    x=np.load(cache/'inputs.npy',mmap_mode='r');y=np.load(cache/'targets.npy',mmap_mode='r')
    wet=np.load(cache/'mask.npy').astype(bool);times=np.load(cache/'times.npy').astype('datetime64[ns]')
    for n,arr in [('inputs.npy',x),('targets.npy',y),('mask.npy',wet),('times.npy',times)]:
        if list(arr.shape)!=complete['shapes'][n]:raise ValueError(f'Cache shape mismatch: {n}')
    events=read(family/'events_2021.json')['events']
    wanted=set(a.events.split(','));events=[e for e in events if str(e['id']) in wanted]
    if {str(e['id']) for e in events}!=wanted:raise ValueError('Requested event missing from frozen manifest')
    output=a.output/'models'/name;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    provenance=dict(version=1,entry=entry,checkpoint_sha256=sha(checkpoint),prepared=prepared,
        events=events,mask_sha256=sha(cache/'mask.npy'),
        code={n:sha(HERE/n) for n in ('run.py','analysis.py','evaluate_2021.py','campaign.py')},
        channel='Hs in meters only',precision='float32_no_autocast',scope='2021 held-out diagnostic; no training')
    freeze(output/'provenance.json',provenance)
    with (rd/'hourly.csv').open() as f:
        reference={int(r['target_index']):r for r in csv.DictReader(f)}
    import evaluate_2021 as ev
    model=None
    length=int(entry['job']['hyperparams']['seq_length'])
    def predict(idx):
        nonlocal model
        import torch
        if model is None:
            available_gpu(a.gpu)
            mods=ev.import_modules(a.server_root,a.server_root/'swan_repaired_v1')
            if mods['repair_support'].code_hashes(a.server_root/'swan_repaired_v1')!=entry['job']['code_hashes']:
                raise ValueError('Training code hashes changed')
            train=mods['train_repaired'];hp=entry['job']['hyperparams']
            args=ev.model_arguments(hp,train.LegacyCompatibleBenchmarkModel)
            model=train.LegacyCompatibleBenchmarkModel(model_name=a.model,input_channels=10,output_channels=4,**args)
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True);del state
            if not torch.cuda.is_available():raise RuntimeError('CUDA is not available')
            model=model.cuda().eval()
        with torch.inference_mode():
            xb=torch.from_numpy(np.array(x[idx-length:idx],copy=True)).unsqueeze(0).cuda()
            p=model(xb)[0][0].float().cpu().numpy()[0]
        return p*(norm['hs'][1]-norm['hs'][0])+norm['hs'][0]
    for event in events:
        folder=output/'events'/str(event['id']);frames=folder/'frames';frames.mkdir(parents=True,exist_ok=True)
        np.save(folder/'mask.npy',wet)
        ids=np.flatnonzero((times>=np.datetime64(event['start']))&(times<=np.datetime64(event['end'])))
        expected=int((np.datetime64(event['end'])-np.datetime64(event['start']))/np.timedelta64(1,'h'))+1
        if len(ids)!=expected or not len(ids):raise ValueError('Incomplete event time coverage')
        reused=0
        for position,i in enumerate(ids):
            idx=int(i)
            if idx<length or np.any(np.diff(times[idx-length:idx+1])!=np.timedelta64(1,'h')):
                raise ValueError('Input sequence crosses a time gap')
            target=np.asarray(y[idx,0])*(norm['hs'][1]-norm['hs'][0])+norm['hs'][0]
            dst=frames/f'{idx}.npz'
            if dst.exists():
                with np.load(dst,allow_pickle=False) as z:
                    if str(z['time'])!=str(times[idx]) or not np.array_equal(z['true_hs'],target):
                        raise ValueError('Saved frame has changed truth/time')
                    p=z['pred_hs'].copy()
            else:
                snap=rd/'snapshots'/f'{idx}.npz'
                if snap.exists():
                    with np.load(snap,allow_pickle=False) as z:
                        if str(z['time'])!=str(times[idx]) or not np.array_equal(z['kcs'].astype(bool),wet) or not np.array_equal(z['true'][0],target):
                            raise ValueError('Snapshot time, mask, or target mismatch')
                        if z['channel_units'].tolist()!=['m','s','sin','cos']:raise ValueError('Snapshot units mismatch')
                        p=z['pred'][0].copy();reused+=1
                else:p=predict(idx)
            if not np.isfinite(p[wet]).all():raise ValueError('Nonfinite Hs prediction')
            # Compare each re-inferred frame against the original scalar evaluation.
            if idx not in reference:raise ValueError('Frame absent from original evaluation')
            actual=float(abs(p[wet].astype(float)-target[wet].astype(float)).mean())
            if not np.isclose(actual,float(reference[idx]['hs_mae']),rtol=2e-4,atol=2e-6):
                raise ValueError(f'Re-inference differs from original hourly MAE at {idx}: {actual} vs {reference[idx]["hs_mae"]}')
            if not dst.exists():
                tmp=frames/f'{idx}.tmp.npz'
                np.savez_compressed(tmp,pred_hs=p.astype('float32'),true_hs=target.astype('float32'),time=str(times[idx]),units='m')
                tmp.replace(dst)
            atomic(output/'status.json',dict(stage='arrays',event=event['id'],completed=position+1,total=len(ids),updated=time.time()))
            if (position+1)%24==0:print(f'[{name}] {event["id"]}: {position+1}/{len(ids)}',flush=True)
        atomic(folder/'arrays_complete.json',dict(indices=ids.tolist(),event=event,reused_original_snapshots_this_attempt=reused))
        analyze_event(folder,event,a.model,a.seed,make_plots=(a.seed==42 or a.plot_all_seeds))
        print(f'[DONE EVENT] {name} {event["id"]}',flush=True)
    atomic(output/'completed.json',dict(model=a.model,seed=a.seed,events=[e['id'] for e in events]))
    atomic(output/'status.json',dict(stage='complete',updated=time.time()))

def main(a):
    if a.mode=='worker':return worker(a)
    if a.mode=='report':return aggregate(a.output)
    tasks=selected_tasks(a)
    if a.mode=='inspect':
        for m,s in tasks:
            rd=a.eval_root/m/'models'/f'{m}_s{s}'
            print(m,s,'original snapshots:',len(list((rd/'snapshots').glob('*.npz'))),
                  'prepared cache:',read(a.eval_root/m/'prepared'/f'{m}_s{s}.json')['cache'])
        print('Only selected peak snapshots were saved by v4. Missing event hours require inference.')
        return
    gpus=[int(x) for x in a.gpus.split(',')]
    if len(gpus)!=len(set(gpus)):raise ValueError('Duplicate GPUs')
    for gpu in gpus:available_gpu(gpu)
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze(a.output/'run_plan.json',dict(tasks=tasks,events=a.events,eval_root=str(a.eval_root.resolve())))
    logs=a.output/'logs';logs.mkdir(exist_ok=True)
    pending=list(tasks);active={};failures=[]
    try:
        while pending or active:
            for gpu,(proc,log,task) in list(active.items()):
                if proc.poll() is not None:
                    if proc.returncode:failures.append(dict(task=task,exit_code=proc.returncode))
                    log.close();del active[gpu]
            if failures:
                pending=[]
            for gpu in gpus:
                if gpu in active or not pending:continue
                available_gpu(gpu)
                m,s=pending.pop(0)
                cmd=[sys.executable,str(HERE/'run.py'),'worker','--server-root',str(a.server_root),'--eval-root',str(a.eval_root),
                     '--output',str(a.output),'--model',m,'--seed',str(s),'--gpu',str(gpu),'--events',a.events]
                if a.plot_all_seeds:cmd.append('--plot-all-seeds')
                log=(logs/f'{m}_s{s}.log').open('a')
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MPLBACKEND='Agg',OMP_NUM_THREADS='4'))
                active[gpu]=(proc,log,(m,s));print(f'[START] {m} s{s} GPU {gpu}',flush=True)
            atomic(a.output/'status.json',dict(active={str(g):v[2] for g,v in active.items()},queued=pending,failures=failures,updated=time.time()))
            if active:time.sleep(3)
        if failures:raise RuntimeError(f'Worker failures; inspect {logs}: {failures}')
        aggregate(a.output)
        atomic(a.output/'completed.json',dict(tasks=tasks,events=a.events,analysis_complete=True))
        print('[COMPLETE]',a.output,flush=True)
    finally:
        for proc,log,_ in active.values():
            if proc.poll() is None:proc.terminate()
            log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['inspect','run','worker','report'])
    p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'))
    p.add_argument('--eval-root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--models',default='fno,ffno,tno');p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--events',default='2109,2112,2114');p.add_argument('--gpus',default='2,3')
    p.add_argument('--model');p.add_argument('--seed',type=int);p.add_argument('--gpu',type=int)
    p.add_argument('--plot-all-seeds',action='store_true')
    a=p.parse_args()
    a.eval_root=a.eval_root or a.server_root/'runs/iclr_parallel_v4/evaluation'
    a.output=a.output or a.server_root/'runs/iclr_event_diagnostics_v1'
    main(a)
