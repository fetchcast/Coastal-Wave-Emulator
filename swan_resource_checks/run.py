#!/usr/bin/env python3
"""Priority queue: event arrays -> inference resources -> input sensitivity -> optional LR pilots."""
import argparse,copy,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from common import read,atomic,freeze,inventory,sha
HERE=Path(__file__).resolve().parent
MODELS='fno,ffno,tno,convnext_lstm,conv_swin,unet_lstm,u_ffno,swin,convlstm,vit'

def discover(a):
    tasks=[];missing=[];event_reference=None;shape_reference=None
    for model in a.models.split(','):
        for seed in map(int,a.seeds.split(',')):
            p=a.eval_root/model/'models'/f'{model}_s{seed}'/'result.json'
            if not p.exists() or not read(p).get('complete'):
                missing.append(f'{model}_s{seed}');continue
            result=read(p);entry=result['signature']['entry']
            if entry['job']['model']!=model or entry['job']['seed']!=seed:raise ValueError('Frozen entry mismatch')
            ev=read(a.eval_root/model/'events_2021.json')['events']
            ev=sorted((str(e['id']),e['start'],e['end']) for e in ev if str(e['id']) in a.events.split(','))
            if {e[0] for e in ev}!=set(a.events.split(',')):raise ValueError('Requested event missing')
            cache=Path(result['signature']['prepared']['cache'])
            shape=read(cache/'complete.json')['shapes']['inputs.npy']
            dims=[entry['job']['hyperparams']['seq_length'],*shape[1:]]
            if event_reference is None:event_reference=ev;shape_reference=dims
            if event_reference!=ev or shape_reference!=dims:raise ValueError('Cross-model event windows or input dimensions differ')
            tasks.append(dict(model=model,seed=seed,entry=entry,signature=result['signature']))
    if not tasks:raise ValueError('No completed per-family evaluations; verify --eval-root')
    return tasks,missing

def launch_command(a,t,gpu):
    stage=t['stage'];m=t['model'];s=t['seed']
    base=['--server-root',str(a.server_root),'--eval-root',str(a.eval_root),'--model',m,'--seed',str(s),'--gpu',str(gpu)]
    if stage=='diagnostics':return [sys.executable,str(HERE/'_diagnostics/run.py'),'worker',*base,'--output',str(a.diagnostics_root),'--events',a.events]
    if stage=='pilots':return [sys.executable,str(HERE/'pilot.py'),'--server-root',str(a.server_root),'--job',str(t['job_file']),'--root',str(a.output/'pilot_training'),'--gpu',str(gpu)]
    return [sys.executable,str(HERE/'worker.py'),'--stage',stage,*base,'--output',str(a.output),'--events',a.events,'--stride',str(a.stride)]

def main(a):
    if a.mode=='status':
        p=a.output/'status.json';print(json.dumps(read(p),indent=2) if p.exists() else 'No status file yet');return
    tasks,missing=discover(a)
    print('Ready frozen evaluations:',len(tasks),'Unavailable:',missing,flush=True)
    if a.mode=='inspect':
        print(json.dumps(dict(gpus=inventory(),entries=[f'{t["model"]}_s{t["seed"]}' for t in tasks],missing=missing),indent=2));return
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scripts={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py')}
    plan=dict(version='2.1.1',tasks=tasks,missing=missing,events=a.events,stride=a.stride,
        eval_root=str(a.eval_root),diagnostics_root=str(a.diagnostics_root),scripts=scripts,lr_pilots=a.lr_pilots,
        pilot_updates=a.pilot_updates,source_revision=read(HERE/'SOURCE_REVISION.json') if (HERE/'SOURCE_REVISION.json').exists() else None,policy='Stage priority with per-model dependencies; no 2021 selection')
    freeze(a.output/'plan.json',plan)
    queue=[]
    for stage in ['diagnostics','resources','robustness']:
        for t in tasks:queue.append(dict(t,stage=stage,id=f'{stage}_{t["model"]}_s{t["seed"]}'))
    if a.lr_pilots:
        for t in tasks:
            if t['seed']!=42 or t['model'] in ('fno','ffno','tno'):continue
            for lr in (5e-5,1e-4,2e-4):
                job=copy.deepcopy(t['entry']['job']);job['hyperparams']['max_lr']=lr
                job.update(seed=42,config_id=f'{t["model"]}_lrpilot_{lr:g}_u{a.pilot_updates}_s42',
                    stage='pilot',max_updates=a.pilot_updates,eval_every_updates=2565,early_stop_patience=0)
                jf=a.output/'pilot_jobs'/(job['config_id']+'.json');freeze(jf,job)
                queue.append(dict(t,stage='pilots',id=job['config_id'],job_file=jf))
    done=set();failed=[];active={};stop=False;start=time.monotonic()
    gpus=list(inventory()) if a.gpus=='auto' else [int(g) for g in a.gpus.split(',')]
    if len(gpus)!=len(set(gpus)) or not set(gpus)<=set(inventory()):raise ValueError('Invalid GPU list')
    def halt(*_):
        nonlocal stop
        stop=True;print('Stop requested: no new work; waiting for owned workers to finish.',flush=True)
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def eligible(t):
        previous={'resources':'diagnostics','robustness':'resources'}.get(t['stage'])
        if previous and f'{previous}_{t["model"]}_s{t["seed"]}' not in done:return False
        if t['stage']=='pilots':return not any(q['stage']!='pilots' for q in queue) and not any(v[3]['stage']!='pilots' for v in active.values())
        return True
    while queue or active:
        for gpu,(proc,log,glock,t) in list(active.items()):
            if proc.poll() is None:continue
            log.close();glock.close();del active[gpu]
            if proc.returncode:
                failed.append(dict(task=t['id'],returncode=proc.returncode));print('[FAILED]',t['id'],flush=True)
            else:done.add(t['id']);print('[DONE]',t['id'],flush=True)
        if failed:stop=True
        if time.monotonic()-start>a.launch_hours*3600:stop=True
        if not stop:
            inv=inventory()
            for gpu in gpus:
                if gpu in active or inv[gpu]['busy'] or len(active)>=a.max_workers:continue
                t=next((q for q in queue if eligible(q)),None)
                if t is None:continue
                glock=Path('/tmp',f'swan-resource-{inv[gpu]["uuid"]}.lock').open('a+')
                try:fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:glock.close();continue
                if inventory()[gpu]['busy']:glock.close();continue
                logdir=a.output/'logs';logdir.mkdir(exist_ok=True)
                log=(logdir/(t['id']+'.log')).open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=inv[gpu]['uuid'],MPLBACKEND='Agg',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                proc=subprocess.Popen(launch_command(a,t,gpu),env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(proc,log,glock,t);queue.remove(t);print('[START]',t['id'],'GPU',gpu,flush=True)
        atomic(a.output/'status.json',dict(updated=time.time(),completed=len(done),active={str(g):v[3]['id'] for g,v in active.items()},
            queued=[t['id'] for t in queue],failed=failed,missing_evaluations=missing,draining=stop))
        if stop and not active:break
        if active or queue:time.sleep(5)
    subprocess.run([sys.executable,str(HERE/'report.py'),'--root',str(a.output)],check=True)
    if not queue and not failed:atomic(a.output/'completed.json',dict(tasks=sorted(done),missing_evaluations=missing))
    lock.close()
    if failed:raise RuntimeError('A worker failed. Inspect logs; unchanged outputs can be resumed.')
    if queue:print('Launch window ended. Re-run the same command to continue; existing valid results are reused.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['inspect','run','status'])
    p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'))
    p.add_argument('--eval-root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--diagnostics-root',type=Path)
    p.add_argument('--models',default=MODELS);p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--gpus',default='auto');p.add_argument('--max-workers',type=int,default=7)
    p.add_argument('--events',default='2109,2112,2114');p.add_argument('--stride',type=int,default=6)
    p.add_argument('--launch-hours',type=float,default=12);p.add_argument('--lr-pilots',action='store_true')
    p.add_argument('--pilot-updates',type=int,default=7695)
    a=p.parse_args();a.server_root=a.server_root.resolve()
    a.eval_root=(a.eval_root or a.server_root/'runs/iclr_parallel_v4/evaluation').resolve()
    a.output=(a.output or a.server_root/'runs/resource_checks_2_1_1').resolve()
    a.diagnostics_root=(a.diagnostics_root or a.server_root/'runs/iclr_event_diagnostics_v1').resolve()
    if a.stride<1 or a.max_workers<1 or a.launch_hours<=0 or a.pilot_updates<2565 or a.pilot_updates%2565:p.error('Invalid budget/stride/concurrency')
    main(a)
