#!/usr/bin/env python3
"""Evaluate a complete frozen family without waiting for other architectures."""
import argparse,fcntl,os,signal,subprocess,sys,time
from pathlib import Path
import campaign as c
from common import context
HERE=Path(__file__).resolve().parent

def wait(cmd,env):
    proc=subprocess.Popen(cmd,env=env)
    try:
        rc=proc.wait()
        if rc:raise RuntimeError(f'Evaluator failed ({rc}): {cmd}')
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:raise RuntimeError('Evaluator still exiting; inspect processes before restart')

def main(a):
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    server=a.server_root.resolve();work=server/'runs/iclr_parallel_v4';root=work/'evaluation'/a.model
    selected=c.read(root/'selected_2021.json');r,protocol,bases,ref=context(server)
    if len(selected)!=3 or {(e['job']['model'],e['job']['seed']) for e in selected}!={(a.model,s) for s in (42,43,44)}:raise ValueError('Three frozen model seeds required')
    summaries=[];norms=[]
    for e in selected:
        j=e['job'];rd=Path(e['checkpoint']).parent
        summary=r.checked_summary(rd.parent,j)
        if summary is None or summary['best_weight']!=e['checkpoint']:raise ValueError('Checkpoint is not verified')
        summaries.append(summary);norms.append(c.read(e['normalization']))
    r.compare_protocols([ref]+summaries)
    if any(n!=c.read(Path(ref['best_weight']).parent/'normalization.json') for n in norms):raise ValueError('Normalization differs from reference')
    c.freeze(root/'evaluation_protocol.json',c.read(work/'evaluation_protocol.json'))
    c.freeze(root/'events_2021.json',c.read(work/'events_2021.json'))
    shared=work/'shared_2021';shared.mkdir(exist_ok=True)
    env=dict(os.environ,MPLBACKEND='Agg',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    import events
    for e in selected:
        name=f'{a.model}_s{e["job"]["seed"]}';ep=root/'entries'/(name+'.json');c.freeze(ep,e)
        prepared=root/'prepared'/(name+'.json')
        # One cache writer at a time; all families share the same content-keyed cache.
        with (shared/'prepare.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            c.atomic(root/'status.json',dict(stage='prepare',model=name,updated=time.time()))
            wait([sys.executable,str(HERE/'evaluate_2021.py'),'prepare','--server-root',str(server),
                '--package',str(server/'swan_repaired_v1'),'--root',str(shared),'--entry',str(ep),'--result',str(prepared),
                '--nc',str(server/'swan_2021_nc_v2/wavm-Waves.nc'),'--bnd',str(server/'bnd_2021_v2'),
                '--reference',str(server/'wavm-Waves_2019_2020_v2.nc'),'--max-bnd-gap-hours','6'],env)
            events.freeze_snapshots(prepared,root/'events_2021.json',root/'snapshots.json')
        result=root/'models'/name/'result.json';result.parent.mkdir(parents=True,exist_ok=True)
        c.atomic(root/'status.json',dict(stage='evaluate',model=name,gpu=a.gpu,updated=time.time()))
        wait([sys.executable,str(HERE/'evaluate_2021.py'),'evaluate','--server-root',str(server),
            '--package',str(server/'swan_repaired_v1'),'--root',str(root),'--entry',str(ep),
            '--prepared',str(prepared),'--result',str(result)],dict(env,CUDA_VISIBLE_DEVICES=str(a.gpu)))
    from report_events import report
    report(root)
    c.atomic(root/'completed.json',dict(model=a.model,seeds=[42,43,44],year=2021))
    c.atomic(root/'status.json',dict(stage='complete',updated=time.time()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--server-root',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--gpu',type=int,required=True)
    try:main(p.parse_args())
    except KeyboardInterrupt:raise SystemExit(130)
