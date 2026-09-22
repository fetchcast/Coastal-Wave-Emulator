#!/usr/bin/env python3
"""Evaluate one frozen cohort/family with its own normalization and trainer package."""
import argparse,fcntl,hashlib,os,signal,subprocess,sys,time
from pathlib import Path
import campaign as c
from settings import load,paths,HERE

def wait(cmd,env):
    p=subprocess.Popen(cmd,env=env)
    try:
        if p.wait():raise RuntimeError('2022 evaluator failed')
    finally:
        if p.poll() is None:p.terminate();p.wait(timeout=45)
def main(a):
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    config=load();server,work=paths(config);root=work/'evaluation'/a.group
    meta=c.read(root/'cohort.json');package=Path(meta['package']);selected=c.read(root/'selected_2022.json')
    os.environ.update(SWAN_SERVER_ROOT=str(server),SWAN_DATA_PATH=meta['reference'])
    for year in (2019,2020,2021):os.environ[f'SWAN_BND_DIR_{year}']=str(server/f'bnd_{year}_v2')
    r=c.load_runner(package)
    for e in selected:
        j=e['job'];rd=Path(e['checkpoint']).parent
        r.repair.verify_job(j,package)
        s=r.checked_summary(rd.parent,j)
        if s is None or s['best_weight']!=e['checkpoint']:raise ValueError('Unverified checkpoint')
        if r.repair.code_hashes(package)!=j['code_hashes']:raise ValueError('Cohort trainer code changed')
        if hashlib.sha256(Path(e['normalization']).read_bytes()).hexdigest()!=e['normalization_sha256']:raise ValueError('Normalization changed')
    shared=work/'shared_2022';shared.mkdir(exist_ok=True)
    import events
    c.freeze(root/'evaluation_protocol.json',c.read(work/'evaluation_protocol.json'))
    c.freeze(root/'events_2022.json',c.read(work/'events_2022.json'))
    env=dict(os.environ,SWAN_SERVER_ROOT=str(server),SWAN_DATA_PATH=meta['reference'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MPLBACKEND='Agg')
    for year in (2019,2020,2021):env[f'SWAN_BND_DIR_{year}']=str(server/f'bnd_{year}_v2')
    for e in selected:
        name=f'{e["job"]["model"]}_s{e["job"]["seed"]}';ep=root/'entries'/(name+'.json');c.freeze(ep,e)
        prepared=root/'prepared'/(name+'.json')
        with (shared/'prepare.lock').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            c.atomic(root/'status.json',dict(stage='prepare',run=name,updated=time.time()))
            wait([sys.executable,str(HERE/'evaluate_2022.py'),'prepare','--server-root',str(server),'--package',str(package),
                '--root',str(shared),'--entry',str(ep),'--result',str(prepared),'--nc',config['source_2022'],
                '--bnd',config['boundary_2022'],'--reference',meta['reference'],'--max-bnd-gap-hours',str(config['max_boundary_gap_hours'])],env)
            events.freeze_snapshots(prepared,root/'events_2022.json',root/'snapshots.json')
        result=root/'models'/name/'result.json';result.parent.mkdir(parents=True,exist_ok=True)
        c.atomic(root/'status.json',dict(stage='evaluate',run=name,gpu=a.gpu,updated=time.time()))
        wait([sys.executable,str(HERE/'evaluate_2022.py'),'evaluate','--server-root',str(server),'--package',str(package),
            '--root',str(root),'--entry',str(ep),'--prepared',str(prepared),'--result',str(result)],dict(env,CUDA_VISIBLE_DEVICES=str(a.gpu)))
    from report_events import report
    report(root);c.atomic(root/'completed.json',dict(group=a.group,year=2022,seeds=[42,43,44]))
    c.atomic(root/'status.json',dict(stage='complete',updated=time.time()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--gpu',type=int,required=True)
    try:main(p.parse_args())
    except KeyboardInterrupt:raise SystemExit(130)
