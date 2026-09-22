#!/usr/bin/env python3
"""Controlled handover to disjoint GPU lanes and per-family held-out evaluation."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
import campaign as c
from common import context,original_selected,verified_entry
from plans import BASELINES,jobs
HERE=Path(__file__).resolve().parent

def lock(stack,path):
    path.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(path.open('a+'))
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)

def proc_info():
    result={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            args=(p/'cmdline').read_bytes().decode().strip('\0').split('\0');cwd=(p/'cwd').resolve()
            resolved=[str((cwd/x).resolve()) if x.endswith('.py') else x for x in args]
            result[int(p.name)]=dict(ppid=int(fields[1]),start=fields[19],args=resolved)
        except (OSError,UnicodeError,IndexError):pass
    return result

def old_controller(server,info):
    script=str(server/'swan_bc_typhoon_v3/run_campaign.py')
    found=[pid for pid,v in info.items() if script in v['args']]
    if len(found)>1:raise RuntimeError('Multiple v3 controllers; no handover')
    return found[0] if found else None

def descendants(info,pid):
    found={pid}
    while True:
        nxt=found|{p for p,v in info.items() if v['ppid'] in found}
        if nxt==found:return found
        found=nxt

def stop_old(server,work):
    info=proc_info();pid=old_controller(server,info)
    if pid is None:return
    status=c.read(server/'runs/iclr_bc_typhoon_v3/status.json')
    if status.get('stage')!='B_and_C':raise RuntimeError('v3 already left B/C; no automatic handover performed')
    owned={p:info[p]['start'] for p in descendants(info,pid)}
    c.atomic(work/'handover.json',dict(pid=pid,process_start=info[pid]['start'],descendants=owned,prior_status=status,time=time.time()))
    print('[HANDOVER] Stop verified v3 controller; active training resumes saved checkpoints.',flush=True)
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+90
    while True:
        now=proc_info();remaining=[p for p,start in owned.items() if p in now and now[p]['start']==start]
        if not remaining:break
        if time.monotonic()>deadline:raise RuntimeError(f'Old processes still exiting: {remaining}. No new workers started. Inspect and rerun.')
        time.sleep(2)

def start(cmd,path):
    path.parent.mkdir(parents=True,exist_ok=True);log=path.open('a')
    return subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1',MPLBACKEND='Agg')),log

def baseline_selected(server,model,r,plan):
    source=server/'runs/iclr_parallel_v4/baselines/fixed'
    rows=[r.checked_summary(source,j) for j in plan if j['model']==model]
    return rows if len(rows)==3 and all(s is not None for s in rows) else None

def aggregate(work):
    import csv
    for filename in ('event_metrics_by_seed.csv','event_metrics_seed_summary.csv'):
        rows=[]
        for p in sorted((work/'evaluation').glob('*/'+filename)):
            if not (p.parent/'completed.json').exists():continue
            with p.open() as f:rows.extend(csv.DictReader(f))
        if rows:
            keys=list(dict.fromkeys(k for row in rows for k in row));target=work/('available_'+filename);tmp=target.with_suffix('.tmp')
            with tmp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
            tmp.replace(target)

def evaluation_gpus(finished_lanes):
    # Evaluation waits until the original lane has exited.
    return (0,1) if 'original' in finished_lanes else ()

def main(a):
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    server=a.server_root.resolve();work=server/'runs/iclr_parallel_v4';oldroot=server/'runs/iclr_bc_typhoon_v3'
    r,protocol,bases,ref=context(server)
    for name,digest in c.read(HERE/'v3_source_hashes.json').items():
        if hashlib.sha256((server/'swan_bc_typhoon_v3'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Installed v3 changed: '+name)
    for p in (server/'swan_2021_nc_v2/wavm-Waves.nc',server/'bnd_2021_v2'):
        if not p.exists():raise FileNotFoundError(p)
    if c.read(oldroot/'protocol.json')!=protocol:raise ValueError('v3 protocol mismatch')
    for m in ('fno','ffno'):
        if original_selected(server,m,r,bases) is None:raise ValueError(f'{m} C repetitions are not complete')
    inv=r.gpu_inventory()
    if not set(range(8))<=set(inv):raise ValueError('This launcher requires eight physical GPUs, indices 0..7')
    plan=jobs(bases['fno']);print('GPU 0,1: existing B/C, then evaluation. GPU 2..7: seven fixed baselines, 21 fits.',flush=True)
    print('Baseline configs:',json.dumps({m:next(j['hyperparams'] for j in plan if j['model']==m) for m in BASELINES},indent=2),flush=True)
    if not a.apply:
        print('Inspection only. START.sh uses --apply for the handover.');return
    work.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        lock(stack,work/'controller.lock')
        c.freeze(work/'protocol.json',protocol);c.freeze(work/'baseline_plan.json',plan)
        c.freeze(work/'controller_config_v41.json',dict(version=41,server=str(server),gpu_lanes=dict(original=[0,1],evaluation=[0,1],baselines=[2,3,4,5,6,7]),
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))},selection='original v3 validation-only selection; remaining seven fixed configs, no search'))
        c.freeze(work/'evaluation_protocol.json',c.read(oldroot/'evaluation_protocol.json'))
        for m in ('fno','ffno'):
            rows=original_selected(server,m,r,bases)
            c.freeze(work/'evaluation'/m/'selected_2021.json',[verified_entry(s,r) for s in rows])
        import events
        # Reuse the already downloaded JMA source; verify it before stopping any process.
        source=server/'swan_bc_typhoon_v3/jma_besttrack.txt'
        if source.exists() and not (HERE/'jma_besttrack.txt').exists():(HERE/'jma_besttrack.txt').write_bytes(source.read_bytes())
        events.verify_source(HERE)
        events.build_events(server/'swan_2021_nc_v2/wavm-Waves.nc',HERE,c.read(work/'evaluation_protocol.json'),work/'events_2021.json')
        stop_old(server,work)
        for path in (oldroot/'controller.lock',server/'runs/iclr_expanded_v1/campaign.lock',server/'runs/iclr_2021_v2/v2.lock',server/'runs/iclr_typhoon_2021_v3/v2.lock'):
            lock(stack,path)
        # Refuse unowned CUDA work before starting the disjoint lanes.
        occupied=[g for g,v in r.gpu_inventory().items() if g in range(8) and v['busy']]
        if occupied:raise RuntimeError(f'Unexpected GPU processes on {occupied}. No new jobs started; inspect before rerunning.')
        c.atomic(work/'handover_complete.json',dict(time=time.time(),previous_root=str(oldroot)))
        lanes={};evaluations={};errors={};failed_models=set();finished_lanes=set()
        try:
            for kind in ('original','baselines'):
                proc,log=start([sys.executable,str(HERE/'lane.py'),kind,'--server-root',str(server)],work/'logs'/(kind+'.log'))
                lanes[kind]=(proc,log)
            while True:
                for kind,(proc,log) in list(lanes.items()):
                    rc=proc.poll()
                    if rc is None:continue
                    log.close();del lanes[kind];finished_lanes.add(kind)
                    if rc:errors[kind]=dict(returncode=rc,log=str(work/'logs'/(kind+'.log')))
                for gpu,(proc,log,model) in list(evaluations.items()):
                    rc=proc.poll()
                    if rc is None:continue
                    log.close();del evaluations[gpu]
                    if rc or not (work/'evaluation'/model/'completed.json').exists():
                        errors[model]=dict(returncode=rc,log=str(work/'logs'/('eval_'+model+'.log')));failed_models.add(model)
                    aggregate(work)
                active_models={m for p,l,m in evaluations.values()};ready=[];all_summaries=[]
                for model in ('fno','ffno','tno',*BASELINES):
                    rows=original_selected(server,model,r,bases) if model in c.MODELS else baseline_selected(server,model,r,plan)
                    if rows is None:continue
                    r.compare_protocols([ref]+rows)
                    for result in rows:
                        audit=result['repair_audit']
                        if audit.get('updates')!=ref['repair_audit']['target_updates'] or audit.get('early_stop'):
                            raise ValueError('Full-run successful-update budgets differ')
                    all_summaries.extend(rows)
                    root=work/'evaluation'/model
                    c.freeze(root/'selected_2021.json',[verified_entry(s,r) for s in rows])
                    if model not in active_models and model not in failed_models and not (root/'completed.json').exists():ready.append(model)
                if all_summaries:c.export(work,all_summaries)
                inv=r.gpu_inventory()
                for gpu in evaluation_gpus(finished_lanes):
                    if not ready:break
                    if gpu in evaluations or inv[gpu]['busy']:continue
                    model=ready.pop(0);proc,log=start([sys.executable,str(HERE/'family_eval.py'),'--server-root',str(server),'--model',model,'--gpu',str(gpu)],work/'logs'/('eval_'+model+'.log'))
                    evaluations[gpu]=(proc,log,model);print('[EVAL START]',model,'GPU',gpu,flush=True)
                completed=sorted(p.parent.name for p in (work/'evaluation').glob('*/completed.json'))
                c.atomic(work/'status.json',dict(updated=time.time(),training_lanes=list(lanes),finished_lanes=sorted(finished_lanes),
                    evaluation_active={str(g):m for g,(p,l,m) in evaluations.items()},evaluation_ready=ready,
                    evaluation_completed=completed,errors=errors,target_families=10))
                if not lanes and not evaluations and not ready:
                    aggregate(work)
                    if len(completed)!=10 or errors:
                        c.atomic(work/'partial_completed.json',dict(completed_families=completed,errors=errors))
                        raise RuntimeError('Some families are incomplete. Successful results are preserved. Inspect status.json and logs.')
                    c.atomic(work/'completed.json',dict(families=10,seeds_per_family=3,year=2021));break
                time.sleep(10)
        finally:
            children=[(p,l) for p,l in lanes.values()]+[(p,l) for p,l,m in evaluations.values()]
            for proc,log in children:
                if proc.poll() is None:proc.terminate()
            for proc,log in children:
                try:proc.wait(timeout=60)
                except subprocess.TimeoutExpired:print('Child still exiting:',proc.pid,flush=True)
                log.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'));p.add_argument('--apply',action='store_true')
    try:main(p.parse_args())
    except KeyboardInterrupt:raise SystemExit(130)
