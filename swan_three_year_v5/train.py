#!/usr/bin/env python3
"""Train fresh three-year-development models in an isolated package and result root."""
import argparse,copy,fcntl,hashlib,os,signal,sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import campaign as c
from settings import load,paths,environment,HERE
from prepare_data import stamp

def plan(a,r,root):
    prepared=c.read(root/'data/prepared.json');pkg=HERE/'trainer'
    if prepared['output_stamp']!=stamp(root/'data/training_2019_2021.nc'):raise ValueError('Merged file changed')
    hashes=r.repair.code_hashes(pkg);expected={k:v['v5_sha256'] for k,v in c.read(HERE/'trainer_changes.json').items()}
    if hashes!=expected:raise ValueError('Isolated v5 trainer changed')
    asset=r.repair.asset_signature(Path(a['server_root']),str(root/'data/training_2019_2021.nc'))
    jobs=[]
    for model in a['models']:
        control=c.read(root/'controls'/f'{model}.json');base=control['entries'][0]['job']
        for seed in a['seeds']:
            j=copy.deepcopy(base);j.update(config_id=f'{model}_three_year_v5_s{seed}',seed=seed,stage='pilot',
                time_steps=prepared['frames'],epochs=30,early_stop_patience=0,train_fraction=1.,
                max_updates=control['max_updates'],eval_every_updates=control['eval_every_updates'],
                code_hashes=hashes,asset_signature=asset,protocol_version=r.repair.VERSION)
            jobs.append(j)
    return jobs,dict(version=r.repair.VERSION,code_hashes=hashes,asset_signature=asset,direction_policy=jobs[0]['bnd_dir_transform'])
def main():
    a=load();server,root=paths(a)
    if a['require_v4_complete'] and not (server/'runs/iclr_parallel_v4/completed.json').exists():
        raise RuntimeError('v4 is not complete. v5 training has not started. CPU preparation can be run separately.')
    if not a['require_v4_complete'] and set(a['gpus']) & set(range(8)):
        raise ValueError('Concurrent v4 mode requires extra physical GPUs with indices >=8; GPU 0..7 remain reserved for v4')
    environment(a);os.environ['CUDA_VISIBLE_DEVICES']=''
    r=c.load_runner(HERE/'trainer');jobs,protocol=plan(a,r,root)
    c.freeze(root/'config.json',a);c.freeze(root/'plan.json',jobs);c.freeze(root/'protocol.json',protocol)
    c.freeze(root/'evaluation_protocol.json',c.read(HERE/'evaluation_protocol.json'))
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    with ExitStack() as stack:
        f=stack.enter_context((root/'v5.lock').open('a+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a['require_v4_complete']:
            # Hold the v4 controller lock to prevent its relaunch during training.
            path=server/'runs/iclr_parallel_v4/controller.lock'
            old=stack.enter_context(path.open('a+'));fcntl.flock(old,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpus=a['gpus'][:a['training_workers']]
        inv=r.gpu_inventory()
        if any(g not in inv for g in gpus):raise ValueError('Requested GPU is absent')
        args=SimpleNamespace(root=root,package=HERE/'trainer',gpus=gpus)
        probes=[]
        for j in jobs[::3]:
            q=copy.deepcopy(j);q.update(config_id='probe_'+j['config_id'],max_updates=2,eval_every_updates=2);probes.append(q)
        # The reference is set by the first new smoke result, never the old split.
        original_compare=r.compare_protocols
        def compare(rows):
            clean=[s for s in rows if s is not None]
            if clean:original_compare(clean)
        r.compare_protocols=compare
        checks=c.run_stage(args,r,'preflight',probes,None,True)
        if len(checks)!=2:raise RuntimeError('v5 preflight failed; do not silently resize models')
        ref=checks[0];r.compare_protocols=original_compare
        results=c.run_stage(args,r,'training',jobs,ref)
        if len(results)!=6:raise RuntimeError('Some v5 fits failed; successful results preserved')
        for s in results:
            if s['repair_audit']['updates']!=s['repair_job']['max_updates']:raise ValueError('Update budget mismatch')
        c.export(root,results)
        c.freeze(root/'selected_6.json',[dict(job=s['repair_job'],checkpoint=s['best_weight']) for s in sorted(results,key=lambda s:(s['model'],s['seed']))])
        c.atomic(root/'training_completed.json',dict(models=2,runs=6,training_years=[2019,2020,2021],heldout_year=2022))
        print('[TRAINING COMPLETE] Run EVALUATE_2022.sh once 2022 output and boundaries are ready.',flush=True)
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
