#!/usr/bin/env python3
"""Run disjoint training lanes using the unchanged repaired trainer."""
import argparse,copy,signal,os
from pathlib import Path
from types import SimpleNamespace
import campaign as c
from common import context
from plans import jobs

def main(a):
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    server=a.server_root.resolve();r,protocol,bases,ref=context(server)
    old=server/'runs/iclr_expanded_v1';root=server/'runs/iclr_bc_typhoon_v3';package=server/'swan_repaired_v1'
    if a.kind=='original':
        from training_schedule import run_schedule
        pilots={}
        for source,plan in [(server/'runs/repaired_timegap_v1',list(bases.values())),(server/'runs/repaired_timegap_extra_v1',c.read(server/'runs/repaired_timegap_extra_v1/extra_plan.json')['jobs'])]:
            for j in plan:
                s=r.checked_summary(source,j)
                if s is None:raise ValueError('Parent pilot incomplete')
                pilots[j['model'],j['seed']]=s
        arows={j['config_id']:r.checked_summary(old/'A',j) for j in c.read(old/'A/plan.json')}
        if len(arows)!=29 or any(s is None for s in arows.values()):raise ValueError('A incomplete')
        run_schedule(a,r,root,old,package,c.read(old/'B/plan.json'),bases,pilots,[0,1],protocol,arows)
    else:
        root=server/'runs/iclr_parallel_v4/baselines';root.mkdir(parents=True,exist_ok=True)
        plan=jobs(bases['fno']);c.freeze(root/'fixed_plan.json',plan);c.freeze(root/'protocol.json',protocol)
        for label in ('preflight','fixed'):
            for p in (old/'A/_source_checks').glob('*.json'):
                v=c.read(p)
                if v.get('checked') is True and v.get('signature')==protocol['asset_signature']:c.freeze(root/label/'_source_checks'/p.name,v)
        probes=[]
        for j in plan[:7]:
            q=copy.deepcopy(j);q.update(config_id='probe_'+j['config_id'],max_updates=2,eval_every_updates=2);probes.append(q)
        args=SimpleNamespace(root=root,package=package,gpus=[2,3,4,5,6,7])
        checks=c.run_stage(args,r,'preflight',probes,ref,True)
        if len(checks)!=7:raise RuntimeError('Preflight resource failure; no silent model downsizing. Inspect baselines/preflight.')
        # Verify evaluator construction against each actual smoke checkpoint before full training.
        import torch
        import train_repaired as train
        from evaluate_2021 import model_arguments
        layouts=[]
        for check in checks:
            j=check['repair_job'];hp=j['hyperparams']
            with torch.device('meta'):
                model=train.LegacyCompatibleBenchmarkModel(model_name=j['model'],input_channels=10,output_channels=4,**model_arguments(hp,train.LegacyCompatibleBenchmarkModel))
            state=torch.load(check['best_weight'],map_location='meta',weights_only=True)
            model.load_state_dict(state,strict=True)
            layouts.append(dict(model=j['model'],n_parameters=sum(p.numel() for p in model.parameters()),strict_checkpoint_layout=True))
            del state,model
        c.atomic(root/'evaluator_layout_checks.json',layouts)
        rows=c.run_stage(args,r,'fixed',plan,ref);c.export(root,rows)
        c.atomic(root/'completed.json',dict(completed=len(rows),target=21,resource_skips=21-len(rows)))
        if len(rows)!=21:raise RuntimeError('Some baseline runs failed resource checks')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['original','baselines']);p.add_argument('--server-root',type=Path,required=True)
    try:main(p.parse_args())
    except KeyboardInterrupt:raise SystemExit(130)
