"""Verify local data/trainer provenance before scheduling work."""
import hashlib,os
from pathlib import Path
import campaign as c
HERE=Path(__file__).resolve().parent

def context(server):
    package=server/'swan_repaired_v1';parent=server/'runs/repaired_timegap_v1';old=server/'runs/iclr_expanded_v1'
    os.environ.update(SWAN_SERVER_ROOT=str(server),SWAN_DATA_PATH=str(server/'wavm-Waves_2019_2020_v2.nc'))
    r=c.load_runner(package);expected=c.read(HERE/'expected_hashes.json');protocol=c.read(parent/'protocol.json')
    if r.repair.code_hashes(package)!=expected:raise ValueError('Trainer hashes changed')
    actual=dict(version=r.repair.VERSION,code_hashes=expected,asset_signature=r.repair.asset_signature(server,str(server/'wavm-Waves_2019_2020_v2.nc')),direction_policy=protocol['direction_policy'])
    if actual!=protocol or c.read(old/'protocol.json')!=protocol:raise ValueError('Training protocol/assets changed')
    bases={j['model']:j for j in c.read(parent/'plan_pilot.json')}
    ref=r.checked_summary(parent,bases['fno'])
    if ref is None:raise ValueError('Missing verified reference pilot')
    return r,protocol,bases,ref

def verified_entry(summary,r):
    job=summary['repair_job'];rd=Path(summary['best_weight']).resolve().parent
    if rd.name!=r.run_name(job):raise ValueError('Checkpoint/run-name mismatch')
    check=r.checked_summary(rd.parent,job)
    if check is None or check['best_weight']!=summary['best_weight']:raise ValueError('Unverified checkpoint')
    norm=rd/'normalization.json';direction=c.read(rd/'direction_manifest.json')
    if direction!=summary['repair_audit']['direction']:raise ValueError('Direction audit mismatch')
    return dict(job=job,checkpoint=summary['best_weight'],normalization=str(norm),direction=direction['chosen'],
        source_summary=str(rd/'run_summary.json'),normalization_sha256=hashlib.sha256(norm.read_bytes()).hexdigest())

def original_selected(server,model,r,bases):
    root=server/'runs/iclr_bc_typhoon_v3';old=server/'runs/iclr_expanded_v1';parent=server/'runs/repaired_timegap_v1'
    path=root/f'selection_{model}.json'
    if not path.exists():return None
    base=c.read(path)['job'];summaries=[]
    for source in (parent,old/'A',old/'B'):
        s=r.checked_summary(source,base)
        if s is not None:summaries.append(s)
    if len(summaries)!=1:raise ValueError('Selected seed42 missing or ambiguous')
    if base['config_id']==bases[model]['config_id']:
        extra=server/'runs/repaired_timegap_extra_v1'
        jobs=[j for j in c.read(extra/'extra_plan.json')['jobs'] if j['model']==model]
        for j in jobs:
            s=r.checked_summary(extra,j)
            if s is None:return None
            summaries.append(s)
    else:
        for seed in (43,44):
            j=c.make_job(base,'C',seed=seed)
            s=r.checked_summary(root/'C',j) or r.checked_summary(old/'C',j)
            if s is None:return None
            summaries.append(s)
    if sorted(s['seed'] for s in summaries)!=[42,43,44]:raise ValueError('Expected three seeds')
    r.compare_protocols(summaries)
    return sorted(summaries,key=lambda s:s['seed'])
