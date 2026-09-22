#!/usr/bin/env python3
"""Run seed replications on GPUs 3-7 without changing the parent campaign."""
import argparse
import copy
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys


def make_jobs(pilots):
    models = {j['model']: j for j in pilots}
    if len(pilots) != 3 or set(models) != {'fno', 'tno', 'ffno'}:
        raise ValueError('Expected exactly three parent pilot jobs')
    if any(j['seed'] != 42 or j['stage'] != 'pilot' for j in pilots):
        raise ValueError('Expected seed-42 parent pilots')
    order = [('tno', 43), ('fno', 43), ('ffno', 43),
             ('tno', 44), ('ffno', 44), ('fno', 44)]
    result = []
    for model, seed in order:
        j = copy.deepcopy(models[model])
        if not j['config_id'].endswith('_s42'):
            raise ValueError('Unexpected parent config ID')
        j['seed'] = seed
        j['config_id'] = j['config_id'][:-4] + f'_s{seed}'
        result.append(j)
    return result


def validate_parent(runner, parent, package, server, data):
    repair = runner.repair
    protocol = json.loads((parent / 'protocol.json').read_text())
    current = dict(version=repair.VERSION, code_hashes=repair.code_hashes(package),
                   asset_signature=repair.asset_signature(server, data),
                   direction_policy=protocol['direction_policy'])
    if current != protocol:
        raise ValueError('Parent code/data fingerprints differ; no extra jobs started')
    if not hasattr(repair, 'continuous_starts'):
        raise ValueError('Install the time-gap update first')
    smoke = json.loads((parent / 'plan_smoke.json').read_text())
    pilots = json.loads((parent / 'plan_pilot.json').read_text())
    if len(smoke) != 3 or {j['model'] for j in smoke} != {'fno', 'tno', 'ffno'}:
        raise ValueError('Parent smoke plan is incomplete')
    for j in smoke + pilots:
        if (j.get('code_hashes') != protocol['code_hashes'] or
                j.get('asset_signature') != protocol['asset_signature'] or
                j.get('protocol_version') != protocol['version'] or
                j.get('bnd_dir_transform') != protocol['direction_policy']):
            raise ValueError('Parent job does not match its protocol')
    summaries = [runner.checked_summary(parent, j) for j in smoke]
    if any(s is None for s in summaries):
        raise ValueError('All three parent smoke jobs must have verified successful summaries')
    runner.compare_protocols(summaries)
    return protocol, pilots, summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server-root', default='/home/jovyan/swan')
    ap.add_argument('--package')
    ap.add_argument('--parent-root')
    ap.add_argument('--root')
    ap.add_argument('--data')
    ap.add_argument('--gpus', default='3,4,5,6,7')
    ap.add_argument('--plan', action='store_true')
    a = ap.parse_args()
    server = Path(a.server_root).resolve()
    package = Path(a.package or server / 'swan_repaired_v1').resolve()
    parent = Path(a.parent_root or server / 'runs/repaired_timegap_v1').resolve()
    root = Path(a.root or server / 'runs/repaired_timegap_extra_v1').resolve()
    if root == parent or root in parent.parents or parent in root.parents:
        raise ValueError('Extra results must be in a separate sibling directory')
    data = str(Path(a.data or server / 'wavm-Waves_2019_2020_v2.nc').resolve())
    gpus = [int(x.strip()) for x in a.gpus.split(',') if x.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or not set(gpus) <= {3, 4, 5, 6, 7}:
        raise ValueError('This addon accepts only distinct GPU IDs from 3,4,5,6,7')
    sys.path.insert(0, str(package))
    runner = importlib.import_module('run_repaired')
    if Path(runner.__file__).resolve() != package / 'run_repaired.py':
        raise ValueError('Wrong runner module imported')
    os.environ.update(SWAN_SERVER_ROOT=str(server), SWAN_DATA_PATH=data,
                      SWAN_RESULTS_ROOT=str(root))
    protocol, pilots, smoke_summaries = validate_parent(runner, parent, package, server, data)
    jobs = make_jobs(pilots)
    for i, j in enumerate(jobs):
        slot = str(gpus[i]) if i < len(gpus) else 'next free selected GPU'
        print(f'[PLAN] {j["config_id"]} slot={slot}', flush=True)
    if a.plan:
        return
    root.mkdir(parents=True, exist_ok=True)
    with open(root / 'launcher.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.freeze(root / 'protocol.json', protocol)
        runner.freeze(root / 'extra_plan.json', dict(parent_root=str(parent), jobs=jobs))
        # Reuse only completed checks for exactly the verified parent assets.
        for p in sorted((parent / '_source_checks').glob('*.json')):
            record = json.loads(p.read_text())
            if record.get('checked') is True and record.get('signature') == protocol['asset_signature']:
                runner.freeze(root / '_source_checks' / p.name, record)
        inv = runner.gpu_inventory()
        if any(g not in inv for g in gpus):
            raise ValueError('A selected GPU does not exist')
        minimum = min(inv[g]['mib'] for g in gpus)
        if any(runner.parameter_bytes_floor(j) > minimum * .85 for j in jobs):
            raise ValueError('A job exceeds the selected GPU memory floor')
        print('[READY] Parent smoke results verified; no parent jobs or source files modified', flush=True)
        summaries = runner.launch(root, jobs, gpus)
        runner.compare_protocols(smoke_summaries + summaries)
        runner.repair.atomic_json(root / 'extra_completed.json',
                                  dict(protocol=protocol, jobs=[j['config_id'] for j in jobs]))
        runner.report(root)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
