#!/usr/bin/env python3
"""Resume existing A runs and schedule per-family C repeats without B/D/E."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
import campaign as c

HERE = Path(__file__).resolve().parent

def processes(server):
    """Return only processes whose argument is an exact known SWAN script path."""
    scripts = {str(server / 'swan_iclr_campaign_v2' / n) for n in ('run_v2.py', 'campaign.py', 'evaluate_2021.py')}
    scripts |= {str(server / 'swan_repaired_v1' / n) for n in ('run_repaired.py', 'train_repaired.py')}
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if scripts.intersection(args):
                result.append((int(p.name), args))
        except (OSError, UnicodeError):
            pass
    return result

def lock(stack, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = stack.enter_context(path.open('a+'))
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

def ready_candidates(model, bases, pilots, plan, rows):
    expected = [j for j in plan if j['model'] == model]
    if not all(j['config_id'] in rows for j in expected):
        return None
    return [pilots[model, 42]] + [rows[j['config_id']] for j in expected]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server-root', type=Path, default=Path('/home/jovyan/swan'))
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--apply', action='store_true', help='Stop the verified default v2 wrapper and start the new controller.')
    a = p.parse_args()
    server = a.server_root.resolve()
    old = server / 'runs/iclr_expanded_v1'
    root = server / 'runs/iclr_c_direct_v1'
    package = server / 'swan_repaired_v1'
    data = server / 'wavm-Waves_2019_2020_v2.nc'
    parent = server / 'runs/repaired_timegap_v1'
    extra = server / 'runs/repaired_timegap_extra_v1'
    gpus = [int(x) for x in a.gpus.split(',')]
    if not gpus or min(gpus) < 0 or len(gpus) != len(set(gpus)):
        raise ValueError('Invalid GPU list')
    os.environ.update(SWAN_SERVER_ROOT=str(server), SWAN_DATA_PATH=str(data))
    r = c.load_runner(package)
    hashes = c.read(HERE / 'expected_hashes.json')
    if r.repair.code_hashes(package) != hashes:
        raise ValueError('Training code differs from reviewed code; no processes stopped.')
    protocol = c.read(parent / 'protocol.json')
    current = dict(version=r.repair.VERSION, code_hashes=hashes,
                   asset_signature=r.repair.asset_signature(server, str(data)),
                   direction_policy=protocol['direction_policy'])
    if current != protocol or c.read(old / 'protocol.json') != protocol:
        raise ValueError('Protocol/assets differ; no processes stopped.')
    bases = {j['model']: j for j in c.read(parent / 'plan_pilot.json')}
    plan = c.read(old / 'A/plan.json')
    if plan != c.architecture_jobs(bases):
        raise ValueError('Expected the complete original 29-job A plan.')
    if list((old / 'A').glob('*.resource_skip.json')):
        raise ValueError('A has resource skips; selection requires review.')
    pilots = {}
    for source, jobs in [(parent, list(bases.values())), (extra, c.read(extra / 'extra_plan.json')['jobs'])]:
        for j in jobs:
            s = r.checked_summary(source, j)
            if s is None:
                raise ValueError(f'Unverified pilot: {j["config_id"]}')
            pilots[j['model'], j['seed']] = s
    if set(pilots) != {(m, seed) for m in c.MODELS for seed in (42, 43, 44)}:
        raise ValueError('Nine verified pilots required')
    r.compare_protocols(list(pilots.values()))
    rows = {j['config_id']: s for j in plan if (s := r.checked_summary(old / 'A', j)) is not None}
    r.compare_protocols(list(pilots.values()) + list(rows.values()))
    print(f'A completed={len(rows)}/29; remaining={29-len(rows)}', flush=True)
    for m in c.MODELS:
        candidates = ready_candidates(m, bases, pilots, plan, rows)
        print(m, 'ready for C: ' + c.choose(candidates, 1)[0]['config_id'] if candidates else 'waiting for its A runs', flush=True)
    print('B/D/E skipped. C uses seeds 43 and 44 of each A-selected model. No 2021 evaluation in this package.', flush=True)
    expected_wrapper = c.read(HERE/'wrapper_hashes.json')
    for name, digest in expected_wrapper.items():
        if hashlib.sha256((server/'swan_iclr_campaign_v2'/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Original wrapper code differs; no processes stopped.')
    old_procs = processes(server)
    for pid, args in old_procs:
        print('OLD PROCESS', pid, ' '.join(args), flush=True)
    if not a.apply:
        print('Inspection only. Use --apply to perform the controlled handover.', flush=True)
        return
    with ExitStack() as stack:
        lock(stack, root / 'controller.lock')
        c.freeze(root / 'protocol.json', protocol)
        c.freeze(root / 'controller_config.json', dict(version=1, source=str(old), gpus=gpus,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256((HERE/'campaign.py').read_bytes()).hexdigest(),
                  selection='best_validation_Hs_MAE_from_pilot_seed42_and_completed_A', skipped=['B','D','E']))
        wrappers = [(pid, args) for pid, args in old_procs if str(server / 'swan_iclr_campaign_v2/run_v2.py') in args]
        if old_procs:
            if len(wrappers) != 1:
                raise ValueError('Expected exactly one default v2 wrapper; nothing stopped.')
            pid, args = wrappers[0]
            index = args.index(str(server / 'swan_iclr_campaign_v2/run_v2.py'))
            if args[index + 1:]:
                raise ValueError('Custom wrapper arguments detected; nothing stopped.')
            if any(str(server / 'swan_iclr_campaign_v2/evaluate_2021.py') in xs for _, xs in old_procs):
                raise ValueError('2021 evaluation already running; nothing stopped.')
            c.atomic(root / 'handover.json', dict(wrapper_pid=pid, prior_status=c.read(old/'status.json'), time=time.time()))
            print(f'Sending SIGTERM to v2 wrapper {pid}; saved checkpoints will be reused.', flush=True)
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 60
            while processes(server):
                if time.monotonic() > deadline:
                    raise RuntimeError('Old processes still exiting. No new workers started; inspect and rerun.')
                time.sleep(2)
        # Holding both original locks prevents an accidental old-wrapper restart.
        lock(stack, old / 'campaign.lock')
        lock(stack, server / 'runs/iclr_2021_v2/v2.lock')
        if processes(server):
            raise RuntimeError('Old processes detected; no new workers started.')
        c.atomic(root / 'handover_complete.json', dict(time=time.time(), source=str(old)))
        run_schedule(a, r, root, old, package, plan, bases, pilots, gpus, protocol)

def run_schedule(a, r, root, old, package, plan, bases, pilots, gpus, protocol):
    active = {}
    selected = {}
    crows = {}
    reference = pilots['fno', 42]
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            for gpu, (proc, job, source, log) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                del active[gpu]
                s = r.checked_summary(source, job)
                if rc != 0 or s is None:
                    raise RuntimeError(f'Worker failed: {job["config_id"]}; inspect {source/r.run_name(job)}')
                r.compare_protocols([reference, s])
                print('[DONE]', job['config_id'], flush=True)
            rows = {j['config_id']: s for j in plan if (s := r.checked_summary(old/'A', j)) is not None}
            r.compare_protocols([reference] + list(rows.values()))
            pending = [(j, old/'A') for j in plan if j['config_id'] not in rows]
            for model in c.MODELS:
                candidates = ready_candidates(model, bases, pilots, plan, rows)
                if candidates is None:
                    continue
                chosen = c.choose(candidates, 1)[0]
                c.freeze(root / f'selection_{model}.json', dict(job=chosen['repair_job'], validation_hs_mae=c.validation(chosen), source='pilot42+A'))
                selected[model] = chosen
                if chosen['config_id'] == bases[model]['config_id']:
                    for seed in (43, 44):
                        crows[model, seed] = pilots[model, seed]
                    continue
                jobs = [c.make_job(chosen['repair_job'], 'C', seed=seed) for seed in (43, 44)]
                c.freeze(root / f'plan_C_{model}.json', jobs)
                for job in jobs:
                    s = r.checked_summary(root/'C', job)
                    if s is None:
                        pending.append((job, root/'C'))
                    else:
                        r.compare_protocols([reference, s])
                        crows[model, job['seed']] = s
            running = {j['config_id'] for _, j, _, _ in active.values()}
            pending = [(j,s) for j,s in pending if j['config_id'] not in running]
            inv = r.gpu_inventory()
            for gpu in gpus:
                if not pending:
                    break
                if gpu in active or inv[gpu]['busy']:
                    continue
                job, source = pending.pop(0)
                if r.parameter_bytes_floor(job) > inv[gpu]['mib'] * .85:
                    raise RuntimeError(f'Insufficient estimated memory: {job["config_id"]}')
                source.mkdir(parents=True, exist_ok=True)
                for cache in (old/'A/_source_checks').glob('*.json'):
                    value = c.read(cache)
                    if value.get('checked') is True and value.get('signature') == protocol['asset_signature']:
                        c.freeze(source/'_source_checks'/cache.name, value)
                path = source/'_jobs'/(job['config_id']+'.json')
                c.freeze(path, job)
                log = (source/(job['config_id']+'.direct_supervisor.log')).open('a')
                proc = subprocess.Popen([sys.executable, str(HERE/'campaign.py'), '--worker', '--package', str(package),
                                        '--root', str(source), '--job', str(path), '--gpu', str(gpu)], stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (proc, job, source, log)
                print('[START]', job['config_id'], 'GPU', gpu, flush=True)
            c.atomic(root/'status.json', dict(updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), A_completed=len(rows),
                C_completed=len(crows), C_target=6, selected={m:s['config_id'] for m,s in selected.items()},
                queued=[j['config_id'] for j,_ in pending], active={str(g):j['config_id'] for g,(_,j,_,_) in active.items()},
                skipped_stages=['B','D','E']))
            if len(rows) == len(plan) and len(crows) == 6 and not active:
                summaries = list(selected.values()) + list(crows.values())
                c.export(root, summaries)
                c.freeze(root/'selected_9.json', [dict(job=s['repair_job'], checkpoint=s['best_weight'],
                         validation_hs_mae=c.validation(s)) for s in summaries])
                c.atomic(root/'completed.json', dict(models=3, runs=9, year2021_used=False))
                print('[COMPLETE] Nine selected runs verified. 2021 evaluation has not been started.', flush=True)
                break
            time.sleep(5)
    finally:
        for proc, _, _, _ in active.values():
            proc.terminate()
        for proc, _, _, log in active.values():
            try:
                proc.wait(timeout=35)
            except subprocess.TimeoutExpired:
                print('Worker still exiting:', proc.pid, flush=True)
            log.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
