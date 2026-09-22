#!/usr/bin/env python3
"""Finish frozen B jobs, schedule family-specific C repeats, and evaluate held-out 2021."""
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
    """Resolve relative script paths using each process working directory."""
    scripts = {str(server / 'swan_iclr_campaign_v2' / n) for n in
               ('run_v2.py', 'campaign.py', 'evaluate_2021.py')}
    scripts |= {str(server / 'swan_repaired_v1' / n) for n in
                ('run_repaired.py', 'train_repaired.py')}
    scripts |= {str(HERE / 'campaign.py'), str(HERE / 'evaluate_2021.py'), str(HERE / 'run_v2.py')}
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            cwd = (p / 'cwd').resolve()
            resolved = [str((cwd / arg).resolve()) if arg.endswith('.py') else arg for arg in args]
            if scripts.intersection(resolved):
                result.append((int(p.name), resolved))
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
    p.add_argument('--eval-workers', type=int, default=2)
    p.add_argument('--apply', action='store_true', help='Stop the verified default v2 wrapper and start the new controller.')
    a = p.parse_args()
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server = a.server_root.resolve()
    old = server / 'runs/iclr_expanded_v1'
    root = server / 'runs/iclr_bc_typhoon_v3'
    package = server / 'swan_repaired_v1'
    data = server / 'wavm-Waves_2019_2020_v2.nc'
    parent = server / 'runs/repaired_timegap_v1'
    extra = server / 'runs/repaired_timegap_extra_v1'
    gpus = [int(x) for x in a.gpus.split(',')]
    if not 1 <= a.eval_workers <= len(gpus):
        raise ValueError('Invalid evaluation worker count')
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
    if len(rows) != len(plan):
        raise ValueError('This handover requires all 29 A runs to be verified complete.')
    bplan = c.read(old / 'B/plan.json')
    candidates = [s for (m, seed), s in pilots.items() if seed == 42] + list(rows.values())
    if bplan != c.stage_b(candidates):
        raise ValueError('Original B plan differs from validation-selected A candidates.')
    if list((old / 'B').glob('*.resource_skip.json')):
        raise ValueError('B has resource skips; no automatic selection is allowed.')
    brows = {j['config_id']: s for j in bplan if (s := r.checked_summary(old / 'B', j)) is not None}
    r.compare_protocols(candidates + list(brows.values()))
    print(f'A complete=29/29; B complete={len(brows)}/{len(bplan)}', flush=True)
    for model in c.MODELS:
        missing = [j['config_id'] for j in bplan if j['model'] == model and j['config_id'] not in brows]
        print(model, 'waiting for B: ' + str(missing) if missing else 'ready for C', flush=True)
    print('Selection: lowest validation Hs MAE from pilot42+A+B. D/E omitted.', flush=True)
    # Freeze evaluation rules before accessing held-out fields or model predictions.
    evaluation_protocol = c.read(HERE / 'evaluation_protocol.json')
    import events
    for required in (server/'swan_2021_nc_v2/wavm-Waves.nc', server/'bnd_2021_v2'):
        if not required.exists():
            raise FileNotFoundError(required)
    event_source = events.verify_source(HERE)
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
        c.freeze(root / 'controller_config.json', dict(version=3, source=str(old), gpus=gpus,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256((HERE/'campaign.py').read_bytes()).hexdigest(),
                  selection='best_validation_Hs_MAE_from_pilot_seed42_A_B', skipped=['D','E'],
                  evaluation_protocol=evaluation_protocol, event_source=event_source,
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}))
        c.freeze(root/'evaluation_protocol.json', evaluation_protocol)
        wrappers = [(pid, args) for pid, args in old_procs if str(server / 'swan_iclr_campaign_v2/run_v2.py') in args]
        if old_procs:
            if len(wrappers) != 1:
                raise ValueError('Expected exactly one default v2 wrapper; nothing stopped.')
            pid, args = wrappers[0]
            index = args.index(str(server / 'swan_iclr_campaign_v2/run_v2.py'))
            if args[index + 1:]:
                raise ValueError('Custom wrapper arguments detected; nothing stopped.')
            if any(any(Path(x).name == 'evaluate_2021.py' for x in xs) for _, xs in old_procs):
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
        run_schedule(a, r, root, old, package, bplan, bases, pilots, gpus, protocol, rows)
        from run_v2 import wait_child
        cmd = [sys.executable, str(HERE/'run_v2.py'), '--eval-only',
               '--server-root', str(server), '--train-root', str(root),
               '--eval-root', str(server/'runs/iclr_typhoon_2021_v3'),
               '--gpus', a.gpus, '--eval-workers', str(a.eval_workers)]
        c.atomic(root/'status.json', dict(stage='heldout_2021', updated=time.time()))
        if wait_child(cmd):
            raise RuntimeError('Held-out evaluation failed. Training is preserved; restart the same launcher.')
        c.atomic(root/'completed.json', dict(training_runs=9, evaluation_year=2021, skipped=['D','E']))
        c.atomic(root/'status.json', dict(stage='complete', updated=time.time()))


def run_schedule(a, r, root, old, package, plan, bases, pilots, gpus, protocol, arows):
    active = {}
    selected = {}
    crows = {}
    reference = pilots['fno', 42]
    c.freeze(root/'plan_B.json', plan)
    c.freeze(root/'plan_A.json', c.read(old/'A/plan.json'))
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
            rows = {j['config_id']: s for j in plan if (s := r.checked_summary(old/'B', j)) is not None}
            r.compare_protocols([reference] + list(rows.values()))
            pending = [(j, old/'B') for j in plan if j['config_id'] not in rows]
            for model in c.MODELS:
                candidates = ready_candidates(model, bases, pilots, plan, rows)
                if candidates is None:
                    continue
                candidates += [s for s in arows.values() if s['model'] == model]
                chosen = c.choose(candidates, 1)[0]
                c.freeze(root / f'selection_{model}.json', dict(job=chosen['repair_job'], validation_hs_mae=c.validation(chosen), source='pilot42+A+B'))
                ranked = c.choose(candidates, len(candidates))
                c.freeze(root/f'candidate_comparison_{model}.json', [dict(
                    config_id=s['config_id'], validation_hs_mae=c.validation(s),
                    hyperparams=s['repair_job']['hyperparams'],
                    training_wallclock_s=s.get('training_wallclock_s'),
                    parameters=s['repair_audit'].get('n_parameters'),
                    selected=s['config_id']==chosen['config_id']) for s in ranked])
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
                        s = r.checked_summary(old/'C', job)
                    if s is None:
                        source = old/'C' if (old/'C'/r.run_name(job)).exists() else root/'C'
                        pending.append((job, source))
                    else:
                        r.compare_protocols([reference, s])
                        crows[model, job['seed']] = s
            running = {j['config_id'] for _, j, _, _ in active.values()}
            pending = [(j,s) for j,s in pending if j['config_id'] not in running]
            inv = r.gpu_inventory()
            for gpu in gpus:
                if not pending:
                    break
                if gpu not in inv:
                    raise ValueError(f'GPU not present: {gpu}')
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
            c.atomic(root/'status.json', dict(updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), A_completed=29, B_completed=len(rows), B_target=len(plan), stage='B_and_C',
                C_completed=len(crows), C_target=6, selected={m:s['config_id'] for m,s in selected.items()},
                queued=[j['config_id'] for j,_ in pending], active={str(g):j['config_id'] for g,(_,j,_,_) in active.items()},
                active_roots={str(g):str(source) for g,(_,j,source,_) in active.items()},
                skipped_stages=['D','E']))
            if len(rows) == len(plan) and len(crows) == 6 and not active:
                summaries = sorted(list(selected.values()) + list(crows.values()), key=lambda s:(s['model'],s['seed']))
                c.export(root, summaries)
                (root/'search').mkdir(exist_ok=True)
                c.export(root/'search', list(pilots.values()) + list(arows.values()) + list(rows.values()))
                c.freeze(root/'selected_9.json', [dict(job=s['repair_job'], checkpoint=s['best_weight'],
                         validation_hs_mae=c.validation(s)) for s in summaries])
                c.atomic(root/'training_completed.json', dict(models=3, runs=9, year2021_used=False))
                print('[TRAINING COMPLETE] Nine selected runs verified.', flush=True)
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
