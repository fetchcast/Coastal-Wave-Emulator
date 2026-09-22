"""Unmodified v3 scheduling functions, invoked under v4 GPU ownership."""
import os, signal, subprocess, sys, time
from pathlib import Path
import campaign as c
HERE=Path(__file__).resolve().parent

def ready_candidates(model, bases, pilots, plan, rows):
    expected = [j for j in plan if j['model'] == model]
    if not all(j['config_id'] in rows for j in expected):
        return None
    return [pilots[model, 42]] + [rows[j['config_id']] for j in expected]

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
