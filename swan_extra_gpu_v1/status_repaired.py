#!/usr/bin/env python3
"""Read pilot progress from both result roots without importing training code."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail_records(path):
    # Read a bounded tail even when tqdm has generated a multi-gigabyte log.
    with path.open('rb') as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 262144))
        text = f.read().decode('utf-8', errors='replace')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    return [line.strip() for line in text.replace('\r', '\n').splitlines() if line.strip()]


def snapshot(roots):
    print(datetime.now().astimezone().isoformat(timespec='seconds'))
    for root in roots:
        print(f'\nRESULT ROOT: {root}')
        jobs = {}
        for p in sorted((root / '_jobs').glob('*.json')):
            j = read_json(p)
            if j.get('stage') == 'pilot':
                jobs[j['config_id']] = j
        for p in (root / 'extra_plan.json', root / 'plan_pilot.json'):
            doc = read_json(p)
            rows = doc if isinstance(doc, list) else doc.get('jobs', [])
            for j in rows:
                jobs[j['config_id']] = j
        if not jobs:
            print('No pilot plan found yet.')
        for name, j in sorted(jobs.items()):
            h = j['hyperparams']
            run = root / (f"{j['stage']}_{j['model']}_{name}_seed{j['seed']}_seq{h['seq_length']}"
                          f"_lr{h['max_lr']:.0e}_wd{h['weight_decay']:.0e}")
            summary = read_json(run / 'run_summary.json')
            completed = (summary.get('repair_job') == j and not summary.get('failed')
                         and summary.get('repair_audit', {}).get('updates') is not None
                         and summary['repair_audit'].get('updates') == summary['repair_audit'].get('target_updates'))
            logs = list(run.glob('attempt_*.log'))
            print(f'\n{name}: {"COMPLETED" if completed else "NOT COMPLETED"}')
            audit = read_json(run / 'training_audit.json')
            if audit:
                print(f"  Last validation: cycle={audit.get('completed_cycles')}, "
                      f"updates={audit.get('updates')}/{audit.get('target_updates')}, "
                      f"best validation Hs MAE={audit.get('best_val_hs_mae')}")
            if not logs:
                print('  No attempt log yet (queued or not launched).')
                continue
            p = max(logs, key=lambda q: q.stat().st_mtime)
            print(f'  Log age: {max(0, time.time()-p.stat().st_mtime):.0f}s | {p.name}')
            lines = tail_records(p)
            progress = [x for x in lines if 'Successful updates:' in x]
            errors = [x for x in lines if 'Traceback (most recent call last)' in x or '[ERROR]' in x]
            if progress:
                print('  ' + progress[-1][:500])
            if errors:
                print('  ERROR FOUND: ' + errors[-1][:500])
            if not progress or errors:
                for line in lines[-3:]:
                    print('  ' + line[:500])
            if completed:
                print('  Hs RMSE:', summary.get('legacy_metrics', {}).get('rmse_m'))
    print('\nProgress-bar ETA does not fully account for later validation and final evaluation.', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', default=[
        '/home/jovyan/swan/runs/repaired_timegap_v1',
        '/home/jovyan/swan/runs/repaired_timegap_extra_v1'])
    ap.add_argument('--watch', type=float, default=0, metavar='SECONDS')
    a = ap.parse_args()
    if a.watch and a.watch < 5:
        ap.error('--watch must be at least 5 seconds')
    while True:
        if a.watch and sys.stdout.isatty():
            print('\033[2J\033[H', end='')
        snapshot([Path(p) for p in a.roots])
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
