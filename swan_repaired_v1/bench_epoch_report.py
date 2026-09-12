#!/usr/bin/env python3
"""Read-only report. Corrected runs have exact counts; old logs give estimates."""
import argparse
import csv
import json
import math
from pathlib import Path
import re


def old_log_record(paths,acc):
    text='\n'.join(p.read_text(errors='replace') for p in paths).replace('\r','\n')
    completed={int(v) for v in re.findall(r'^Ep(\d+) Train ',text,re.M)}
    batches={int(k):int(n) for k,n in re.findall(r'Epoch (\d+): 100%\|[^|]*\|\s*(\d+)/\2\s*\[',text)}
    matched=completed & batches.keys()
    # Never label an incomplete log reconstruction as the total update count.
    updates=sum(math.ceil(batches[k]/acc) for k in matched) if completed and matched==completed else None
    early=re.search(r'\[EARLY STOP\].*?Stop at ep (\d+)',text)
    return dict(cycles=len(completed) or None,updates=updates,count_kind='log estimate' if updates is not None else 'unknown',
                early_stop=bool(early),best_update=None)


def collect(root):
    rows=[]
    for p in sorted(root.glob('*/run_summary.json')):
        try:s=json.loads(p.read_text())
        except (ValueError,OSError):continue
        if s.get('failed'):continue
        a=s.get('repair_audit');hp=s.get('hyperparams',{})
        if a:
            counts=dict(cycles=a['completed_cycles'],updates=a['updates'],count_kind='optimizer hook',
                        early_stop=a['early_stop'],best_update=a['best_update'])
        else:
            paths=list(p.parent.glob('*_stdout.log'))+list(p.parent.glob('*_stderr.log'))
            counts=old_log_record(paths,int(hp.get('acc_steps',1)))
        rows.append(dict(config=s.get('config_id'),model=s.get('model'),seed=s.get('seed'),
            width=hp.get('fno_width'),depth=hp.get('fno_depth'),modes=hp.get('modes_x'),
            **counts,rmse_hs=s.get('legacy_metrics',{}).get('rmse_m'),
            checkpoint_policy=a.get('selection_metric') if a else 'not verified from summary',
            direction=(a.get('direction') or {}).get('chosen') if a else None))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='/home/jovyan/swan/runs/v2_focused_all');ap.add_argument('--csv')
    a=ap.parse_args();rows=collect(Path(a.root))
    for row in rows:print(json.dumps(row,ensure_ascii=False))
    if a.csv and rows:
        with open(a.csv,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'{len(rows)} completed summaries. Historical log estimates do not detect AMP skips or guarantee the same training recipe.')

if __name__=='__main__':main()
