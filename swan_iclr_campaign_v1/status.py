#!/usr/bin/env python3
"""Show campaign stage and bounded per-worker progress logs."""
import argparse
import json
from pathlib import Path
import time
from campaign import tail
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('/home/jovyan/swan/runs/iclr_expanded_v1'))
p.add_argument('--watch',type=float,default=0)
a=p.parse_args()
if a.watch and a.watch<5:
    p.error('Use --watch >=5')
try:
    while True:
        print(time.strftime('%Y-%m-%d %H:%M:%S'))
        status=a.root/'status.json'
        if status.exists():
            print(status.read_text())
        for stage in ('preflight','A','B','C','D','E'):
            done=a.root/stage/'completed.json'
            if done.exists():
                v=json.loads(done.read_text())
                print(stage,'complete:',len(v['successful']),'resource skips:',len(v['resource_skips']))
        if status.exists():
            v=json.loads(status.read_text())
            for gpu,name in v.get('active',{}).items():
                logs=list((a.root/v['stage']).glob(f'*{name}*/attempt_*.log'))
                if logs:
                    last=max(logs,key=lambda x:x.stat().st_mtime)
                    lines=tail(last).replace('\r','\n').splitlines()
                    progress=[x for x in lines if 'Successful updates:' in x or '[CYCLE ' in x]
                    print('GPU',gpu,name,progress[-1] if progress else 'Loading/preprocessing')
        if (a.root/'campaign_completed.json').exists():
            print('CAMPAIGN COMPLETE; results.csv contains full-run results.')
        if not a.watch:
            break
        time.sleep(a.watch)
except KeyboardInterrupt:
    pass
