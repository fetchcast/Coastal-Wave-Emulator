#!/usr/bin/env python3
"""Display training, held-out evaluation, and recent worker progress."""
import argparse
import json
from pathlib import Path
import time
from campaign import tail


def show(server):
    training=server/'runs/iclr_bc_typhoon_v3'
    evaluation=server/'runs/iclr_typhoon_2021_v3'
    print(time.strftime('%Y-%m-%d %H:%M:%S %Z'),flush=True)
    for label,root in [('TRAINING',training),('2021',evaluation)]:
        p=root/'status.json'
        if not p.exists():
            print(label,': not started')
            continue
        status=json.loads(p.read_text())
        print(label,json.dumps(status,indent=2))
        for gpu,config in status.get('active',{}).items():
            if label=='TRAINING':
                stage=Path(status.get('active_roots',{}).get(gpu,str(training/'C')))
                logs=list(stage.glob(f'pilot_*_{config}_seed*/attempt_*.log'))
            else:logs=list((evaluation/'models'/config).glob('evaluate.log'))
            if logs:
                log=max(logs,key=lambda p:p.stat().st_mtime)
                lines=tail(log).replace('\r','\n').splitlines()
                progress=[x for x in lines if 'Successful updates:' in x or '[CYCLE' in x or '[EVAL]' in x]
                print('GPU',gpu,config,'log age',round(time.time()-log.stat().st_mtime),'s')
                print(progress[-1] if progress else '\n'.join(lines[-2:]))
    if (evaluation/'analysis_completed.json').exists():
        print('REPORT:',evaluation/'event_metrics_seed_summary.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'))
    p.add_argument('--watch',type=float,default=0)
    a=p.parse_args()
    try:
        while True:
            show(a.server_root)
            if a.watch<=0:break
            time.sleep(max(1,a.watch))
    except KeyboardInterrupt:pass
