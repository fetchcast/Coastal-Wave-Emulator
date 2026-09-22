#!/usr/bin/env python3
"""Show all v4 lanes and recent progress without importing torch."""
import argparse,json,time
from pathlib import Path
from campaign import tail

def show(server):
    root=server/'runs/iclr_parallel_v4'
    print(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
    for label,path in [('V4',root/'status.json'),('ORIGINAL B/C',server/'runs/iclr_bc_typhoon_v3/status.json'),('BASELINES',root/'baselines/status.json')]:
        print(label)
        if path.exists():
            try:print(json.dumps(json.loads(path.read_text()),indent=2))
            except (OSError,ValueError) as e:print('Read failed:',e)
        else:print('not started')
    for p in sorted((root/'evaluation').glob('*/status.json')):
        print('EVAL',p.parent.name,p.read_text())
    for label,folder in [('ORIGINAL',server/'runs/iclr_expanded_v1/B'),('C',server/'runs/iclr_bc_typhoon_v3/C'),('BASELINE',root/'baselines/fixed')]:
        for rd in sorted(folder.glob('*')):
            if not rd.is_dir() or (rd/'run_summary.json').exists():continue
            logs=list(rd.glob('attempt_*.log'))
            if not logs:continue
            log=max(logs,key=lambda p:p.stat().st_mtime)
            lines=tail(log).replace('\r','\n').splitlines()
            progress=[x for x in lines if 'Successful updates:' in x]
            print(label,rd.name,'log age',round(time.time()-log.stat().st_mtime),'s')
            print(progress[-1] if progress else '\n'.join(lines[-2:]))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'));p.add_argument('--watch',type=float,default=0)
    a=p.parse_args()
    try:
        while True:
            show(a.server_root)
            if a.watch<=0:break
            time.sleep(max(a.watch,2))
    except KeyboardInterrupt:pass
