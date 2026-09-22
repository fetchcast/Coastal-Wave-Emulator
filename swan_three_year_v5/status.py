#!/usr/bin/env python3
"""Read isolated v5 progress."""
import argparse,time,json
from settings import load,paths

def show():
    _,root=paths(load());print(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
    for name in ('data/prepared.json','status.json','training_completed.json','evaluation_status.json','completed.json'):
        p=root/name
        if p.exists():print(name,p.read_text())
    for p in sorted((root/'evaluation').glob('*/status.json')):print(p.parent.name,p.read_text())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',type=int,default=0);a=p.parse_args()
    try:
        while True:
            show()
            if a.watch<=0:break
            time.sleep(max(2,a.watch))
    except KeyboardInterrupt:pass
