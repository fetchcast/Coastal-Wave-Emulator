#!/usr/bin/env python3
"""Display status from the direct-C controller."""
import argparse
import json
from pathlib import Path
import time
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('/home/jovyan/swan/runs/iclr_c_direct_v1'))
p.add_argument('--watch',type=float,default=0)
a=p.parse_args()
try:
    while True:
        f=a.root/'status.json'
        print(time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
        if f.exists():print(json.dumps(json.loads(f.read_text()),indent=2),flush=True)
        else:print('Waiting for scheduler status; inspect iclr_c_direct_v1.log.',flush=True)
        if (a.root/'completed.json').exists():print('COMPLETE: selected runs ready; 2021 evaluation not started.',flush=True)
        if a.watch<=0:break
        time.sleep(a.watch)
except KeyboardInterrupt:pass
