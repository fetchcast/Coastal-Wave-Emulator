#!/usr/bin/env python3
"""Display both training progress and independent-year evaluation progress."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from campaign import tail
p=argparse.ArgumentParser()
p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'))
p.add_argument('--watch',type=float,default=0)
a=p.parse_args()
if a.watch and a.watch<5:p.error('Use at least five seconds')
try:
    while True:
        subprocess.run([sys.executable,str(Path(__file__).with_name('status.py')),'--root',str(a.server_root/'runs/iclr_expanded_v1')],check=True)
        root=a.server_root/'runs/iclr_2021_v2'
        if (root/'status.json').exists():print('2021:',(root/'status.json').read_text(),flush=True)
        for log in sorted((root/'models').glob('*/evaluate.log')):
            lines=[s for s in tail(log).splitlines() if '[EVAL]' in s or '[SKIP EVAL]' in s]
            if lines:print(log.parent.name,lines[-1])
        if not a.watch:break
        time.sleep(a.watch)
except KeyboardInterrupt:pass
