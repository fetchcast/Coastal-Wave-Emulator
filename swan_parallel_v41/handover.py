#!/usr/bin/env python3
"""Stop only the verified v4 controller, then start the revised scheduler."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import run as scheduler

def stop_v4(server):
    script=str(server/'swan_parallel_v4/run.py')
    info=scheduler.proc_info()
    found=[p for p,v in info.items() if script in v['args']]
    if len(found)>1:
        raise RuntimeError('Multiple v4 controllers found; no process was stopped')
    if not found:
        return
    pid=found[0]
    owned={p:info[p]['start'] for p in scheduler.descendants(info,pid)}
    current=scheduler.proc_info()
    if pid not in current or current[pid]['start']!=info[pid]['start']:
        raise RuntimeError('Controller changed during inspection; rerun')
    print(f'[HANDOVER] SIGTERM to v4 controller {pid}; resume saved checkpoints.',flush=True)
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+240
    while True:
        now=scheduler.proc_info()
        remaining=[p for p,start in owned.items() if p in now and now[p]['start']==start]
        if not remaining:
            return
        if time.monotonic()>deadline:
            raise RuntimeError(f'Old processes still exiting: {remaining}. No new jobs launched.')
        time.sleep(2)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'))
    a=p.parse_args();server=a.server_root.resolve()
    root=server/'runs/iclr_parallel_v4';root.mkdir(parents=True,exist_ok=True)
    with (root/'v41_handover.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        info=scheduler.proc_info()
        target=str(Path(__file__).resolve().parent/'run.py')
        if any(target in v['args'] for v in info.values()):
            raise RuntimeError('v4.1 is already running')
        if (root/'completed.json').exists():
            print('v4 already completed; nothing to accelerate.');return
        # Validate source compatibility and prerequisites before stopping v4.
        cmd=[sys.executable,str(Path(target)),'--server-root',str(server)]
        subprocess.run(cmd,check=True)
        stop_v4(server)
        subprocess.run(cmd+['--apply'],check=True)
if __name__=='__main__':
    main()
