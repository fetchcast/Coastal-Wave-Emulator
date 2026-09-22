#!/usr/bin/env python3
"""Collect lightweight diagnostics; exclude prediction arrays and checkpoints."""
import argparse
from pathlib import Path
import zipfile
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('/home/jovyan/swan/runs/iclr_event_diagnostics_v1'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise SystemExit('Output already exists; choose a new filename')
files=[x for x in a.root.rglob('*') if x.is_file() and x.suffix in ('.csv','.json','.png','.log')]
if not files:raise SystemExit('No diagnostic files found')
a.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as z:
    for x in sorted(files):z.write(x,x.relative_to(a.root))
print(a.output, 'files=',len(files), 'bytes=',a.output.stat().st_size)
