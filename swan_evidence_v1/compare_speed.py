#!/usr/bin/env python3
"""Compare measured inference cost; optionally use an explicitly matched SWAN baseline."""
import argparse, json
from pathlib import Path
from analyze import read, write_csv

def main(a):
    runs=[read(p) for p in a.benchmarks]
    for key in ('gpu_uuid','precision','batch_size','input_shape','target_indices','output','script_sha256'):
        if any(r[key]!=runs[0][key] for r in runs): raise ValueError('Incomparable benchmark setting: '+key)
    if len({r['model'] for r in runs})!=len(runs):raise ValueError('Supply one benchmark per model')
    baseline=read(a.swan) if a.swan else None
    if baseline:
        if baseline.get('same_domain_forcing_output_contract') is not True:raise ValueError('Matched physical workload must be explicitly confirmed')
        if baseline.get('timing_scope')!='solver_core':raise ValueError('Only solver-core comparison supported; end-to-end needs an actual end-to-end emulator measurement')
        for k in ('domain','forcing_period','grid','hardware','command','outputs_description','timing_log'):
            if not baseline.get(k):raise ValueError('Missing SWAN provenance: '+k)
        for k in ('wall_seconds','output_frames'):
            if not isinstance(baseline.get(k),(int,float)) or baseline[k]<=0:raise ValueError('Invalid '+k)
        if baseline['output_frames']!=int(baseline['output_frames']): raise ValueError('output_frames must be integer')
        if baseline.get('emulator_input_shape')!=runs[0]['input_shape']:raise ValueError('Input grid/sequence contract mismatch')
    ref=next((r for r in runs if r['model']=='fno'),runs[0]);rows=[]
    for r in runs:
        latency=r['cached_host_to_host']['median_ms']/1000
        row=dict(model=r['model'],config_id=r['config_id'],forward_median_ms=r['forward_gpu']['median_ms'],
            cached_host_to_host_median_ms=latency*1000,relative_to_model=ref['model'],
            measured_latency_ratio=ref['cached_host_to_host']['median_ms']/(latency*1000))
        if baseline:
            row.update(estimated_swan_core_to_cached_emulator_ratio=baseline['wall_seconds']/(baseline['output_frames']*latency),
                caveat='Estimated from median latency; excludes emulator startup/preprocessing/I/O; not end-to-end speedup')
        rows.append(row)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);write_csv(a.output,rows);print(a.output)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('benchmarks',nargs='+',type=Path)
    p.add_argument('--swan',type=Path);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
