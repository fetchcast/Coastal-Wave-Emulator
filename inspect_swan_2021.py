#!/usr/bin/env python3
"""Inspect 2021 SWAN and boundary inputs without loading wave fields or training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import xarray as xr

REQUIRED = ['windu','windv','depth','veloc-x','veloc-y','hsign','period','dir','x','y','kcs']


def time_info(da):
    info = dict(dims=list(da.dims), shape=list(da.shape), attrs=dict(da.attrs))
    if da.ndim != 1:
        info['issue'] = 'Time coordinate is not one-dimensional'
        return info
    values = da.values
    info['count'] = len(values)
    if not len(values):
        return info
    info.update(first=str(values[0]),last=str(values[-1]))
    if np.issubdtype(values.dtype, np.datetime64):
        valid = ~np.isnat(values)
        info['missing_timestamps'] = int((~valid).sum())
        if not valid.all():
            return info
        ns = values.astype('datetime64[ns]').astype('int64')
        diff = np.diff(ns)/1e9
        info.update(unique=bool(len(np.unique(ns)) == len(ns)),
                    increasing=bool(np.all(diff > 0)),
                    intervals_seconds={str(k):int(v) for k,v in Counter(diff.tolist()).items()},
                    gap_count=int((diff > 3600).sum()),
                    nonhourly_intervals=int((~np.isclose(diff,3600)).sum()),
                    years=sorted(set(str(v)[:4] for v in values)))
        info['gap_examples'] = [dict(before=str(values[i]),after=str(values[i+1]),seconds=float(diff[i]))
                                for i in np.flatnonzero(diff > 3600)[:20]]
    else:
        info['note'] = 'Non-numpy datetime or numeric time; calendar/units need interpretation'
    return info


def inspect_nc(path, reference=None):
    record = dict(path=str(path), bytes=path.stat().st_size)
    times = {}
    try:
        try:
            ds = xr.open_dataset(path, chunks=None, cache=False)
            record['decoded_times'] = True
        except Exception as e:
            record['decode_error'] = str(e)
            ds = xr.open_dataset(path, chunks=None, cache=False, decode_times=False)
            record['decoded_times'] = False
        with ds:
            record.update(dimensions=dict(ds.sizes),global_attrs=dict(ds.attrs),variables={})
            for name, da in ds.variables.items():
                record['variables'][name] = dict(dims=list(da.dims),shape=list(da.shape),dtype=str(da.dtype),attrs=dict(da.attrs))
                if name.lower() in ('time','times','datetime') or da.attrs.get('standard_name') == 'time' or da.attrs.get('axis') == 'T':
                    record.setdefault('time_axes',{})[name] = time_info(ds[name])
                    if da.ndim==1 and np.issubdtype(da.dtype,np.datetime64):
                        times[name] = da.values.astype('datetime64[ns]').astype('int64')
            record['missing_training_names'] = [n for n in REQUIRED if n not in ds]
            record['grid'] = {}
            for name in ('x','y','kcs'):
                if name not in ds:
                    continue
                da=ds[name]
                # Compare only a static grid or its first time slice, never all wave fields.
                first=da.isel({dim:0 for dim in da.dims if dim.lower()=='time'})
                if first.size > 2000000:
                    record['grid'][name] = dict(note='Grid too large for this lightweight check')
                    continue
                arr=np.asarray(first.values)
                g=dict(dims=list(first.dims),shape=list(arr.shape),dtype=str(arr.dtype),
                       sha256=hashlib.sha256(arr.tobytes()).hexdigest())
                if reference is not None and name in reference:
                    other=reference[name]
                    same_shape=arr.shape==other.shape
                    g['reference_shape_equal']=same_shape
                    g['reference_values_equal']=bool(same_shape and np.array_equal(arr,other,equal_nan=True))
                    if same_shape and np.issubdtype(arr.dtype,np.number):
                        mask=np.isfinite(arr)&np.isfinite(other)
                        g['max_absolute_difference']=float(np.max(np.abs(arr[mask].astype(float)-other[mask].astype(float)))) if mask.any() else None
                record['grid'][name]=g
    except Exception as e:
        record['error']=f'{type(e).__name__}: {e}'
    return record,times


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'))
    p.add_argument('--nc-dir',type=Path)
    p.add_argument('--bnd-dir',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    server=a.server_root.resolve()
    nc=(a.nc_dir or server/'swan_2021_nc_v2').resolve()
    bnd=(a.bnd_dir or server/'bnd_2021_v2').resolve()
    ref=(a.reference or server/'wavm-Waves_2019_2020_v2.nc').resolve()
    out=(a.output or server/'swan_2021_inspection.json').resolve()
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='Metadata inspection only; no model evaluation',
                nc_dir=str(nc),bnd_dir=str(bnd),reference=str(ref),issues=[],netcdf=[],boundary={})
    reference={}
    if ref.is_file():
        try:
            with xr.open_dataset(ref,decode_times=False,cache=False) as ds:
                report['reference_variables']={n:dict(dims=list(v.dims),attrs=dict(v.attrs)) for n,v in ds.variables.items()}
                for name in ('x','y','kcs'):
                    if name in ds:
                        da=ds[name].isel({d:0 for d in ds[name].dims if d.lower()=='time'})
                        if da.size<=2000000:reference[name]=da.values
        except Exception as e:
            report['issues'].append(f'Reference inspection failed: {e}')
    else:
        report['issues'].append('Reference file not found; grid comparison not performed')
    ncfiles=sorted(p for p in nc.rglob('*') if p.is_file() and p.suffix.lower() in ('.nc','.nc4','.cdf')) if nc.is_dir() else []
    if not ncfiles:
        report['issues'].append('No NetCDF files found in requested directory')
    time_groups={}
    for i,path in enumerate(ncfiles,1):
        print(f'[NC {i}/{len(ncfiles)}] {path.name}',flush=True)
        record,axes=inspect_nc(path,reference)
        report['netcdf'].append(record)
        # Report overlap per schema to avoid silently treating variable-separated files as duplicates.
        schema=json.dumps(sorted(record.get('variables',{})))
        for name,values in axes.items():
            if np.any(values==np.iinfo(np.int64).min):continue
            time_groups.setdefault((schema,name),[]).append(values)
    report['combined_time_groups']=[]
    for (schema,name),parts in time_groups.items():
        values=np.concatenate(parts)
        unique=np.unique(values)
        report['combined_time_groups'].append(dict(variables=json.loads(schema),axis=name,files=len(parts),
            timestamps=len(values),unique_timestamps=len(unique),overlap_or_duplicates=len(values)-len(unique),
            first=str(unique[0].astype('datetime64[ns]')) if len(unique) else None,
            last=str(unique[-1].astype('datetime64[ns]')) if len(unique) else None,
            note='Inventory only; files were not merged, deduplicated or reordered for evaluation'))
    bfiles=sorted(p for p in bnd.rglob('*') if p.is_file()) if bnd.is_dir() else []
    report['boundary']['file_count']=len(bfiles)
    report['boundary']['extension_counts']=dict(Counter(p.suffix.lower() or '<none>' for p in bfiles))
    report['boundary']['files']=[dict(path=str(p),bytes=p.stat().st_size) for p in bfiles]
    if not bfiles:
        report['issues'].append('No boundary files found')
    samples=[]
    extensions=sorted(set(p.suffix.lower() for p in bfiles))
    for ext in extensions:
        candidates=[p for p in bfiles if p.suffix.lower()==ext]
        for path in list(dict.fromkeys(candidates[:1]+candidates[-1:])):
            sample=dict(path=str(path))
            if ext in ('.nc','.nc4','.cdf'):
                sample['metadata']=inspect_nc(path)[0]
            else:
                with path.open('rb') as f:raw=f.read(16384)
                if b'\x00' in raw:
                    sample['note']='Binary content; not decoded'
                else:
                    sample['first_lines']=raw.decode('utf-8',errors='replace').splitlines()[:50]
            samples.append(sample)
    report['boundary']['samples']=samples
    report['helper_sources']={}
    for name in ('bnd_features.py','boundspec_segments.py'):
        path=server/name
        if path.is_file():
            # Include the exact helper source for checking boundary parsing without importing it.
            report['helper_sources'][name]=path.read_text(errors='replace') if path.stat().st_size<=250000 else 'Too large; attach separately'
        else:
            report['issues'].append(f'Helper not found at {path}')
    report['limitations']=['Wave arrays were not scanned for missing values.',
        'Boundary timestamps and segment coverage have not yet been parsed or validated.',
        'Coordinates/mask comparison uses static data or the first time slice only.',
        'No normalization, direction calibration, prediction, merging or source edits performed.']
    if out==ref or out in ncfiles or out in bfiles:
        raise ValueError('Output must not overwrite a source file')
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_name(out.name+'.tmp')
    tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str))
    tmp.replace(out)
    print(f'[DONE] NetCDF={len(ncfiles)} boundary files={len(bfiles)}')
    for issue in report['issues']:print('[CHECK]',issue)
    print('[REPORT]',out)
    print('Attach this JSON report. This script does not run the 2021 evaluation.')
    return 1 if not ncfiles or not bfiles or any('error' in r for r in report['netcdf']) else 0

if __name__=='__main__':
    sys.exit(main())
