#!/usr/bin/env python3
"""Stream 2019-2021 into an isolated NetCDF; never read held-out 2022."""
import hashlib,json,os,shutil,fcntl
from pathlib import Path
import numpy as np
from settings import load,paths,HERE
from campaign import atomic,freeze,read
VARS=('windu','windv','depth','veloc-x','veloc-y','hsign','period','dir')
STATIC=('x','y','kcs')

def stamp(p):
    p=Path(p).resolve();s=p.stat();return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
def choose_times(times,years):
    times=np.asarray(times,dtype='datetime64[ns]')
    if np.isnat(times).any() or np.any(np.diff(times.astype('int64'))<=0):raise ValueError('Nonunique or unordered source times')
    y=times.astype('datetime64[Y]').astype(int)+1970
    ids=np.flatnonzero(np.isin(y,years))
    if not len(ids) or set(y[ids])!=set(years):raise ValueError('Requested source year is missing')
    if np.any(times[ids].astype('datetime64[h]').astype('datetime64[ns]')!=times[ids]):raise ValueError('Off-hour timestamps')
    return ids,times[ids]
def time_audit(times):
    d=np.diff(np.asarray(times,dtype='datetime64[ns]').astype('int64'))
    hour=3600*10**9
    if np.any(d<=0) or np.any(d%hour):raise ValueError('Invalid combined hourly time axis')
    return [dict(before=str(times[i]),after=str(times[i+1]),missing_hours=int(d[i]//hour-1)) for i in np.flatnonzero(d!=hour)]
def read_times(ds,nc):
    t=ds.variables['time'];cal=getattr(t,'calendar','standard')
    if cal not in ('standard','gregorian','proleptic_gregorian'):raise ValueError('Unsupported calendar')
    vals=nc.num2date(t[:],t.units,calendar=cal,only_use_cftime_datetimes=False)
    return np.array([str(v).replace(' ','T') for v in vals],dtype='datetime64[ns]')
def main():
    _,root=paths(load());root.mkdir(parents=True,exist_ok=True)
    with (root/'prepare.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return prepare()

def prepare():
    import netCDF4 as nc
    a=load();server,root=paths(a);out=root/'data/training_2019_2021.nc';info=root/'data/prepared.json'
    inputs=[Path(a['source_2019_2020']),Path(a['source_2021'])]
    stamps=[stamp(p) for p in inputs]
    counts={read(root/'controls'/f'{m}.json')['entries'][0]['job']['time_steps'] for m in ('fno','ffno')}
    if len(counts)!=1:raise ValueError('Two-year source lengths differ')
    original_frames=next(iter(counts))
    root.mkdir(parents=True,exist_ok=True);freeze(root/'config.json',a)
    signature=dict(sources=stamps,original_frames=original_frames,years=[2019,2020,2021],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if info.exists():
        value=read(info)
        if value['signature']!=signature or not out.exists() or stamp(out)!=value['output_stamp']:raise ValueError('Prepared data/provenance changed')
        print('[REUSE]',out);return
    if out.exists():raise ValueError('Untracked output exists; preserve and inspect it')
    out.parent.mkdir(parents=True,exist_ok=True);parts=[];times_all=[];reference={};bytes_needed=0
    for p,years in zip(inputs,([2019,2020],[2021])):
        with nc.Dataset(p) as ds:
            source_times=read_times(ds,nc)
            if years==[2019,2020]:
                if len(source_times)<original_frames:raise ValueError('Original source is now shorter')
                source_times=source_times[:original_frames]
            ids,times=choose_times(source_times,years)
            if years==[2021] and not np.array_equal(times,np.arange(np.datetime64('2021-01-01','h'),np.datetime64('2022-01-01','h'))):raise ValueError('2021 must contain the complete calendar year')
            for name in STATIC:
                v=np.asarray(ds.variables[name][:])
                if reference and name in reference and not np.array_equal(v,reference[name],equal_nan=True):raise ValueError('Grid/mask changed: '+name)
                reference[name]=v
            wet=reference['kcs']>0
            if not wet.any():raise ValueError('Empty wet mask')
            for name in VARS:
                v=ds.variables[name]
                if np.dtype(v.dtype).kind!='f' or any(k in v.ncattrs() for k in ('scale_factor','add_offset')):
                    raise ValueError('Only unpacked floating-point physical fields are supported: '+name)
                if v.dimensions!=('time','nmax','mmax'):raise ValueError('Unexpected dimensions: '+name)
                if v.shape[1:]!=wet.shape:raise ValueError('Shape mismatch: '+name)
                if len(parts):
                    with nc.Dataset(inputs[0]) as first:
                        if v.dtype!=first.variables[name].dtype:raise ValueError('Source dtypes differ: '+name)
                        if getattr(v,'units','')!=getattr(first.variables[name],'units',''):raise ValueError('Units changed: '+name)
                bytes_needed+=len(ids)*wet.size*np.dtype(v.dtype).itemsize
            parts.append((p,ids));times_all.extend(times)
    times=np.asarray(times_all);gaps=time_audit(times)
    if shutil.disk_usage(out.parent).free<bytes_needed+2*1024**3:raise RuntimeError(f'Need at least {bytes_needed/1e9+2.15:.1f} GB free (uncompressed estimate)')
    tmp=out.with_suffix('.partial.nc')
    with nc.Dataset(inputs[0]) as first,nc.Dataset(tmp,'w',format='NETCDF4') as dst:
        dst.createDimension('time',None)
        for dim,n in zip(('nmax','mmax'),reference['kcs'].shape):dst.createDimension(dim,n)
        for name in STATIC:
            src=first.variables[name];v=dst.createVariable(name,src.dtype,src.dimensions,fill_value=getattr(src,'_FillValue',False))
            v.setncatts({k:src.getncattr(k) for k in src.ncattrs() if k!='_FillValue'});v[:]=src[:]
        tv=dst.createVariable('time','f8',('time',));tv.units='hours since 1970-01-01 00:00:00';tv.calendar='proleptic_gregorian'
        tv[:]=times.astype('datetime64[h]').astype('int64')
        for name in VARS:
            src=first.variables[name]
            v=dst.createVariable(name,src.dtype,src.dimensions,zlib=True,complevel=1,fill_value=getattr(src,'_FillValue',False),chunksizes=(1,*reference['kcs'].shape))
            v.setncatts({k:src.getncattr(k) for k in src.ncattrs() if k not in ('_FillValue','scale_factor','add_offset')})
        offset=0
        for p,ids in parts:
            with nc.Dataset(p) as ds:
                for start in range(0,len(ids),16):
                    ix=ids[start:start+16]
                    for name in VARS:
                        raw=ds.variables[name][ix,:,:]
                        values=np.asarray(np.ma.filled(raw,np.nan))
                        if not np.isfinite(values[:,wet]).all():raise ValueError(f'Nonfinite wet cells: {p}/{name}/{start}')
                        dst.variables[name][offset+start:offset+start+len(ix)]=values
                    if start%512==0:print('[MERGE]',p.name,offset+start,'/',len(times),flush=True)
                offset+=len(ids)
        dst.setncattr('v5_provenance',json.dumps(signature))
    if [stamp(p) for p in inputs]!=stamps:raise ValueError('Source changed during merge')
    tmp.replace(out)
    atomic(info,dict(signature=signature,output_stamp=stamp(out),frames=len(times),first=str(times[0]),last=str(times[-1]),gaps=gaps,
        note='Gaps are retained and sequence windows crossing them are excluded by repaired trainer; no 2022 inputs read'))
    print('[PREPARED]',out,'frames=',len(times),'gaps=',len(gaps))
if __name__=='__main__':main()
