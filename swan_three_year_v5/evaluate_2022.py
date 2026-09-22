#!/usr/bin/env python3
"""Prepare bounded-memory held-out inputs and evaluate frozen EMA checkpoints."""
import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import numpy as np
from campaign import atomic, read, freeze

VERSION='heldout-2022-typhoon-v3.0'

def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()

def file_stamp(path):
    path=Path(path).resolve();st=path.stat()
    return dict(path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns)

def source_stamp(server,nc,bnd):
    files=sorted(Path(bnd).glob('*.bnd'))
    if not files:raise ValueError('No .bnd files')
    return dict(nc=file_stamp(nc),bnd=[file_stamp(p) for p in files],
        helpers={n:hashlib.sha256((server/n).read_bytes()).hexdigest() for n in ('bnd_features.py','boundspec_segments.py')},
        evaluator=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def import_modules(server,package):
    sys.path.insert(0,str(server));sys.path.insert(0,str(package))
    modules={n:importlib.import_module(n) for n in ('legacy_repaired','train_repaired','repair_support','bnd_features','boundspec_segments')}
    for n,m in modules.items():
        expected=(package if n in ('legacy_repaired','train_repaired','repair_support') else server)/(n+'.py')
        if Path(m.__file__).resolve()!=expected.resolve():raise ValueError(f'Unexpected module {n}')
    return modules

def apply_direction(bnd,chosen):
    match=re.fullmatch(r'(rot|refl)([+-]?\d+(?:\.\d+)?)',chosen)
    if not match:raise ValueError(f'Invalid saved direction transform: {chosen}')
    r=np.deg2rad(float(match[2]));s=bnd[:,2].copy();c=bnd[:,3].copy()
    if match[1]=='refl':
        bnd[:,2]=np.sin(r)*c-np.cos(r)*s;bnd[:,3]=np.cos(r)*c+np.sin(r)*s
    else:
        bnd[:,2]=s*np.cos(r)+c*np.sin(r);bnd[:,3]=c*np.cos(r)-s*np.sin(r)
    return bnd

def year_indices(values,length):
    values=np.asarray(values).astype('datetime64[ns]')
    if np.isnat(values).any() or np.any(np.diff(values.astype('int64'))!=3600*10**9):
        raise ValueError('Expected a unique, continuous hourly source time axis')
    ids=np.flatnonzero((values>=np.datetime64('2022-01-01'))&(values<np.datetime64('2023-01-01')))
    if len(ids)!=8760 or not np.array_equal(values[ids],np.arange(np.datetime64('2022-01-01T00','h'),np.datetime64('2023-01-01T00','h')).astype('datetime64[ns]')):
        raise ValueError('Expected complete calendar year 2022')
    return ids,np.arange(length,len(ids))

def aligned_boundaries(series,names,times,max_gap):
    import pandas as pd
    result={};audit={}
    for name in sorted(set(names)):
        if name not in series:raise ValueError(f'Missing required segment {name}')
        df=series[name][['hs','tm','sin','cos']].copy()
        df.index=pd.DatetimeIndex(df.index)
        if df.index.tz is not None:raise ValueError('Unexpected timezone-aware boundary data')
        if len(df)<2 or not df.index.is_unique or not df.index.is_monotonic_increasing:
            raise ValueError(f'Invalid boundary time axis {name}')
        gaps=np.diff(df.index.asi8)/3.6e12
        if gaps.max()>max_gap:raise ValueError(f'Boundary gap exceeds {max_gap} hours: {name}')
        if df.index[0]>times[0] or df.index[-1]<times[-1]:raise ValueError(f'Boundary endpoint coverage incomplete: {name}')
        if not np.isfinite(df.to_numpy()).all():raise ValueError(f'Nonfinite boundary series {name}')
        # Retain observations bracketing the final evaluation hour; do not extrapolate.
        union=df.index.union(times).sort_values()
        aligned=df.reindex(union).interpolate(method='time',limit_area='inside').reindex(times)
        if not np.isfinite(aligned.to_numpy()).all():raise ValueError(f'Incomplete aligned boundary {name}')
        result[name]=aligned
        audit[name]=dict(first=str(df.index[0]),last=str(df.index[-1]),max_gap_hours=float(gaps.max()),
            interpolated_hours=int((~times.isin(df.index)).sum()))
    return result,audit

def prepare(a):
    import xarray as xr
    import pandas as pd
    entry=read(a.entry);job=entry['job'];norm=read(entry['normalization'])
    if hashlib.sha256(Path(entry['normalization']).read_bytes()).hexdigest()!=entry['normalization_sha256']:raise ValueError('Saved normalization changed')
    mods=import_modules(a.server_root,a.package);legacy=mods['legacy_repaired'];b=mods['bnd_features'];seg=mods['boundspec_segments']
    signature=dict(version=VERSION,sources=source_stamp(a.server_root,a.nc,a.bnd),normalization=norm,
        direction=entry['direction'],max_bnd_gap_hours=a.max_bnd_gap_hours)
    cache=a.root/'cache'/digest(signature)
    cache.mkdir(parents=True,exist_ok=True)
    freeze(cache/'signature.json',signature)
    complete=cache/'complete.json'
    if complete.exists():
        info=read(complete)
        for name in ('inputs.npy','targets.npy','mask.npy','times.npy'):
            arr=np.load(cache/name,mmap_mode='r')
            if list(arr.shape)!=info['shapes'][name]:raise ValueError('Invalid cached array shape')
        atomic(a.result,dict(cache=str(cache),signature=signature));return
    with xr.open_dataset(a.nc,cache=False) as ds,xr.open_dataset(a.reference,cache=False) as ref:
        ids,_=year_indices(ds.time.values,int(job['hyperparams']['seq_length']))
        for n in ('x','y','kcs'):
            if not np.array_equal(ds[n].values,ref[n].values,equal_nan=True):raise ValueError(f'Grid differs: {n}')
        mask=ds.kcs.values>0;H,W=mask.shape;T=len(ids)
        if not mask.any():raise ValueError('Empty wet mask')
        for n in ('windu','windv','depth','veloc-x','veloc-y','hsign','period','dir'):
            if tuple(ds[n].dims)!=('time','nmax','mmax'):raise ValueError(f'Unexpected dimensions {n}')
        times=pd.DatetimeIndex(ds.time.values[ids])
        if (H,W)==(seg.M,seg.N):swap=False
        elif (H,W)==(seg.N,seg.M):swap=True
        else:raise ValueError('Boundary grid dimensions differ')
        b.assert_on_edges(seg.SEGMENTS,M=seg.M,N=seg.N)
        owner,names=b.build_owner_label(H,W,segments=seg.SEGMENTS,exact_M=seg.M,exact_N=seg.N,kcs=ds.kcs.values,swap_ij=swap)
        raw=b.read_all_bnds(a.bnd,direction='from')
        aligned,baudit=aligned_boundaries(raw,names.values(),times,a.max_bnd_gap_hours)
        # Avoid silently invalidating the training helper's missing-segment behavior.
        if any(not np.isfinite(np.asarray(norm[n],dtype=float)).all() for n in norm):raise ValueError('Invalid saved normalization')
        need=T*H*W*14*4
        existing=sum((cache/n).stat().st_size for n in ('inputs.npy','targets.npy') if (cache/n).exists())
        if shutil.disk_usage(cache).free+existing<need+2*1024**3:raise RuntimeError('Need approximately 33 GB free for this held-out cache')
        x=np.lib.format.open_memmap(cache/'inputs.npy',mode='w+',dtype='float32',shape=(T,10,H,W))
        y=np.lib.format.open_memmap(cache/'targets.npy',mode='w+',dtype='float32',shape=(T,4,H,W))
        gradient=legacy._depth_grad_mag(ds.depth.isel(time=int(ids[0])).values)
        for start in range(0,T,16):
            stop=min(start+16,T);chunk=ds.isel(time=ids[start:stop]).load()
            for n in ('windu','windv','depth','veloc-x','veloc-y','hsign','period','dir'):
                if not np.isfinite(chunk[n].values[:,mask]).all():raise ValueError(f'Nonfinite wet-cell data: {n}, frame {start}')
            xx,yy,_,_,_=legacy.load_and_preprocess_data(chunk,norm,time_steps=stop-start)
            # The original full-dataset loader computes this feature once from frame zero.
            xx[:,5]=gradient
            bb=b.make_boundary_feature_maps(times[start:stop],owner,aligned,names,ds.kcs.values,norm['hs'],norm['tm'])
            apply_direction(bb,entry['direction'])
            x[start:stop]=np.concatenate([xx,bb],axis=1);y[start:stop]=yy
            if not np.isfinite(x[start:stop]).all():raise ValueError('Nonfinite preprocessed input')
            if start%256==0:print(f'[PREPARE] {start}/{T}',flush=True)
        x.flush();y.flush();del x,y
        np.save(cache/'mask.npy',mask);np.save(cache/'times.npy',times.values)
        info=dict(shapes={'inputs.npy':[T,10,H,W],'targets.npy':[T,4,H,W],'mask.npy':[H,W],'times.npy':[T]},
            boundary=baudit,unused_boundary_segments=sorted(set(raw)-set(names.values())),
            year='2022',excluded_source_frames=int(ds.sizes['time']-T),normalization_source=entry['normalization'],
            direction_source='saved training manifest; no held-out calibration',signature=signature)
        atomic(complete,info)
    atomic(a.result,dict(cache=str(cache),signature=signature))

def frame_metrics(pred,target,mask,norm):
    output={}
    for ch,name in ((0,'hs'),(1,'tm')):
        scale=norm[name][1]-norm[name][0];lo=norm[name][0]
        pp=pred[ch,mask].astype('float64')*scale+lo;tt=target[ch,mask].astype('float64')*scale+lo
        diff=pp-tt
        output.update({name+'_mse':float(np.mean(diff**2)),name+'_rmse':float(np.sqrt(np.mean(diff**2))),
                       name+'_mae':float(np.mean(abs(diff))),name+'_bias':float(np.mean(diff))})
        if name=='hs':
            output['true_hs_max']=float(tt.max());output['true_hs_mean']=float(tt.mean())
            output['pred_hs_max']=float(pp.max())
            output['pred_hs_at_true_max']=float(pp[int(np.argmax(tt))])
            for threshold in (3,5):
                selected=tt>=threshold;d=diff[selected];tag=f'hs_ge{threshold}'
                output[tag+'_count']=int(selected.sum())
                output[tag+'_sse']=float(np.sum(d*d))
                output[tag+'_sae']=float(np.sum(np.abs(d)))
                output[tag+'_se']=float(np.sum(d))
        else:
            output['true_tm_max']=float(tt.max())
            output['true_tm_gt30_count']=int(np.sum(tt>30))
            output['true_tm_negative_count']=int(np.sum(tt<0))
            output['pred_tm_negative_count']=int(np.sum(pp<0))
    angle=np.rad2deg(np.arctan2(pred[2,mask],pred[3,mask])-np.arctan2(target[2,mask],target[3,mask]))
    diff=(angle+180)%360-180
    output['dir_mse']=float(np.mean(diff.astype('float64')**2));output['dir_mae']=float(np.mean(abs(diff)))
    output['dir_radius_lt_0p1_fraction']=float(np.mean(np.hypot(pred[2,mask],pred[3,mask])<0.1))
    output['dir_low_radius_fraction']=float(np.mean(np.hypot(pred[2,mask],pred[3,mask])<1e-6))
    return output

def model_arguments(hp, constructor):
    args={k:v for k,v in hp.items() if k in inspect.signature(constructor).parameters}
    args['extra']=dict(hp)
    args['feat']=hp.get('unet_feat',[32,64,128,256,512])
    return args


def evaluate(a):
    import torch
    entry=read(a.entry);prepared=read(a.prepared);cache=Path(prepared['cache']);job=entry['job'];norm=read(entry['normalization'])
    if hashlib.sha256(Path(entry['normalization']).read_bytes()).hexdigest()!=entry['normalization_sha256']:raise ValueError('Saved normalization changed')
    mods=import_modules(a.server_root,a.package);train=mods['train_repaired'];legacy=mods['legacy_repaired']
    expected=job['code_hashes']
    if mods['repair_support'].code_hashes(a.package)!=expected:raise ValueError('Training code changed')
    signature=dict(version=VERSION,entry=entry,prepared=prepared,checkpoint=file_stamp(entry['checkpoint']),
                   evaluation_protocol=read(a.root/'evaluation_protocol.json'),
                   events=read(a.root/'events_2022.json'), snapshots=read(a.root/'snapshots.json'),
                   evaluator=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),precision='float32_no_autocast')
    rd=Path(a.result).parent;rd.mkdir(parents=True,exist_ok=True);freeze(rd/'signature.json',signature)
    if a.result.exists():
        old=read(a.result)
        if old.get('signature')==signature and old.get('complete'):print('[SKIP EVAL]',job['config_id']);return
        raise ValueError('Conflicting held-out result')
    x=np.load(cache/'inputs.npy',mmap_mode='r');y=np.load(cache/'targets.npy',mmap_mode='r')
    mask=np.load(cache/'mask.npy');times=np.load(cache/'times.npy');length=job['hyperparams']['seq_length']
    _,targets=year_indices(times,length)
    # Build on CPU first and load the exact saved wrapper state without key rewriting.
    hp=job['hyperparams'];args=model_arguments(hp,train.LegacyCompatibleBenchmarkModel)
    model=train.LegacyCompatibleBenchmarkModel(model_name=job['model'],input_channels=10,output_channels=4,**args)
    state=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);del state
    if not torch.cuda.is_available():raise RuntimeError('Held-out evaluation requires an available CUDA GPU')
    model=model.cuda().eval();torch.cuda.reset_peak_memory_stats()
    fields=['target_index','time','hs_mse','hs_rmse','hs_mae','hs_bias','true_hs_max','true_hs_mean','tm_mse','tm_rmse','tm_mae','tm_bias','dir_mse','dir_mae','dir_low_radius_fraction']
    fields += ['pred_hs_max','pred_hs_at_true_max'] + [f'hs_ge{t}_{key}' for t in (3,5) for key in ('count','sse','sae','se')] + ['true_tm_max','true_tm_gt30_count','true_tm_negative_count','pred_tm_negative_count','dir_radius_lt_0p1_fraction']
    path=rd/'hourly.csv';done={}
    if path.exists():
        with path.open() as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=fields:raise ValueError('Hourly CSV columns differ from this evaluator')
            for row in reader:
                try:
                    idx=int(row['target_index'])
                    if idx in done:raise ValueError('Duplicate hourly row')
                    vals={k:float(v) for k,v in row.items() if k not in ('target_index','time')}
                    if idx not in targets or row['time']!=str(times[idx]) or not all(np.isfinite(list(vals.values()))):raise ValueError('Invalid hourly row')
                    done[idx]=dict(target_index=idx,time=row['time'],**vals)
                except (KeyError,ValueError,TypeError) as e:
                    raise ValueError('Interrupted/malformed hourly CSV; preserve it and remove hourly.csv to rerun this model') from e
    snapshot_ids={int(v['target_index']) for v in read(a.root/'snapshots.json')}
    start_time=time.monotonic()
    with path.open('a',newline='') as f,torch.inference_mode():
        writer=csv.DictWriter(f,fieldnames=fields)
        if path.stat().st_size==0:writer.writeheader();f.flush()
        for idx in targets:
            if int(idx) in done and (int(idx) not in snapshot_ids or (rd/'snapshots'/f'{int(idx)}.npz').exists()):continue
            xb=torch.from_numpy(np.array(x[idx-length:idx],copy=True)).unsqueeze(0).cuda()
            pred=model(xb)[0][0].float().cpu().numpy()
            if not np.isfinite(pred[:,mask]).all():raise ValueError('Nonfinite predictions')
            vals=frame_metrics(pred,y[idx],mask,norm)
            row=dict(target_index=int(idx),time=str(times[idx]),**vals)
            if int(idx) in snapshot_ids:
                folder=rd/'snapshots';folder.mkdir(exist_ok=True)
                pp=pred.copy();tt=np.array(y[idx],copy=True)
                for ch,key in ((0,'hs'),(1,'tm')):
                    scale=norm[key][1]-norm[key][0]
                    pp[ch]=pp[ch]*scale+norm[key][0];tt[ch]=tt[ch]*scale+norm[key][0]
                tmp=folder/f'{int(idx)}.tmp.npz'
                np.savez_compressed(tmp,pred=pp,true=tt,kcs=mask,time=str(times[idx]),
                                    channel_units=np.array(['m','s','sin','cos']))
                tmp.replace(folder/f'{int(idx)}.npz')
            if int(idx) not in done:
                writer.writerow(row);f.flush();done[int(idx)]=row
            if len(done)%100==0:print(f'[EVAL] {job["config_id"]} {len(done)}/{len(targets)}',flush=True)
    ordered=[done[int(idx)] for idx in targets]
    def aggregate(rows):
        out={}
        for name in ('hs','tm'):
            out[name+'_mean_frame_rmse']=float(np.mean([r[name+'_rmse'] for r in rows]))
            out[name+'_pooled_rmse']=float(np.sqrt(np.mean([r[name+'_mse'] for r in rows])))
            out[name+'_mae']=float(np.mean([r[name+'_mae'] for r in rows]))
            out[name+'_bias']=float(np.mean([r[name+'_bias'] for r in rows]))
        out['dir_circular_mae_deg']=float(np.mean([r['dir_mae'] for r in rows]))
        out['dir_circular_rmse_deg']=float(np.sqrt(np.mean([r['dir_mse'] for r in rows])))
        out['dir_low_radius_fraction']=float(np.mean([r['dir_low_radius_fraction'] for r in rows]))
        return out
    months={m:aggregate([r for r in ordered if r['time'][5:7]==m]) for m in sorted(set(r['time'][5:7] for r in ordered))}
    atomic(a.result,dict(complete=True,signature=signature,model=job['model'],seed=job['seed'],config_id=job['config_id'],
        frames=len(ordered),first=ordered[0]['time'],last=ordered[-1]['time'],metrics=aggregate(ordered),monthly=months,
        elapsed_this_attempt_s=time.monotonic()-start_time,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Frozen pre-2022 model; native SWAN direction convention; uniform kcs>0 mask; no new calibration',
        notes=['Mean frame Hs RMSE matches the aggregation used by legacy rmse_m.',
               'Pooled RMSE is also reported and must not be confused with mean frame RMSE.',
               'First 12 hours excluded for input context; no 2021 carry-in; 2023 endpoint excluded.',
               'No additional spin-up exclusion; early-year simulation spin-up must be assessed separately.']))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','evaluate'])
    for name in ('server-root','package','root','entry','result','nc','bnd','reference','prepared'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--max-bnd-gap-hours',type=float,default=6.)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    else:evaluate(a)
if __name__=='__main__':main()
