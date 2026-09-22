#!/usr/bin/env python3
"""One GPU, one frozen model; no changes to the training source or checkpoint."""
import argparse,csv,fcntl,os,sys,time
from pathlib import Path
import numpy as np
from common import read,atomic,freeze,sha,inventory,conditions,perturb,indices,error_stats
HERE=Path(__file__).resolve().parent

def main(a):
    inv=inventory()
    if inv[a.gpu]['busy']:raise RuntimeError('Assigned GPU became busy; existing processes were not stopped')
    os.environ['CUDA_VISIBLE_DEVICES']=inv[a.gpu]['uuid']
    sys.path.insert(0,str(HERE/'_diagnostics'))
    import evaluate_2021 as ev
    import torch
    rd=a.eval_root/a.model/'models'/f'{a.model}_s{a.seed}'
    original=read(rd/'result.json');sig=original['signature'];entry=sig['entry'];job=entry['job']
    if not original.get('complete') or job['model']!=a.model or job['seed']!=a.seed:raise ValueError('Incomplete/wrong frozen evaluation')
    prepared=read(a.eval_root/a.model/'prepared'/f'{a.model}_s{a.seed}.json')
    if prepared!=sig['prepared']:raise ValueError('Prepared metadata changed')
    checkpoint=Path(entry['checkpoint']);stamp=sig['checkpoint'];st=checkpoint.stat()
    if st.st_size!=stamp['size'] or st.st_mtime_ns!=stamp['mtime_ns']:raise ValueError('Checkpoint changed')
    if sha(entry['normalization'])!=entry['normalization_sha256']:raise ValueError('Normalization changed')
    norm=read(entry['normalization']);cache=Path(prepared['cache'])
    if read(cache/'signature.json')!=prepared['signature']:raise ValueError('Cache signature changed')
    x=np.load(cache/'inputs.npy',mmap_mode='r');y=np.load(cache/'targets.npy',mmap_mode='r')
    times=np.load(cache/'times.npy');wet=np.load(cache/'mask.npy').astype(bool)
    shapes=read(cache/'complete.json')['shapes']
    for key,arr in [('inputs.npy',x),('targets.npy',y),('times.npy',times),('mask.npy',wet)]:
        if list(arr.shape)!=shapes[key]:raise ValueError('Cache shape mismatch')
    length=job['hyperparams']['seq_length']
    events=read(a.eval_root/a.model/'events_2021.json')['events']
    events=[e for e in events if str(e['id']) in a.events.split(',')]
    if {str(e['id']) for e in events}!=set(a.events.split(',')):raise ValueError('Frozen event missing')
    event_ids={str(e['id']):indices(times,e,length,a.stride) for e in events}
    mods=ev.import_modules(a.server_root,a.server_root/'swan_repaired_v1')
    if mods['repair_support'].code_hashes(a.server_root/'swan_repaired_v1')!=job['code_hashes']:raise ValueError('Trainer hashes differ')
    out=a.output/a.stage/f'{a.model}_s{a.seed}';out.mkdir(parents=True,exist_ok=True)
    output_lock=(out/'worker.lock').open('a+')
    fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    provenance=dict(version='2.1.1',stage=a.stage,signature=sig,checkpoint_sha256=sha(checkpoint),events=events,
        stride=a.stride,conditions=conditions(),warmup=a.warmup,repeats=a.repeats,
        scripts={p.name:sha(p) for p in HERE.glob('*.py')},precision='FP32_no_autocast_TF32_disabled',
        scope='Exploratory input-error sensitivity against unchanged SWAN truth; no candidate selection')
    # JSON roundtrip converts tuples so repeated runs compare identically.
    import json
    freeze(out/'provenance.json',json.loads(json.dumps(provenance)))
    if (out/'completed.json').exists():print('[REUSE]',out,flush=True);return
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cls=mods['train_repaired'].LegacyCompatibleBenchmarkModel
    model=cls(model_name=a.model,input_channels=10,output_channels=4,**ev.model_arguments(job['hyperparams'],cls))
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);del state
    model=model.cuda().eval()
    with (rd/'hourly.csv').open() as f:reference={int(r['target_index']):float(r['hs_mae']) for r in csv.DictReader(f)}
    scale=norm['hs'][1]-norm['hs'][0];lo=norm['hs'][0]
    def predict(arr):
        xb=torch.from_numpy(np.array(arr,copy=True)).unsqueeze(0).cuda()
        with torch.inference_mode():p=model(xb)[0][0].float().cpu().numpy()[0]
        return p*scale+lo
    def clean_check(i,p):
        truth=np.asarray(y[i,0])*scale+lo;v=error_stats(p,truth,wet)
        if i not in reference or not np.isclose(v['mae'],reference[i],rtol=2e-4,atol=2e-6):
            raise ValueError(f'Clean inference differs from original at {i}; inspect precision/model restoration before perturbation')
        return truth,v
    if a.stage=='resources':
        valid=np.unique(np.concatenate(list(event_ids.values())))
        ids=valid[np.linspace(0,len(valid)-1,min(8,len(valid)),dtype=int)]
        host=[np.array(x[i-length:i],copy=True) for i in ids]
        for i,arr in zip(ids,host):clean_check(int(i),predict(arr))
        with torch.inference_mode():
            for k in range(a.warmup):predict(host[k%len(host)])
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            core=[];pipeline=[]
            for k in range(a.repeats):
                arr=host[k%len(host)];xb=torch.from_numpy(arr).unsqueeze(0).cuda();torch.cuda.synchronize()
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();pred=model(xb)[0][0];end.record();end.synchronize();core.append(begin.elapsed_time(end))
                del xb,pred
                torch.cuda.synchronize();start=time.perf_counter();predict(arr);torch.cuda.synchronize()
                pipeline.append(1000*(time.perf_counter()-start))
        result=dict(model=a.model,seed=a.seed,config_id=job['config_id'],annual_hs_mae=original['metrics']['hs_mae'],
            gpu=torch.cuda.get_device_name(),gpu_uuid=inv[a.gpu]['uuid'],torch=str(torch.__version__),cuda=torch.version.cuda,
            input_shape=[1,*host[0].shape],precision=provenance['precision'],
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
            forward_median_ms=float(np.median(core)),cached_pipeline_median_ms=float(np.median(pipeline)),
            forward_ms=core,cached_pipeline_ms=pipeline,indices=ids.tolist(),warmup=a.warmup,
            measurement='Inference only, batch 1. Peak includes model and input/output tensors. Cached pipeline includes host copies, transfers and Hs denormalization; excludes disk/preprocessing/startup. Not SWAN speedup or training peak.')
        atomic(out/'result.json',result)
    else:
        # Atomically checkpoint each event/hour; resume validates the frozen provenance.
        rows=[]
        for event,ids in event_ids.items():
            for n,i in enumerate(ids):
                i=int(i);dst=out/'frames'/f'{event}_{i}.json'
                if dst.exists():rows.extend(read(dst));continue
                base=np.array(x[i-length:i],copy=True);p=predict(base);truth,clean=clean_check(i,p);block=[]
                for kind,v in conditions():
                    if kind=='clean':stats=clean
                    else:
                        delay=int(v) if kind=='boundary_delay' else 0
                        older=np.array(x[i-length-delay:i-delay,6:10],copy=True) if delay else None
                        pp=predict(perturb(base,kind,v,norm,older));stats=error_stats(pp,truth,wet)
                    block.append(dict(model=a.model,seed=a.seed,event=event,target_index=i,time=str(times[i]),
                        condition=kind,level=v,delta_mae=stats['mae']-clean['mae'],**stats))
                atomic(dst,block);rows.extend(block)
                print(f'[{a.model} s{a.seed}] {event} {n+1}/{len(ids)}',flush=True)
        with (out/'hourly.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        groups=[]
        for event in event_ids:
            for kind,v in conditions():
                rr=[r for r in rows if r['event']==event and r['condition']==kind and r['level']==v]
                count=sum(r['high_count'] for r in rr)
                groups.append(dict(model=a.model,seed=a.seed,event=event,condition=kind,level=v,frames=len(rr),stride_hours=a.stride,
                    hs_mae=float(np.mean([r['mae'] for r in rr])),delta_mae=float(np.mean([r['delta_mae'] for r in rr])),
                    hs_ge5_mae=sum(r['high_sae'] for r in rr)/count if count else None,high_count=count))
        atomic(out/'result.json',groups)
    atomic(out/'completed.json',dict(complete=True,stage=a.stage,model=a.model,seed=a.seed))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['resources','robustness'],required=True)
    for key in ['server-root','eval-root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--events',default='2109,2112,2114');p.add_argument('--stride',type=int,default=6)
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--repeats',type=int,default=40)
    a=p.parse_args()
    if a.stride<1 or a.warmup<1 or a.repeats<10:p.error('Invalid sampling/timing settings')
    main(a)
