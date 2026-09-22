#!/usr/bin/env python3
"""Benchmark one frozen v3 model on a free physical GPU; no training."""
import argparse, hashlib, inspect, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(v):
    return dict(median_ms=float(np.median(v)),p10_ms=float(np.quantile(v,.1)),p90_ms=float(np.quantile(v,.9)),mean_ms=float(np.mean(v)),n=len(v))
def main(a):
    if a.output.exists(): raise FileExistsError(a.output)
    if a.gpu<0: raise ValueError('Use a physical GPU index >=0')
    # Probe before creating our own CUDA context. Refuse a busy device.
    gpu_uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(line.split(',')[0].strip()==gpu_uuid for line in running.splitlines()):
        raise RuntimeError('GPU has running compute processes. Use a free GPU after training. Do not stop training for this measurement.')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_uuid
    import torch
    s=read(a.signature); e=s['entry']; job=e['job']; cache=Path(s['prepared']['cache'])
    ckpt=Path(e['checkpoint']); stamp=s['checkpoint']
    if ckpt.stat().st_size!=stamp['size'] or ckpt.stat().st_mtime_ns!=stamp['mtime_ns']: raise ValueError('Checkpoint changed')
    sys.path.insert(0,str(a.server_root));sys.path.insert(0,str(a.package))
    import train_repaired as train
    import repair_support as repair
    if Path(train.__file__).resolve()!=(a.package/'train_repaired.py').resolve(): raise ValueError('Unexpected trainer')
    if repair.code_hashes(a.package)!=job['code_hashes']: raise ValueError('Training code hash mismatch')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    hp=job['hyperparams']; args={k:v for k,v in hp.items() if k in inspect.signature(train.LegacyCompatibleBenchmarkModel).parameters}
    start=time.perf_counter()
    model=train.LegacyCompatibleBenchmarkModel(model_name=job['model'],input_channels=10,output_channels=4,**args)
    weights=torch.load(ckpt,map_location='cpu',weights_only=True);model.load_state_dict(weights,strict=True);del weights
    model=model.cuda().eval();torch.cuda.synchronize(); startup=time.perf_counter()-start
    data=np.load(cache/'inputs.npy',mmap_mode='r'); times=np.load(cache/'times.npy'); length=hp['seq_length']
    valid=np.flatnonzero((times>=np.datetime64('2021-01-01T12'))&(times<np.datetime64('2022-01-01')))
    valid=valid[valid>=length]
    if len(valid)<a.samples: raise ValueError('Insufficient inputs')
    indices=valid[np.linspace(0,len(valid)-1,a.samples,dtype=int)]
    # Preload host arrays outside timing; explicitly excludes disk I/O and feature preparation.
    host=[np.array(data[i-length:i],copy=True) for i in indices]
    core=[]; pipeline=[]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(a.warmup):
            xb=torch.from_numpy(host[i%len(host)]).unsqueeze(0).cuda(); out=model(xb)[0][0]
            torch.cuda.synchronize();del xb,out
        for i in range(a.repeats):
            arr=host[i%len(host)];xb=torch.from_numpy(arr).unsqueeze(0).cuda();torch.cuda.synchronize()
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();out=model(xb)[0][0];end.record();end.synchronize()
            core.append(begin.elapsed_time(end));del xb,out
            torch.cuda.synchronize();t=time.perf_counter()
            xb=torch.from_numpy(arr.copy()).unsqueeze(0).cuda();out=model(xb)[0][0].float().cpu().numpy()
            torch.cuda.synchronize();pipeline.append((time.perf_counter()-t)*1000)
            if not np.isfinite(out).all(): raise ValueError('Nonfinite inference')
            del xb,out
    result=dict(model=job['model'],seed=job['seed'],config_id=job['config_id'],checkpoint=str(ckpt),
        checkpoint_sha256=sha(ckpt),signature_sha256=sha(a.signature),script_sha256=sha(__file__),
        gpu=torch.cuda.get_device_name(0),gpu_uuid=gpu_uuid,torch_version=torch.__version__,cuda_version=torch.version.cuda,
        precision='FP32_no_autocast_TF32_disabled',batch_size=1,input_shape=[1,*host[0].shape],
        output='one four-channel spatial frame per sequence',target_indices=indices.tolist(),
        warmup=a.warmup,forward_gpu=summary(core),cached_host_to_host=summary(pipeline),
        forward_samples_ms=core,cached_host_to_host_samples_ms=pipeline,startup_seconds=startup,
        peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
        limitations=['Cached pipeline excludes disk I/O, input feature construction, denormalization and output writing.',
        'Forward timing excludes transfers and startup; neither number is full deployment wall time.',
        'Idle check is point-in-time; keep this GPU free throughout timing.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--signature',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--server-root',type=Path,default=Path('/home/jovyan/swan'))
    p.add_argument('--package',type=Path,default=Path('/home/jovyan/swan/swan_repaired_v1'))
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--repeats',type=int,default=40);p.add_argument('--samples',type=int,default=8)
    a=p.parse_args()
    if a.warmup<1 or a.repeats<10 or a.samples<1:p.error('Require warmup>=1, repeats>=10, samples>=1')
    main(a)
