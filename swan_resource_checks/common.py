"""Pure helpers for frozen-checkpoint resource and input-error diagnostics."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np

def read(p): return json.loads(Path(p).read_text())
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)
def freeze(p,v):
    if Path(p).exists():
        if read(p)!=v: raise ValueError(f'Frozen configuration differs: {p}; use a new output root')
    else: atomic(p,v)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def inventory():
    def query(*args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=20)
    rows=query('--query-gpu=index,uuid','--format=csv,noheader').strip().splitlines()
    apps=query('--query-compute-apps=gpu_uuid,pid','--format=csv,noheader').strip().splitlines()
    used={x.split(',')[0].strip() for x in apps if x.strip()}
    return {int(i):{'uuid':u.strip(),'busy':u.strip() in used} for i,u in (x.split(',') for x in rows)}

def conditions():
    return [('clean',0.)]+[(kind,v) for kind,vs in [('wind',[-.1,-.05,.05,.1]),('boundary_hs',[-.1,-.05,.05,.1]),('boundary_direction',[-10.,-5.,5.,10.]),('boundary_delay',[1.,3.,6.])] for v in vs]

def perturb(x,kind,value,norm,delayed=None):
    """Input shape L,10,H,W. Transform physical values, preserve absent boundary cells."""
    z=np.array(x,copy=True)
    if kind=='clean':return z
    if kind=='wind':
        for ch,key in [(0,'wind_u'),(1,'wind_v')]:
            lo,hi=norm[key];scale=hi-lo
            if scale<=0:raise ValueError('Invalid wind normalization')
            z[:,ch]=((z[:,ch]*scale+lo)*(1+value)-lo)/scale
    elif kind=='boundary_delay':
        if delayed is None or delayed.shape!=z[:,6:10].shape:raise ValueError('Invalid delayed boundary sequence')
        z[:,6:10]=delayed
    else:
        support=np.hypot(z[:,8],z[:,9])>1e-6
        if kind=='boundary_hs':
            lo,hi=norm['hs'];scale=hi-lo
            if scale<=0:raise ValueError('Invalid Hs normalization')
            z[:,6]=np.where(support,((z[:,6]*scale+lo)*(1+value)-lo)/scale,z[:,6])
        elif kind=='boundary_direction':
            angle=np.deg2rad(value);s=z[:,8].copy();c=z[:,9].copy()
            z[:,8]=np.where(support,s*np.cos(angle)+c*np.sin(angle),s)
            z[:,9]=np.where(support,c*np.cos(angle)-s*np.sin(angle),c)
        else:raise ValueError(kind)
    return z

def indices(times,event,length,stride):
    times=np.asarray(times).astype('datetime64[ns]')
    ids=np.flatnonzero((times>=np.datetime64(event['start']))&(times<=np.datetime64(event['end'])))
    expected=int((np.datetime64(event['end'])-np.datetime64(event['start']))/np.timedelta64(1,'h'))+1
    if len(ids)!=expected or expected<1:raise ValueError('Incomplete event coverage')
    ids=ids[::stride]
    for i in ids:
        if i<length+6 or np.any(np.diff(times[i-length-6:i+1])!=np.timedelta64(1,'h')):
            raise ValueError('Input/delayed sequence crosses a gap or lacks context')
    return ids

def error_stats(p,t,wet):
    p=p[wet].astype(float);t=t[wet].astype(float)
    if not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError('Nonfinite wet-cell values')
    d=p-t;high=t>=5
    return dict(mae=float(abs(d).mean()),bias=float(d.mean()),rmse=float(np.sqrt((d*d).mean())),
        high_count=int(high.sum()),high_sae=float(abs(d[high]).sum()),
        true_max=float(t.max()),pred_max=float(p.max()),pred_at_true_max=float(p[t.argmax()]),negative_prediction_fraction=float((p<0).mean()))
