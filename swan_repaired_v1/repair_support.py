"""Audited splitting, direction handling, and training for SWAN repair v1.

All indices are sequence starts. Validation/test partitions are frozen before
training subsampling. This module never reads an old benchmark result.
"""
import hashlib
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader, Subset

VERSION = 'swan-repair-1'
SPLIT_CONTEXT = {}


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path, obj):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    os.close(fd)
    try:
        torch.save(obj, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a, dtype='<i8')
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def fraction():
    raw = os.getenv('SWAN_TRAIN_FRACTION', '1')
    try: value = float(raw)
    except ValueError as exc: raise ValueError('Invalid SWAN_TRAIN_FRACTION') from exc
    if not math.isfinite(value) or not 0.25 <= value <= 1:
        raise ValueError('This campaign supports fractions in [0.25, 1]; calibration uses the nested 25% subset')
    return value


def selected_blocks(blocks, labels, q, seed, f):
    rng = np.random.default_rng(seed + 1000)
    kept = []
    for label in range(q):
        b = sorted(int(b) for b in blocks if labels[b] == label)
        rng.shuffle(b)
        kept.extend(b[:max(1, int(round(f * len(b))))])
    return sorted(kept)


def split_hook(blocks, labels, q, seed):
    f = fraction()
    SPLIT_CONTEXT.clear()
    SPLIT_CONTEXT.update(base_train_blocks=sorted(int(x) for x in blocks),
                         calibration_blocks=selected_blocks(blocks, labels, q, seed, 0.25),
                         selected_train_blocks=selected_blocks(blocks, labels, q, seed, f),
                         fraction=f)
    return SPLIT_CONTEXT['selected_train_blocks']


def hourly_time_axis(values):
    """Accept missing hourly records, but reject duplicate or sub-hourly times."""
    import pandas as pd
    times = pd.DatetimeIndex(values)
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Invalid time ordering or missing timestamps')
    seconds = np.asarray((times[1:] - times[:-1]).total_seconds())
    hours = seconds / 3600.0
    if np.any(hours < 1) or not np.allclose(hours, np.rint(hours), rtol=0, atol=1e-9):
        raise ValueError('Expected hourly records with optional whole-hour gaps')
    gaps = np.flatnonzero(hours > 1)
    records = [dict(left_index=int(i), before=str(times[i]), after=str(times[i+1]),
                    interval_seconds=float(seconds[i]), missing_hours=int(round(hours[i]))-1)
               for i in gaps]
    return times, gaps, records


def continuous_starts(values, seq_length):
    """Check every interval from the first input through the target frame."""
    times, gaps, records = hourly_time_axis(values)
    if seq_length < 1 or len(times) <= seq_length:
        raise ValueError('Time axis is too short for the sequence length')
    broken = np.zeros(len(times)-1, dtype=np.int64)
    broken[gaps] = 1
    cumulative = np.concatenate(([0], np.cumsum(broken)))
    starts = np.arange(len(times)-seq_length)
    good = cumulative[starts+seq_length] == cumulative[starts]
    return good, records


def strict_split(make_split, wave, seq_length, times=None, **kwargs):
    fraction()
    tr, va, te = make_split(wave, seq_length, block_hours=168, q=5, seed=42,
                           embargo_hours=seq_length, **kwargs)
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError('The fixed block split is empty; no random/candidate fallback is allowed')
    # The original default embargo already enforces these bounds. Check the
    # forward sequence window explicitly, not t - sequence_length.
    for idx in (tr, va, te):
        if np.any(idx // 168 != (idx + seq_length) // 168):
            raise ValueError('A sequence crosses a fixed block boundary')
    if any(np.intersect1d(a, b).size for a, b in ((tr, va), (tr, te), (va, te))):
        raise ValueError('Split overlap')
    n = len(wave) - seq_length
    all_idx = np.arange(n)
    pos = all_idx % 168
    valid = (pos >= seq_length) & (pos <= 168 - 2 * seq_length - 1)
    full_tr = all_idx[valid & np.isin(all_idx // 168, SPLIT_CONTEXT['base_train_blocks'])]
    time_record = dict(policy='exclude_gap_crossing_input_and_target', checked=False)
    if times is not None:
        if len(times) != len(wave): raise ValueError('Time and wave lengths differ')
        good, gaps = continuous_starts(times, seq_length)
        before = dict(train=len(tr), val=len(va), test=len(te), base_train=len(full_tr))
        tr, va, te, full_tr = [idx[good[idx]] for idx in (tr, va, te, full_tr)]
        after = dict(train=len(tr), val=len(va), test=len(te), base_train=len(full_tr))
        time_record.update(checked=True, gaps=gaps, excluded_starts_total=int((~good).sum()),
                           excluded_by_split={k:before[k]-after[k] for k in before})
        if min(after.values()) == 0: raise ValueError('Empty split after time-gap filtering')
        print(f'[TIME GAP] gaps={len(gaps)} excluded_by_split={time_record["excluded_by_split"]}', flush=True)
    calibration = tr[np.isin(tr // 168, SPLIT_CONTEXT['calibration_blocks'])]
    if not len(calibration): raise ValueError('Empty direction calibration subset')
    SPLIT_CONTEXT.update(full_train_indices=full_tr, calibration_indices=calibration)
    record = {k:v for k,v in SPLIT_CONTEXT.items() if not isinstance(v, np.ndarray)}
    record.update(version=VERSION, seq_length=seq_length, block_hours=168, embargo=seq_length,
                  block_unit="stored_samples", time_continuity=time_record,
                  n_train=len(tr), n_base_train=len(full_tr), n_val=len(va), n_test=len(te),
                  sample_fraction=len(tr)/len(full_tr),
                  base_split_hash=array_hash(full_tr, va, te),
                  selected_split_hash=array_hash(tr, va, te),
                  calibration_hash=array_hash(calibration))
    atomic_json('split_manifest.json', record)
    np.savez_compressed('split_indices.npz', train=tr, val=va, test=te,
                        base_train=full_tr, calibration=calibration)
    print(f'[SPLIT] tr/va/te={len(tr)}/{len(va)}/{len(te)} fraction={record["sample_fraction"]:.6f}', flush=True)
    return tr, va, te, 'block168_q5_emb' + str(seq_length)


def align_bnd(bnd, ds, kcs, time_index, seq_length):
    """Calibrate using the common nested training subset, or use a fixed convention.

    Boundary channel contract is [Hs, Tm, sin(theta), cos(theta)]. No heuristic
    sin/cos swapping is performed. Full target fields are not materialized.
    """
    import pandas as pd
    if bnd.ndim != 4 or bnd.shape[1] != 4: raise ValueError('Invalid boundary channels')
    times = pd.DatetimeIndex(time_index)
    raw_times = pd.DatetimeIndex(ds['time'].values[:len(times)]).tz_localize(None)
    if not times.equals(raw_times) or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Boundary and target times are not identical, unique and increasing')
    mask = np.asarray(kcs > 0)
    if mask.ndim != 2 or mask.shape != bnd.shape[2:]: raise ValueError('Boundary mask mismatch')
    sel = SPLIT_CONTEXT['calibration_indices'] + seq_length
    candidates = [(0.,False),(90.,False),(-90.,False),(180.,False),
                  (0.,True),(90.,True),(180.,True),(270.,True)]
    totals = np.zeros(8); counts = np.zeros(8, dtype=np.int64)
    # Chunking keeps memory proportional to eight frames, not all targets.
    for start in range(0, len(sel), 8):
        ids = sel[start:start+8]
        target = np.deg2rad(ds['dir'].isel(time=ids).values.astype(np.float64))
        ts, tc = np.sin(target), np.cos(target)
        bs, bc = bnd[ids,2].astype(np.float64), bnd[ids,3].astype(np.float64)
        radius = np.hypot(bs, bc)
        valid = mask[None] & np.isfinite(target) & np.isfinite(radius) & (radius > 1e-6)
        bs = bs / np.maximum(radius, 1e-6); bc = bc / np.maximum(radius, 1e-6)
        for i, (deg, reflect) in enumerate(candidates):
            r = np.deg2rad(deg)
            ss = np.sin(r)*bc - np.cos(r)*bs if reflect else bs*np.cos(r)+bc*np.sin(r)
            cc = np.cos(r)*bc + np.sin(r)*bs if reflect else bc*np.cos(r)-bs*np.sin(r)
            totals[i] += np.sum((ss*ts+cc*tc)[valid]); counts[i] += valid.sum()
    if np.any(counts == 0): raise ValueError('No valid training directions for calibration')
    scores = totals/counts
    auto = candidates[int(np.argmax(scores))]
    policy = os.getenv('SWAN_BND_DIR_TRANSFORM', 'train_auto')
    chosen = auto
    if policy != 'train_auto':
        m = re.fullmatch(r'(rot|refl)([+-]?\d+(?:\.\d+)?)', policy)
        if not m: raise ValueError('Direction policy must be train_auto, rot-90, refl+270, etc.')
        chosen = (float(m.group(2)), m.group(1)=='refl')
    deg, reflect = chosen; r = np.deg2rad(deg)
    for start in range(0, len(bnd), 16):
        bs = bnd[start:start+16,2].copy(); bc = bnd[start:start+16,3].copy()
        bnd[start:start+16,2] = np.sin(r)*bc-np.cos(r)*bs if reflect else bs*np.cos(r)+bc*np.sin(r)
        bnd[start:start+16,3] = np.cos(r)*bc+np.sin(r)*bs if reflect else bc*np.cos(r)-bs*np.sin(r)
    name = lambda c: ('refl' if c[1] else 'rot') + f'{c[0]:+g}'
    table = {name(c):float(s) for c,s in zip(candidates,scores)}
    atomic_json('direction_manifest.json', dict(policy=policy, chosen=name(chosen), search_pick=name(auto),
                calibration_hash=array_hash(sel), n_calibration=len(sel), scores=table,
                channel_order=['hs','tm','sin','cos']))
    print(f'[DIRECTION] policy={policy} chosen={name(chosen)} train-calibration={len(sel)}', flush=True)
    return (deg + 1000 if reflect else deg), table


class PeakSampler(Sampler):
    """Fixed peak oversampling with a serializable permutation and cursor."""
    def __init__(self, allowed_indices, wave_data, seq_len, pct=95, up_factor=2, seed=42):
        self.allowed = np.asarray(allowed_indices, dtype=np.int64)
        hs = wave_data[seq_len:,0].reshape(len(wave_data)-seq_len,-1).max(axis=1)
        values = hs[self.allowed]
        if not len(values) or not np.isfinite(values).all(): raise ValueError('Invalid peak sampler scores')
        threshold = np.percentile(values, pct)
        self.peak_idx = self.allowed[values >= threshold]
        self.norm_idx = self.allowed[values < threshold]
        self.up_factor = int(up_factor)
        self.length = len(self.norm_idx)+self.up_factor*len(self.peak_idx)
        self.rng = np.random.default_rng(seed)
        self.order = np.empty(0, dtype=np.int64); self.cursor = 0
    def __len__(self): return self.length
    def __iter__(self):
        if self.cursor >= len(self.order):
            normal = self.rng.permutation(self.norm_idx)
            peaks = self.rng.choice(self.peak_idx, size=self.up_factor*len(self.peak_idx), replace=True)
            self.order = np.concatenate([normal,peaks]); self.rng.shuffle(self.order); self.cursor = 0
        while self.cursor < len(self.order):
            i = int(self.order[self.cursor]); self.cursor += 1
            yield i
    def state_dict(self):
        return dict(order=self.order.copy(), cursor=self.cursor, rng=self.rng.bit_generator.state)
    def load_state_dict(self, state):
        order = np.asarray(state['order'], dtype=np.int64)
        if len(order) not in (0,self.length) or not np.isin(order,self.allowed).all():
            raise ValueError('Incompatible sampler resume state')
        self.order=order; self.cursor=int(state['cursor']); self.rng.bit_generator.state=state['rng']


def strict_collate(batch):
    items=[]
    for item in batch:
        if item is None: raise ValueError('Missing sample; refusing to insert a zero sample')
        x,y = (torch.as_tensor(v).contiguous().float() for v in item)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite sample; refusing to change evaluation sample alignment')
        items.append((x,y))
    if not items: raise ValueError('Empty batch')
    return torch.stack([i[0] for i in items]), torch.stack([i[1] for i in items])


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] is not None: torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])


def train_loop(g, model, dl_tr, dl_va, w, *, epochs, acc_steps, norm_params,
               ckpt_prefix='ckpt', freeze_logvars_epochs=1, early_stop_patience=0, lambda_tv=2e-3):
    """Train with explicit successful-update budget and update-based validation.

    One reported cycle equals one full-data-equivalent epoch. Native sampler
    passes and successful optimizer updates are recorded separately. Default
    early stopping is disabled for controlled data-fraction comparisons.
    """
    from tqdm import tqdm
    device=g['device']; scaler=g['SCALER']
    if dl_tr.num_workers != 0: raise ValueError('Exact sampler resume requires num_workers=0')
    if len(dl_tr) == 0: raise ValueError('No complete training batches')
    job=json.loads(os.getenv('SWAN_REPAIR_JOB','{}'))
    smoke=job.get('stage')=='smoke'
    # Base sampler counts use raw Hs, so changing normalization does not change
    # the reference update budget.
    split = json.loads(Path('split_manifest.json').read_text()) if Path('split_manifest.json').exists() else {}
    base_n=int(split.get('base_sampler_length',len(dl_tr.sampler)))
    base_updates=math.ceil(base_n/(dl_tr.batch_size*acc_steps))
    interval=int(job.get('eval_every_updates',base_updates))
    budget=int(job.get('max_updates',epochs*base_updates))
    if interval < 1 or budget < 1: raise ValueError('Empty update budget')
    patience=int(job.get('early_stop_patience',early_stop_patience))
    if fraction()<1 and patience: raise ValueError('Data-fraction comparisons require early stopping disabled')
    opt=torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if n!='log_vars'], lr=g['max_lr'], weight_decay=g['weight_decay']),
        dict(params=[model.log_vars], lr=g['max_lr']*.1, weight_decay=0.)])
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[g['max_lr'],g['max_lr']*.1],
        total_steps=budget,pct_start=g['pct_start'],div_factor=g['div_factor'],final_div_factor=g['final_div_factor'])
    ema=g['EMA'](model,decay=.999)
    history={k:[] for k in ['train_losses','val_losses','train_mae_hs','train_mae_tm','train_mae_dir',
                           'val_mae_hs','val_mae_tm','val_mae_dir','epochs']}
    counts=dict(updates=0,attempts=0,skipped_updates=0,train_batches=0,native_passes_completed=0)
    best=float('inf'); best_update=0; bad=0; cycle=0; running=0.; nb=0; iterator=None
    resume=Path(f'{ckpt_prefix}_resume.pt'); best_path=Path(f'{ckpt_prefix}_best_ema.pth')
    audit_path=Path('training_audit.json')
    signature=dict(job=job,split_hash=split.get('selected_split_hash'),budget=budget,interval=interval,
                   batch_size=dl_tr.batch_size,acc_steps=acc_steps,version=VERSION)
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False)
        if state['signature'] != signature: raise ValueError('Resume configuration differs; use a new output root')
        model.load_state_dict(state['model']); opt.load_state_dict(state['opt']); sched.load_state_dict(state['sched'])
        if state['scaler']: scaler.load_state_dict(state['scaler'])
        ema.shadow={k:v.to(device) for k,v in state['ema'].items()}
        dl_tr.sampler.load_state_dict(state['sampler'])
        if dl_tr.generator is not None: dl_tr.generator.set_state(state['loader_rng'].cpu())
        history=state['history']; counts=state['counts']; best=state['best']; best_update=state['best_update']
        bad=state['bad']; cycle=state['cycle']; restore_rng(state['rng'])
        print(f'[RESUME] successful_updates={counts["updates"]}',flush=True)
    # A post-step hook distinguishes a real optimizer step from an AMP skip.
    stepped=[0]
    handle=opt.register_step_post_hook(lambda *args,**kwargs: stepped.__setitem__(0,stepped[0]+1))
    eval_idx=dl_tr.sampler.allowed[:8] if smoke else dl_tr.sampler.allowed
    eval_train=DataLoader(Subset(dl_tr.dataset,eval_idx.tolist()),batch_size=dl_tr.batch_size,
                          shuffle=False,drop_last=False,num_workers=0,collate_fn=strict_collate,
                          generator=torch.Generator().manual_seed(91))
    if smoke:
        dl_va=DataLoader(Subset(dl_va.dataset,list(range(min(8,len(dl_va.dataset))))),batch_size=1,
                         collate_fn=strict_collate,generator=torch.Generator().manual_seed(92))
    bar=tqdm(total=budget,initial=counts['updates'],desc='Successful updates')
    try:
        while counts['updates'] < budget and not (patience and bad>=patience):
            if iterator is None: iterator=iter(dl_tr)
            group=[]
            for _ in range(acc_steps):
                try: group.append(next(iterator))
                except StopIteration:
                    counts['native_passes_completed']+=1; iterator=None; break
            if iterator is not None and len(dl_tr.sampler.order)-dl_tr.sampler.cursor < dl_tr.batch_size:
                dl_tr.sampler.cursor=len(dl_tr.sampler.order)
                iterator=None; counts['native_passes_completed']+=1
            if not group: continue
            model.train(); model.log_vars.requires_grad_(counts['updates'] >= freeze_logvars_epochs*interval)
            virtual_epoch=1+counts['updates']//interval
            use_peak=virtual_epoch>=5; quantile=.90 if virtual_epoch<8 else .95
            opt.zero_grad(set_to_none=True)
            # Every yielded batch is full-sized (drop_last=True). Dividing by
            # the actual number of batches fixes the partial accumulation tail.
            for xb,yb in group:
                xb=xb.to(device); yb=yb.to(device)
                wb=w
                if wb.dim()==2: wb=wb[None,None]
                elif wb.dim()==3: wb=wb[None]
                we=wb.expand(xb.size(0),-1,-1,-1).squeeze(1)
                if use_peak:
                    hs=yb[:,0]; threshold=torch.quantile(hs.reshape(hs.size(0),-1).float(),quantile,dim=1).view(-1,1,1)
                    we=we*(1+.2*(hs>=threshold).float())
                with torch.amp.autocast(device_type=device.type,dtype=g['AMP_DTYPE'],enabled=g['AMP_ENABLED']):
                    outs=model(xb)
                    loss,_=g['deep_supervised_loss'](outs[0],outs[1:],yb,we,
                        model.log_vars if model.log_vars.requires_grad else None,lambda_tv=lambda_tv)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss/len(group)).backward()
                running+=float(loss.detach()); nb+=1; counts['train_batches']+=1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=not scaler.is_enabled())
            before=stepped[0]; scaler.step(opt); scaler.update(); counts['attempts']+=1
            if stepped[0]==before:
                counts['skipped_updates']+=1
                if counts['skipped_updates']>20: raise FloatingPointError('Too many AMP skipped updates')
                continue
            sched.step(); counts['updates']+=1; bar.update(1)
            with torch.no_grad(): model.log_vars.clamp_(-5.,5.)
            ema.update(model)
            if counts['updates']%interval and counts['updates']!=budget: continue
            cycle+=1
            ema.apply_to(model)
            try:
                vl=g['compute_loss'](model,dl_va,w,lambda_tv=lambda_tv)
                va=g['compute_mae_phys'](model,dl_va,w,norm_params)
                tr=g['compute_mae_phys'](model,eval_train,w,norm_params)
                metric=float(va['hs'])
                if not all(math.isfinite(v) for v in [vl,metric,*va.values(),*tr.values()]):
                    raise FloatingPointError('Nonfinite evaluation metric')
                improved=metric < best-1e-9
                if improved:
                    best=metric; best_update=counts['updates']; bad=0
                    atomic_torch(best_path,model.state_dict())
            finally: ema.restore(model)
            if improved: atomic_torch(f'{ckpt_prefix}_best_raw.pth',model.state_dict())
            else: bad+=1
            atomic_torch(f'{ckpt_prefix}_last_raw.pth',model.state_dict())
            history['epochs'].append(cycle); history['train_losses'].append(running/max(nb,1)); history['val_losses'].append(vl)
            for k in ['hs','tm','dir']:
                history['train_mae_'+k].append(float(tr[k])); history['val_mae_'+k].append(float(va[k]))
            running=0.; nb=0
            early=bool(patience and bad>=patience)
            report=dict(version=VERSION,signature=signature,**counts,completed_cycles=cycle,
                target_updates=budget,eval_every_updates=interval,early_stop=early,
                best_update=best_update,best_val_hs_mae=best,evaluated_checkpoint=str(best_path.resolve()),
                selection_metric='val_hs_mae_ema',history=history,
                n_parameters=sum(p.numel() for p in model.parameters()),
                sampler_length=len(dl_tr.sampler),n_train=len(dl_tr.sampler.allowed),
                base_split_hash=split.get('base_split_hash'),selected_split_hash=split.get('selected_split_hash'),
                direction=json.loads(Path('direction_manifest.json').read_text()) if Path('direction_manifest.json').exists() else None)
            atomic_json(audit_path,report)
            atomic_torch(resume,dict(signature=signature,model=model.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),
                scaler=scaler.state_dict(),ema=ema.shadow,sampler=dl_tr.sampler.state_dict(),
                loader_rng=dl_tr.generator.get_state() if dl_tr.generator is not None else None,
                rng=rng_state(),history=history,counts=counts,best=best,best_update=best_update,bad=bad,cycle=cycle))
            print(f'[CYCLE {cycle}] updates={counts["updates"]}/{budget} val_Hs_MAE={metric:.6f} best={best:.6f}',flush=True)
            if early: break
    finally:
        handle.remove(); bar.close()
    if not best_path.exists(): raise RuntimeError('No best EMA checkpoint exists')
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=True))
    history['audit']=json.loads(audit_path.read_text())
    # Retain the full resume state until evaluation and summary collection succeed.
    return history


def code_hashes(package):
    names=['train_repaired.py','legacy_repaired.py','repair_support.py','run_repaired.py']
    return {n:hashlib.sha256((Path(package)/n).read_bytes()).hexdigest() for n in names}


def asset_signature(server, data, bnd_on=True):
    """Content hashes for helpers, metadata fingerprints for large source files."""
    server=Path(server); data=Path(data)
    if not data.is_file(): raise FileNotFoundError(data)
    st=data.stat(); files={str(data.resolve()):[st.st_size,st.st_mtime_ns]}
    helpers={}
    if bnd_on:
        for name in ['bnd_features.py','boundspec_segments.py']:
            p=server/name
            if not p.is_file(): raise FileNotFoundError(p)
            helpers[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for year in [2019,2020]:
            folder=Path(os.getenv(f'SWAN_BND_DIR_{year}',str(server/f'bnd_{year}_v2')))
            if not folder.is_dir(): raise FileNotFoundError(folder)
            paths=sorted(p for p in folder.rglob('*') if p.is_file())
            if not paths: raise ValueError(f'Empty BND directory: {folder}')
            for p in paths:
                st=p.stat(); files[str(p.resolve())]=[st.st_size,st.st_mtime_ns]
    record=dict(files=files,helpers=helpers)
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()


def verify_job(job, package):
    if job.get('code_hashes')!=code_hashes(package): raise ValueError('Code changed after plan creation')
    actual=asset_signature(os.environ['SWAN_SERVER_ROOT'],os.environ['SWAN_DATA_PATH'],job.get('use_bnd')=='on')
    if job.get('asset_signature')!=actual: raise ValueError('Data/helpers changed after plan creation')
    fraction_value=float(job.get('train_fraction',1.))
    if not math.isfinite(fraction_value) or not .25 <= fraction_value <= 1.: raise ValueError('Invalid fraction')
    if job.get('protocol_version')!=VERSION: raise ValueError('Wrong protocol version')


def validate_source(ds, time_steps):
    """Check physical wet cells once per source fingerprint, before sanitization."""
    import fcntl
    import pandas as pd
    job=json.loads(os.getenv('SWAN_REPAIR_JOB','{}'))
    key=hashlib.sha256(json.dumps([job.get('asset_signature'),time_steps,VERSION]).encode()).hexdigest()
    cache=Path(os.getenv('SWAN_RESULTS_ROOT','.'))/'_source_checks'; cache.mkdir(parents=True,exist_ok=True)
    with open(cache/(key+'.lock'),'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        done=cache/(key+'.json')
        if done.exists() and job.get('asset_signature'):
            print('[SOURCE CHECK] cached',flush=True); return
        if 'time' not in ds or ds.sizes['time']<time_steps: raise ValueError('Missing/short time axis')
        times, gaps, records = hourly_time_axis(ds['time'].values[:time_steps])
        print(f'[TIME] hourly records with {len(gaps)} gap(s); crossing sequences will be excluded', flush=True)
        for record in records:
            print(f'[TIME GAP] {record}', flush=True)
        mask=ds['kcs'].values
        if mask.ndim==3:
            if not np.all(mask==mask[0]): raise ValueError('Time-varying wet mask is unsupported')
            mask=mask[0]
        wet=mask>0
        if not wet.any(): raise ValueError('Empty wet mask')
        for name in ['windu','windv','depth','veloc-x','veloc-y','hsign','period','dir']:
            da=ds[name]
            for start in range(0,time_steps if 'time' in da.dims else 1,64):
                if start%1024==0:print(f'[SOURCE CHECK] {name} frame {start}/{time_steps}',flush=True)
                a=da.isel(time=slice(start,min(start+64,time_steps))).values if 'time' in da.dims else da.values
                if a.shape[-2:]!=wet.shape or not np.isfinite(a[...,wet]).all():
                    raise ValueError(f'Nonfinite wet-cell data or shape mismatch in {name}, frame {start}')
            print(f'[SOURCE CHECK] {name} OK',flush=True)
        for name in ['x','y']:
            if not np.isfinite(ds[name].values[...,wet]).all(): raise ValueError(f'Invalid coordinates {name}')
        if job.get('asset_signature'): atomic_json(done,dict(checked=True,signature=job['asset_signature']))
