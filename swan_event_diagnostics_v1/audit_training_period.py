#!/usr/bin/env python3
"""Audit saved split indices; no training-period inference or holdout claims."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from analysis import write_csv

def read(path):return json.loads(Path(path).read_text())

def split_timeline(times, starts, length):
    n=len(times);labels=np.full(n,'excluded',dtype=object)
    seen_input=np.zeros(n,dtype=bool);seen_target=np.zeros(n,dtype=bool)
    for split in ('train','val','test'):
        ids=np.asarray(starts[split],dtype=int)
        if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate split starts')
        if len(ids) and (ids.min()<0 or ids.max()+length>=n):raise ValueError('Split exceeds source times')
        target=ids+length
        if np.any(labels[target]!='excluded'):raise ValueError('Overlapping target splits')
        labels[target]=split
        for i in ids:
            if np.any(np.diff(times[i:i+length+1])!=np.timedelta64(1,'h')):
                raise ValueError('Saved split crosses a time gap')
            if split=='train':seen_input[i:i+length]=True
        if split=='train':seen_target[target]=True
    rows=[]
    for i in range(length,n):
        continuous=bool(np.all(np.diff(times[i-length:i+1])==np.timedelta64(1,'h')))
        rows.append(dict(target_index=i,time=str(times[i]),target_split=labels[i],
            target_time_seen_as_training_input=bool(seen_input[i]),
            input_times_overlap_training_input=bool(seen_input[i-length:i].any()),
            input_times_overlap_training_target=bool(seen_target[i-length:i].any()),
            continuous_input_and_target=continuous))
    return rows

def main(a):
    import xarray as xr
    if a.output.exists():raise ValueError('Use a new audit output directory')
    with xr.open_dataset(a.nc) as ds:all_times=ds.time.values.astype('datetime64[ns]')
    a.output.mkdir(parents=True)
    manifests=[];summary=[]
    events=[]
    if a.events_csv:
        with a.events_csv.open() as f:events=list(csv.DictReader(f))
        for event in events:
            if np.datetime64(event['start'])>np.datetime64(event['end']):raise ValueError('Invalid event window')
            if int(event['start'][:4]) not in (2019,2020):raise ValueError('Expected historical event')
    for model in a.models.split(','):
        for entry in read(a.eval_root/model/'selected_2021.json'):
            job=entry['job'];folder=Path(entry['checkpoint']).parent
            split_file=folder/'split_indices.npz';manifest=read(folder/'split_manifest.json')
            length=int(job['hyperparams']['seq_length']);times=all_times[:int(job['time_steps'])]
            if manifest['seq_length']!=length:raise ValueError('Sequence length mismatch')
            with np.load(split_file,allow_pickle=False) as z:
                rows=split_timeline(times,{k:z[k] for k in ('train','val','test')},length)
            name=f'{model}_s{job["seed"]}'
            write_csv(a.output/(name+'_timeline.csv'),rows)
            manifests.append(dict(model=model,seed=job['seed'],source_nc=str(a.nc),
                split_file=str(split_file),split_sha256=hashlib.sha256(split_file.read_bytes()).hexdigest(),
                split_manifest=manifest,note='Train means included in training split, not proof of sampling frequency. Validation influenced selection.'))
            for e in events:
                selected=[r for r in rows if np.datetime64(e['start'])<=np.datetime64(r['time'])<=np.datetime64(e['end'])]
                if not selected:raise ValueError(f'Event outside source time axis: {e}')
                r=dict(model=model,seed=job['seed'],event=e['id'],name=e['name'],frames=len(selected),
                       classification='training-era diagnostic; not an independent held-out cyclone')
                r.update({k+'_targets':sum(x['target_split']==k for x in selected) for k in ('train','val','test','excluded')})
                r['input_overlap_training_frames']=sum(x['input_times_overlap_training_input'] for x in selected)
                summary.append(r)
    (a.output/'provenance.json').write_text(json.dumps(manifests,indent=2))
    if summary:write_csv(a.output/'historical_event_exposure.csv',summary)
    print('Audit complete:',a.output)
    print('No historical predictions were generated. Do not label mixed historical events as independent tests.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--eval-root',type=Path,default=Path('/home/jovyan/swan/runs/iclr_parallel_v4/evaluation'))
    p.add_argument('--nc',type=Path,default=Path('/home/jovyan/swan/wavm-Waves_2019_2020_v2.nc'))
    p.add_argument('--models',default='fno,ffno,tno')
    p.add_argument('--events-csv',type=Path)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
