#!/usr/bin/env python3
"""Summarize externally evaluated regions; never relabel a new year as a new sea."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from analyze import read, write_csv, sha

def main(a):
    manifest=read(a.manifest);source=manifest['training_region'];records=manifest.get('regions',[])
    if not records:raise ValueError('No other-region predictions provided. Geographic generalization is NOT EVALUATED.')
    result=[]; provenance=[]
    for region in records:
        if region['region_id']==source:raise ValueError('Same region is temporal generalization, not geographic')
        if not region.get('excluded_from_training_and_selection'):raise ValueError('Region participated in training/selection')
        if region.get('adaptation')!='none':raise ValueError('This report supports zero-shot only; fine-tuning requires a separate study')
        for k in ('grid_mapping','forcing_contract','target_units','mask_policy','normalization_policy','checkpoint_sha256_by_model','evaluation_code_sha256'):
            if not region.get(k):raise ValueError('Missing region audit field: '+k)
        p=Path(region['metrics_csv']);p=p if p.is_absolute() else a.manifest.parent/p
        with p.open() as f:rows=list(csv.DictReader(f))
        keys=[(r['model'],int(r['seed']),r['scope']) for r in rows]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate region/model/seed/scope')
        for scope in sorted({r['scope'] for r in rows}):
            for model in ('fno','ffno','tno'):
                rr=[r for r in rows if r['scope']==scope and r['model']==model]
                if sorted(int(r['seed']) for r in rr)!=[42,43,44]:raise ValueError('Need three seeds per region/model/scope')
                for r in rr:
                    expected=region['checkpoint_sha256_by_model'][model][str(r['seed'])]
                    if len(expected)!=64 or r['checkpoint_sha256']!=expected:raise ValueError('Checkpoint hash mismatch')
                x=np.array([float(r['hs_mae']) for r in rr])
                if not np.isfinite(x).all() or (x<0).any():raise ValueError('Invalid physical Hs MAE')
                result.append(dict(region=region['region_id'],scope=scope,model=model,hs_mae_m_mean=x.mean(),seed_sd=x.std(ddof=1),n_seeds=3,
                    claim='External supplied metrics; region audit is user-attested; no universal geographic claim'))
        provenance.append(dict(region=region['region_id'],metrics_path=str(p),sha256=sha(p)))
    a.output.mkdir(parents=True,exist_ok=False);write_csv(a.output/'geographic_summary.csv',result)
    (a.output/'provenance.json').write_text(json.dumps(dict(manifest=manifest,inputs=provenance),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
