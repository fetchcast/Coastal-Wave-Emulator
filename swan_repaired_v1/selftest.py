#!/usr/bin/env python3
"""CPU regression tests. Use --extended for a synthetic NetCDF worker test."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

import repair_support as r
import legacy_repaired as l
import train_repaired as t

torch.set_num_threads(2)

@contextlib.contextmanager
def workspace():
    cwd=Path.cwd();env=dict(os.environ)
    with tempfile.TemporaryDirectory(prefix='swan_repair_test_') as d:
        os.chdir(d)
        try:yield Path(d)
        finally:
            os.chdir(cwd);os.environ.clear();os.environ.update(env)


def unit_weights(layer):
    with torch.no_grad():
        for p in layer.parameters():p[...,0].fill_(1);p[...,1].zero_()

class RepairTests(unittest.TestCase):
    def test_fft2_signed_modes_and_odd_grid(self):
        for h,w in [(9,10),(8,9),(1,8)]:
            layer=t.SpectralConv2d(1,1,h,w);unit_weights(layer)
            x=torch.randn(2,1,h,w,requires_grad=True);y=layer(x)
            torch.testing.assert_close(y,x,atol=2e-6,rtol=2e-6)
            y.square().mean().backward();self.assertTrue(torch.isfinite(x.grad).all())
    def test_fft3_all_sign_regions(self):
        for tt,h,w in [(5,7,8),(4,6,9),(1,1,8)]:
            layer=t.SpectralConv3d(1,1,tt,h,w);unit_weights(layer)
            x=torch.randn(1,1,tt,h,w,requires_grad=True);y=layer(x)
            torch.testing.assert_close(y,x,atol=2e-6,rtol=2e-6)
            y.square().mean().backward();self.assertTrue(torch.isfinite(x.grad).all())
    def test_model_forward_backward(self):
        for name in ('fno','tno','ffno'):
            model=t.LegacyCompatibleBenchmarkModel(name,6,4,seq_length=4,fno_width=8,fno_depth=2,modes_x=3,modes_y=3,modes_t=2,width=8)
            x=torch.randn(1,4,6,7,9);outs=model(x)
            self.assertEqual(outs[0].shape,(1,4,7,9))
            sum(o.square().mean() for o in outs).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    def test_circular_loss_and_layout(self):
        p=torch.zeros(1,4,3,5);y=p.clone();p[:,2:]=1
        y[:,2]=math.sin(math.pi/3);y[:,3]=math.cos(math.pi/3)
        p.requires_grad_();loss,parts=l.ds_loss(p,y)
        self.assertAlmostEqual(float(parts[2]),1-math.cos(math.pi/12),places=6)
        loss.backward();self.assertGreater(float(p.grad[:,2:].abs().sum()),0)
        z=torch.zeros_like(p,requires_grad=True);l.ds_loss(z,y)[0].backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(abs((359-1+180)%360-180),2)
    def test_strict_split_nested_and_excluded(self):
        with workspace():
            wave=np.random.default_rng(302).random((17498,1,1,1),dtype=np.float32)
            splits={}
            for f in (1.,.5,.25):
                os.environ['SWAN_TRAIN_FRACTION']=str(f)
                splits[f]=r.strict_split(l.make_block_stratified_split,wave,12)[:3]
            self.assertEqual(tuple(map(len,splits[1.])),(9770,1980,1980))
            for f in (.5,.25):
                for k in (1,2):np.testing.assert_array_equal(splits[f][k],splits[1.][k])
                stamps=[set((idx[:,None]+np.arange(13)).ravel()) for idx in splits[f]]
                self.assertFalse(stamps[0]&stamps[1] or stamps[0]&stamps[2] or stamps[1]&stamps[2])
            self.assertTrue(set(splits[.25][0])<=set(splits[.5][0])<=set(splits[1.][0]))
            for raw in ('0','-1','1.1','NaN','oops'):
                os.environ['SWAN_TRAIN_FRACTION']=raw
                with self.assertRaises(ValueError):r.strict_split(l.make_block_stratified_split,wave,12)
    def test_direction_does_not_read_heldout_targets(self):
        with workspace():
            os.environ['SWAN_BND_DIR_TRANSFORM']='train_auto'
            times=pd.date_range('2019-01-01',periods=10,freq='h')
            ds=xr.Dataset({'dir':(('time','y','x'),np.zeros((10,1,2)))},coords={'time':times})
            b=np.zeros((10,4,1,2),np.float32);b[:,3]=1
            r.SPLIT_CONTEXT['calibration_indices']=np.array([0,1])
            choice1=r.align_bnd(b.copy(),ds,np.ones((1,2)),times,2)
            ds['dir'].values[[0,1,4,5,6,7,8,9]]=180
            choice2=r.align_bnd(b.copy(),ds,np.ones((1,2)),times,2)
            self.assertEqual(choice1,choice2)
    def test_sampler_resume(self):
        wave=np.arange(112,dtype=np.float32).reshape(112,1,1,1)
        a=r.PeakSampler(np.arange(100),wave,12);it=iter(a);first=[next(it) for _ in range(17)]
        state=copy.deepcopy(a.state_dict());rest=list(it)
        b=r.PeakSampler(np.arange(100),wave,12);b.load_state_dict(state)
        self.assertEqual(rest,list(b));self.assertEqual(list(a),list(b))
        self.assertEqual(len(first)+len(rest),105)
    def test_collate_refuses_nonfinite(self):
        with self.assertRaises(ValueError):r.strict_collate([(np.array([np.nan]),np.array([0.]))])
    def test_optimizer_tail_and_lr_ratio(self):
        p=torch.nn.Parameter(torch.tensor(1.));q=torch.nn.Parameter(torch.tensor(1.))
        opt=torch.optim.AdamW([dict(params=[p],weight_decay=.01),dict(params=[q],weight_decay=0.)])
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[1e-4,1e-5],total_steps=10)
        for _ in range(10):
            self.assertAlmostEqual(opt.param_groups[1]['lr']/opt.param_groups[0]['lr'],.1)
            opt.zero_grad();((p+q)/1).backward();opt.step();sched.step()
        # The repaired tail divides by its actual batch count, not acc_steps.
        z=torch.tensor(2.,requires_grad=True)
        (z.square()/1).backward();self.assertEqual(float(z.grad),4.)
    def test_train_resume_is_equivalent(self):
        with workspace() as work:
            os.environ['SWAN_TRAIN_FRACTION']='1'
            os.environ['SWAN_REPAIR_JOB']=json.dumps(dict(stage='smoke',max_updates=6,eval_every_updates=2))
            class Tiny(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.conv=torch.nn.Conv2d(2,4,1);self.drop=torch.nn.Dropout(.2);self.log_vars=torch.nn.Parameter(torch.zeros(3))
                def forward(self,x):return [self.conv(self.drop(x[:,-1]))]
            def run(folder,interrupt=False,skip=False):
                folder.mkdir(exist_ok=True);os.chdir(folder)
                torch.manual_seed(71);np.random.seed(71);random.seed(71)
                x=np.random.default_rng(0).normal(size=(19,2,4,5)).astype(np.float32)
                y=np.random.default_rng(1).random((19,4,4,5),dtype=np.float32);y[:,2]=0.;y[:,3]=1.
                ds=l.WindWaveDataset(x,y,2,0,17)
                sampler=r.PeakSampler(np.arange(7),y,2)
                dl=DataLoader(ds,batch_size=1,sampler=sampler,drop_last=True,collate_fn=r.strict_collate,generator=torch.Generator().manual_seed(7))
                va=DataLoader(l.SubsetIndicesDataset(ds,np.arange(8,12)),batch_size=1,collate_fn=r.strict_collate,generator=torch.Generator().manual_seed(8))
                model=Tiny();g=dict(vars(l));g.update(device=torch.device('cpu'),AMP_ENABLED=False,SCALER=torch.amp.GradScaler(enabled=False))
                if skip:
                    class SkipOnce:
                        def __init__(self):self.pending=True
                        def is_enabled(self):return False
                        def scale(self,loss):return loss
                        def unscale_(self,opt):pass
                        def step(self,opt):
                            if self.pending:self.pending=False
                            else:opt.step()
                        def update(self):pass
                        def state_dict(self):return {}
                    g['SCALER']=SkipOnce()
                original=r.atomic_torch
                def save(path,obj):
                    original(path,obj)
                    if interrupt and str(path).endswith('_resume.pt'):raise RuntimeError('TEST INTERRUPTION')
                r.atomic_torch=save
                try:hist=r.train_loop(g,model,dl,va,torch.ones(4,5)/20,epochs=1,acc_steps=3,norm_params={'hs':(0.,1.),'tm':(0.,1.)})
                finally:r.atomic_torch=original;os.chdir(work)
                return model.state_dict(),hist
            uninterrupted,h1=run(work/'full')
            with self.assertRaisesRegex(RuntimeError,'TEST INTERRUPTION'):run(work/'resumed',True)
            resumed,h2=run(work/'resumed')
            for k in uninterrupted:torch.testing.assert_close(uninterrupted[k],resumed[k],rtol=0,atol=0)
            for k in ['updates','attempts','train_batches','native_passes_completed','best_update']:
                self.assertEqual(h1['audit'][k],h2['audit'][k])
            for k in ['train_losses','val_losses','train_mae_hs']:self.assertEqual(h1[k],h2[k])
            self.assertEqual(h1['audit']['updates'],6)
            _,skipped=run(work/'skipped',skip=True)
            self.assertEqual(skipped['audit']['updates'],6)
            self.assertEqual(skipped['audit']['attempts'],7)
            self.assertEqual(skipped['audit']['skipped_updates'],1)
            best=torch.load(work/'full/ckpt_best_ema.pth',weights_only=True)
            for k in best:torch.testing.assert_close(best[k],uninterrupted[k])


def extended():
    from run_repaired import job
    with workspace() as work:
        nt,ny,nx=5000,5,6;rng=np.random.default_rng(3)
        wet=np.ones((ny,nx),np.int16);wet[0,0]=0
        shape=(nt,ny,nx)
        fields={v:(('time','lat','lon'),rng.random(shape,dtype=np.float32)+.1) for v in ['windu','windv','veloc-x','veloc-y','hsign','period']}
        fields.update(dir=(('time','lat','lon'),rng.random(shape,dtype=np.float32)*360),
                      depth=(('lat','lon'),np.ones((ny,nx),np.float32)*30),
                      kcs=(('lat','lon'),wet),x=(('lat','lon'),np.tile(np.linspace(126,130,nx),(ny,1))),
                      y=(('lat','lon'),np.tile(np.linspace(33,37,ny)[:,None],(1,nx))))
        ds=xr.Dataset(fields,coords={'time':pd.date_range('2019-01-01',periods=nt,freq='h')})
        path=work/'tiny.nc';ds.to_netcdf(path)
        out=work/'runs';out.mkdir()
        os.environ.update(SWAN_SERVER_ROOT=str(work),SWAN_DATA_PATH=str(path),SWAN_RESULTS_ROOT=str(out),SWAN_BND_DIR_TRANSFORM='train_auto')
        t.CONFIG.update(data_path=str(path),results_root=str(out),station_root=str(work),benchmark_after_train=False)
        package=Path(__file__).resolve().parent
        for name in ('fno','tno','ffno'):
            j=job(name,8,2,3,'smoke','synthetic');j.update(use_bnd='off',time_steps=nt,max_updates=2,eval_every_updates=2,
                   code_hashes=r.code_hashes(package),asset_signature=r.asset_signature(work,path,False),bnd_dir_transform='train_auto')
            jf=work/f'{name}.json';jf.write_text(json.dumps(j))
            rc=t.worker_main(str(jf));assert rc==0,(name,rc)
            if name=='tno':assert j['hyperparams']['width']==8 and j['hyperparams']['depth']==2
        summaries=list(out.glob('*/run_summary.json'));assert len(summaries)==3
        for p in summaries:
            s=json.loads(p.read_text());a=s['repair_audit'];assert a['updates']==2
            assert a['selection_metric']=='val_hs_mae_ema'
            aweights=torch.load(s['best_weight'],weights_only=True)
            saved=next(p.parent.glob('*model_weights*.pth'))
            bweights=torch.load(saved,weights_only=True)
            for k in aweights:torch.testing.assert_close(aweights[k],bweights[k],rtol=0,atol=0)
        print('[PASS] End-to-end CPU workers: FNO, TNO, FFNO, NetCDF, training, best-EMA evaluation, summaries')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--extended',action='store_true');args=ap.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RepairTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.extended:extended()
