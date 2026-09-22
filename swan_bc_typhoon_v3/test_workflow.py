"""CPU tests for scheduling, frozen selection, event definitions, and metrics."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import run_campaign as d
import events
import evaluate_2021 as ev
import report_events as report


def job(model, tag, seed=42, width=64, lr=.0001):
    return dict(model=model, config_id=f'{model}_{tag}_w{width}d4m24_s{seed}', seed=seed,
                epochs=30, train_fraction=1., hyperparams=dict(fno_width=width,fno_depth=4,
                modes_x=24,modes_y=24,batch_size=1,acc_steps=4,use_checkpoint=True,max_lr=lr))


def summary(j,value):
    return dict(model=j['model'],seed=j['seed'],config_id=j['config_id'],repair_job=j,
                repair_audit=dict(best_val_hs_mae=value,updates=76950),best_weight='/test/checkpoint',
                legacy_metrics=dict(rmse_m=999))


class WorkflowTests(unittest.TestCase):
    def test_wait_for_family_B(self):
        b=job('tno','B0');pilot=job('tno','pilot')
        self.assertIsNone(d.ready_candidates('tno',{}, {('tno',42):summary(pilot,.2)},[b],{}))

    def test_selection_ignores_test_error(self):
        a=summary(job('fno','A'),.1);b=summary(job('fno','B'),.2)
        a['legacy_metrics']['rmse_m']=100;b['legacy_metrics']['rmse_m']=.001
        self.assertEqual(d.c.choose([a,b],1)[0],a)

    def test_seed_preserves_learning_rate_and_batch(self):
        j=job('tno','B',width=256,lr=.0002);q=d.c.make_job(j,'C',seed=43)
        self.assertEqual(j['hyperparams'],q['hyperparams']);self.assertEqual(q['seed'],43)

    def test_completed_B_reused_and_other_C_overlaps_TNO_B(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';old=Path(tmp)/'old';root.mkdir()
            bases={m:job(m,'pilot') for m in d.c.MODELS}
            pilots={(m,s):summary(job(m,'pilot',s),.5) for m in d.c.MODELS for s in (42,43,44)}
            ap=[job(m,'A',width=w) for m in d.c.MODELS for w in (128,256)]
            ar={j['config_id']:summary(j,.2 if j['hyperparams']['fno_width']==128 else .3) for j in ap}
            d.c.atomic(old/'A/plan.json',ap)
            bp=d.c.stage_b(list(ar.values()))
            saved={j['config_id']:summary(j,.1 if j['model']=='fno' else .4) for j in bp if j['model']!='tno'}
            launched=[]
            class Proc:
                pid=99999
                def __init__(self,cmd,**kwargs):
                    self.j=d.c.read(Path(cmd[cmd.index('--job')+1]));self.polls=0;launched.append(self.j)
                def poll(self):
                    self.polls+=1
                    if self.polls<3 and '_B' in self.j['config_id']:return None
                    saved[self.j['config_id']]=summary(self.j,.15);return 0
                def terminate(self):pass
                def wait(self,timeout=None):return 0
            r=SimpleNamespace(checked_summary=lambda source,j:saved.get(j['config_id']),
                compare_protocols=lambda rows:None,gpu_inventory=lambda:{g:dict(busy=False,mib=200000) for g in range(8)},
                parameter_bytes_floor=lambda j:1,run_name=lambda j:j['config_id'])
            with patch.object(d.subprocess,'Popen',Proc),patch.object(d.time,'sleep'),patch.object(d.signal,'signal'):
                d.run_schedule(None,r,root,old,Path(tmp),bp,bases,pilots,list(range(8)),{'asset_signature':{}},ar)
                first_count=len(launched)
                d.run_schedule(None,r,root,old,Path(tmp),bp,bases,pilots,list(range(8)),{'asset_signature':{}},ar)
            self.assertEqual(first_count,10)
            self.assertEqual(len(launched),first_count)
            self.assertTrue(all(j['model']=='tno' and '_B' in j['config_id'] for j in launched[:4]))
            self.assertEqual({j['model'] for j in launched[4:8]},{'fno','ffno'})
            self.assertTrue(all('_C_' in j['config_id'] for j in launched[4:]))
            self.assertEqual(len(d.c.read(root/'selected_9.json')),9)
            self.assertIn('_B',d.c.read(root/'selection_fno.json')['job']['config_id'])
            self.assertIn('_A_',d.c.read(root/'selection_ffno.json')['job']['config_id'])

    def test_relative_wrapper_process_is_detected(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            server=Path(tmp);folder=server/'swan_iclr_campaign_v2';folder.mkdir()
            (folder/'run_v2.py').write_text('import time; time.sleep(30)')
            proc=subprocess.Popen([sys.executable,'-u','run_v2.py'],cwd=folder)
            try:
                detected=d.processes(server)
                self.assertTrue(any(pid==proc.pid and str(folder/'run_v2.py') in args for pid,args in detected))
            finally:
                proc.terminate();proc.wait(timeout=5)

    def test_freeze_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'selection.json';d.c.freeze(p,{'a':1})
            with self.assertRaises(ValueError):d.c.freeze(p,{'a':2})

    def test_jma_parser(self):
        header='66666 2114    2 0045 2114 0 6             CHANTHU               20220101'
        lines='21091700 002 5 325 1293  935     095\n21091706 002 6 330 1300  980     000\n'
        storms=events.parse_jma(header+'\n'+lines)
        self.assertEqual(storms[0]['id'],'2114')
        self.assertTrue(storms[0]['lifetime_typhoon'])
        self.assertEqual(storms[0]['points'][0]['lon'],129.3)
        self.assertEqual(storms[0]['points'][0]['time'],'2021-09-17T00:00:00')

    def test_high_wave_metrics_and_circular_angles(self):
        truth=np.zeros((4,2,2));truth[0]=[[1,3],[5,6]];truth[1]=[[10,31],[0,5]]
        truth[2]=np.sin(np.deg2rad(359));truth[3]=np.cos(np.deg2rad(359))
        pred=truth.copy();pred[0]+=1;pred[2]=np.sin(np.deg2rad(1));pred[3]=np.cos(np.deg2rad(1))
        out=ev.frame_metrics(pred,truth,np.ones((2,2),bool),{'hs':[0,1],'tm':[0,1]})
        self.assertEqual(out['hs_ge3_count'],3);self.assertEqual(out['hs_ge5_count'],2)
        self.assertEqual(out['hs_ge3_sse'],3);self.assertEqual(out['true_tm_gt30_count'],1)
        self.assertAlmostEqual(out['dir_mae'],2);self.assertEqual(out['pred_hs_at_true_max'],7)
        result=report.summarize([out,out]);self.assertEqual(result['hs_ge5_rmse'],1)

    def test_no_tail_samples_not_zero_error(self):
        truth=np.zeros((4,2,2));truth[3]=1
        out=ev.frame_metrics(truth,truth,np.ones((2,2),bool),{'hs':[0,1],'tm':[0,1]})
        result=report.summarize([out]);self.assertIsNone(result['hs_ge3_mae'])

    def test_peak_timing_sign(self):
        rows=[dict(time='2021-01-01T00',true_hs_max=4,pred_hs_max=3,pred_hs_at_true_max=2),
              dict(time='2021-01-01T02',true_hs_max=3,pred_hs_max=5,pred_hs_at_true_max=3)]
        out=report.peak_metrics(rows)
        self.assertEqual(out['domain_peak_time_error_h'],2)
        self.assertEqual(out['true_peak_location_time_bias_m'],-2)

    def test_event_geometry_and_year_clipping(self):
        import sys
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);nc=p/'grid.nc'
            dataset=SimpleNamespace(x=SimpleNamespace(values=np.array([[129.3,129.4]])),
                y=SimpleNamespace(values=np.array([[32.5,32.5]])), kcs=SimpleNamespace(values=np.ones((1,2))))
            xr=SimpleNamespace(open_dataset=lambda path:nullcontext(dataset))
            text='66666 2101    2 0045 2101 0 6             TEST                 20220101\n21010100 002 3 325 1293  990     040\n21010106 002 5 330 1300  980     065\n'
            (p/'jma_besttrack.txt').write_text(text)
            proto=dict(event_distance_km=400,event_padding_hours=24,tc_grades=[3,4,5,9])
            with patch.dict(sys.modules,{'xarray':xr}):
                out=events.build_events(nc,p,proto,p/'events.json')
            self.assertEqual(out['events'][0]['start'],'2021-01-01T12')
            self.assertEqual(out['events'][0]['end'],'2021-01-02T06')


if __name__=='__main__':unittest.main()
