"""Synthetic complete-year report integration; no scientific results."""
import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from evaluate_2021 import frame_metrics
from report_events import report
class Integration(unittest.TestCase):
    def test_single_nonfourier_family_full_year(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model='conv_swin'
            event=dict(id='synthetic',name='Synthetic event',start='2021-01-01T12',end='2021-01-01T14',lifetime_typhoon=True)
            (root/'events_2021.json').write_text(json.dumps(dict(events=[event])))
            (root/'snapshots.json').write_text(json.dumps([dict(event_id='synthetic',target_index=12)]))
            entries=[dict(job=dict(model=model,seed=s,config_id=f'synthetic_{s}')) for s in (42,43,44)]
            (root/'selected_2021.json').write_text(json.dumps(entries))
            times=np.arange(np.datetime64('2021-01-01T12','h'),np.datetime64('2022-01-01','h'))
            truth=np.ones((4,3,3),dtype='float32');truth[0]=4;truth[1]=7;truth[2]=0
            wet=np.ones((3,3),bool)
            for seed in (42,43,44):
                pred=truth.copy();pred[0]+=(seed-41)*.01
                vals=frame_metrics(pred,truth,wet,dict(hs=[0,1],tm=[0,1]))
                folder=root/'models'/f'{model}_s{seed}';(folder/'snapshots').mkdir(parents=True)
                with (folder/'hourly.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['target_index','time',*vals]);w.writeheader()
                    for idx,t in enumerate(times,12):w.writerow(dict(target_index=idx,time=str(t),**vals))
                np.savez(folder/'snapshots/12.npz',pred=pred,true=truth,kcs=wet,time=str(times[0]))
            report(root)
            self.assertTrue((root/'analysis_completed.json').exists())
            with (root/'event_metrics_by_seed.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),15)
            annual=[r for r in rows if r['scope']=='annual']
            self.assertTrue(all(int(r['frames'])==8748 for r in annual))
            self.assertTrue(all(r['hs_ge5_mae']=='' for r in rows))
            self.assertEqual(len(list((root/'figures').glob('*.png'))),4)
if __name__=='__main__':unittest.main()
