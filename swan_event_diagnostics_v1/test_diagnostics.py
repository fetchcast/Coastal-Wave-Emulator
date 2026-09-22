"""CPU tests for reconstruction, threshold scoring, peak timing, and split indexing."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from analysis import analyze_event,aggregate,detection
from audit_training_period import split_timeline

class DiagnosticsTests(unittest.TestCase):
    def test_detection(self):
        d=detection(np.array([4,4,1,1]),np.array([4,1,4,1]),3)
        self.assertEqual([d[k] for k in ('tp','fp','fn','tn')],[1,1,1,1])
        self.assertEqual(d['recall'],.5)
        self.assertIsNone(detection(np.zeros(4),np.zeros(4),5)['recall'])
    def test_target_offset_and_exposure(self):
        times=np.arange(np.datetime64('2019-01-01T00'),np.datetime64('2019-01-01T12'),np.timedelta64(1,'h'))
        r=split_timeline(times,dict(train=[0],val=[4],test=[8]),2)
        d={x['target_index']:x for x in r}
        self.assertEqual(d[2]['target_split'],'train')
        self.assertEqual(d[6]['target_split'],'val')
        self.assertEqual(d[10]['target_split'],'test')
        self.assertTrue(d[2]['input_times_overlap_training_input'])
        self.assertFalse(d[10]['input_times_overlap_training_input'])
    def test_event_metrics_and_plots(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for event_id in ['a','b']:
                folder=root/'models/fno_s42/events'/event_id
                (folder/'frames').mkdir(parents=True)
                mask=np.array([[True,True],[True,False]])
                np.save(folder/'mask.npy',mask)
                ids=[0,1,2]
                for i in ids:
                    true=np.array([[1.,2.+i],[0.5,999.]],dtype='float32')
                    pred=true+.5;pred[~mask]=-999
                    np.savez_compressed(folder/'frames'/f'{i}.npz',pred_hs=pred,true_hs=true,time=f'2021-09-15T0{i}:00:00.000000000')
                (folder/'arrays_complete.json').write_text(json.dumps(dict(indices=ids)))
                event=dict(id=event_id,name='SYNTHETIC')
                s=analyze_event(folder,event,'fno',42,make_plots=True)
                self.assertAlmostEqual(s['mae'],.5)
                self.assertAlmostEqual(s['bias_at_true_peak'],.5)
                bins=pd.read_csv(folder/'wave_bins.csv')
                self.assertAlmostEqual(bins.contribution_to_event_mae.sum(),.5)
                self.assertAlmostEqual(bins['count'].sum(),9)
                self.assertTrue((folder/'time_series.png').exists())
            aggregate(root)
            stats=pd.read_csv(root/'summary_event_metrics.csv')
            self.assertTrue((stats.n_seeds==1).all())
            self.assertTrue(stats.mae_std.isna().all())
            common=pd.read_csv(root/'common_bin_standardized_mae.csv')
            self.assertTrue(np.allclose(common.standardized_mae,.5))
    def test_time_gap_rejected(self):
        times=np.array(['2019-01-01T00','2019-01-01T01','2019-01-01T03'],dtype='datetime64[h]')
        with self.assertRaises(ValueError):split_timeline(times,dict(train=[0],val=[],test=[]),2)

if __name__=='__main__':unittest.main()
