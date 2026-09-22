import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from plans import jobs,BASELINES
from evaluate_2021 import model_arguments
from run import descendants,aggregate
from report_events import seed_statistics,peak_metrics
class Tests(unittest.TestCase):
    def base(self):return dict(code_hashes={'trainer':'hash'},asset_signature='frozen',bnd_dir_transform='train_auto',hyperparams={},max_updates=2)
    def test_plan_provenance_and_seeds(self):
        base=self.base();plan=jobs(base)
        self.assertEqual(len(plan),21);self.assertEqual(len({j['config_id'] for j in plan}),21)
        self.assertTrue(all(j['seed']==42 for j in plan[:7]))
        for j in plan:
            self.assertEqual(j['asset_signature'],'frozen');self.assertNotIn('max_updates',j)
            self.assertEqual(j['hyperparams']['batch_size']*j['hyperparams']['acc_steps'],4)
        self.assertEqual(base,self.base())
    def test_constructor_preserves_family_settings(self):
        def ctor(hidden_dim=128,feat=None,extra=None,embed_dim=96):pass
        h=dict(hidden_dim=768,unet_feat=[64,128],vit_depth=6,convnext_dims=[96,192,384])
        a=model_arguments(h,ctor)
        self.assertEqual(a['feat'],[64,128]);self.assertEqual(a['hidden_dim'],768)
        self.assertEqual(a['extra']['vit_depth'],6);self.assertEqual(a['extra']['convnext_dims'],[96,192,384])
    def test_new_family_summary(self):
        rows=[dict(scope='annual',model='conv_swin',seed=s,hs_mae=x) for s,x in zip((42,43,44),(.1,.2,.3))]
        result=seed_statistics(rows)
        self.assertEqual(len(result),1);self.assertAlmostEqual(result[0]['hs_mae_mean'],.2)
        self.assertAlmostEqual(result[0]['hs_mae_seed_sd'],.1)
    def test_missing_seed_rejected(self):
        with self.assertRaises(ValueError):seed_statistics([dict(scope='annual',model='vit',seed=42,hs_mae=.1)])
    def test_process_tree_scope(self):
        info={1:dict(ppid=0),2:dict(ppid=1),3:dict(ppid=2),4:dict(ppid=0)}
        self.assertEqual(descendants(info,1),{1,2,3})
    def test_aggregate_ignores_incomplete_family(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for m in ('fno','ffno'):
                p=root/'evaluation'/m;p.mkdir(parents=True)
                (p/'event_metrics_by_seed.csv').write_text('model,seed,hs_mae\n'+m+',42,0.1\n')
            (root/'evaluation/fno/completed.json').write_text('{}')
            aggregate(root)
            output=(root/'available_event_metrics_by_seed.csv').read_text()
            self.assertIn('fno,42',output);self.assertNotIn('ffno',output)
if __name__=='__main__':unittest.main()
