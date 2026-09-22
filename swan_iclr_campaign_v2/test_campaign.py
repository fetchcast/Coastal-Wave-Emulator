"""CPU-only tests for planning, selection and scheduler failure isolation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import campaign as c


def base(model):
    return dict(model=model,config_id=model+'_pilot',seed=42,stage='pilot',train_fraction=1.,epochs=30,
                hyperparams=dict(fno_width=64,fno_depth=4,modes_x=24,modes_y=24,max_lr=1e-4,
                                 batch_size=4,acc_steps=1))

class Tests(unittest.TestCase):
    def test_plan_and_copy(self):
        bases={m:base(m) for m in c.MODELS};before=copy.deepcopy(bases)
        jobs=c.architecture_jobs(bases)
        self.assertEqual(len(jobs),29)
        self.assertEqual(len({j['config_id'] for j in jobs}),29)
        self.assertEqual(bases,before)
        self.assertFalse(any(j['model']=='tno' and j['hyperparams']['fno_width']==256 and j['hyperparams']['fno_depth']>4 for j in jobs))
        for j in jobs:
            self.assertEqual(j['hyperparams']['batch_size']*j['hyperparams']['acc_steps'],4)
        self.assertTrue(any(j['model']=='fno' and j['hyperparams']['fno_width']==256 and j['hyperparams']['fno_depth']==8 for j in jobs))

    def test_selection_does_not_use_test(self):
        rows=[dict(config_id='a',repair_audit=dict(best_val_hs_mae=.1),legacy_metrics=dict(rmse_m=99)),
              dict(config_id='b',repair_audit=dict(best_val_hs_mae=.2),legacy_metrics=dict(rmse_m=0))]
        self.assertEqual(c.choose(rows,1)[0]['config_id'],'a')

    def test_followup_preserves_settings(self):
        j=base('tno');j['hyperparams']['use_checkpoint']=False
        k=c.make_job(j,'E',fraction=.25,seed=43)
        self.assertEqual(k['hyperparams'],j['hyperparams'])
        self.assertEqual(k['train_fraction'],.25)
        self.assertEqual(k['seed'],43)

    def test_freeze(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'plan.json';c.freeze(p,[1]);c.freeze(p,[1])
            with self.assertRaises(ValueError):c.freeze(p,[2])

    def test_scheduler_oom_continues_and_resume(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); successes={}; calls=[]
            jobs=[dict(config_id=x) for x in ('oom','good')]
            def popen(cmd,**kw):
                j=c.read(cmd[cmd.index('--job')+1]);name=j['config_id'];calls.append(name)
                if name=='oom':
                    p=root/'A'/name/'attempt_1.log';p.parent.mkdir(parents=True);p.write_text('CUDA out of memory')
                else:
                    successes[name]=dict(config_id=name)
                return SimpleNamespace(poll=lambda:1 if name=='oom' else 0)
            r=SimpleNamespace(checked_summary=lambda root,j:successes.get(j['config_id']),
                compare_protocols=lambda s:None,gpu_inventory=lambda:{0:dict(busy=False,mib=180000)},
                parameter_bytes_floor=lambda j:0,run_name=lambda j:j['config_id'])
            a=SimpleNamespace(root=root,package=root,gpus=[0])
            with patch.object(c.subprocess,'Popen',popen),patch.object(c.time,'sleep',lambda _:None):
                rows=c.run_stage(a,r,'A',jobs,{})
                self.assertEqual([x['config_id'] for x in rows],['good'])
                c.run_stage(a,r,'A',jobs,{})
                self.assertEqual(calls,['oom','good'])

if __name__=='__main__':
    unittest.main()
