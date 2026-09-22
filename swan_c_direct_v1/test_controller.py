"""CPU tests for family selection, seed preservation, and scheduling."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import run_c_direct as d


def job(model, tag, seed=42):
    return dict(model=model, config_id=f'{model}_{tag}_w64d4m24_s{seed}', seed=seed,
                epochs=30, train_fraction=1., hyperparams=dict(fno_width=64, fno_depth=4,
                modes_x=24, modes_y=24, batch_size=1, acc_steps=4, use_checkpoint=True))

def summary(j, value):
    return dict(model=j['model'], seed=j['seed'], config_id=j['config_id'], repair_job=j,
                repair_audit=dict(best_val_hs_mae=value, updates=76950), best_weight='/test/checkpoint')

class Tests(unittest.TestCase):
    def test_wait_for_all_family_candidates(self):
        j=job('tno','A'); base=job('tno','pilot')
        self.assertIsNone(d.ready_candidates('tno', {}, {('tno',42):summary(base,.2)}, [j], {}))
    def test_seed_preserves_training_settings(self):
        j=job('tno','A'); q=d.c.make_job(j,'C',seed=43)
        self.assertEqual(j['hyperparams'],q['hyperparams'])
        self.assertEqual(q['seed'],43)
    def test_scheduler_resumes_A_and_starts_other_C(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';old=Path(tmp)/'old';root.mkdir()
            plan=[job(m,'A') for m in d.c.MODELS]
            pilots={(m,s):summary(job(m,'pilot',s),.2) for m in d.c.MODELS for s in (42,43,44)}
            bases={m:job(m,'pilot') for m in d.c.MODELS}
            saved={j['config_id']:summary(j,.1) for j in plan if j['model']!='tno'}
            launched=[]
            class Proc:
                pid=99999
                def __init__(self,cmd,**kwargs):
                    self.j=d.c.read(Path(cmd[cmd.index('--job')+1]));launched.append(self.j)
                def poll(self):
                    saved[self.j['config_id']]=summary(self.j,.1);return 0
                def terminate(self):pass
                def wait(self,timeout=None):return 0
            r=SimpleNamespace(checked_summary=lambda source,j:saved.get(j['config_id']),
                compare_protocols=lambda rows:None, gpu_inventory=lambda:{g:dict(busy=False,mib=200000) for g in range(8)},
                parameter_bytes_floor=lambda j:1, run_name=lambda j:j['config_id'])
            with patch.object(d.subprocess,'Popen',Proc),patch.object(d.time,'sleep'),patch.object(d.signal,'signal'):
                d.run_schedule(None,r,root,old,Path(tmp),plan,bases,pilots,list(range(8)),{'asset_signature':{}})
            self.assertEqual(len(launched),7)
            self.assertEqual(launched[0]['config_id'],job('tno','A')['config_id'])
            self.assertTrue(all('_C_' in j['config_id'] for j in launched[1:]))
            self.assertEqual(len(d.c.read(root/'selected_9.json')),9)
            self.assertTrue((root/'completed.json').exists())
    def test_freeze_rejects_changed_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'selection.json';d.c.freeze(p,{'a':1})
            with self.assertRaises(ValueError):d.c.freeze(p,{'a':2})
if __name__=='__main__':unittest.main()
