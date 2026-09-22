import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from common import perturb,indices,error_stats,conditions,freeze
import run

class Checks(unittest.TestCase):
    def setUp(self):
        self.x=np.zeros((2,10,2,2),dtype=np.float32);self.x[:,9,0,0]=1
        self.norm={'wind_u':[-10,30],'wind_v':[-20,20],'hs':[2,12]}
    def test_wind_in_physical_units(self):
        self.x[:,0]=.5;self.x[:,1]=.25
        z=perturb(self.x,'wind',.1,self.norm)
        np.testing.assert_allclose(z[:,0]*40-10,11)
        np.testing.assert_allclose(z[:,1]*40-20,-11)
        np.testing.assert_array_equal(z[:,2:],self.x[:,2:])
    def test_boundary_hs_preserves_absent_cells(self):
        z=perturb(self.x,'boundary_hs',.1,self.norm)
        np.testing.assert_allclose(z[:,6,0,0]*10+2,2.2)
        np.testing.assert_array_equal(z[:,6,1,1],0)
        np.testing.assert_array_equal(self.x[:,6],0)
    def test_direction_rotates_unit_vector(self):
        z=perturb(self.x,'boundary_direction',10,self.norm)
        np.testing.assert_allclose(z[:,8,0,0],np.sin(np.deg2rad(10)),rtol=1e-6)
        np.testing.assert_allclose(np.hypot(z[:,8],z[:,9]),np.hypot(self.x[:,8],self.x[:,9]),rtol=1e-6)
    def test_delay_only_boundary(self):
        delayed=np.ones_like(self.x[:,6:10]);z=perturb(self.x,'boundary_delay',3,self.norm,delayed)
        np.testing.assert_array_equal(z[:,:6],self.x[:,:6]);np.testing.assert_array_equal(z[:,6:],delayed)
    def test_delayed_context_gap_rejected(self):
        times=np.arange(np.datetime64('2021-01-01T00'),np.datetime64('2021-01-03T00'),np.timedelta64(1,'h'))
        e={'start':str(times[30]),'end':str(times[36])}
        self.assertEqual(indices(times,e,12,6).tolist(),[30,36])
        times[15]+=np.timedelta64(1,'h')
        with self.assertRaises(ValueError):indices(times,e,12,6)
    def test_tail_sufficient_statistics(self):
        r=error_stats(np.array([2,7,1]),np.array([1,5,999]),np.array([True,True,False]))
        self.assertEqual(r['mae'],1.5);self.assertEqual(r['high_count'],1);self.assertEqual(r['high_sae'],2)
    def test_freeze_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'plan.json';freeze(p,{'stride':6})
            with self.assertRaises(ValueError):freeze(p,{'stride':1})
    def test_all_conditions_have_one_clean_control(self):
        self.assertEqual(len(conditions()),16);self.assertEqual(conditions().count(('clean',0.)),1)
    def test_scheduler_avoids_busy_gpu_and_orders_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=SimpleNamespace(mode='run',output=root/'out',server_root=root,eval_root=root/'eval',diagnostics_root=root/'diag',
                events='2114',stride=6,lr_pilots=False,pilot_updates=7695,gpus='auto',max_workers=7,launch_hours=12)
            tasks=[dict(model=m,seed=42,entry={},signature={}) for m in ['fno','ffno']]
            started=[]
            class Proc:
                def __init__(self,cmd,**kwargs):started.append(cmd)
                def poll(self):return 0
                returncode=0
            inv={0:{'uuid':'TEST_BUSY','busy':True},1:{'uuid':'TEST_FREE','busy':False}}
            with patch.object(run,'discover',return_value=(tasks,[])),patch.object(run,'inventory',return_value=inv),patch.object(run.subprocess,'Popen',Proc),patch.object(run.subprocess,'run'),patch.object(run.time,'sleep'),patch.object(run.signal,'signal'):
                run.main(a)
            self.assertEqual(len(started),6)
            self.assertTrue(all(c[c.index('--gpu')+1]=='1' for c in started))
            self.assertEqual([c[c.index('--stage')+1] if '--stage' in c else 'diagnostics' for c in started],['diagnostics','diagnostics','resources','resources','robustness','robustness'])

if __name__=='__main__':unittest.main()
