"""CPU checks for held-out year boundaries, transforms and metric conventions."""
import unittest
import numpy as np
import pandas as pd
from evaluate_2021 import year_indices,apply_direction,aligned_boundaries,frame_metrics

class Tests(unittest.TestCase):
    def test_endpoint_and_context(self):
        times=np.arange(np.datetime64('2021-01-01T00','h'),np.datetime64('2022-01-01T01','h'))
        ids,targets=year_indices(times,12)
        self.assertEqual(len(ids),8760);self.assertEqual(len(targets),8748)
        self.assertEqual(str(times[targets[0]]),'2021-01-01T12')
        self.assertEqual(str(times[targets[-1]]),'2021-12-31T23')
    def test_gap_rejected(self):
        times=np.arange(np.datetime64('2021-01-01T00','h'),np.datetime64('2022-01-01T01','h'))
        with self.assertRaises(ValueError):year_indices(np.delete(times,14),12)
    def test_saved_transform(self):
        theta=np.array([0.,30.,179.,270.]);x=np.zeros((4,4,1,1),dtype=np.float32)
        x[:,2,0,0]=np.sin(np.deg2rad(theta));x[:,3,0,0]=np.cos(np.deg2rad(theta))
        for name,expected in [('rot+90',theta+90),('refl+270',270-theta)]:
            y=apply_direction(x.copy(),name)
            np.testing.assert_allclose(y[:,2,0,0],np.sin(np.deg2rad(expected)),atol=1e-6)
            np.testing.assert_allclose(y[:,3,0,0],np.cos(np.deg2rad(expected)),atol=1e-6)
    def test_boundary_interpolation_and_endpoint(self):
        index=pd.date_range('2021-01-01',periods=3,freq='3h')
        df=pd.DataFrame({'hs':[0.,3.,6.],'tm':[2.,2.,2.],'sin':[0.,0.,0.],'cos':[1.,1.,1.]},index=index)
        wanted=pd.date_range(index[0],periods=6,freq='h')
        aligned,audit=aligned_boundaries({'A':df},['A'],wanted,6)
        np.testing.assert_allclose(aligned['A'].hs,np.arange(6))
        self.assertEqual(audit['A']['interpolated_hours'],4)
        with self.assertRaises(ValueError):aligned_boundaries({'A':df},['A'],pd.date_range(index[0],periods=8,freq='h'),6)
        with self.assertRaises(ValueError):aligned_boundaries({},['A'],wanted,6)
    def test_physical_and_circular_metrics(self):
        pred=np.zeros((4,1,1));truth=np.zeros_like(pred)
        pred[0]=.2;truth[0]=.1
        pred[2]=np.sin(np.deg2rad(359));pred[3]=np.cos(np.deg2rad(359))
        truth[2]=np.sin(np.deg2rad(1));truth[3]=np.cos(np.deg2rad(1))
        m=frame_metrics(pred,truth,np.ones((1,1),bool),{'hs':[0,10],'tm':[0,20]})
        self.assertAlmostEqual(m['hs_rmse'],1.)
        self.assertAlmostEqual(m['dir_mae'],2.)

if __name__=='__main__':unittest.main()
