import unittest
import numpy as np
from analyze import exact_p,holm,moving_block_ci,seed_tests
class Tests(unittest.TestCase):
    def test_three_seed_resolution(self):
        self.assertEqual(exact_p([1,2,3]),.25)
        self.assertEqual(exact_p([0,0,0]),1)
    def test_holm(self):
        np.testing.assert_allclose(holm([.03,.01,.2]),[.06,.03,.2])
    def test_block_constant(self):
        np.testing.assert_allclose(moving_block_ci(np.ones(100)*.2,24,100,np.random.default_rng(7)),[.2,.2])
    def test_missing_seed_rejected(self):
        with self.assertRaises(ValueError):seed_tests([dict(scope='annual',model='fno',seed=42)])
    def test_missing_tail_not_zero(self):
        rows=[dict(scope='annual',model=m,seed=s,hs_mae=.1+i*.01+s*.0001,hs_ge3_mae='',hs_ge5_mae='') for i,m in enumerate(('fno','ffno','tno')) for s in (42,43,44)]
        results=seed_tests(rows)
        self.assertEqual(len(results),3)
        self.assertTrue(all(r['metric']=='hs_mae' for r in results))
if __name__=='__main__':unittest.main()
