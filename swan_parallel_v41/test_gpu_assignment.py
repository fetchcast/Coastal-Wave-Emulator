"""Check that the new lanes cannot assign the same GPU concurrently."""
import ast
from pathlib import Path
import unittest
import run

class AssignmentTests(unittest.TestCase):
    def test_evaluation_waits_for_original_exit(self):
        self.assertEqual(run.evaluation_gpus(set()),())
        self.assertEqual(run.evaluation_gpus({'baselines'}),())
        self.assertEqual(run.evaluation_gpus({'original'}),(0,1))
    def test_training_and_evaluation_are_disjoint(self):
        tree=ast.parse((Path(__file__).parent/'lane.py').read_text())
        groups=[ast.literal_eval(k.value) for n in ast.walk(tree) if isinstance(n,ast.Call)
                for k in n.keywords if k.arg=='gpus']
        self.assertIn([2,3,4,5,6,7],groups)
        self.assertFalse(set(groups[0])&set(run.evaluation_gpus({'original'})))
if __name__=='__main__':unittest.main()
