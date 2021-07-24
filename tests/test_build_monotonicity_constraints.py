from ror.Dataset import read_dataset_from_txt
import unittest
from ror.monotonicity_constraints import create_monotonicity_constraints


class TestMonotonicityConstraints(unittest.TestCase):
    def test_creating_monotonicity_constraints_failed(self):
        with self.assertRaises(AssertionError):
            create_monotonicity_constraints(None)

    def test_creating_monotonicity_constraints_success(self):
        data = read_dataset_from_txt("tests/datasets/example2.txt")

        constraints = create_monotonicity_constraints(data)

        self.assertEqual(len(constraints), 2)
        self.assertTrue("gain criterion" in constraints)
        self.assertTrue("cost criterion" in constraints)

        self.assertEqual(len(constraints['gain criterion']), 3)
        self.assertEqual(constraints['gain criterion'][0].name,
                         "mono_u_gain criterion_1_u_gain criterion_3")

        self.assertEqual(len(constraints['cost criterion']), 3)
        self.assertEqual(constraints['cost criterion'][0].name,
                         "mono_u_cost criterion_0_u_cost criterion_3")
