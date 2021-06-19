import unittest
from ror.build_model import create_monotonicity_constraints
import numpy as np


class TestMain(unittest.TestCase):
    def test_creating_monotonicity_constraints_failed(self):
        with self.assertRaises(AssertionError):
            create_monotonicity_constraints([], [])

    def test_creating_monotonicity_constraints_success(self):
        criteria = [
            ("gain criterion", "g"),
            ("cost criterion", "c")
        ]
        data = np.array(
            [
                [1, 0],
                [5, 2],
                [3, 1],
                [6, -1]
            ]
        )

        constraints = create_monotonicity_constraints(data, criteria)

        self.assertEqual(len(constraints), 2)
        self.assertTrue("gain criterion" in constraints)
        self.assertTrue("cost criterion" in constraints)

        self.assertEqual(len(constraints['gain criterion']), 3)
        # a_1 <= a_3
        self.assertEqual(constraints['gain criterion'][0].name,
                         "mono_u_gain criterion_1_u_gain criterion_3")

        # a_0 <= a_3
        self.assertEqual(len(constraints['cost criterion']), 3)
        self.assertEqual(constraints['cost criterion'][0].name,
                         "mono_u_cost criterion_0_u_cost criterion_3")
