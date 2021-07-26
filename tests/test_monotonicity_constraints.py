from ror.data_loader import read_dataset_from_txt
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

        # gain criterion
        gain_criterion_constraints = constraints['gain criterion']

        self.assertEqual(len(gain_criterion_constraints), 3)

        # first constraint for gain criterion
        self.assertEqual(
            gain_criterion_constraints[0].name, "mono_u_gain criterion_a4_u_gain criterion_a2")
        best_value_first_constraint_first_criterion = gain_criterion_constraints[0].get_variable(
            'u_gain criterion_a4')
        self.assertIsNotNone(best_value_first_constraint_first_criterion)
        self.assertAlmostEqual(
            best_value_first_constraint_first_criterion.coefficient, -1.0)

        worst_value_first_constraint_first_criterion = gain_criterion_constraints[0].get_variable(
            'u_gain criterion_a2')
        self.assertIsNotNone(worst_value_first_constraint_first_criterion)
        self.assertAlmostEqual(
            worst_value_first_constraint_first_criterion.coefficient, 1.0)

        # last constraint for gain criterion
        self.assertEqual(
            gain_criterion_constraints[2].name, "mono_u_gain criterion_a4_u_gain criterion_a1")
        best_value_third_constraint_first_criterion = gain_criterion_constraints[2].get_variable(
            'u_gain criterion_a4')
        self.assertIsNotNone(best_value_third_constraint_first_criterion)
        self.assertAlmostEqual(
            best_value_third_constraint_first_criterion.coefficient, -1.0)

        worst_value_third_constraint_first_criterion = gain_criterion_constraints[2].get_variable(
            'u_gain criterion_a1')
        self.assertIsNotNone(worst_value_third_constraint_first_criterion)
        self.assertAlmostEqual(
            worst_value_third_constraint_first_criterion.coefficient, 1.0)

        # cost criterion
        cost_criterion_constraints = constraints['cost criterion']

        self.assertEqual(len(cost_criterion_constraints), 3)

        # first constraint for cost criterion
        self.assertEqual(
            cost_criterion_constraints[0].name, "mono_u_cost criterion_a4_u_cost criterion_a1")
        best_value_first_constraint_second_criterion = cost_criterion_constraints[0].get_variable(
            'u_cost criterion_a4')
        self.assertIsNotNone(best_value_first_constraint_second_criterion)
        self.assertAlmostEqual(
            best_value_first_constraint_second_criterion.coefficient, -1.0)

        worst_value_first_constraint_first_criterion = cost_criterion_constraints[0].get_variable(
            'u_cost criterion_a1')
        self.assertIsNotNone(worst_value_first_constraint_first_criterion)
        self.assertAlmostEqual(
            worst_value_first_constraint_first_criterion.coefficient, 1.0)

        # first constraint for cost criterion
        self.assertEqual(
            cost_criterion_constraints[2].name, "mono_u_cost criterion_a4_u_cost criterion_a2")
        best_value_third_constraint_second_criterion = cost_criterion_constraints[2].get_variable(
            'u_cost criterion_a4')
        self.assertIsNotNone(best_value_third_constraint_second_criterion)
        self.assertAlmostEqual(
            best_value_third_constraint_second_criterion.coefficient, -1.0)

        worst_value_third_constraint_first_criterion = cost_criterion_constraints[2].get_variable(
            'u_cost criterion_a2')
        self.assertIsNotNone(worst_value_third_constraint_first_criterion)
        self.assertAlmostEqual(
            worst_value_third_constraint_first_criterion.coefficient, 1.0)
