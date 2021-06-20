from ror.min_max_value_constraints import create_max_value_constraint, create_min_value_constraints
from ror.Dataset import read_dataset_from_txt
import unittest


class TestMinMaxValueConstraint(unittest.TestCase):
    def test_creating_min_value_constraints(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        min_constraints = create_min_value_constraints(data)

        self.assertEqual(len(min_constraints), len(data.criteria))

        # min value for criterion no. 1
        self.assertEqual(len(min_constraints[0].get_constraints_variables), 1)
        self.assertAlmostEqual(
            min_constraints[0].get_variable('b05').coefficient, 1.0)
        self.assertAlmostEqual(
            min_constraints[0].free_variable.coefficient, 0.0)
        self.assertEqual(min_constraints[0].relation.sign, '==')

        # min value for criterion no. 2
        self.assertEqual(len(min_constraints[1].get_constraints_variables), 1)
        self.assertIsNotNone(min_constraints[1].get_variable('b02'))
        self.assertAlmostEqual(
            min_constraints[1].get_variable('b02').coefficient, 1.0)
        self.assertAlmostEqual(
            min_constraints[1].free_variable.coefficient, 0.0)
        self.assertEqual(min_constraints[1].relation.sign, '==')

    def test_creating_max_value_constraints(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        max_constraint = create_max_value_constraint(data)

        self.assertEqual(len(max_constraint.get_constraints_variables), 2)
        self.assertIsNotNone(max_constraint.get_variable('b02'))
        self.assertAlmostEqual(
            max_constraint.get_variable('b02').coefficient, 1.0)
        self.assertIsNotNone(max_constraint.get_variable('b03'))
        self.assertAlmostEqual(
            max_constraint.get_variable('b03').coefficient, 1.0)
        self.assertAlmostEqual(max_constraint.free_variable.coefficient, 1.0)
        self.assertEqual(max_constraint.relation.sign, '==')
