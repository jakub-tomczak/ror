from ror.Relation import Relation
from ror.inner_maximization_constraints import create_inner_maximization_constraints
import unittest
from ror.Dataset import Dataset
import numpy as np


class TestInnerMaximization(unittest.TestCase):
    def test_creating_inner_maximization_constraints_for_none_dataset(self):
        with self.assertRaises(AssertionError):
            create_inner_maximization_constraints(None)

    def test_creating_inner_maximization_constraints_for_small_dataset(self):
        data = Dataset(
            ['a1', 'a2'],
            np.array([
                [10, 11],
                [9, 12]
            ]),
            [("c1", "g"), ("c2", "c")]
        )
        inner_max_constraints = create_inner_maximization_constraints(data)

        self.assertEqual(len(inner_max_constraints), len(
            data.alternatives) * len(data.criteria) * 3 + len(data.alternatives))

        first_alternative_constraints_length = len(data.criteria) * 3 + 1
        first_alternative_constraints = inner_max_constraints[:
                                                              first_alternative_constraints_length]

        # check first alternative, first criterion
        self.assertTrue(all([constraint.get_variable(
            "u_c1_a1") is not None for constraint in first_alternative_constraints[:3]]))
        self.assertTrue(all([constraint.get_variable(
            "lambda_all_a1") is not None for constraint in first_alternative_constraints[:3]]))
        self.assertTrue(all([constraint.get_variable(
            "c_c1_a1") is not None for constraint in first_alternative_constraints[1:3]]))

        # check individual coefficients for the first alternative on first criterion
        self.assertAlmostEqual(set(first_alternative_constraints[0].variables_names), set(
            ['lambda_all_a1', 'u_c1_a1']))
        self.assertAlmostEqual(
            first_alternative_constraints[0].get_variable('u_c1_a1').coefficient, -1.0)
        self.assertAlmostEqual(first_alternative_constraints[0].get_variable(
            'lambda_all_a1').coefficient, -1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[0].free_variable.coefficient, -1.0)
        self.assertEqual(
            first_alternative_constraints[0].relation, Relation("<="))

        self.assertAlmostEqual(set(first_alternative_constraints[1].variables_names), set(
            ['lambda_all_a1', 'u_c1_a1', 'c_c1_a1']))
        self.assertAlmostEqual(
            first_alternative_constraints[1].get_variable('u_c1_a1').coefficient, -1.0)
        self.assertAlmostEqual(first_alternative_constraints[1].get_variable(
            'c_c1_a1').coefficient, -Dataset.DEFAULT_M)
        self.assertAlmostEqual(first_alternative_constraints[1].get_variable(
            'lambda_all_a1').coefficient, -1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[1].free_variable.coefficient, -1.0)
        self.assertEqual(
            first_alternative_constraints[1].relation, Relation("<="))

        self.assertAlmostEqual(set(first_alternative_constraints[2].variables_names), set(
            ['lambda_all_a1', 'u_c1_a1', 'c_c1_a1']))
        self.assertAlmostEqual(
            first_alternative_constraints[2].get_variable('u_c1_a1').coefficient, 1.0)
        self.assertAlmostEqual(first_alternative_constraints[2].get_variable(
            'c_c1_a1').coefficient, -Dataset.DEFAULT_M)
        self.assertAlmostEqual(first_alternative_constraints[2].get_variable(
            'lambda_all_a1').coefficient, 1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[2].free_variable.coefficient, 1.0)
        self.assertEqual(
            first_alternative_constraints[2].relation, Relation("<="))

        # check first alternative, second criterion
        self.assertTrue(all([constraint.get_variable(
            "u_c2_a1") is not None for constraint in first_alternative_constraints[3:6]]))
        self.assertTrue(all([constraint.get_variable(
            "lambda_all_a1") is not None for constraint in first_alternative_constraints[3:6]]))
        self.assertTrue(all([constraint.get_variable(
            "c_c2_a1") is not None for constraint in first_alternative_constraints[4:6]]))

        # check individual coefficients for the first alternative on second criterion
        self.assertAlmostEqual(set(first_alternative_constraints[3].variables_names), set(
            ['lambda_all_a1', 'u_c2_a1']))
        self.assertAlmostEqual(
            first_alternative_constraints[3].get_variable('u_c2_a1').coefficient, -1.0)
        self.assertAlmostEqual(first_alternative_constraints[3].get_variable(
            'lambda_all_a1').coefficient, -1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[3].free_variable.coefficient, -1.0)
        self.assertEqual(
            first_alternative_constraints[3].relation, Relation("<="))

        self.assertAlmostEqual(set(first_alternative_constraints[4].variables_names), set(
            ['lambda_all_a1', 'u_c2_a1', 'c_c2_a1']))
        self.assertAlmostEqual(
            first_alternative_constraints[4].get_variable('u_c2_a1').coefficient, -1.0)
        self.assertAlmostEqual(first_alternative_constraints[4].get_variable(
            'c_c2_a1').coefficient, -Dataset.DEFAULT_M)
        self.assertAlmostEqual(first_alternative_constraints[4].get_variable(
            'lambda_all_a1').coefficient, -1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[4].free_variable.coefficient, -1.0)
        self.assertEqual(
            first_alternative_constraints[4].relation, Relation("<="))

        self.assertAlmostEqual(set(first_alternative_constraints[5].variables_names), set(
            ['lambda_all_a1', 'u_c2_a1', 'c_c2_a1']))
        self.assertAlmostEqual(
            first_alternative_constraints[5].get_variable('u_c2_a1').coefficient, 1.0)
        self.assertAlmostEqual(first_alternative_constraints[5].get_variable(
            'c_c2_a1').coefficient, -Dataset.DEFAULT_M)
        self.assertAlmostEqual(first_alternative_constraints[5].get_variable(
            'lambda_all_a1').coefficient, 1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[5].free_variable.coefficient, 1.0)
        self.assertEqual(
            first_alternative_constraints[5].relation, Relation("<="))

        # only c_c1_a1, c_c2_a1 variables in last, c function sum, constraint
        self.assertEqual(
            first_alternative_constraints[6].name, "sum_binary_var_c_(a1)")
        self.assertSetEqual(set(first_alternative_constraints[6].variables_names), set(
            ["c_c1_a1", "c_c2_a1"]))
        # check free variable coeff in c function sum constraint
        self.assertEqual(
            first_alternative_constraints[6].free_variable.coefficient, len(data.criteria)-1)
