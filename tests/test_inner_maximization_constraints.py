from ror.Relation import PREFERENCE, Relation
from ror.dataset_constants import DEFAULT_M
from ror.inner_maximization_constraints import create_inner_maximization_constraints
import unittest
from ror.Dataset import RORDataset
from ror.PreferenceRelations import PreferenceRelation
import numpy as np


class TestInnerMaximization(unittest.TestCase):
    def test_creating_inner_maximization_constraints_for_none_dataset(self):
        with self.assertRaises(AssertionError):
            create_inner_maximization_constraints(None)

    def test_creating_inner_maximization_constraints_for_small_dataset_with_no_relations(self):
        data = RORDataset(
            ['a1', 'a2'],
            np.array([
                [10, 11],
                [9, 12]
            ]),
            [("c1", "g"), ("c2", "c")]
        )
        inner_max_constraints = create_inner_maximization_constraints(data)

        # no relations, no inner maximization constraints in the first step
        self.assertEqual(len(inner_max_constraints), 0)

    def test_creating_inner_maximization_constraints_for_small_dataset(self):
        
        data = RORDataset(
            ['a1', 'a2'],
            np.array([
                [10, 11],
                [9, 12]
            ]),
            [("c1", "g"), ("c2", "c")],
            [
                PreferenceRelation('a1', 'a2', PREFERENCE)
            ]
        )
        inner_max_constraints = create_inner_maximization_constraints(data)

        constraints_count = (3*len(data.criteria) + 1) * len(data.alternatives)
        self.assertEqual(len(inner_max_constraints), constraints_count)

        first_alternative_constraints_length = len(data.criteria) * 3 + 1
        first_alternative_constraints = inner_max_constraints[:
                                                              first_alternative_constraints_length]

        # check first alternative, first criterion
        self.assertTrue(all([constraint.get_variable(
            "u_{c1}(a1)") is not None for constraint in first_alternative_constraints[:3]]))
        self.assertTrue(all([constraint.get_variable(
            "lambda_{all}(a1)") is not None for constraint in first_alternative_constraints[:3]]))
        self.assertTrue(all([constraint.get_variable(
            "c_{c1}(a1)") is not None for constraint in first_alternative_constraints[1:3]]))

        # check individual coefficients for the first alternative on first criterion
        self.assertSetEqual(set(first_alternative_constraints[0].variables_names), set(
            ['lambda_{all}(a1)', 'u_{c1}(a1)']))
        self.assertAlmostEqual(
            first_alternative_constraints[0].get_variable('u_{c1}(a1)').coefficient, -1.0)
        self.assertAlmostEqual(first_alternative_constraints[0].get_variable(
            'lambda_{all}(a1)').coefficient, -1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[0].free_variable.coefficient, -1.0)
        self.assertEqual(
            first_alternative_constraints[0].relation, Relation("<="))

        self.assertSetEqual(set(first_alternative_constraints[1].variables_names), set(
            ['lambda_{all}(a1)', 'u_{c1}(a1)', 'c_{c1}(a1)']))
        self.assertAlmostEqual(
            first_alternative_constraints[1].get_variable('u_{c1}(a1)').coefficient, -1.0)
        self.assertAlmostEqual(first_alternative_constraints[1].get_variable(
            'c_{c1}(a1)').coefficient, -DEFAULT_M)
        self.assertAlmostEqual(first_alternative_constraints[1].get_variable(
            'lambda_{all}(a1)').coefficient, -1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[1].free_variable.coefficient, -1.0)
        self.assertEqual(
            first_alternative_constraints[1].relation, Relation("<="))

        self.assertSetEqual(set(first_alternative_constraints[2].variables_names), set(
            ['lambda_{all}(a1)', 'u_{c1}(a1)', 'c_{c1}(a1)']))
        self.assertAlmostEqual(
            first_alternative_constraints[2].get_variable('u_{c1}(a1)').coefficient, 1.0)
        self.assertAlmostEqual(first_alternative_constraints[2].get_variable(
            'c_{c1}(a1)').coefficient, -DEFAULT_M)
        self.assertAlmostEqual(first_alternative_constraints[2].get_variable(
            'lambda_{all}(a1)').coefficient, 1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[2].free_variable.coefficient, 1.0)
        self.assertEqual(
            first_alternative_constraints[2].relation, Relation("<="))

        # check first alternative, second criterion
        self.assertTrue(all([constraint.get_variable(
            "u_{c2}(a1)") is not None for constraint in first_alternative_constraints[3:6]]))
        self.assertTrue(all([constraint.get_variable(
            "lambda_{all}(a1)") is not None for constraint in first_alternative_constraints[3:6]]))
        self.assertTrue(all([constraint.get_variable(
            "c_{c2}(a1)") is not None for constraint in first_alternative_constraints[4:6]]))

        # check individual coefficients for the first alternative on second criterion
        self.assertSetEqual(set(first_alternative_constraints[3].variables_names), set(
            ['lambda_{all}(a1)', 'u_{c2}(a1)']))
        self.assertAlmostEqual(
            first_alternative_constraints[3].get_variable('u_{c2}(a1)').coefficient, -1.0)
        self.assertAlmostEqual(first_alternative_constraints[3].get_variable(
            'lambda_{all}(a1)').coefficient, -1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[3].free_variable.coefficient, -1.0)
        self.assertEqual(
            first_alternative_constraints[3].relation, Relation("<="))

        self.assertSetEqual(set(first_alternative_constraints[4].variables_names), set(
            ['lambda_{all}(a1)', 'u_{c2}(a1)', 'c_{c2}(a1)']))
        self.assertAlmostEqual(
            first_alternative_constraints[4].get_variable('u_{c2}(a1)').coefficient, -1.0)
        self.assertAlmostEqual(first_alternative_constraints[4].get_variable(
            'c_{c2}(a1)').coefficient, -DEFAULT_M)
        self.assertAlmostEqual(first_alternative_constraints[4].get_variable(
            'lambda_{all}(a1)').coefficient, -1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[4].free_variable.coefficient, -1.0)
        self.assertEqual(
            first_alternative_constraints[4].relation, Relation("<="))

        self.assertSetEqual(set(first_alternative_constraints[5].variables_names), set(
            ['lambda_{all}(a1)', 'u_{c2}(a1)', 'c_{c2}(a1)']))
        self.assertAlmostEqual(
            first_alternative_constraints[5].get_variable('u_{c2}(a1)').coefficient, 1.0)
        self.assertAlmostEqual(first_alternative_constraints[5].get_variable(
            'c_{c2}(a1)').coefficient, -DEFAULT_M)
        self.assertAlmostEqual(first_alternative_constraints[5].get_variable(
            'lambda_{all}(a1)').coefficient, 1.0)
        self.assertAlmostEqual(
            first_alternative_constraints[5].free_variable.coefficient, 1.0)
        self.assertEqual(
            first_alternative_constraints[5].relation, Relation("<="))

        # only c_{c1}(a1), c_{c2}(a1) variables in last, c function sum, constraint
        self.assertEqual(
            first_alternative_constraints[6].name, "sum_binary_var_c_(a1)")
        self.assertSetEqual(set(first_alternative_constraints[6].variables_names), set(
            ["c_{c1}(a1)", "c_{c2}(a1)"]))
        # check free variable coeff in c function sum constraint
        self.assertEqual(
            first_alternative_constraints[6].free_variable.coefficient, len(data.criteria)-1)
