from ror.Relation import PREFERENCE
import unittest
from ror.Dataset import read_dataset_from_txt
from ror.PreferenceRelations import PreferenceRelation, create_lambda_constraints


class TestLambdaConstraintRelations(unittest.TestCase):
    def test_creating_lambda_constraints_with_no_preferences(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")
        lambda_constraints = create_lambda_constraints(data, [])
        self.assertEqual(len(lambda_constraints), 0)

    def test_creating_lambda_constraints(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        preference_constraint = PreferenceRelation('b01', 'b02', PREFERENCE, 0.0).to_constraint(data)
        lambda_constraints = create_lambda_constraints(data, [preference_constraint])

        self.assertEqual(len(lambda_constraints), 14)

        # lower 1
        self.assertEqual(lambda_constraints[0].name, f'lower_lambda1_MaxSpeed_b01')
        # 2*lambda + 1*u_i
        self.assertEqual(len(lambda_constraints[0].variables), len(data.criteria) + 1)

        # lower 2
        self.assertEqual(lambda_constraints[1].name, f'lower_lambda2_MaxSpeed_b01')
        # 2*lambda + 1*u_i + M*c_i
        self.assertEqual(len(lambda_constraints[1].variables), len(data.criteria) + 1 + 1)

        # upper
        self.assertEqual(lambda_constraints[2].name, f'upper_lambda_MaxSpeed_b01')
        # 2*lambda + 1*u_i + M*c_i
        self.assertEqual(len(lambda_constraints[2].variables), len(data.criteria) + 1 + 1)

        # indices 3-5 are for alternative b01, second criterion

        # c_i
        self.assertEqual(lambda_constraints[6].name, f'c_sum_constraint_b01')
        # len(criteria)
        self.assertEqual(len(lambda_constraints[6].variables), len(data.criteria))
