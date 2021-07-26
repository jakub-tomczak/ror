import unittest
from ror.Dataset import read_dataset_from_txt
from ror.PreferenceRelations import PreferenceRelation
from ror.Relation import INDIFFERENCE, PREFERENCE, Relation, WEAK_PREFERENCE


class TestPreferenceRelations(unittest.TestCase):
    def test_creating_preference_invalid_alpha(self):
        with self.assertRaises(AssertionError):
            PreferenceRelation('b01', 'b02', PREFERENCE, -1e-10)
        with self.assertRaises(AssertionError):
            PreferenceRelation('b01', 'b02', PREFERENCE, 1+1e-10)

    def test_creating_preference_invalid_relation(self):
        with self.assertRaises(AssertionError):
            PreferenceRelation('b01', 'b02', "<=", 0)

        with self.assertRaises(AssertionError):
            PreferenceRelation('b01', 'b02', Relation('<='), 0)

    def test_strong_preference_alpha_0(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        preference = PreferenceRelation('b01', 'b02', PREFERENCE, 0.0)
        preference_constraint = preference.to_constraint(data)

        self.assertEqual(preference_constraint._relation, PREFERENCE)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint._name,
                         'preference_all_b02_<=_b01')

        self.assertTrue(
            'lambda_all_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b01').coefficient, 1.0)
        self.assertTrue(
            'u_MaxSpeed_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b01').coefficient, 0.0)

        self.assertTrue(
            'lambda_all_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b02').coefficient, -1.0)
        self.assertTrue(
            'u_MaxSpeed_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b02').coefficient, 0.0)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient, data.eps)

    def test_strong_preference_alpha_1(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        alpha = 1.0
        preference = PreferenceRelation(
            'b01', 'b02', PREFERENCE, alpha)
        preference_constraint = preference.to_constraint(data)

        self.assertEqual(preference_constraint._relation, PREFERENCE)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint._name,
                         'preference_all_b02_<=_b01')

        self.assertTrue(
            'lambda_all_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b01').coefficient, 0.0)
        self.assertTrue(
            'u_MaxSpeed_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b01').coefficient, -1.0)

        self.assertTrue(
            'lambda_all_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b02').coefficient, 0.0)
        self.assertTrue(
            'u_MaxSpeed_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b02').coefficient, 1.0)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient,
            # this is just data.eps, but write how it evaluates
            data.eps - alpha * len(data.criteria) + alpha * len(data.criteria)
        )

    def test_weak_preference_alpha_1(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        alpha = 1.0
        preference = PreferenceRelation(
            'b01', 'b02', WEAK_PREFERENCE, alpha)
        preference_constraint = preference.to_constraint(data)

        self.assertEqual(preference_constraint._relation,
                         WEAK_PREFERENCE)
        # 2 * len(data.criteria) -> u_i(a_k); +2 -> lambda(a_k)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint.name,
                         'weak preference_all_b02_<=_b01')

        self.assertTrue(
            'lambda_all_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b01').coefficient, 0.0)
        self.assertTrue(
            'u_MaxSpeed_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b01').coefficient, -1.0)

        self.assertTrue(
            'lambda_all_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b02').coefficient, 0.0)
        self.assertTrue(
            'u_MaxSpeed_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b02').coefficient, 1.0)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient, 0)

    def test_strong_preference_alpha_0_5(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        alpha = 0.5
        preference = PreferenceRelation(
            'b01', 'b02', PREFERENCE, alpha)
        preference_constraint = preference.to_constraint(data)

        self.assertEqual(preference_constraint._relation, PREFERENCE)
        # 2 * len(data.criteria) -> u_i(a_k); +2 -> lambda(a_k)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint.name,
                         'preference_all_b02_<=_b01')

        self.assertTrue(
            'lambda_all_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b01').coefficient, 0.5)
        self.assertTrue(
            'u_MaxSpeed_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b01').coefficient, -0.5)

        self.assertTrue(
            'lambda_all_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b02').coefficient, -0.5)
        self.assertTrue(
            'u_MaxSpeed_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b02').coefficient, 0.5)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient,
            # this is just data.eps, but write how it evaluates
            data.eps - alpha * len(data.criteria) + alpha * len(data.criteria)
        )

    def test_weak_preference_alpha_0_5(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        alpha = 0.5
        preference = PreferenceRelation(
            'b01', 'b02', WEAK_PREFERENCE, alpha)
        preference_constraint = preference.to_constraint(data)

        self.assertEqual(preference_constraint._relation,
                         WEAK_PREFERENCE)
        # 2 * len(data.criteria) -> u_i(a_k); +2 -> lambda(a_k)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint._name,
                         'weak preference_all_b02_<=_b01')

        # all constraints are the same as in the strong preference, except free variable
        # that should be equal to 0 (there is no epsilion value)
        self.assertTrue(
            'lambda_all_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b01').coefficient, 0.5)
        self.assertTrue(
            'u_MaxSpeed_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b01').coefficient, -0.5)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient,
            # this is just 0, but write how it evaluates
            - alpha * len(data.criteria) + alpha * len(data.criteria)
        )

    def test_indifference_preference_alpha_0_5(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        alpha = 0.5
        preference = PreferenceRelation(
            'b01', 'b02', INDIFFERENCE, alpha)
        preference_constraint = preference.to_constraint(data)

        self.assertEqual(preference_constraint._relation,
                         INDIFFERENCE)
        # 2 * len(data.criteria) -> u_i(a_k); +2 -> lambda(a_k)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint._name,
                         'indifference_all_b02_==_b01')

        # all constraints are the same as in the strong preference, except free variable
        # that should be equal to 0 (there is no epsilion value)
        self.assertTrue(
            'lambda_all_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b01').coefficient, 0.5)
        self.assertTrue(
            'u_MaxSpeed_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b01').coefficient, -0.5)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient,
            # this is just 0, but write how it evaluates
            - alpha * len(data.criteria) + alpha * len(data.criteria)
        )
