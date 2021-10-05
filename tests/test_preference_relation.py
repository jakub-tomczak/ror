from ror.data_loader import read_dataset_from_txt
import unittest
from ror.PreferenceRelations import PreferenceRelation
from ror.Relation import INDIFFERENCE, PREFERENCE, Relation, WEAK_PREFERENCE


class TestPreferenceRelations(unittest.TestCase):
    def test_creating_preference_invalid_alpha(self):
        with self.assertRaises(AssertionError):
            relation = PreferenceRelation('b01', 'b02', PREFERENCE)
            relation.alpha = -1e-10
        with self.assertRaises(AssertionError):
            relation = PreferenceRelation('b01', 'b02', PREFERENCE)
            relation.alpha = 1+1e-10

    def test_creating_preference_invalid_relation(self):
        with self.assertRaises(AssertionError):
            PreferenceRelation('b01', 'b02', "<=")

        with self.assertRaises(AssertionError):
            PreferenceRelation('b01', 'b02', Relation('<='))

    def test_strong_preference_alpha_0(self):
        loading_result = read_dataset_from_txt("tests/datasets/example.txt")
        data = loading_result.dataset

        preference = PreferenceRelation('b01', 'b02', PREFERENCE)
        preference_constraint = preference.to_constraint(data, 0.0)

        self.assertEqual(preference_constraint._relation, PREFERENCE)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint._name,
                         'preference_{all}_(b02) <= preference_{all}_(b01)')

        self.assertTrue(
            'lambda_{all}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b01)').coefficient, 1.0)
        self.assertTrue(
            'u_{MaxSpeed}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b01)').coefficient, 0.0)

        self.assertTrue(
            'lambda_{all}_(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b02)').coefficient, -1.0)
        self.assertTrue(
            'u_{MaxSpeed}_(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b02)').coefficient, 0.0)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient, data.eps)

    def test_strong_preference_alpha_1(self):
        loading_result = read_dataset_from_txt("tests/datasets/example.txt")
        data = loading_result.dataset

        alpha = 1.0
        preference = PreferenceRelation('b01', 'b02', PREFERENCE)
        preference_constraint = preference.to_constraint(data, alpha)

        self.assertEqual(preference_constraint._relation, PREFERENCE)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint._name,
                         'preference_{all}_(b02) <= preference_{all}_(b01)')

        self.assertTrue(
            'lambda_{all}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b01)').coefficient, 0.0)
        self.assertTrue(
            'u_{MaxSpeed}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b01)').coefficient, -1.0)

        self.assertTrue(
            'lambda_{all}_(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b02)').coefficient, 0.0)
        self.assertTrue(
            'u_{MaxSpeed}_(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b02)').coefficient, 1.0)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient,
            # this is just data.eps, but write how it evaluates
            data.eps - alpha * len(data.criteria) + alpha * len(data.criteria)
        )

    def test_weak_preference_alpha_1(self):
        loading_result = read_dataset_from_txt("tests/datasets/example.txt")
        data = loading_result.dataset

        alpha = 1.0
        preference = PreferenceRelation('b01', 'b02', WEAK_PREFERENCE)
        preference_constraint = preference.to_constraint(data, alpha)

        self.assertEqual(preference_constraint._relation,
                         WEAK_PREFERENCE)
        # 2 * len(data.criteria) -> u_i(a_k); +2 -> lambda(a_k)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint.name,
                         'weak preference_{all}_(b02) <= weak preference_{all}_(b01)')

        self.assertTrue(
            'lambda_{all}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b01)').coefficient, 0.0)
        self.assertTrue(
            'u_{MaxSpeed}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b01)').coefficient, -1.0)

        self.assertTrue(
            'lambda_{all}_(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b02)').coefficient, 0.0)
        self.assertTrue(
            'u_{MaxSpeed}_(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b02)').coefficient, 1.0)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient, 0)

    def test_strong_preference_alpha_0_5(self):
        loading_result = read_dataset_from_txt("tests/datasets/example.txt")
        data = loading_result.dataset

        alpha = 0.5
        preference = PreferenceRelation('b01', 'b02', PREFERENCE)
        preference_constraint = preference.to_constraint(data, alpha)

        self.assertEqual(preference_constraint._relation, PREFERENCE)
        # 2 * len(data.criteria) -> u_i(a_k); +2 -> lambda(a_k)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint.name,
                         'preference_{all}_(b02) <= preference_{all}_(b01)')

        self.assertTrue(
            'lambda_{all}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b01)').coefficient, 0.5)
        self.assertTrue(
            'u_{MaxSpeed}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b01)').coefficient, -0.5)

        self.assertTrue(
            'lambda_{all}_(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b02)').coefficient, -0.5)
        self.assertTrue(
            'u_{MaxSpeed}_(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b02)').coefficient, 0.5)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient,
            # this is just data.eps, but write how it evaluates
            data.eps - alpha * len(data.criteria) + alpha * len(data.criteria)
        )

    def test_weak_preference_alpha_0_5(self):
        loading_result = read_dataset_from_txt("tests/datasets/example.txt")
        data = loading_result.dataset

        alpha = 0.5
        preference = PreferenceRelation('b01', 'b02', WEAK_PREFERENCE)
        preference_constraint = preference.to_constraint(data, alpha)

        self.assertEqual(preference_constraint._relation,
                         WEAK_PREFERENCE)
        # 2 * len(data.criteria) -> u_i(a_k); +2 -> lambda(a_k)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint._name,
                         'weak preference_{all}_(b02) <= weak preference_{all}_(b01)')

        # all constraints are the same as in the strong preference, except free variable
        # that should be equal to 0 (there is no epsilion value)
        self.assertTrue(
            'lambda_{all}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b01)').coefficient, 0.5)
        self.assertTrue(
            'u_{MaxSpeed}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b01)').coefficient, -0.5)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient,
            # this is just 0, but write how it evaluates
            - alpha * len(data.criteria) + alpha * len(data.criteria)
        )

    def test_indifference_preference_alpha_0_5(self):
        loading_result = read_dataset_from_txt("tests/datasets/example.txt")
        data = loading_result.dataset

        alpha = 0.5
        preference = PreferenceRelation('b01', 'b02', INDIFFERENCE)
        preference_constraint = preference.to_constraint(data, alpha)

        self.assertEqual(preference_constraint._relation,
                         INDIFFERENCE)
        # 2 * len(data.criteria) -> u_i(a_k); +2 -> lambda(a_k)
        self.assertEqual(len(preference_constraint.variables),
                         2 * len(data.criteria) + 2)
        self.assertEqual(preference_constraint._name,
                         'indifference_{all}_(b02) == indifference_{all}_(b01)')

        # all constraints are the same as in the strong preference, except free variable
        # that should be equal to 0 (there is no epsilion value)
        self.assertTrue(
            'lambda_{all}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}_(b01)').coefficient, 0.5)
        self.assertTrue(
            'u_{MaxSpeed}_(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}_(b01)').coefficient, -0.5)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient,
            # this is just 0, but write how it evaluates
            - alpha * len(data.criteria) + alpha * len(data.criteria)
        )
