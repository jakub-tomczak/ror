from ror.Relation import PREFERENCE
import unittest
from ror.Dataset import read_dataset_from_txt
from ror.PreferenceRelations import PreferenceIntensityRelation


class TestIntensityPreferenceRelations(unittest.TestCase):
    def test_strong_intensity_preference_alpha_0(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        preference = PreferenceIntensityRelation(
            'b01', 'b02', 'b03', 'b04', PREFERENCE, 0.0)
        preference_constraint = preference.to_constraint(data)

        self.assertEqual(preference_constraint.relation, PREFERENCE)
        self.assertEqual(len(preference_constraint.variables),
                         4 * len(data.criteria) + 4)

        self.assertTrue(
            'lambda_all_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b01').coefficient, -1.0)
        self.assertTrue(
            'u_MaxSpeed_b01' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b01').coefficient, 0.0)

        self.assertTrue(
            'lambda_all_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b02').coefficient, 1.0)
        self.assertTrue(
            'u_MaxSpeed_b02' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b02').coefficient, 0.0)

        self.assertTrue(
            'lambda_all_b03' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b03').coefficient, 1.0)
        self.assertTrue(
            'u_MaxSpeed_b03' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b03').coefficient, 0.0)

        self.assertTrue(
            'lambda_all_b04' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_all_b04').coefficient, -1.0)
        self.assertTrue(
            'u_MaxSpeed_b04' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_MaxSpeed_b04').coefficient, 0.0)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient, -data.eps)
