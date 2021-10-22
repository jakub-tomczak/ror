from ror.data_loader import read_dataset_from_txt
from ror.Relation import PREFERENCE
import unittest
from ror.PreferenceRelations import PreferenceIntensityRelation


class TestIntensityPreferenceRelations(unittest.TestCase):
    def test_strong_intensity_preference_alpha_0(self):
        loading_result = read_dataset_from_txt("tests/datasets/example.txt")
        data = loading_result.dataset

        preference = PreferenceIntensityRelation('b01', 'b02', 'b03', 'b04', PREFERENCE)
        preference_constraint = preference.to_constraint(data, 0.0)

        self.assertEqual(preference_constraint.relation, PREFERENCE)
        self.assertEqual(len(preference_constraint.variables),
                         4 * len(data.criteria) + 4)

        self.assertTrue(
            'lambda_{all}(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}(b01)').coefficient, -1.0)
        self.assertTrue(
            'u_{MaxSpeed}(b01)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}(b01)').coefficient, 0.0)

        self.assertTrue(
            'lambda_{all}(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}(b02)').coefficient, 1.0)
        self.assertTrue(
            'u_{MaxSpeed}(b02)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}(b02)').coefficient, 0.0)

        self.assertTrue(
            'lambda_{all}(b03)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}(b03)').coefficient, 1.0)
        self.assertTrue(
            'u_{MaxSpeed}(b03)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}(b03)').coefficient, 0.0)

        self.assertTrue(
            'lambda_{all}(b04)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'lambda_{all}(b04)').coefficient, -1.0)
        self.assertTrue(
            'u_{MaxSpeed}(b04)' in preference_constraint.variables_names)
        self.assertAlmostEqual(preference_constraint.get_variable(
            'u_{MaxSpeed}(b04)').coefficient, 0.0)

        self.assertAlmostEqual(
            preference_constraint.free_variable.coefficient, -data.eps)
