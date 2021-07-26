from ror.data_loader import read_dataset_from_txt
from ror.Constraint import ValueConstraintVariable
import unittest
from ror.Dataset import Dataset
from ror.d_function import d


class TestDFunction(unittest.TestCase):
    def  test_creating_d_function_invalid_alpha(self):
        with self.assertRaises(AssertionError):
            d('', -1e-10, None)
        with self.assertRaises(AssertionError):
            d('', 1.0 + 1e-10, None)

    def test_creating_d_function_invalid_alternative(self):
        dataset = read_dataset_from_txt("tests/datasets/example.txt")
        alternative = 'non-existing_alternative'
        with self.assertRaises(AssertionError):
            d(alternative, 0, dataset)

    def test_creating_d_function_none_dataset(self):
        alternative = 'non-existing_alternative'
        with self.assertRaises(AssertionError):
            d(alternative, 0, None)

    def test_creating_d_function_alpha_0(self):
        dataset = read_dataset_from_txt("tests/datasets/example.txt")
        alternative = "b01"
        alpha = 0.0

        d_constraint = d(alternative, alpha, dataset)

        # when alpha is 0 then only lambda variables are present
        self.assertAlmostEqual(d_constraint[ValueConstraintVariable.name].coefficient, 0.0)
        self.assertAlmostEqual(d_constraint['u_MaxSpeed_b01'].coefficient, 0.0)
        self.assertAlmostEqual(d_constraint['u_FuelCons_b01'].coefficient, 0.0)
        self.assertTrue('lambda_all_b01' in d_constraint.variables_names)
        self.assertAlmostEqual(d_constraint['lambda_all_b01'].coefficient, 1.0)
        self.assertTrue('lambda_all_b01' in d_constraint.variables_names)
        self.assertAlmostEqual(d_constraint['lambda_all_b01'].coefficient, 1.0)

    def test_creating_d_function_alpha_0_5(self):
        dataset = read_dataset_from_txt("tests/datasets/example.txt")
        alternative = "b01"
        alpha = 0.5

        d_constraint = d(alternative, alpha, dataset)

        self.assertAlmostEqual(d_constraint[ValueConstraintVariable.name].coefficient, 1.0)
        self.assertAlmostEqual(d_constraint['u_MaxSpeed_b01'].coefficient, -0.5)
        self.assertAlmostEqual(d_constraint['u_FuelCons_b01'].coefficient, -0.5)
        self.assertTrue('lambda_all_b01' in d_constraint.variables_names)
        self.assertAlmostEqual(d_constraint['lambda_all_b01'].coefficient, 0.5)
        self.assertTrue('lambda_all_b01' in d_constraint.variables_names)
        self.assertAlmostEqual(d_constraint['lambda_all_b01'].coefficient, 0.5)

    def test_multiplying_d_function(self):
        dataset = read_dataset_from_txt("tests/datasets/example.txt")
        alternative = "b01"
        alpha = 0.5

        d_constraint = d(alternative, alpha, dataset)
        d_constraint.multiply_by_scalar(2.0)

        self.assertAlmostEqual(d_constraint[ValueConstraintVariable.name].coefficient, 2.0)
        self.assertAlmostEqual(d_constraint['u_MaxSpeed_b01'].coefficient, -1.0)
        self.assertAlmostEqual(d_constraint['u_FuelCons_b01'].coefficient, -1.0)
        self.assertTrue('lambda_all_b01' in d_constraint.variables_names)
        self.assertAlmostEqual(d_constraint['lambda_all_b01'].coefficient, 1.0)
        self.assertTrue('lambda_all_b01' in d_constraint.variables_names)
        self.assertAlmostEqual(d_constraint['lambda_all_b01'].coefficient, 1.0)
