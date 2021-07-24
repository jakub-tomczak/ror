from ror.Constraint import ValueConstraintVariable
import unittest
from ror.Dataset import read_dataset_from_txt
from ror.d_function import d


class TestDFunction(unittest.TestCase):
    def test_creating_d_function_alpha_0(self):
        dataset = read_dataset_from_txt("tests/datasets/example.txt")
        alternative = "b01"
        alpha = 0.0

        d_constraint = d(alternative, alpha, dataset)

        # when alpha is 0 then only lambda variables are present
        self.assertAlmostEqual(d_constraint[ValueConstraintVariable.name].coefficient, 0.0)
        self.assertAlmostEqual(d_constraint['u_MaxSpeed_b01'].coefficient, 0.0)
        self.assertAlmostEqual(d_constraint['u_FuelCons_b01'].coefficient, 0.0)
        self.assertAlmostEqual(d_constraint['lambda_MaxSpeed_b01'].coefficient, 1.0)
        self.assertAlmostEqual(d_constraint['lambda_FuelCons_b01'].coefficient, 1.0)

    def test_creating_d_function_alpha_0_5(self):
        dataset = read_dataset_from_txt("tests/datasets/example.txt")
        alternative = "b01"
        alpha = 0.5

        d_constraint = d(alternative, alpha, dataset)

        self.assertAlmostEqual(d_constraint[ValueConstraintVariable.name].coefficient, 1.0)
        self.assertAlmostEqual(d_constraint['u_MaxSpeed_b01'].coefficient, -45.0)
        self.assertAlmostEqual(d_constraint['u_FuelCons_b01'].coefficient, 13.5)
        self.assertAlmostEqual(d_constraint['lambda_MaxSpeed_b01'].coefficient, 0.5)
        self.assertAlmostEqual(d_constraint['lambda_FuelCons_b01'].coefficient, 0.5)

    def test_multiplying_d_function(self):
        dataset = read_dataset_from_txt("tests/datasets/example.txt")
        alternative = "b01"
        alpha = 0.5

        d_constraint = d(alternative, alpha, dataset)
        d_constraint.multiply_by_scalar(2.0)

        self.assertAlmostEqual(d_constraint[ValueConstraintVariable.name].coefficient, 2.0)
        self.assertAlmostEqual(d_constraint['u_MaxSpeed_b01'].coefficient, -90.0)
        self.assertAlmostEqual(d_constraint['u_FuelCons_b01'].coefficient, 27.0)
        self.assertAlmostEqual(d_constraint['lambda_MaxSpeed_b01'].coefficient, 1.0)
        self.assertAlmostEqual(d_constraint['lambda_FuelCons_b01'].coefficient, 1.0)
