import unittest
from ror.Dataset import read_dataset_from_txt, Dataset
from ror.slope_constraints import create_slope_constraints
import numpy as np


class TestSlopeConstraints(unittest.TestCase):
    def test_creating_slope_contraint_success(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        constraints = create_slope_constraints(data)

        # there should be 4 constraints for MaxSpeed and 2 for FuelCons
        # as there is 1 repetition in g value out of 3 possibilities in MaxSpeed
        # and there are 2 repetitions in g value out of 3 possibilities in FuelCons
        self.assertEqual(len(constraints), 6)

        first_constraint = constraints[0]
        # there should be b02, b03 and b04, delta
        self.assertEqual(len(first_constraint.variables), 4)
        self.assertEqual(
            set(first_constraint.variables_names),
            set(["u_MaxSpeed_b02", "u_MaxSpeed_b03", "u_MaxSpeed_b04", "delta"])
        )

        first_constraint_slope_first_coeff = \
            1 / (data.get_data_for_alternative_and_criterion("b04", "MaxSpeed").coefficient -
                 data.get_data_for_alternative_and_criterion("b03", "MaxSpeed").coefficient)

        first_constraint_slope_second_coeff = \
            1 / (data.get_data_for_alternative_and_criterion("b03", "MaxSpeed").coefficient -
                 data.get_data_for_alternative_and_criterion("b02", "MaxSpeed").coefficient)

        self.assertAlmostEqual(first_constraint.get_variable(
            "u_MaxSpeed_b02").coefficient, first_constraint_slope_second_coeff)
        self.assertAlmostEqual(first_constraint.get_variable(
            "u_MaxSpeed_b03").coefficient, -first_constraint_slope_first_coeff - first_constraint_slope_second_coeff)
        self.assertAlmostEqual(first_constraint.get_variable(
            "u_MaxSpeed_b04").coefficient, first_constraint_slope_first_coeff)

        second_constraint = constraints[1]
        second_constraint_slope_first_coeff = \
            1 / (data.get_data_for_alternative_and_criterion("b03", "MaxSpeed").coefficient -
                   data.get_data_for_alternative_and_criterion("b02", "MaxSpeed").coefficient)
        second_constraint_slope_second_coeff = \
            1 / (data.get_data_for_alternative_and_criterion("b04", "MaxSpeed").coefficient -
                   data.get_data_for_alternative_and_criterion("b03", "MaxSpeed").coefficient)

        self.assertEqual(first_constraint_slope_first_coeff, second_constraint_slope_second_coeff)
        self.assertEqual(first_constraint_slope_second_coeff, second_constraint_slope_first_coeff)

        self.assertAlmostEqual(second_constraint.get_variable(
            "u_MaxSpeed_b02").coefficient, -second_constraint_slope_first_coeff)
        self.assertAlmostEqual(second_constraint.get_variable("u_MaxSpeed_b03").coefficient,
                               second_constraint_slope_first_coeff + second_constraint_slope_second_coeff)
        self.assertAlmostEqual(second_constraint.get_variable(
            "u_MaxSpeed_b04").coefficient, -second_constraint_slope_second_coeff)

    def test_creating_one_slope_constraints_failes_too_few_alternatives(self):
        '''
        At least 3 alternatives are reuqired to create a slope constraints.
        There will be an empty list when there are no slope constraints.
        '''
        data = Dataset(
            ["a1", "a2"],
            np.array([
                [10, 11],
                [9, 12]
            ]),
            [("First criterion", "g"), ("Second criterion", "c")]
        )
        slope_constraints = create_slope_constraints(data)
        self.assertEqual(len(slope_constraints), 0)

    def test_creating_one_slope_contraint_success(self):
        data = Dataset(
            ["a1", "a2", "a3"],
            np.array([
                [10, 11],
                [9, 12],
                [7, 4]
            ]),
            [("First criterion", "g"), ("Second criterion", "c")]
        )

        slope_constraints = create_slope_constraints(data)
        # no repetition in criterion values so there will be
        # 2 x number of criteria + (n-2)*2 slope constraints
        # where n is the number of alternatives
        self.assertEqual(len(slope_constraints), 4)

        first_criterion_first_slope, first_criterion_second_slope = slope_constraints[:2]

        first_coeff = 1 / (data.matrix[2, 0] - data.matrix[1, 0]) # -0.5
        second_coeff = 1 / (data.matrix[1, 0] - data.matrix[0, 0]) # -1.0
        self.assertAlmostEqual(first_coeff, -0.5)
        self.assertAlmostEqual(second_coeff, -1.0)
        self.assertAlmostEqual(first_criterion_first_slope.get_variable("delta").coefficient, -1.0)
        self.assertAlmostEqual(first_criterion_first_slope.get_variable("u_First criterion_a1").coefficient, second_coeff)
        self.assertAlmostEqual(first_criterion_first_slope.get_variable("u_First criterion_a2").coefficient, -first_coeff - second_coeff)
        self.assertAlmostEqual(first_criterion_first_slope.get_variable("u_First criterion_a3").coefficient, first_coeff)

        self.assertAlmostEqual(first_criterion_second_slope.get_variable("delta").coefficient, -1.0)
        self.assertAlmostEqual(first_criterion_second_slope.get_variable("u_First criterion_a1").coefficient, -second_coeff)
        self.assertAlmostEqual(first_criterion_second_slope.get_variable("u_First criterion_a2").coefficient, first_coeff + second_coeff)
        self.assertAlmostEqual(first_criterion_second_slope.get_variable("u_First criterion_a3").coefficient, -first_coeff)

        second_criterion_first_slope, second_criterion_second_slope = slope_constraints[2:]

        first_coeff = 1 / (data.matrix[2, 1] - data.matrix[1, 1]) # 1/8
        second_coeff = 1 / (data.matrix[1, 1] - data.matrix[0, 1]) # -1.0
        self.assertAlmostEqual(first_coeff, 1/8)
        self.assertAlmostEqual(second_coeff, -1.0)
        self.assertAlmostEqual(second_criterion_first_slope.get_variable("delta").coefficient, -1.0)
        self.assertAlmostEqual(second_criterion_first_slope.get_variable("u_Second criterion_a1").coefficient, second_coeff)
        self.assertAlmostEqual(second_criterion_first_slope.get_variable("u_Second criterion_a2").coefficient, -first_coeff - second_coeff)
        self.assertAlmostEqual(second_criterion_first_slope.get_variable("u_Second criterion_a3").coefficient, first_coeff)

        self.assertAlmostEqual(second_criterion_second_slope.get_variable("delta").coefficient, -1.0)
        self.assertAlmostEqual(second_criterion_second_slope.get_variable("u_Second criterion_a1").coefficient, -second_coeff)
        self.assertAlmostEqual(second_criterion_second_slope.get_variable("u_Second criterion_a2").coefficient, first_coeff + second_coeff)
        self.assertAlmostEqual(second_criterion_second_slope.get_variable("u_Second criterion_a3").coefficient, -first_coeff)