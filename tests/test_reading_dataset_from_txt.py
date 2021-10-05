from ror.Relation import INDIFFERENCE, PREFERENCE
from ror.data_loader import AvailableParameters, read_dataset_from_txt
import unittest
from ror.Dataset import Dataset, RORDataset
import numpy as np


class TestTxtDatasetReader(unittest.TestCase):
    def test_reading_dataset_from_txt(self):
        loading_result = read_dataset_from_txt("tests/datasets/example.txt")
        data = loading_result.dataset

        self.assertIs(type(data), RORDataset)
        self.assertEqual(len(data.criteria), 2)
        self.assertEqual(data.criteria[0][0], "MaxSpeed")
        self.assertEqual(data.criteria[0][1], "g")
        self.assertEqual(data.criteria[1][0], "FuelCons")
        self.assertEqual(data.criteria[1][1], "c")

        self.assertEqual(len(data.alternatives), 5)
        self.assertEqual(data.alternatives[0], "b01")
        self.assertEqual(data.alternatives[4], "b05")

        self.assertIs(type(data.matrix[0, 0]), np.float64)
        self.assertEqual(data.matrix[0, 0], 90)
        self.assertIs(type(data.matrix[4, 0]), np.float64)
        self.assertEqual(data.matrix[4, 0], 83)

        # cost type criteria are reversed
        # (multiplied by -1 so we can treat them as gain type criteria)
        self.assertIs(type(data.matrix[4, 1]), np.float64)
        self.assertEqual(data.matrix[4, 1], -26)
        self.assertIs(type(data.matrix[0, 1]), np.float64)
        self.assertEqual(data.matrix[0, 1], -27)

    def test_reading_dataset_from_txt(self):
        loading_result = read_dataset_from_txt("tests/datasets/example.txt")
        data = loading_result.dataset

        self.assertIs(type(data), RORDataset)
        self.assertEqual(len(data.criteria), 2)
        self.assertEqual(data.criteria[0][0], "MaxSpeed")
        self.assertEqual(data.criteria[0][1], "g")
        self.assertEqual(data.criteria[1][0], "FuelCons")
        self.assertEqual(data.criteria[1][1], "c")

        self.assertEqual(len(data.alternatives), 5)
        self.assertEqual(data.alternatives[0], "b01")
        self.assertEqual(data.alternatives[4], "b05")

        self.assertIs(type(data.matrix[0, 0]), np.float64)
        self.assertEqual(data.matrix[0, 0], 90)
        self.assertIs(type(data.matrix[4, 0]), np.float64)
        self.assertEqual(data.matrix[4, 0], 83)

        # cost type criteria are reversed
        # (multiplied by -1 so we can treat them as gain type criteria)
        self.assertIs(type(data.matrix[4, 1]), np.float64)
        self.assertEqual(data.matrix[4, 1], -26)
        self.assertIs(type(data.matrix[0, 1]), np.float64)
        self.assertEqual(data.matrix[0, 1], -27)


    def test_reading_dataset_from_txt_with_preferences(self):
        loading_result = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        data = loading_result.dataset

        self.assertIs(type(data), RORDataset)
        self.assertEqual(len(data.criteria), 2)
        self.assertEqual(data.criteria[0][0], "MaxSpeed")
        self.assertEqual(data.criteria[0][1], "g")
        self.assertEqual(data.criteria[1][0], "FuelCons")
        self.assertEqual(data.criteria[1][1], "c")

        self.assertEqual(len(data.alternatives), 14)
        self.assertEqual(data.alternatives[0], "b01")
        self.assertEqual(data.alternatives[4], "b05")

        self.assertIs(type(data.matrix[0, 0]), np.float64)
        self.assertEqual(data.matrix[0, 0], 90)
        self.assertIs(type(data.matrix[4, 0]), np.float64)
        self.assertEqual(data.matrix[4, 0], 83)

        self.assertEqual(len(data.preferenceRelations), 3)
        self.assertEqual(data.preferenceRelations[0].relation, INDIFFERENCE)
        self.assertEqual(data.preferenceRelations[0].alternative_1, "b01")
        self.assertEqual(data.preferenceRelations[0].alternative_2, "b02")
        self.assertEqual(data.preferenceRelations[1].relation, PREFERENCE)
        self.assertEqual(data.preferenceRelations[1].alternative_1, "b06")
        self.assertEqual(data.preferenceRelations[1].alternative_2, "b03")
        self.assertEqual(data.preferenceRelations[2].relation, PREFERENCE)
        self.assertEqual(data.preferenceRelations[2].alternative_1, "b08")
        self.assertEqual(data.preferenceRelations[2].alternative_2, "b07")
        
        self.assertEqual(len(data.intensityRelations), 1)
        self.assertEqual(data.intensityRelations[0].relation, PREFERENCE)
        self.assertEqual(data.intensityRelations[0].alternative_1, "b04")
        self.assertEqual(data.intensityRelations[0].alternative_2, "b08")
        self.assertEqual(data.intensityRelations[0].alternative_3, "b07")
        self.assertEqual(data.intensityRelations[0].alternative_4, "b06")
        
    def test_reading_with_preferences(self):
        loading_result = read_dataset_from_txt("tests/datasets/ror_dataset_with_parameters.txt")
        data = loading_result.dataset
        parameters = loading_result.parameters

        self.assertIs(type(data), RORDataset)
        self.assertEqual(len(data.criteria), 2)
        self.assertEqual(len(data.alternatives), 14)
        self.assertEqual(len(data.preferenceRelations), 3)
        self.assertEqual(len(data.intensityRelations), 1)

        self.assertAlmostEqual(parameters[AvailableParameters.EPS], 2e-11)
        self.assertAlmostEqual(parameters[AvailableParameters.INITIAL_ALPHA], 0.1)