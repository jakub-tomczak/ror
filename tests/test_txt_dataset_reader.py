import unittest
import numpy as np
from ror.Dataset import read_dataset_from_txt, Dataset


class TestTxtDatasetReader(unittest.TestCase):
    def test_reading_dataset(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")

        self.assertIs(type(data), Dataset)
        self.assertEqual(len(data.criteria), 2)
        self.assertEqual(data.criteria[0][0], "MaxSpeed")
        self.assertEqual(data.criteria[0][1], "g")
        self.assertEqual(data.criteria[1][0], "FuelCons")
        self.assertEqual(data.criteria[1][1], "c")

        self.assertEqual(len(data.alternatives), 5)
        self.assertEqual(data.alternatives[0], "b01")
        self.assertEqual(data.alternatives[4], "b05")

        self.assertIs(type(data.data[0, 0]), np.int64)
        self.assertEqual(data.data[0, 0], 90)

        self.assertIs(type(data.data[4, 1]), np.int64)
        self.assertEqual(data.data[4, 1], 26)