from typing import List
from ror.Relation import INDIFFERENCE, PREFERENCE
from ror.data_loader import RORParameter, read_dataset_from_txt
import unittest
from ror.Dataset import Dataset, RORDataset
import numpy as np
import tempfile
import os


class TestTxtDatasetWriter(unittest.TestCase):
    def test_saving_dataset_to_txt(self):
        loading_result = read_dataset_from_txt(
            "tests/datasets/ror_dataset_with_parameters.txt")
        data = loading_result.dataset
        parameters = loading_result.parameters
        dir = tempfile.mkdtemp()

        filename = os.path.join(dir, 'test_saving_dataset_to_txt.txt')
        data.save_to_file(filename, {RORParameter.INITIAL_ALPHA: 0.01})

        read_data: List[str] = []
        with open(filename, 'r') as file:
            for line in file:
                read_data.append(line.strip())

        self.assertTrue('#Data' in read_data)
        self.assertTrue('alternative id,MaxSpeed[g],FuelCons[c]' in read_data)
        
        self.assertTrue('#Preferences' in read_data)
        self.assertTrue('b01,b02,indifference' in read_data)
        self.assertTrue('b08,b07,preference' in read_data)
        self.assertTrue('b04,b08,b07,b06,preference' in read_data)
        
        self.assertTrue('#Parameters' in read_data)
        self.assertTrue(f'eps={parameters[RORParameter.EPS]}' in read_data)
        self.assertTrue(f'initial_alpha=0.01' in read_data)

    def read_save_and_read(self):
        loading_result = read_dataset_from_txt(
            "tests/datasets/ror_dataset_with_parameters.txt")
        data = loading_result.dataset
        parameters = loading_result.parameters
        dir = tempfile.mkdtemp()

        filename = os.path.join(dir, 'test_saving_dataset_to_txt.txt')
        data.save_to_file(filename, {RORParameter.INITIAL_ALPHA: 0.01})

        loading_result_2 = read_dataset_from_txt(
            "tests/datasets/ror_dataset_with_parameters.txt")
        data_2 = loading_result_2.dataset
        parameters_2 = loading_result_2.parameters

        self.assertEqual(len(data_2.alternatives), len(data.alternatives))
        self.assertEqual(len(data_2.criteria), len(data.criteria))
        self.assertEqual(len(data_2.preferenceRelations), len(data.preferenceRelations))
        self.assertEqual(len(data_2.intensityRelations), len(data.intensityRelations))
        self.assertAlmostEqual(data_2.eps, data.eps)
        self.assertAlmostEqual(parameters_2[RORParameter.INITIAL_ALPHA], parameters[RORParameter.INITIAL_ALPHA])
