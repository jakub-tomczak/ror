from ror.data_loader import read_dataset_from_txt
import unittest
from ror.RORModel import RORModel


class TestRORModel(unittest.TestCase):
    def test_creating_ror_model(self):
        data = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        model = RORModel(data, 0.0, "Model with alpha 0.0")

        self.assertEqual(len(model.constraints), 171)
        self.assertEqual(model.target, "delta")
