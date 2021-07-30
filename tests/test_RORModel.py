from ror.data_loader import read_dataset_from_txt
import unittest
from ror.RORModel import RORModel
from gurobipy import GRB


class TestRORModel(unittest.TestCase):
    def test_creating_ror_model(self):
        data = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        model = RORModel(data, 0.0, "Model with alpha 0.0")

        self.assertEqual(len(model.constraints), 171)
        self.assertEqual(model.target, "delta")

    def test_creating_gurobi_model_from_ror_model(self):
        data = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        model = RORModel(data, 0.0, "Model with alpha 0.0")

        self.assertEqual(len(model.constraints), 171)
        self.assertEqual(model.target, "delta")

        model.to_gurobi_model()

    def test_saving_model_to_lp(self):
        data = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        model = RORModel(data, 0.0, "Model with alpha 0.0")

        self.assertEqual(len(model.constraints), 171)
        self.assertEqual(model.target, "delta")

        model.save_model()

    def test_solving_model(self):
        data = read_dataset_from_txt("tests/datasets/ror_full_dataset.txt")
        model = RORModel(data, 0.0, "Model with alpha 0.0")

        gurobi_model = model.to_gurobi_model()
        gurobi_model.optimize()

        self.assertEqual(gurobi_model.status, GRB.OPTIMAL)
        self.assertAlmostEqual(gurobi_model.objVal, 0.0)
