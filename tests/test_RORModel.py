from ror.Constraint import ConstraintVariable, ConstraintVariablesSet
from ror.data_loader import read_dataset_from_txt
import unittest
from ror.RORModel import RORModel


class TestRORModel(unittest.TestCase):
    def test_creating_ror_model(self):
        data = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        model = RORModel(data, 0.0, "Model with alpha 0.0")

        self.assertEqual(len(model.constraints), 171)
        self.assertIsNone(model.target)

    def test_creating_gurobi_model_from_ror_model(self):
        data = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        model = RORModel(data, 0.0, "Model with alpha 0.0")

        self.assertEqual(len(model.constraints), 171)
        model.target = ConstraintVariablesSet([
            ConstraintVariable("delta", 1.0)
        ])

        model.to_gurobi_model()

    def test_saving_model_to_lp(self):
        data = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        model = RORModel(data, 0.0, "Model with alpha 0.0")

        self.assertEqual(len(model.constraints), 171)
        model.target = ConstraintVariablesSet([
            ConstraintVariable("delta", 1.0)
        ])

        model.save_model()

    def test_solving_model(self):
        data = read_dataset_from_txt("tests/datasets/ror_full_dataset.txt")
        model = RORModel(data, 0.0, "Model with alpha 0.0")
        model.target = ConstraintVariablesSet([
            ConstraintVariable("delta", 1.0)
        ])

        result = model.solve()

        self.assertAlmostEqual(result.objective_value, 0.0)
