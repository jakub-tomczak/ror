from typing import Set
from ror.Constraint import ConstraintVariable, ConstraintVariablesSet
from ror.data_loader import read_dataset_from_txt
import unittest
from ror.solvers.GurobiSolver import GurobiSolver
from ror.RORModel import RORModel


class TestRORModel(unittest.TestCase):
    def test_creating_ror_model(self):
        loading_result = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        data = loading_result.dataset
        model = RORModel(data, 0.0, "Model with alpha 0.0")
        
        reference_alternatives: Set[str] = set()
        for relation in data.preferenceRelations:
            reference_alternatives.update(relation.alternatives)
        for intensity_relation in data.intensityRelations:
            reference_alternatives.update(intensity_relation.alternatives)
        inner_maximization_constraints = (3*len(data.criteria) + 1)* len(reference_alternatives)

        for constraints_name in model.constraints_dict:
            from ror.constraints_constants import ConstraintsName
            if constraints_name == ConstraintsName.INNER_MAXIMIZATION.value:
                self.assertEqual(len(model.constraints_dict[constraints_name]), inner_maximization_constraints)
            elif constraints_name == ConstraintsName.PREFERENCE_INFORMATION.value:
                self.assertEqual(len(model.constraints_dict[constraints_name]), 3)
            elif constraints_name == ConstraintsName.PREFERENCE_INTENSITY_INFORMATION.value:
                self.assertEqual(len(model.constraints_dict[constraints_name]), 1)
            elif constraints_name == ConstraintsName.MIN_CONSTRAINTS.value:
                self.assertEqual(len(model.constraints_dict[constraints_name]), len(data.criteria))
            elif constraints_name == ConstraintsName.MAX_CONSTRAINTS.value:
                self.assertEqual(len(model.constraints_dict[constraints_name]), 1)

        self.assertEqual(len(model.constraints), 122)
        self.assertIsNone(model.target)


    def test_saving_model_to_lp(self):
        loading_result = read_dataset_from_txt("tests/datasets/ror_dataset.txt")
        data = loading_result.dataset
        model = RORModel(data, 0.0, "Model with alpha 0.0")
        model.solver = GurobiSolver()

        self.assertEqual(len(model.constraints), 122)
        model.target = ConstraintVariablesSet([
            ConstraintVariable("delta", 1.0)
        ])
        model.solver.create_model(model)
        model.save_model()

    def test_solving_model(self):
        loading_result = read_dataset_from_txt("tests/datasets/ror_full_dataset.txt")
        data = loading_result.dataset
        model = RORModel(data, 0.0, "Model with alpha 0.0")
        model.target = ConstraintVariablesSet([
            ConstraintVariable("delta", 1.0)
        ])
        model.solver = GurobiSolver()
        result = model.solve()

        self.assertAlmostEqual(result.objective_value, 0.0)
