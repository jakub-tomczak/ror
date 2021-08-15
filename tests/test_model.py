from ror.Relation import Relation
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet, ValueConstraintVariable
from ror.data_loader import read_dataset_from_txt
import unittest
from ror.Model import Model


class TestModel(unittest.TestCase):
    def test_creating_ror_model_with_incorrect_target(self):
        model = Model()
        # raise assertion error as an objective cannot be set
        # when there is any constraint
        with self.assertRaises(AssertionError):
            model.target = ConstraintVariablesSet([
                ConstraintVariable("delta", 1.0)
            ])

    def test_creating_ror_model_with_correct_target(self):
        model = Model([
            # only constraint
            # 3.0*delta + 3.0*u_1_a1 <= 0
            Constraint(
                ConstraintVariablesSet([
                    ConstraintVariable("delta", 3.0),
                    ConstraintVariable("u_1_a1", 3.0),
                    ValueConstraintVariable(0.0)
                ]),
                Relation("<=", "Some relation")
            )
        ])
        model.target = ConstraintVariablesSet([
            ConstraintVariable("delta", 1.0)
        ])