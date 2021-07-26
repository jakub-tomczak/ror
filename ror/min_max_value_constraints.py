from ror.Relation import INDIFFERENCE, Relation
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet, ValueConstraintVariable
from ror.Dataset import Dataset
from typing import List
import numpy as np


def create_min_value_constraints(dataset: Dataset) -> List[Constraint]:
    assert dataset is not None, "dataset cannot be None"
    assert len(
        dataset.alternatives) > 0, "number of alternatives in the dataset must be greater than 0"
    worst_values = []

    for column, (criterion_name, _) in zip(dataset.matrix.T, dataset.criteria):
        # sort data in a column and save indices
        _data = np.sort(column)
        _data_indices = np.argsort(column)

        # reverse data in sorted column,
        # if criterion is of gain type (ascending sort by default)
        # if criterion is of cost type (cost criterion has all values multiplied by -1)
        _data = _data[::-1]
        _data_indices = _data_indices[::-1]

        worst_value_index = _data_indices[-1]

        worst_values.append(Constraint(
            ConstraintVariablesSet([
                ConstraintVariable(
                    Constraint.create_variable_name(
                        'u', criterion_name, dataset.alternatives[worst_value_index]),
                    1.0
                )
            ]),
            Relation("=="),
            f"worst_value_on_criterion_{criterion_name}"
        ))
    return worst_values


def create_max_value_constraint(dataset: Dataset) -> Constraint:
    assert dataset is not None, "dataset cannot be None"
    assert len(
        dataset.alternatives) > 0, "number of alternatives in the dataset must be greater than 0"

    constraint_variables = []
    for column, (criterion_name, _) in zip(dataset.matrix.T, dataset.criteria):
        # sort data in a column and save indices
        _data = np.sort(column)
        _data_indices = np.argsort(column)

        # reverse data in sorted column,
        # if criterion is of gain type (ascending sort by default)
        # if criterion is of cost type (cost criterion has all values multiplied by -1)
        _data = _data[::-1]
        _data_indices = _data_indices[::-1]

        best_value_index = _data_indices[0]

        constraint_variables.append(ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, dataset.alternatives[best_value_index]),
            1.0,
            dataset.alternatives[best_value_index]
        ))

    constraint = Constraint(
        ConstraintVariablesSet(constraint_variables),
        Relation('=='),
        f"max_value_constraint"
    )
    constraint.add_variable(ValueConstraintVariable(1.0))

    return constraint
