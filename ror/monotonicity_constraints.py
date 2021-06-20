from ror.Relation import Relation
import numpy as np
from typing import List, Dict
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet
from ror.Dataset import Dataset, criterion_types


def create_monotonicity_constraints(dataset: Dataset) -> Dict[str, List[Constraint]]:
    assert dataset is not None, "dataset cannot be None"
    data = dataset.matrix
    criteria = dataset.criteria
    constraints = dict()

    for column, (criterion_name, criterion_type) in zip(data.T, criteria):
        _constraints = []
        # sort data in a column and save indices
        _data = np.sort(column)
        _data_indices = np.argsort(column)

        # reverse data in sorted column,
        # if criterion is of gain type (ascending sort by default)
        if criterion_type == criterion_types["gain"]:
            _data = _data[::-1]
            _data_indices = _data_indices[::-1]

        best_value_index = _data_indices[0]
        # iterate over all values, skipping the best (first) value
        # for row, row_index in zip(_data[1:], _data_indices[1:]):
        for row_index in _data_indices[1:]:
            _constraints.append(
                Constraint(ConstraintVariablesSet([
                    ConstraintVariable(
                        Constraint.create_variable_name(
                            'u', criterion_name, row_index),
                        -1,
                    ),
                    ConstraintVariable(
                        Constraint.create_variable_name(
                            'u', criterion_name, best_value_index),
                        -1,
                    )
                ]),
                    Relation('<=', 'monotonicity relation'),
                    f"mono_{Constraint.create_variable_name('u', criterion_name, row_index)}_{Constraint.create_variable_name('u', criterion_name, best_value_index)}"
                )
            )
        constraints[criterion_name] = _constraints
    return constraints
