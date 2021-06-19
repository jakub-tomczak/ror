from ror.Relation import Relation
import numpy as np
from typing import List, Tuple, Dict
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet
from ror.Dataset import criterion_types
from ror.PreferenceRelations import PreferenceRelation


def create_monotonicity_constraints(data, criteria: List[Tuple[str, str]]) -> Dict[str, List[Constraint]]:
    assert type(data) is np.ndarray, "data must be a numpy array"
    assert len(criteria) == data.shape[1], \
        "number of columns in data must correspond to the number of criteria"

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
                    Relation('monotonicity relation', '<='),
                    f"mono_{Constraint.create_variable_name('u', criterion_name, row_index)}_{Constraint.create_variable_name('u', criterion_name, best_value_index)}"
                )
            )
        constraints[criterion_name] = _constraints
    return constraints


def create_preference_constraints(
    data,
    criteria: List[Tuple[str, str]],
    preferences: List[PreferenceRelation]
) -> Dict[str, List[Constraint]]:
    assert type(data) is np.ndarray, "data must be a numpy array"
    assert len(criteria) == data.shape[1], \
        "number of columns in data must correspond to the number of criteria"

    return [preference.to_constraint() for preference in preferences]


def create_preference__intensities_constraints(
    data,
    criteria: List[Tuple[str, str]],
    preferences: Tuple[List[str], List[str], str]
) -> Dict[str, List[Constraint]]:
    pass
