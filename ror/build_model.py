import numpy as np
from typing import List, Tuple, Dict
from ror.Constraint import Constraint
from ror.Dataset import criterion_types


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
                Constraint(
                    [1, -1],
                    [
                        f'{Constraint.alternatives_prefix}{row_index}',
                        f'{Constraint.alternatives_prefix}{best_value_index}'
                    ],
                    "<=",
                    0.0,
                    f"mono_{Constraint.alternatives_prefix}{row_index}_{Constraint.alternatives_prefix}{best_value_index}"
                )
            )
        constraints[criterion_name] = _constraints
    return constraints
