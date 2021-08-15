from ror.Relation import Relation
import numpy as np
from typing import List, Dict
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet
from ror.Dataset import Dataset


def create_monotonicity_constraints(dataset: Dataset) -> Dict[str, List[Constraint]]:
    '''
    Sort each criterion in descending order. Assume best value is on the 0th index,
    create constraints by taking all alternatives and comparing it with the best value.
    If values are [a1:0, a2:6, a3: 5, a4: 10], then they will be sorted to
    [a4: 10, a2: 6, a3: 5, a1: 0], and there will be 3 constraints returned, starting from
    u1(a4) >= u1(a2) and then normalized to -u1(a4) + u1(a2) <= 0
    '''
    assert dataset is not None, "dataset cannot be None"
    assert len(
        dataset.alternatives) > 1, "number of alternatives in the dataset must be greater than 1"
    data = dataset.matrix
    criteria = dataset.criteria
    constraints = dict()

    for column, (criterion_name, _) in zip(data.T, criteria):
        _constraints = []
        # sort data (asceding order) in a column
        _data = np.sort(column)
        # sort indices (asceding order) in a column
        _data_indices = np.argsort(column)

        # reverse data in sorted column
        _data = _data[::-1]
        _data_indices = _data_indices[::-1]

        # iterate over all alternatives' values in the criterion,
        # skipping the best (first) value
        for idx in range(len(_data_indices)-1):
            better_variable_name = Constraint.create_variable_name(
                'u', criterion_name, dataset.alternatives[_data_indices[idx]])
            worst_variable_name = Constraint.create_variable_name(
                'u', criterion_name, dataset.alternatives[_data_indices[idx+1]])
            _constraints.append(
                Constraint(
                    ConstraintVariablesSet([
                        ConstraintVariable(worst_variable_name, 1.0,
                                           dataset.alternatives[_data_indices[idx]]),
                        ConstraintVariable(
                            better_variable_name, -1.0, dataset.alternatives[_data_indices[idx+1]])
                    ]),
                    Relation('<=', 'monotonicity relation'),
                    f"mono_{better_variable_name}_{worst_variable_name}"
                )
            )
        constraints[criterion_name] = _constraints
    return constraints
