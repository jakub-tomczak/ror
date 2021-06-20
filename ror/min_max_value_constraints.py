
from ror.Relation import INDIFFERENCE, Relation
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet, ValueConstraintVariable
from ror.Dataset import Dataset, criterion_types
from typing import List
import numpy as np


def create_min_value_constraints(dataset: Dataset) -> List[Constraint]:
    worst_values = []

    for column, (criterion_name, criterion_type) in zip(dataset.matrix.T, dataset.criteria):
        # sort data in a column and save indices
        _data = np.sort(column)
        _data_indices = np.argsort(column)

        # reverse data in sorted column,
        # if criterion is of gain type (ascending sort by default)
        if criterion_type == criterion_types["gain"]:
            _data = _data[::-1]
            _data_indices = _data_indices[::-1]

        worst_value_index = _data_indices[-1]

        worst_values.append(Constraint(
            ConstraintVariablesSet([
                ConstraintVariable(dataset.alternatives[worst_value_index], 1.0)
            ]),
            Relation("=="),
            f"worst_value_on_criterion_{criterion_name}"
        ))
    return worst_values


def create_max_value_constraint(dataset: Dataset) -> Constraint:
    best_alternative_names = []

    for column, (_, criterion_type) in zip(dataset.matrix.T, dataset.criteria):
        # sort data in a column and save indices
        _data = np.sort(column)
        _data_indices = np.argsort(column)

        # reverse data in sorted column,
        # if criterion is of gain type (ascending sort by default)
        if criterion_type == criterion_types["gain"]:
            _data = _data[::-1]
            _data_indices = _data_indices[::-1]

        best_value_index = _data_indices[0]

        best_alternative_names.append(dataset.alternatives[best_value_index])
        
    constraint = Constraint(ConstraintVariablesSet(
        [ConstraintVariable(variable_name, 1.0) for variable_name in best_alternative_names],
    ), Relation('=='), "max value constraint")
    constraint.add_variable(ValueConstraintVariable(1.0))

    return constraint
