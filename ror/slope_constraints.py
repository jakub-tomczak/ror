import logging
from ror.Relation import Relation
from ror.Dataset import Dataset
from typing import List, Tuple
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet, ValueConstraintVariable
import numpy as np


# difference of 2 values greater than DIFF_EPS indicates that they are different
DIFF_EPS = 1e-10


def check_preconditions(data: Dataset) -> bool:
    if len(data.alternatives) < 3:
        logging.info('number of alternatives is lower than 3, skipping slope constraint')
        return False
    return True


def _create_slope_constraint(
        alternative_index: int,
        data: Dataset,
        criterion_name: str,
        relation: Relation,
        alternatives: List[str],
        alternative_scores: List[float]) -> Tuple[Constraint, Constraint]:
    '''
    Returns slope constraint or None if there would be division by 0 (in case when g_i(l) == g_i(l-1) or g_i(l-1) == g_i(l-2))
    Slope constraint is meeting the requirement | z - w | <= rho
    This constraint minimizes the differences between 2 consecutive characteristic points.
    This constraint requires partial utility function to be monotonic, non-decreasing
    '''
    first_diff = alternative_scores[alternative_index] - alternative_scores[alternative_index-1]
    # check if the 2 following points are not in the same place
    if abs(first_diff) < DIFF_EPS:
        logging.info(
            f'Criterion {criterion_name} for alternative {alternatives[alternative_index]} has the same value ({alternative_scores[alternative_index-1]}) as alternative {alternatives[alternative_index-1]} on this criterion.')
        return None
    first_coeff = 1 / (first_diff)
    second_diff = alternative_scores[alternative_index-1] - alternative_scores[alternative_index-2]
    # check if the 2 following points are not in the same place
    if abs(second_diff) < DIFF_EPS:
        logging.info(
            f'Criterion {criterion_name} for alternative {alternatives[alternative_index-1]} has the same value ({alternatives[alternative_index-2]}) as alternative {alternatives[alternative_index-2]} on this criterion.')
        return None
    second_coeff = 1 / (second_diff)

    delta_constraint = ConstraintVariable(
        "delta",
        -1.0
    ) if data.delta is None else ValueConstraintVariable(
        data.delta
    )

    # create constraint
    first_constraint = Constraint(ConstraintVariablesSet([
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, alternatives[alternative_index]),
            first_coeff,
            alternatives[alternative_index]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, alternatives[alternative_index-1]),
            -first_coeff,
            alternatives
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, alternatives[alternative_index-1]),
            -second_coeff,
            alternatives[alternative_index-1]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, alternatives[alternative_index-2]),
            second_coeff,
            alternatives[alternative_index-2]
        ),
        delta_constraint
    ]), relation, Constraint.create_variable_name("first_slope", criterion_name, alternative_index))

    second_constraint = Constraint(ConstraintVariablesSet([
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, alternatives[alternative_index]),
            -first_coeff,
            alternatives[alternative_index]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, alternatives[alternative_index-1]),
            first_coeff,
            alternatives[alternative_index-1]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, alternatives[alternative_index-1]),
            second_coeff,
            alternatives[alternative_index-1]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, alternatives[alternative_index-2]),
            -second_coeff,
            alternatives[alternative_index-2]
        ),
        delta_constraint
    ]), relation, Constraint.create_variable_name("second_slope", criterion_name, alternative_index))

    return (first_constraint, second_constraint)


def create_slope_constraints(data: Dataset, relation: Relation = None) -> List[Constraint]:
    '''
    Returns slope constraints for all alternatives except the ones that have duplicated
    values in the criterion space.
    So the number of constraints will be
    2 x criteria + (m-2)*2
    where 'm' is the number of alternatives without duplicated data on each criterion
    and 'criteria' is the number of criteria in the data.
    '''
    if not check_preconditions(data):
        return []

    if relation is None:
        relation = Relation('<=')
    constraints = []
    for criterion_index, (criterion_name, _) in enumerate(data.criteria):
        alternative_score_on_criterion = data.matrix[:, criterion_index]

        for l in range(2, len(data.alternatives)):
            slope_constraints = _create_slope_constraint(
                l, data, criterion_name, relation, data.alternatives, alternative_score_on_criterion 
            )
            if slope_constraints is not None:
                first_constraint, second_constraint = slope_constraints
                constraints.append(first_constraint)
                constraints.append(second_constraint)
    return constraints
