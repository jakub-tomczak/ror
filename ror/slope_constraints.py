from ror.Relation import Relation
from ror.Dataset import Dataset
from typing import Tuple
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet

DIFF_EPS = 1e-10


def check_preconditions(data: Dataset) -> bool:
    if len(data.alternatives) < 3:
        print('number of alternatives is lower than 3, skipping slope constraint')
        return False
    return True


def _create_slope_constraint(l: int, data: Dataset, criterion_name: str) -> Tuple[Constraint, Constraint]:
    '''
    Returns slope constraint or None if there would be division by 0 (in case when g_i(l) == g_i(l-1) or g_i(l-1) == g_i(l-2))
    Slope constraint is meeting the requirement | z - w | <= rho
    This constraint minimizes the differences between 2 consecutive characteristic points.
    This constraint requires partial utility function to be monotonic, non-decreasing
    '''
    first_diff = data.get_data_for_alternative_and_criterion(data.alternatives[l], criterion_name).coefficient -\
        data.get_data_for_alternative_and_criterion(
            data.alternatives[l-1], criterion_name).coefficient
    # check if the 2 following points are not in the same place
    if abs(first_diff) < DIFF_EPS:
        print(
            f'Criterion {criterion_name} for alternative {data.alternatives[l]} has the same value ({data.get_data_for_alternative_and_criterion(data.alternatives[l], criterion_name).coefficient}) as alternative {data.alternatives[l-1]} on this criterion.')
        return None
    first_coeff = 1 / (first_diff)
    second_diff = data.get_data_for_alternative_and_criterion(data.alternatives[l-1], criterion_name).coefficient -\
        data.get_data_for_alternative_and_criterion(
            data.alternatives[l-2], criterion_name).coefficient
    # check if the 2 following points are not in the same place
    if abs(second_diff) < DIFF_EPS:
        print(
            f'Criterion {criterion_name} for alternative {data.alternatives[l-1]} has the same value ({data.get_data_for_alternative_and_criterion(data.alternatives[l], criterion_name).coefficient}) as alternative {data.alternatives[l-2]} on this criterion.')
        return None
    second_coeff = 1 / (second_diff)
    # create constraint
    first_constraint = Constraint(ConstraintVariablesSet([
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, data.alternatives[l]),
            first_coeff,
            data.alternatives[l]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, data.alternatives[l-1]),
            -first_coeff,
            data.alternatives[l-1]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, data.alternatives[l-1]),
            -second_coeff,
            data.alternatives[l-1]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, data.alternatives[l-2]),
            second_coeff,
            data.alternatives[l-2]
        ),
        ConstraintVariable(
            "delta",
            -1.0
        )
    ]), Relation("<="), Constraint.create_variable_name("first_slope", criterion_name, l))

    second_constraint = Constraint(ConstraintVariablesSet([
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, data.alternatives[l]),
            -first_coeff,
            data.alternatives[l]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, data.alternatives[l-1]),
            first_coeff,
            data.alternatives[l-1]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, data.alternatives[l-1]),
            second_coeff,
            data.alternatives[l-1]
        ),
        ConstraintVariable(
            Constraint.create_variable_name(
                'u', criterion_name, data.alternatives[l-2]),
            -second_coeff,
            data.alternatives[l-2]
        ),
        ConstraintVariable(
            "delta",
            -1.0
        )
    ]), Relation("<="), Constraint.create_variable_name("second_slope", criterion_name, l))

    return (first_constraint, second_constraint)


def create_slope_constraints(data: Dataset):
    '''
    Returns slope constraints for all alternatives except the ones that have duplicated
    values in the criterion space.
    So the number of constraints will be
    2 x criteria + (n-2)*2
    where 'n' is the number of alternatives without duplicated data on each criterion
    and 'criteria' is the number of criteria in the data.
    '''
    if not check_preconditions(data):
        return []

    constraints = []
    for criterion_name, _ in data.criteria:
        for l in range(2, len(data.alternatives)):
            slope_constraints = _create_slope_constraint(
                l, data, criterion_name
            )
            if slope_constraints is not None:
                first_constraint, second_constraint = slope_constraints
                constraints.append(first_constraint)
                constraints.append(second_constraint)
    return constraints
