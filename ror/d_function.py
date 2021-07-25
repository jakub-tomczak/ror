from ror.auxiliary_variables import get_vector
from typing import List
from ror.Constraint import Constraint, ConstraintVariablesSet, ConstraintVariable, ValueConstraintVariable
from ror.Dataset import Dataset


def d_sum(alternative: str, alpha: float, dataset: Dataset) -> List[ConstraintVariable]:
    return [
        ConstraintVariable(
            Constraint.create_variable_name('u', criterion_name, alternative),
            alpha * (-1) * alternative_values._coefficient
        )
        for (criterion_name, _), alternative_values
        in zip(dataset.criteria, dataset.get_data_for_alternative(alternative))
    ]


def d(alternative: str, alpha: float, dataset: Dataset) -> ConstraintVariablesSet:
    '''
    Creates a set of variables that represents d* function.
    '''
    sum_constraint_variables = d_sum(alternative, alpha, dataset)
    lambda_constraint_variables = get_vector(
        'lambda', alternative, 1-alpha, dataset)
    variables = ConstraintVariablesSet(
        [*sum_constraint_variables, *lambda_constraint_variables],
        f'd*({alternative})'
    )
    free_variable = ValueConstraintVariable(alpha * len(dataset.criteria))
    variables.add_variable(free_variable)
    return variables
