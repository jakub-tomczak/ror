from ror.auxiliary_variables import get_lambda_variable
from typing import List
from ror.Constraint import Constraint, ConstraintVariablesSet, ConstraintVariable, ValueConstraintVariable
from ror.Dataset import Dataset


def d_sum(alternative: str, alpha: float, dataset: Dataset) -> List[ConstraintVariable]:
    return [
        ConstraintVariable(
            Constraint.create_variable_name('u', criterion_name, alternative),
            alpha * (-1)
        )
        for (criterion_name, _)
        in dataset.criteria
    ]


def d(alternative: str, alpha: float, dataset: Dataset) -> ConstraintVariablesSet:
    '''
    Creates a set of variables that represents d* function.
    '''
    assert 0 <= alpha <= 1.0, f'alpha value must be in range <0, 1>, {alpha} provided.'
    assert dataset is not None, 'dataset must not be None'
    assert alternative in dataset.alternatives,\
        f'provided alternative "{alternative}" doesn\'t exist in provided dataset'

    sum_constraint_variables = d_sum(alternative, alpha, dataset)
    lambda_constraint_variables: List[ConstraintVariable] = [
        get_lambda_variable(alternative, criterion_name, coefficient=1-alpha)
        for criterion_name, _
        in dataset.criteria
    ]
    variables = ConstraintVariablesSet(
        [*sum_constraint_variables, *lambda_constraint_variables],
        f'd*({alternative})'
    )
    free_variable = ValueConstraintVariable(alpha * len(dataset.criteria))
    variables.add_variable(free_variable)
    return variables
