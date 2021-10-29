from typing import List
from ror.Constraint import Constraint, ConstraintVariable
from ror.Dataset import Dataset


def get_vector(
        vector_variable_name: str,
        alternative: str,
        coefficient: float,
        dataset: Dataset,
        is_binary: bool = False) -> List[ConstraintVariable]:
    return [
        ConstraintVariable(
            Constraint.create_variable_name(
                vector_variable_name, criterion_name, alternative),
            coefficient,
            alternative,
            is_binary
        )
        for (criterion_name, _)
        in dataset.criteria
    ]


def get_lambda_variable(alternative: str, coefficient: float = 1.0) -> ConstraintVariable:
    return ConstraintVariable(
        Constraint.create_variable_name('lambda', 'all', alternative),
        coefficient,
        alternative
    )
