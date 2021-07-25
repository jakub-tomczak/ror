from typing import List
from ror.Constraint import Constraint, ConstraintVariable
from ror.Dataset import Dataset


def get_vector(vector_variable_name: str, alternative: str, coefficient: float, dataset: Dataset) -> List[ConstraintVariable]:
    return [
        ConstraintVariable(
            Constraint.create_variable_name(
                vector_variable_name, criterion_name, alternative),
            coefficient
        )
        for (criterion_name, _)
        in dataset.criteria
    ]
