from ror.Relation import Relation
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet, ValueConstraintVariable
from ror.auxiliary_variables import get_lambda_variable, get_vector
from ror.Dataset import Dataset, RORDataset
from typing import List
from ror.dataset_constants import DEFAULT_M


def _create_inner_maximization_constraint_for_alternative(data: Dataset, alternative: str) -> List[Constraint]:
    # c vector is a vector of binary value variables
    c_vector = get_vector("c", alternative, 1.0, data, True)

    constraints: List[Constraint] = []

    for criterion_index in range(len(data.criteria)):
        criterion_name, _ = data.criteria[criterion_index]

        first_constraint = Constraint(
            ConstraintVariablesSet([
                get_lambda_variable(alternative, coefficient=-1.0),
                ConstraintVariable(
                    Constraint.create_variable_name(
                        "u", criterion_name, alternative),
                    -1.0,
                    alternative
                ),
                ValueConstraintVariable(-1.0)
            ]),
            Relation("<="),
            f"1st_inner_maximization_criterion_{criterion_name}_alternative_{alternative}"
        )
        constraints.append(first_constraint)

        second_constraint = Constraint(
            ConstraintVariablesSet([
                get_lambda_variable(alternative, coefficient=-1.0),
                ConstraintVariable(
                    Constraint.create_variable_name(
                        "u", criterion_name, alternative),
                    -1.0,
                    alternative
                ),
                c_vector[criterion_index].with_coefficient(-DEFAULT_M),
                ValueConstraintVariable(-1.0)
            ]),
            Relation("<="),
            f"2nd_inner_maximization_criterion_{criterion_name}_alternative_{alternative}"
        )
        constraints.append(second_constraint)

        third_constraint = Constraint(
            ConstraintVariablesSet([
                get_lambda_variable(alternative, coefficient=1.0),
                ConstraintVariable(
                    Constraint.create_variable_name(
                        "u", criterion_name, alternative),
                    1.0,
                    alternative
                ),
                c_vector[criterion_index].with_coefficient(-DEFAULT_M),
                ValueConstraintVariable(1.0)
            ]),
            Relation("<="),
            f"3rd_inner_maximization_criterion_{criterion_name}_alternative_{alternative}"
        )
        constraints.append(third_constraint)

    sum_constraint = Constraint(
        ConstraintVariablesSet([
            # unpack all c variables from vector
            *c_vector,
            ValueConstraintVariable(len(data.criteria)-1)
        ]),
        Relation("<="),
        f"sum_binary_var_c_({alternative})"
    )
    constraints.append(sum_constraint)

    return constraints


def create_inner_maximization_constraints(data: RORDataset) -> List[Constraint]:
    assert data is not None, "dataset must not be none"

    constraints = []
    for alternative_name in data.alternatives:
        for constraint in _create_inner_maximization_constraint_for_alternative(data, alternative_name):
            constraints.append(constraint)
    return constraints
