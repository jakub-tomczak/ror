from ror.Relation import Relation
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet, ValueConstraintVariable
from ror.auxiliary_variables import get_lambda_variable, get_vector
from ror.Dataset import Dataset, RORDataset
from typing import List, Set
from ror.dataset_constants import DEFAULT_M


def create_inner_maximization_constraint_for_alternative(data: Dataset, alternative: str) -> List[Constraint]:
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
                ConstraintVariable(
                    Constraint.create_variable_name('c', criterion_name, alternative),
                    -DEFAULT_M,
                    alternative,
                    is_binary=True
                ),
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
                ConstraintVariable(
                    Constraint.create_variable_name('c', criterion_name, alternative),
                    -DEFAULT_M,
                    alternative,
                    is_binary=True
                ),
                ValueConstraintVariable(1.0)
            ]),
            Relation("<="),
            f"3rd_inner_maximization_criterion_{criterion_name}_alternative_{alternative}"
        )
        constraints.append(third_constraint)

    c_vector = [
        ConstraintVariable(
            Constraint.create_variable_name('c', _criterion, alternative),
            1.0,
            alternative,
            is_binary=True
        ) for _criterion, _ in data.criteria
    ]
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

    constraints: List[Constraint] = []
    # for inner maximization take only alternatives a_k such that
    # a_k \in A^{R}
    reference_alternatives: Set[str] = set()
    for relation in data.preferenceRelations:
        reference_alternatives.update(relation.alternatives)
    for intensity_relation in data.intensityRelations:
        reference_alternatives.update(intensity_relation.alternatives)
    for alternative_name in reference_alternatives:
        for constraint in create_inner_maximization_constraint_for_alternative(data, alternative_name):
            constraints.append(constraint)
    return constraints
