from ror.auxiliary_variables import get_lambda_variable
from ror.helpers import reduce_lists
from typing import List
from ror.Relation import INDIFFERENCE, PREFERENCE, Relation, WEAK_PREFERENCE
from ror.Constraint import Constraint, ConstraintVariable, ValueConstraintVariable, ConstraintVariablesSet
from ror.Dataset import Dataset


class PreferenceCriterion:
    ALL = -1


class Preference:
    def __init__(self, function_name: str, alpha: float) -> None:
        assert function_name != '',\
            'Name of the function that this preference represents must not be empty'
        assert 0 <= alpha <= 1, "alpha must be in range <0, 1>"

        self._function_name = function_name
        self._alpha = alpha


class PreferenceRelation(Preference):
    def __init__(
            self,
            alternative_1: str,
            alternative_2: str,
            relation: Relation,
            alpha: float,
            preference_criterion: int = PreferenceCriterion.ALL):
        assert type(relation) is Relation,\
            f"relation object must be of Relation type"
        assert any([relation == WEAK_PREFERENCE, relation == PREFERENCE, relation == INDIFFERENCE]),\
            f"relation object must be of WEAK_PREFERENCE, PREFERENCE or INDIFFERENCE"
        super().__init__(relation._name, alpha)

        self._alternative_1: str = alternative_1
        self._alternative_2: str = alternative_2
        self._relation: Relation = relation
        self._criterion: int = preference_criterion

    def __repr__(self):
        return f"<Preference: {self._alternative_1} {self._relation} {self._alternative_2}>"

    def to_constraint(self, dataset: Dataset) -> Constraint:
        '''
        Creates Constraint object for this relation on a specific criterion.
        '''

        constraint = Constraint(
            ConstraintVariablesSet(),
            self._relation,
            Constraint.create_variable_name(
                self._function_name, Dataset.ALL_CRITERIA,
                f'{self._alternative_2}_{self._relation.sign}_{self._alternative_1}'
            )
        )

        # create constraints for d_k
        constraint.add_variables(ConstraintVariablesSet([
            ConstraintVariable(
                Constraint.create_variable_name(
                    'u', criterion_name, self._alternative_1),
                self._alpha * (-1),
                self._alternative_1
            )
            for criterion_name, _
            in dataset.criteria
        ]))
        constraint.add_variable(get_lambda_variable(
            self._alternative_1, 1 - self._alpha))

        constraint.add_variable(
            ValueConstraintVariable(-1 * self._alpha * len(dataset.criteria))
        )

        # create constraints for d_l
        constraint.add_variables(ConstraintVariablesSet([
            ConstraintVariable(
                Constraint.create_variable_name(
                    'u', criterion_name, self._alternative_2),
                self._alpha,
                self._alternative_2
            )
            for criterion_name, _
            in dataset.criteria
        ]))
        constraint.add_variable(get_lambda_variable(
            self._alternative_2, -1 * (1 - self._alpha)))

        constraint.add_variable(
            ValueConstraintVariable(self._alpha * len(dataset.criteria))
        )

        # use eps for PREFERENCE relation, 0 otherwise
        rhs = dataset.eps if self._relation == PREFERENCE else 0
        constraint.add_variable(ValueConstraintVariable(rhs))

        return constraint


class PreferenceIntensityRelation(Preference):
    def __init__(
            self,
            alternative_1: str,
            alternative_2: str,
            alternative_3: str,
            alternative_4: str,
            relation: Relation,
            alpha: float,
            preference_criterion: int = PreferenceCriterion.ALL):
        super().__init__('d_intens', alpha)

        assert type(relation) is Relation, f"relation must be Relation type"

        self._alternative_1 = alternative_1
        self._alternative_2 = alternative_2
        self._alternative_3 = alternative_3
        self._alternative_4 = alternative_4
        self._relation = relation
        self._preference_criterion = preference_criterion

    def __repr__(self):
        return f"<Preference: {self._alternative_1} - {self._alternative_2} {self._relation} {self._alternative_3} - {self._alternative_4}>"

    def to_constraint(self, dataset: Dataset) -> Constraint:
        '''
        Creates Constraint object for this relation on a specific criterion.
        PREFERENCE relation creates a relation
        '''
        # -d(a_k) + d(a_l) + d(a_p) - d(a_q) <= -eps
        # -d(alternative_1) + d(alternative_2) + d(alternative_3) - d(alternative_4) <= -eps

        constraint = Constraint(
            ConstraintVariablesSet(),
            self._relation,
            Constraint.create_variable_name(
                self._function_name, Dataset.ALL_CRITERIA,
                f'{self._alternative_2}_{self._alternative_1}'
            )
        )

        # create constraints for -d(a_k)
        constraint.add_variables(ConstraintVariablesSet([
            ConstraintVariable(
                Constraint.create_variable_name(
                    'u', criterion_name, self._alternative_1),
                self._alpha,
                self._alternative_1
            )
            for criterion_name, _
            in dataset.criteria
        ]))
        constraint.add_variable(get_lambda_variable(
            self._alternative_1, -1 * (1 - self._alpha)))

        constraint.add_variable(
            ValueConstraintVariable(self._alpha * len(dataset.criteria))
        )

        # create constraints for d(a_l)
        constraint.add_variables(ConstraintVariablesSet([
            ConstraintVariable(
                Constraint.create_variable_name(
                    'u', criterion_name, self._alternative_2),
                -1 * self._alpha,
                self._alternative_2
            )
            for criterion_name, _
            in dataset.criteria
        ]))
        constraint.add_variable(get_lambda_variable(
            self._alternative_2, (1 - self._alpha)))
        constraint.add_variable(
            ValueConstraintVariable(-1 * self._alpha * len(dataset.criteria))
        )

        # create constraints for d(a_p)
        constraint.add_variables(ConstraintVariablesSet(
            [
                ConstraintVariable(
                    Constraint.create_variable_name(
                        'u', criterion_name, self._alternative_3),
                    -1 * self._alpha,
                    self._alternative_3
                )
                for criterion_name, _
                in dataset.criteria
            ])
        )
        constraint.add_variable(get_lambda_variable(
            self._alternative_3, (1 - self._alpha)))
        constraint.add_variable(
            ValueConstraintVariable(-1 * self._alpha * len(dataset.criteria))
        )

        # create constraints for -d(a_q)
        constraint.add_variables(ConstraintVariablesSet(
            [
                ConstraintVariable(
                    Constraint.create_variable_name(
                        'u', criterion_name, self._alternative_4),
                    self._alpha,
                    self._alternative_4
                )
                for criterion_name, _
                in dataset.criteria
            ]
        ))
        constraint.add_variable(get_lambda_variable(
            self._alternative_4, -1 * (1 - self._alpha)))
        constraint.add_variable(
            ValueConstraintVariable(self._alpha * len(dataset.criteria))
        )

        # use eps for PREFERENCE relation, 0 otherwise (WEAK PREFERENCE and INDIFFERENCE)
        rhs = -1 * dataset.eps if self._relation == PREFERENCE else 0
        constraint.add_variable(ValueConstraintVariable(rhs))

        return constraint
