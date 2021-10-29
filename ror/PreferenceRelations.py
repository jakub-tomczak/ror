from typing import Set
from ror.auxiliary_variables import get_lambda_variable
from ror.dataset_constants import ALL_CRITERIA
from ror.Relation import INDIFFERENCE, PREFERENCE, Relation, WEAK_PREFERENCE
from ror.Constraint import Constraint, ConstraintVariable, ValueConstraintVariable, ConstraintVariablesSet


class PreferenceCriterion:
    ALL = -1


class Preference:
    def __init__(self, function_name: str, relation: Relation) -> None:
        assert function_name != '',\
            'Name of the function that this preference represents must not be empty'
        self._function_name = function_name
        self._alpha = None

        assert type(relation) is Relation, f"relation must be Relation type"
        self._relation = relation

    @property
    def relation(self) -> Relation:
        return self._relation

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha: float):
        assert new_alpha is not None, "alpha must not be None"
        assert 0 <= new_alpha <= 1, "alpha must be in range <0, 1>"
        self._alpha = new_alpha


class PreferenceRelation(Preference):
    def __init__(
            self,
            alternative_1: str,
            alternative_2: str,
            relation: Relation,
            preference_criterion: int = PreferenceCriterion.ALL):
        assert any([relation == WEAK_PREFERENCE, relation == PREFERENCE, relation == INDIFFERENCE]),\
            f"relation object must be of WEAK_PREFERENCE, PREFERENCE or INDIFFERENCE"
        super().__init__(relation._name, relation)

        self._alternative_1: str = alternative_1
        self._alternative_2: str = alternative_2
        self._criterion: int = preference_criterion

    @property
    def alternative_1(self) -> str:
        return self._alternative_1

    @property
    def alternative_2(self) -> str:
        return self._alternative_2

    @property
    def alternatives(self) -> Set[str]:
        return set([self._alternative_1, self._alternative_2])

    def __repr__(self):
        return f"<Preference: {self._alternative_1} {self._relation} {self._alternative_2}>"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PreferenceRelation):
            return False
        return o.alternative_1 == self.alternative_1 \
            and o.alternative_2 == self.alternative_2 \
            and o.relation == self.relation \
            and o._criterion == self._criterion

    def __hash__(self) -> int:
        return 13 * self._alternative_1.__hash__() + 19 * self._alternative_2.__hash__() \
            + 23 * self._criterion.__hash__() + 31 * self._relation.__hash__()

    def to_constraint(self, dataset: 'Dataset', alpha: float) -> Constraint:
        '''
        Creates Constraint object for this relation on a specific criterion.
        '''
        self.alpha = alpha

        constraint = Constraint(
            ConstraintVariablesSet(),
            self._relation,
            '{} {} {}'.format(
                Constraint.create_variable_name(
                    self._function_name,
                    ALL_CRITERIA,
                    self._alternative_2
                ),
                self._relation.sign,
                Constraint.create_variable_name(
                    self._function_name,
                    ALL_CRITERIA,
                    self._alternative_1
                )
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
            preference_criterion: int = PreferenceCriterion.ALL):
        super().__init__('d_intens', relation)

        self._alternative_1 = alternative_1
        self._alternative_2 = alternative_2
        self._alternative_3 = alternative_3
        self._alternative_4 = alternative_4
        self._preference_criterion = preference_criterion

    @property
    def alternative_1(self) -> str:
        return self._alternative_1

    @property
    def alternative_2(self) -> str:
        return self._alternative_2

    @property
    def alternative_3(self) -> str:
        return self._alternative_3

    @property
    def alternative_4(self) -> str:
        return self._alternative_4
    
    @property
    def alternatives(self) -> Set[str]:
        return set([
            self._alternative_1,
            self._alternative_2,
            self._alternative_3,
            self._alternative_4
        ])

    def __repr__(self):
        return f"<Preference: {self._alternative_1} - {self._alternative_2} {self._relation} {self._alternative_3} - {self._alternative_4}>"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PreferenceIntensityRelation):
            return False
        return o.alternative_1 == self.alternative_1 \
            and o.alternative_2 == self.alternative_2 \
            and o.alternative_3 == self.alternative_3 \
            and o.alternative_4 == self.alternative_4 \
            and o.relation == self.relation \
            and o._preference_criterion == self._preference_criterion

    def __hash__(self) -> int:
        return 13 * self._alternative_1.__hash__() + 19 * self._alternative_2.__hash__() \
            + 37 * self._alternative_3.__hash__() + 31 * self._alternative_4.__hash__() \
            + 23 * self._preference_criterion.__hash__() + 41 * self._relation.__hash__()

    def to_constraint(self, dataset: 'Dataset', alpha: float) -> Constraint:
        '''
        Creates Constraint object for this relation on a specific criterion.
        PREFERENCE relation creates a relation
        '''
        self.alpha = alpha
        # -d(a_k) + d(a_l) + d(a_p) - d(a_q) <= -eps
        # -d(alternative_1) + d(alternative_2) + d(alternative_3) - d(alternative_4) <= -eps

        constraint = Constraint(
            ConstraintVariablesSet(),
            self._relation,
            Constraint.create_variable_name(
                self._function_name, ALL_CRITERIA,
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
