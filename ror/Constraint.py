from __future__ import annotations
from ror.Relation import Relation
from typing import Dict, List, Set
from io import StringIO


class ConstraintVariable:
    '''
    Class that stores a coefficient and the name of the variable.
    '''

    def __init__(self, name: str, coefficient: float, alternative: str = None, is_binary: bool = False) -> None:
        self._name = name
        self._coefficient = coefficient
        self._alternative = alternative
        self._is_binary = is_binary

    def __hash__(self) -> int:
        return hash(self.__attributes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConstraintVariable) and self.__attributes == other.__attributes

    def __repr__(self) -> str:
        return f'<Variable[name: {self._name}, coeff: {self._coefficient}]>'

    def multiply(self, factor: float) -> ConstraintVariable:
        self.coefficient *= factor
        return self

    def with_coefficient(self, coefficient: float) -> ConstraintVariable:
        self.coefficient = coefficient
        return self

    @property
    def __attributes(self):
        return (self._name, self._coefficient)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def coefficient(self):
        return self._coefficient

    @coefficient.setter
    def coefficient(self, value):
        self._coefficient = value

    @property
    def alternative(self):
        return self._alternative

    @property
    def is_binary(self):
        return self._is_binary


class ValueConstraintVariable(ConstraintVariable):
    '''
    Class that stores a coefficient of the free variable.
    '''
    name = 'free'

    def __init__(self, coefficient: float) -> None:
        super().__init__(ValueConstraintVariable.name, coefficient)

    def __repr__(self) -> str:
        return f'{self.coefficient}'


class ConstraintVariablesSet:
    '''
    Class that stores a variables for the constraint.
    '''

    def __init__(self, variables: List[ConstraintVariable] = None, name: str = None):
        # key: str -> value: ConstraintVariable
        self._variables: Dict[str, ConstraintVariable] = dict()
        self._name = name
        if variables is not None:
            self.add_variables(variables)

    def add_variable(self, variable: ConstraintVariable):
        assert variable is not None, "Variable must not be None"

        if variable.name in self._variables:
            self._variables[variable.name].coefficient += variable.coefficient
        else:
            self._variables[variable.name] = variable

    def add_variables(self, variables: List[ConstraintVariable]):
        assert variables is not None, "Variables list must not be None"

        for variable in variables:
            self.add_variable(variable)

    @property
    def variables(self) -> List[ConstraintVariable]:
        return self._variables.values()

    @property
    def variables_names(self) -> List[str]:
        return list(self._variables.keys())

    @property
    def alternative_names(self):
        return [var.alternative for var in self._variables.values()]

    def __getitem__(self, key: str):
        return self._variables[key]

    def __setitem__(self, key: str, value: ConstraintVariable):
        self.add_variable(value)

    def __repr__(self):
        return 'no variables' if len(self._variables) < 1 else ",".join([v for v in self._variables])

    def __eq__(self, o: object) -> bool:
        if type(o) is not ConstraintVariablesSet:
            return False
        return o._name == self._name and self._variables == o._variables

    def __hash__(self) -> int:
        return self._name.__hash__ + 13 * self._variables.__hash__

    def multiply_by_scalar(self, scalar: float) -> ConstraintVariablesSet:
        for variable in self._variables.values():
            variable.multiply(scalar)
        return self


def merge_variables(new_variable: ConstraintVariable, variables: List[ConstraintVariable]) -> Set[ConstraintVariable]:
    '''
    Adds new_variable to the list with variables. If the new_variable already exists in variables
    then sum coefficient of the new_variable with the coefficient from the variable in the list.
    '''
    for var in variables:
        if var._name == new_variable._name:
            var._coefficient += new_variable._coefficient
            return variables
    variables.append(new_variable)
    return variables


class Constraint:
    '''
    Class that stores a constraint for the model.
    '''

    # class members
    def create_variable_name(function_name: str, criterion_name: str, alternative_name: str):
        return f'{function_name}_{{{criterion_name}}}({alternative_name})'

    # instance members
    def __init__(self, variables: ConstraintVariablesSet, relation: Relation, name: str = "constr"):
        assert variables is not None, "VariablesSet cannot be None"
        assert type(variables) is ConstraintVariablesSet,\
            "variables must be of type ConstraintVariablesSet"
        assert type(relation) is Relation,\
            f"relation must be a Relation object"

        # define fields
        self._variables_set: ConstraintVariablesSet = ConstraintVariablesSet()
        self._rhs = ValueConstraintVariable(0.0)
        self._relation = relation
        self._name = name

        self.add_variables(variables)

    def add_variables(self, variables_set: ConstraintVariablesSet):
        for variable in variables_set.variables:
            self.add_variable(variable)

    def add_variable(self, variable: ConstraintVariable):
        assert variable._name is not None, 'Variable name must not be None'

        if type(variable) is ValueConstraintVariable:
            self._rhs.coefficient += variable.coefficient
        else:
            # add variable to constraint variables set
            self._variables_set.add_variable(variable)

    def get_variable(self, variable_name: str) -> ConstraintVariable:
        if variable_name in self._variables_set.variables_names:
            return self._variables_set[variable_name]
        return None

    @property
    def alternatives(self) -> List[str]:
        '''
        Returns names of all alternatives
        in this Constraint.
        '''
        return [
            variable.alternative
            for variable in self._variables_set.variables
        ]

    @property
    def name(self):
        return self._name

    @property
    def free_variable(self):
        return self._rhs

    @property
    def variables(self):
        return self._variables_set.variables

    @property
    def variables_names(self):
        return self._variables_set.variables_names

    @property
    def relation(self):
        return self._relation

    @property
    def get_constraints_variables(self):
        return self._variables_set.variables

    @property
    def number_of_variables(self) -> int:
        return len(self._variables_set.variables)

    def multiply_by_scalar(self, scalar: float):
        self._variables_set.multiply_by_scalar(scalar)
        self._rhs._coefficient *= scalar
        if scalar < 0.0:
            if self._relation.sign == '<':
                self._relation.sign = '>'
            elif self._relation.sign == '<=':
                self._relation.sign = '>='
            elif self._relation.sign == '>':
                self._relation.sign = '<'
            elif self._relation.sign == '>=':
                self._relation.sign = '<='

    def __repr__(self):
        variables = '+'.join(
            [f'{variable.coefficient}*{variable.name}' for variable in self._variables_set.variables])
        return f'<Constraint:[name: {self._name}, variables: {variables}, relation: {self._relation.sign}, rhs: {self._rhs}]>'

    def __eq__(self, o: object) -> bool:
        if type(o) is not Constraint:
            return False
        return o._relation == self._relation and o._name == self._name and o._rhs == self._rhs and o._variables_set == self._variables_set

    def __hash__(self) -> int:
        return self._name.__hash__ \
            + 13 * self._relation.__hash__ \
            + 19 * self._rhs.__hash__ \
            + 23 * self._variables_set.__hash__

    def to_latex(self) -> str:
        constraint_str = StringIO()
        first_added = False
        for var in self._variables_set.variables:
            coeff = round(var.coefficient, 3)
            if var.coefficient == 0:
                # don't add variable with coefficient equal to 0
                continue
            if var.coefficient > 0 and first_added:
                # + coeff * variable, if coeff is positive and this is not the first variable
                constraint_str.write(f'+{coeff} * {var.name}')
            else:
                # coeff * variable, if coeff < 0 or we add first positive variable
                constraint_str.write(f'{coeff} * {var.name}')
                first_added = True
        constraint_str.write(self._relation.sign)
        constraint_str.write(f'{self._rhs.coefficient}')
        return constraint_str.getvalue()


def merge_constraints(constraints: List[Constraint]) -> Constraint:
    assert len(constraints) > 1,\
        'There must be at least 2 constraints for merging constraints'
    assert len(set([constraint._relation for constraint in constraints])) == 1,\
        'All constraints must have the same relation'

    merged_constraint = Constraint(
        ConstraintVariablesSet(),
        constraints[0]._relation,
        f'merged_constraint_{"_".join([c._name for c in constraints])}'
    )
    for constraint in constraints:
        for variable in constraint.variables:
            merged_constraint.add_variable(variable)
        # add free variable
        merged_constraint.add_variable(constraint.free_variable)

    return merged_constraint
