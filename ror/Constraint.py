from typing import List

class Constraint:
    alternatives_prefix = 'a_'
    valid_relations = ['<', '<=', '>', '>=', '==']

    def __init__(self, coefficients: List[float], variables: List[str], relation: str, value: float, name: str = "constr"):
        assert len(coefficients) == len(variables),\
            f'Number of coefficients must be equal to the number of variables, {len(coefficients)} != {len(variables)}'
        assert relation in Constraint.valid_relations,\
            f"Relation must be one of {Constraint.valid_relations}, provided value: {relation}"
        assert type(value) is float,\
            f"Value must be of float type, provided: {type(value)}"

        self.coefficients = coefficients
        self.variables = variables
        if coefficients is not None and len(coefficients) > 0:
            self.variable_to_coefficient = \
                {variable: coeff for variable, coeff in zip(variables, coefficients)}
        self.relation = relation
        self.value = value
        self.name = name

    @property
    def get_constraints_variables(self):
        return set(self.variable_to_coefficient.keys())

    def get_constraint(self, variable: str):
        if variable not in self.variable_to_coefficient:
            return None
        return (self.variable_to_coefficient[variable], variable)

    def __repr__(self):
        variables = self.variable_to_coefficient.keys()
        coefficients = self.variable_to_coefficient.values()
        lhs = '+'.join([f'({a}*{b})' for a,b in zip(coefficients, variables)])
        return f'Constraint: {lhs}{self.relation}{self.value}'
