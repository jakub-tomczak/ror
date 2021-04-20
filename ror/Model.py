from ror.Constraint import Constraint
from typing import List

class Model:
    def __init__(self, constraints: List[Constraint] = None, target: str = None):
        assert constraints is None or type(constraints) is list,\
            "constrains must be an array of Constraint class or None"
        self.constrains = constraints if constraints is not None else []
        self.target = target

    def add_constraint(self, constraint: Constraint):
        assert type(constraint) is Constraint,\
            f"constraint must be of Constraint type, provided: {type(constraint)}"
        self.constraints.append(constraint)
        return self

    def add_target(self, target: str):
        self.target = target
        return self

    def __repr__(self):
        data = ['Model', f"target: {self.target}", ""] + [c.__repr__() for c in self.constrains]
        return '\n'.join(data)


def create_monotonicity_constraints(data) -> List[Constraint]:
    constraints = []
    for column in data.T:
        for row in range(len(column)):
            for next_row in range(row+1, len(column)):
                if column[row] >= column[next_row]:
                   constraints.append(
                       Constraint(
                           [1, -1],
                           [
                               f'{Constraint.alternatives_prefix}{row}',
                               f'{Constraint.alternatives_prefix}{next_row}'
                           ],
                           "<=",
                           0.0
                        )
                    )
    return constraints
