from ror.Constraint import Constraint
from typing import List


class Model:
    def __init__(self, constraints: List[Constraint] = None, target: str = None):
        assert constraints is None or type(constraints) is list,\
            "constrains must be an array of Constraint class or None"
        self._constrains = constraints if constraints is not None else []
        self._target = target

    def add_constraint(self, constraint: Constraint):
        assert type(constraint) is Constraint,\
            f"constraint must be of Constraint type, provided: {type(constraint)}"
        self._constraints.append(constraint)
        return self

    def add_target(self, target: str):
        self._target = target
        return self

    def __repr__(self):
        data = ['Model', f"target: {self._target}",
                ""] + [c.__repr__() for c in self._constrains]
        return '\n'.join(data)
