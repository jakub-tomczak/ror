from ror.helpers import reduce_lists
from ror.Constraint import Constraint
from typing import List
import gurobipy as gp
from gurobipy import GRB


class Model:
    def __init__(self, constraints: List[Constraint] = None, target: str = None, notes: str = None):
        assert constraints is None or type(constraints) is list,\
            "constrains must be an array of Constraint class or None"
        self._constraints: List[Constraint] = constraints if constraints is not None else [
        ]
        self._target: str = target
        self._notes: str = notes

    def add_constraints(self, constraints: List[Constraint]):
        for constraint in constraints:
            self.add_constraint(constraint)

    def add_constraint(self, constraint: Constraint):
        assert type(constraint) is Constraint,\
            f"constraint must be of Constraint type, provided: {type(constraint)}"
        self._constraints.append(constraint)
        return self

    def __repr__(self):
        data = ['Model', f"target: {self._target}",
                ""] + [c.__repr__() for c in self._constraints]
        return '\n'.join(data)
    
    @property
    def variables_names(self):
        variables_name = set()
        for constraint in self._constraints:
            variables_name.update(constraint.variables_names)
        return variables_name

    @property
    def constraints(self) -> List[Constraint]:
        return self._constraints

    @property
    def target(self) -> str:
        return self._target

    @target.setter
    def target(self, target: str):
        assert target in reduce_lists([constr.variables_names for constr in self._constraints]),\
            f"target variable '{target}' doesn't exist in any constraint"
        self._target = target

    @property
    def notes(self) -> str:
        return self._notes

    def to_gurobi_model(self):
        gurobi_model = gp.Model("model")
        distinct_variables = self.variables_names
        gurobi_variables = {variable_name: gurobi_model.addVar(name=variable_name) for variable_name in distinct_variables}
        gurobi_operators = {
            "<=": GRB.LESS_EQUAL,
            "==": GRB.EQUAL
        }

        for constraint in self._constraints:
            variables = constraint.variables
            expr = gp.LinExpr(
                [variable.coefficient for variable in variables],
                [gurobi_variables[variable.name] for variable in variables]
            )
            gurobi_model.addLConstr(
                lhs=expr,
                sense=gurobi_operators[constraint.relation.sign],
                rhs=constraint.free_variable.coefficient,
                name=constraint.name
            )
            gurobi_model.update()

        return gurobi_model

