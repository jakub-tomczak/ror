from ror.CalculationsException import CalculationsException
from ror.helpers import reduce_lists
from ror.Constraint import Constraint, ConstraintVariable, ConstraintVariablesSet
from typing import Dict, List, Set
import gurobipy as gp
from gurobipy import GRB
from ror.OptimizationResult import OptimizationResult
import logging
from functools import reduce

from ror.latex_exporter import export_latex, export_latex_pdf


class Model:
    DEFAULT_CONSTRAINTS_KEY = 'default'
    def __init__(self, constraints: List[Constraint] = None, name: str = None):
        assert constraints is None or type(constraints) is list,\
            "constrains must be an array of Constraint class or None"
        new_constraints = dict()
        if constraints is not None and len(constraints) > 0:
            new_constraints[Model.DEFAULT_CONSTRAINTS_KEY] = constraints
        self._constraints: Dict[str, List[Constraint]] = new_constraints
        # target (objective) is a set of variables
        self._target: ConstraintVariablesSet = None
        self._name: str = name
        self.gurobi_model: gp.Model = None

    def add_constraints(self, constraints: List[Constraint], name:str = None):
        for constraint in constraints:
            self.add_constraint(constraint, name=name)

    def add_constraint(self, constraint: Constraint, name: str = None):
        assert type(constraint) is Constraint,\
            f"constraint must be of Constraint type, provided: {type(constraint)}"
        key = Model.DEFAULT_CONSTRAINTS_KEY
        for constraints in self._constraints.values():
            if constraint in constraints:
                logging.debug(f'Constraint {constraint.name} already in model, skipping.')
                return
        if name is not None:
            key = name
        if key not in self._constraints:
            self._constraints[key] = [constraint]
        else:
            self._constraints[key].append(constraint)
        return self

    def __repr__(self):
        constraints_str = [c.__repr__() for c in self.__get_constraints_list()]
        data = ['Model', f"target: {self._target}",
                ""] + constraints_str
        return '\n'.join(data)

    @property
    def variables(self) -> Set[ConstraintVariable]:
        '''
        Returns all variables in model.
        '''
        variables = set()
        for constraint in self.__get_constraints_list():
            variables.update(constraint.variables)
        return variables

    def __get_constraints_list(self) -> List[Constraint]:
        return list(reduce(lambda a, b: a+b, self._constraints.values(), []))

    @property
    def constraints(self) -> List[Constraint]:
        return self.__get_constraints_list()

    @property
    def constraints_dict(self) -> Dict[str, List[Constraint]]:
        return self._constraints

    @property
    def target(self) -> ConstraintVariablesSet:
        return self._target

    @target.setter
    def target(self, target: ConstraintVariablesSet):
        self._validate_target(target)
        self._target = target

    @property
    def name(self) -> str:
        return self._name

    def _validate_target(self, target: ConstraintVariablesSet):
        assert target is not None, "Model's target must not be None"
        all_variables = set(reduce_lists([constr.variables_names for constr in self.__get_constraints_list()]))
        # free variable should be always available in target
        all_variables.add("free")
        variables = set(target.variables_names)
        diff = variables.difference(all_variables)
        assert len(diff) == 0, f"target variables '{diff}' doesn't exist in any constraint"

    def to_gurobi_model(self) -> gp.Model:
        self._validate_target(self.target)

        gurobi_model = gp.Model("model")
        # set lower verbosity
        gurobi_model.Params.OutputFlag = 0
        distinct_variables = self.variables
        # create a dict
        # variable name: str -> variable: gurobi variable object
        gurobi_variables = {variable.name: gurobi_model.addVar(
            name=variable.name, vtype=GRB.BINARY if variable.is_binary else GRB.CONTINUOUS) for variable in distinct_variables}
        gurobi_operators = {
            "<=": GRB.LESS_EQUAL,
            "==": GRB.EQUAL
        }

        for constraint in self.__get_constraints_list():
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

        # add objective

        objective = gp.LinExpr(
            [variable.coefficient for variable in self._target.variables if variable.name != "free"],
            [gurobi_variables[variable.name] for variable in self._target.variables if variable.name != "free"])
        if "free" in self._target.variables_names:
            objective.addConstant(self._target["free"].coefficient)
        gurobi_model.setObjective(objective)
        gurobi_model.update()

        return gurobi_model

    def update_model(self):
        self.gurobi_model = self.to_gurobi_model()
        return self.gurobi_model

    def save_model(self):
        self.update_model()
        self.gurobi_model.write('model.lp')

    def export_to_latex(self, filename: str):
        export_latex(self, filename)
    
    def export_to_latex_pdf(self, filename: str):
        export_latex_pdf(self, filename)

    def solve(self) -> OptimizationResult:
        model = self.to_gurobi_model()

        model.optimize()
        if model.status == GRB.INF_OR_UNBD:
            # Turn presolve off to determine whether model is infeasible
            # or unbounded
            logging.info("Turning presolve off")
            model.setParam(GRB.Param.Presolve, 0)
            model.optimize()

        if model.status == GRB.OPTIMAL:
            logging.info(f'Optimal objective: {model.objVal}')
            variables_values: Dict[str, float] = {}
            # save calculated coefficients
            for v in model.getVars():
                variables_values[v.VarName] = v.X
            return OptimizationResult(self, model.objVal, variables_values)
        elif model.status == GRB.INFEASIBLE:
            logging.error('Model is infeasible.')
            raise CalculationsException(f'Model {self.name} is infeasible.')
