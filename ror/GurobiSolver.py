from typing import Dict
from ror.AbstractSolver import AbstractSolver
from ror.RORModel import RORModel
from ror.OptimizationResult import OptimizationResult
from ror.CalculationsException import CalculationsException
import gurobipy as gp
from gurobipy import GRB
import logging


class GurobiSolver(AbstractSolver):
    def __init__(self) -> None:
        super().__init__()
        self.__model: gp.Model = None
        self.__name: str = 'Gurobi model'    

    def solve(self, model: RORModel) -> OptimizationResult:
        self._create_model(model)

        self.__model.optimize()
        if self.__model.status == GRB.INF_OR_UNBD:
            # Turn presolve off to determine whether model is infeasible
            # or unbounded
            logging.info("Turning presolve off")
            self.__model.setParam(GRB.Param.Presolve, 0)
            self.__model.optimize()

        if self.__model.status == GRB.OPTIMAL:
            logging.debug(f'Optimal objective: {self.__model.objVal}')
            variables_values: Dict[str, float] = {}
            # save calculated coefficients
            for v in self.__model.getVars():
                variables_values[v.VarName] = v.X
            return OptimizationResult(self, self.__model.objVal, variables_values)
        elif self.__model.status == GRB.INFEASIBLE:
            logging.error('Model is infeasible.')
            raise CalculationsException(f'Model {self.name} is infeasible.')


    def _create_model(self, model: RORModel):
        model._validate_target(model.target)

        self.__name = model.name
        gurobi_model = gp.Model(self.__name)
        # set lower verbosity
        gurobi_model.Params.OutputFlag = 0
        distinct_variables = model.variables
        # create a dict
        # variable name: str -> variable: gurobi variable object
        gurobi_variables = {variable.name: gurobi_model.addVar(
            name=variable.name, vtype=GRB.BINARY if variable.is_binary else GRB.CONTINUOUS) for variable in distinct_variables}
        gurobi_operators = {
            "<=": GRB.LESS_EQUAL,
            "==": GRB.EQUAL
        }

        for constraint in model.constraints:
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
            [variable.coefficient for variable in model.target.variables if variable.name != "free"],
            [gurobi_variables[variable.name] for variable in model.target.variables if variable.name != "free"])
        if "free" in model.target.variables_names:
            objective.addConstant(model.target["free"].coefficient)
        gurobi_model.setObjective(objective)
        gurobi_model.update()

        self.__model = gurobi_model
    
    def save_model(self, filename: str) -> str:
        # save with lp extension
        filename += '.lp'
        self.__model.write(filename)
        return filename
