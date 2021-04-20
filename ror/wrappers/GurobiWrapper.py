from ror import Constraint, Model
import gurobipy as gp
from gurobipy import GRB


class GurobiWrapper():
    gurobi_operators = {
        "<=": GRB.LESS_EQUAL,
        "==": GRB.EQUAL
    }

    def __init__(self, model: Model):
        self.model = model
        self._gurobi_model = gp.Model("model")
        self.variables = self.get_variables_from_model(model)
        self.gurobi_constraints = self.add_constraints(model)


    def get_variables_from_model(self, model: Model):
        '''
        Creates and returns dict: variable_name -> gurobi variable
        '''
        variables = [x.get_constraints_variables for x in model.constrains]
        distinct_variables = set()
        for variables_set in variables:
            distinct_variables.update(variables_set)

        return {x: self.gurobi_model.addVar(name=x) for x in distinct_variables}

    def add_constraints(self, model: Model):
        '''
        Adds all constraints from model to gurobi model
        and returns newly added constraint list.
        '''
        constraints = []
        for index, constr in enumerate(model.constrains):
            expr = gp.LinExpr(
                constr.coefficients,
                [self.variables[c] for c in constr.variables]
            )
            new_constraint = self.gurobi_model.addLConstr(
                lhs=expr,
                sense=GurobiWrapper.gurobi_operators[constr.relation],
                rhs=constr.value,
                name=f"constr_{index}"
            )
            constraints.append(new_constraint)
        
        self.gurobi_model.update()
        return constraints

    @property
    def gurobi_model(self):
        return self._gurobi_model
