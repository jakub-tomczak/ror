import unittest
from ror.Constraint import Constraint
from ror.Model import Model, create_monotonicity_constraints
from ror.wrappers.GurobiWrapper import GurobiWrapper
from tests.datasets.buses import buses

class TestMain(unittest.TestCase):
    def test_creating_gurobi_model(self):
        mono_constraints = create_monotonicity_constraints(buses)
        model = Model(mono_constraints, 'gamma')
        
        gurobi_model = GurobiWrapper(model)
        self.assertIsNotNone(gurobi_model.gurobi_model)
