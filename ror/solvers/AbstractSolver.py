from abc import abstractmethod


from ror.RORModel import RORModel
from ror.OptimizationResult import OptimizationResult

class AbstractSolver:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def solve_model(self, model: RORModel) -> OptimizationResult:
        pass

    def save_model(self, model: RORModel) -> str:
        pass

    @abstractmethod
    def create_model(self, model: RORModel):
        pass