from abc import abstractmethod


from ror.RORModel import RORModel
from ror.OptimizationResult import OptimizationResult

class AbstractSolver:
    def __init__(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def solve_model(self, model: RORModel) -> OptimizationResult:
        pass

    def save_model(self, model: RORModel) -> str:
        pass

    @abstractmethod
    def _create_model(self, model: RORModel):
        pass

    @property
    def name(self) -> str:
        return self._name