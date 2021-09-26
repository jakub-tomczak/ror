from __future__ import annotations

from typing import Dict


class OptimizationResult:
    def __init__(self, model: Model, objective_value: float, variables_values: Dict[str, float]) -> None:
        self.model: Model = model
        self.objective_value = objective_value
        self.variables_values = variables_values

class AlternativeOptimizedValue():
    def __init__(self, alternative_name: str, alpha_value: float, alpha_value_name: str) -> None:
        self._alternative_name = alternative_name
        self._alpha_value = alpha_value
        self._alpha_value_name = alpha_value_name