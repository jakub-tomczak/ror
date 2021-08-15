from __future__ import annotations

from typing import Dict


class OptimizationResult:
    def __init__(self, model: Model, objective_value: float, variables_values: Dict[str, float]) -> None:
        self.model: Model = model
        self.objective_value = objective_value
        self.variables_values = variables_values
