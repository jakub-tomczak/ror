from enum import Enum

# name of constraints used for naming a set of constraints
# used i.e. for exporting constraints to latex
class ConstraintsName(Enum):
    PREFERENCE_INFORMATION = 'preference information'
    PREFERENCE_INTENSITY_INFORMATION = 'preference intensity information'
    MONOTONICITY = 'monotonicity'
    MIN_CONSTRAINTS = 'min'
    MAX_CONSTRAINTS = 'max'
    INNER_MAXIMIZATION = 'inner maximization'
    SLOPE = 'slope'

    def monotonicity(criterion: str) -> str:
        return f'{ConstraintsName.MONOTONICITY.value}, criterion: {criterion}'