from enum import Enum


VALID_SEPARATORS = [',', ';']
PARAMETERS_VALUE_SEPARATOR = '='
DATA_SECTION = "#Data"
PREFERENCES_SECTION = "#Preferences"
PARAMETERS_SECTION = "#Parameters"


class RORParameter(Enum):
    EPS = 'eps'
    INITIAL_ALPHA = 'initial_alpha'
    ALPHA_VALUES = 'alpha_values'
    RESULTS_AGGREGATOR="results_aggregator"
    # digit place precision in solutions
    PRECISION = 'precision'
    ALPHA_WEIGHTS = 'alpha_weights'
    NUMBER_OF_ALPHA_VALUES = 'alpha_values_number'
