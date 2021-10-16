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
