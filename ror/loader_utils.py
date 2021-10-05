from enum import Enum


VALID_SEPARATORS = [',', ';']
PARAMETERS_VALUE_SEPARATOR = '='
DATA_SECTION = "#Data"
PREFERENCES_SECTION = "#Preferences"
PARAMETERS_SECTION = "#Parameters"


class AvailableParameters(Enum):
    EPS = 'eps'
    INITIAL_ALPHA = 'initial_alpha'
