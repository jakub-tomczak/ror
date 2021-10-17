from typing import Any, Dict, List, Set, Union

from ror.loader_utils import RORParameter
from ror.Dataset import Dataset

class DataValidationException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def float_validator(value: float, min_value: float = None, max_value: float = None) -> bool:
    # float can be int if fits range
    if type(value) not in [float, int]:
        return False
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True

def int_validator(value: int, min_value: int = None, max_value: int = None) -> bool:
    if type(value) is not int:
        return False
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True

def list_validator(value: List[Any], expectedtype: type = None) -> bool:
    if type(value) is not list:
        return False
    if expectedtype is not None:
        # divide into 2 ifs for clarity
        if not all([type(item) is expectedtype for item in value]):
            return False
    return True

# alias for parameters value
RORParameterValue = Union[list, str, float]

class RORParameters:
    def get_default_parameter_value(parameter: RORParameter) -> RORParameterValue:
        if parameter == RORParameter.EPS:
            return Dataset.DEFAULT_EPS
        elif parameter == RORParameter.INITIAL_ALPHA:
            return 0.0
        elif parameter == RORParameter.ALPHA_VALUES:
            return [0.0, 0.5, 1.0]
        elif parameter == RORParameter.PRECISION:
            return 3
        elif parameter == RORParameter.ALPHA_WEIGHTS:
            return [1.0, 1.0, 1.0]
        elif parameter == RORParameter.RESULTS_AGGREGATOR:
            return 'DefaultResultAggregator'
        else:
            return None

    def __init__(self) -> None:
        self.__parameters: Dict[RORParameter, RORParameterValue] = dict()

    def __validate_parameter_name(self, parameter: str):
        assert parameter in RORParameter, f'Parameter {parameter} not recognized, it is not a member of RORParameter enum'

    def __validate_parameter_value(self, parameter: RORParameter, value: Any):
        if parameter == RORParameter.EPS:
            if not float_validator(value, min_value=0.0):
                raise DataValidationException('Failed to parse EPS value. EPS value must be a float value greater than 0.')
        elif parameter == RORParameter.INITIAL_ALPHA:
            if not float_validator(value, min_value=0.0, max_value=1.0):
                raise DataValidationException('Failed to parse INITIAL_ALPHA value. INITIAL_ALPHA value must be a float value in range <0.0, 1.0>')
        elif parameter == RORParameter.ALPHA_VALUES:
            exception_msg = 'Failed to parse ALPHA_VALUES value. ALPHA_VALUES value must be a list with float values in range <0.0, 1.0>'
            # validate one by one as or condition won't fail if the first condition is False
            # don't add float type for items - they can be float or ints (i.e. 0)
            if not list_validator(value):
                raise DataValidationException(exception_msg)
            if not all([float_validator(item, min_value=0.0, max_value=1.0) for item in value]):
                raise DataValidationException(exception_msg)
        elif parameter == RORParameter.PRECISION:
            if not int_validator(value, min_value=0, max_value=10):
                raise DataValidationException('Failed to parse PRECISION value. PRECISION value must be an int value in range <0, 10>')
        elif parameter == RORParameter.ALPHA_WEIGHTS:
            exception_msg = 'Failed to parse ALPHA_WEIGHTS value. ALPHA_WEIGHTS value must be a list with float values equal or greater 0'
            # validate one by one as or condition won't fail if the first condition is False
            # don't add float type for items - they can be float or ints (i.e. 1)
            if not list_validator(value):
                raise DataValidationException(exception_msg)
            if not all([float_validator(item, min_value=0) for item in value]):
                raise DataValidationException(exception_msg)
        elif parameter == RORParameter.RESULTS_AGGREGATOR:
            pass
            # checked in ror_solver.solve_model method

    def add_parameter(self, parameter: RORParameter, value: RORParameterValue):
        self.__validate_parameter_name(parameter)
        self.__validate_parameter_value(parameter, value)
        self.__parameters[parameter] = value

    def get_parameter(self, parameter_name: RORParameter) -> RORParameterValue:
        if parameter_name in self.__parameters:
            return self.__parameters[parameter_name]
        else:
            return RORParameters.get_default_parameter_value(parameter_name)

    def keys(self) -> Set[RORParameter]:
        return self.__parameters.keys()

    def __getitem__(self, parameter_name: RORParameter) -> RORParameterValue:
        self.__validate_parameter_name(parameter_name)
        if parameter_name in self.__parameters:
            return self.__parameters[parameter_name]
        else:
            return RORParameters.get_default_parameter_value(parameter_name)