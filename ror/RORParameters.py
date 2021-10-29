from typing import Any, Dict, List, Set, Union
from ror.dataset_constants import DEFAULT_EPS

from ror.loader_utils import RORParameter
from ror.Dataset import Dataset

class DataValidationException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def float_validator(value: float, min_value: float = None, max_value: float = None) -> bool:
    # float can be int if fits range
    try:
        float(value)
    except:
        # ints are convertable to float, accept them as well
        if not int_validator(value):
            return False
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True

def int_validator(value: int, min_value: int = None, max_value: int = None) -> bool:
    try:
        int(value)
    except:
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
RORParameterValue = Union[list, str, float, int]

class RORParameters:
    def get_default_parameter_value(parameter: RORParameter) -> RORParameterValue:
        if parameter == RORParameter.EPS:
            return DEFAULT_EPS
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
        elif parameter == RORParameter.NUMBER_OF_ALPHA_VALUES:
            return 3
        elif parameter == RORParameter.TIE_RESOLVER:
            return 'NoResolver'
        else:
            return None

    def __init__(self) -> None:
        self.__parameters: Dict[RORParameter, RORParameterValue] = dict()

    def __validate_parameter_name(self, parameter: str):
        assert parameter in RORParameter, f'Parameter {parameter} not recognized, it is not a member of RORParameter enum'

    def __validate_parameter_value(self, parameter: RORParameter, value: Any):
        if parameter == RORParameter.EPS:
            if not float_validator(value, min_value=0.0):
                raise DataValidationException(f'Failed to parse {RORParameter.EPS} value. {RORParameter.EPS} value must be a float value greater than 0.')
        elif parameter == RORParameter.INITIAL_ALPHA:
            if not float_validator(value, min_value=0.0, max_value=1.0):
                raise DataValidationException(f'Failed to parse {RORParameter.INITIAL_ALPHA.value} value. {RORParameter.INITIAL_ALPHA.value} value must be a float value in range <0.0, 1.0>')
        elif parameter == RORParameter.ALPHA_VALUES:
            exception_msg = f'Failed to parse {RORParameter.ALPHA_VALUES.value} value. {RORParameter.ALPHA_VALUES.value} value must be a list with float values in range <0.0, 1.0>'
            # validate one by one as or condition won't fail if the first condition is False
            # don't add float type for items - they can be float or ints (i.e. 0)
            if not list_validator(value):
                raise DataValidationException(exception_msg)
            if not all([float_validator(item, min_value=0.0, max_value=1.0) for item in value]):
                raise DataValidationException(exception_msg)
        elif parameter == RORParameter.PRECISION:
            if not int_validator(value, min_value=0, max_value=10):
                raise DataValidationException(f'Failed to parse {RORParameter.PRECISION.value} value. {RORParameter.PRECISION.value} value must be an int value in range <0, 10>')
        elif parameter == RORParameter.ALPHA_WEIGHTS:
            exception_msg = f'Failed to parse {RORParameter.ALPHA_WEIGHTS.value} value. {RORParameter.ALPHA_WEIGHTS.value} value must be a list with float values equal or greater 0'
            # validate one by one as or condition won't fail if the first condition is False
            # don't add float type for items - they can be float or ints (i.e. 1)
            if not list_validator(value):
                raise DataValidationException(exception_msg)
            if not all([float_validator(item, min_value=0) for item in value]):
                raise DataValidationException(exception_msg)
        elif parameter == RORParameter.RESULTS_AGGREGATOR:
            pass
            # validated in ror_solver.solve_model method
        elif parameter == RORParameter.NUMBER_OF_ALPHA_VALUES:
            if not int_validator(value, min_value=1, max_value=15):
                raise DataValidationException(f'Failed to parse {RORParameter.NUMBER_OF_ALPHA_VALUES.value} value. {RORParameter.NUMBER_OF_ALPHA_VALUES.value} value must be an int value in <1, 15>')
        elif parameter == RORParameter.TIE_RESOLVER:
            # validated in ror_solver.solve_model method
            pass

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

    def deep_copy(self) -> 'RORParameters':
        import copy
        return copy.deepcopy(self)
