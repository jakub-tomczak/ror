from typing import Dict, Set, Union

from ror.loader_utils import RORParameter
from ror.Dataset import Dataset


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
        assert parameter in RORParameter, f'Parameter {parameter} not recignized, it is not a member of RORParameter enum'

    def add_parameter(self, parameter: RORParameter, value: RORParameterValue):
        self.__validate_parameter_name(parameter)
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