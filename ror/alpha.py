from typing import Dict, List


class AlphaValue:
    def __init__(self, value: float, name: str) -> None:
        self._value: float = value
        self._name: str = name

    @property
    def value(self) -> float:
        return self._value

    @property
    def name(self) -> float:
        return self._name


class AlphaValues:
    def __init__(self, alpha_values: List[AlphaValue]) -> None:
        self.__alpha_values: Dict[str, AlphaValue] = {
            alpha_value.name: alpha_value for alpha_value in alpha_values}

    @property
    def alpha(self) -> List[AlphaValue]:
        return [alpha_value for alpha_value in self.__alpha_values.values()]

    @property
    def values(self) -> List[float]:
        return [alpha_value.value for alpha_value in self.__alpha_values.values()]

    def __getitem__(self, alpha_name: str) -> AlphaValue:
        if self.__alpha_values is not None and alpha_name in self.__alpha_values:
            return self.__alpha_values[alpha_name]
        else:
            return None
