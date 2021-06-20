from ror.Constraint import ConstraintVariable
import numpy as np
from typing import List, Tuple
import pandas as pd
import os


criterion_types = {
    "gain": "g",
    "cost": "c"
}


class Dataset:
    DEFAULT_EPS = 1e-6
    DEFAULT_M = 1e10
    ALL_CRITERIA = 'all'

    def __init__(self, alternatives: List[str], data: any, criteria: List[Tuple[str, str]]):
        assert type(data) is np.ndarray, "Data must be a numpy array"
        assert len(alternatives) == data.shape[0],\
            "Number of alternatives labels doesn't match the number of data rows"
        assert len(criteria) == data.shape[1],\
            "Number of criteria doesn't match the number of data columns"

        # list with names of alternatives
        self._alternatives: List[str] = alternatives
        # matrix with data for each alternative on each criterion
        self._data = data
        self._criteria = criteria
        self._eps = Dataset.DEFAULT_EPS
        self._M = Dataset.DEFAULT_M
        self._alternative_to_variable = dict()
        self._criterion_to_index = {
            criterion: index for criterion, index in enumerate(criteria)
        }

        for alternative_values, alternative_name in zip(data, alternatives):
            self._alternative_to_variable[alternative_name] = [
                ConstraintVariable(
                    f'{criterion_name}_{alternative_name}', value)
                for value, (criterion_name, _)
                in zip(alternative_values, criteria)
            ]

    def get_data_for_alternative(self, alternative_name: str) -> List[ConstraintVariable]:
        assert alternative_name in self._alternative_to_variable,\
            f"Alternative {alternative_name} doesn't exist in alternatives"

        return self._alternative_to_variable[alternative_name]

    def get_data_for_alternative_and_criterion(self, alternative_name: str, criterion: str) -> ConstraintVariable:
        assert criterion in self._criterion_to_index.keys(),\
            f'Criterion {criterion} is unknown'

        alternative_values = self.get_data_for_alternative(alternative_name)

        assert self._criterion_to_index[criterion] < len(alternative_values),\
            'Invalid number of values for the alternative'

        return alternative_values[self._criterion_to_index[criterion]]

    @property
    def criteria(self) -> List[Tuple[str, str]]:
        return self._criteria

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def M(self) -> float:
        return self._M

    @property
    def alternatives(self):
        return self._alternatives

    @property
    def matrix(self):
        return self._data

    @property
    def alternative_to_variable(self):
        return self._alternative_to_variable


def read_dataset_from_txt(filename: str):
    if not os.path.exists(filename):
        print(f"file {filename} doesn't exist")
        return None

    data = pd.read_csv(filename, sep=',')

    def parse_criterion(criterion: str) -> Tuple[str, str]:
        data = criterion.strip().split('[')

        if len(data) != 2 or len(data[0]) < 1 or len(data[1]) < 1:
            print(f"Failed to parse criterion {criterion}")
            return ('', '')
        criterion_type = data[1][0]
        if criterion_type not in criterion_types.values():
            print(
                f"Invalid criterion type: {criterion_type}, expected values: {criterion_type.values()}")
            return ('', '')
        return (data[0], criterion_type)

    alternatives = data.iloc[:, 0]
    values = data.iloc[:, 1:].to_numpy()
    # skip first column - this should be id
    criteria = [parse_criterion(criterion) for criterion in data.columns[1:]]

    return Dataset(
        alternatives=alternatives,
        data=values,
        criteria=criteria
    )
