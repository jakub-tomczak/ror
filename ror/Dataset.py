from __future__ import annotations
import logging
from ror.Constraint import ConstraintVariable
import numpy as np
from typing import List, Tuple


class Dataset:
    DEFAULT_EPS = 1e-6
    DEFAULT_M = 1e10
    ALL_CRITERIA = 'all'
    CRITERION_TYPES = {
        "gain": "g",
        "cost": "c"
    }

    def reverse_cost_type_criteria(data: any, criteria: List[Tuple[str, str]]):
        assert type(data) is np.ndarray, "Data must be a numpy array"
        # reverse values in the cost type criteria - assume that all criteria are of a gain type
        for index, (criterion_name, criterion_type) in enumerate(criteria):
            if criterion_type == Dataset.CRITERION_TYPES["cost"]:
                logging.info(f'Flipping values in criterion {criterion_name}')
                data[:, index] *= -1
        return data

    def __init__(self, alternatives: List[str], data: any, criteria: List[Tuple[str, str]], delta: float = None, eps: float = None):
        assert type(data) is np.ndarray, "Data must be a numpy array"
        assert len(alternatives) == data.shape[0],\
            "Number of alternatives labels doesn't match the number of data rows"
        assert len(criteria) == data.shape[1],\
            "Number of criteria doesn't match the number of data columns"

        # list with names of alternatives
        self._alternatives: List[str] = alternatives
        # matrix with data for each alternative on each criterion
        self._data = Dataset.reverse_cost_type_criteria(data, criteria)
        self._criteria = criteria
        self._eps = eps if eps is not None else Dataset.DEFAULT_EPS
        self._M = Dataset.DEFAULT_M
        self._alternative_to_variable = dict()
        self._criterion_to_index = {
            criterion_name: index for index, (criterion_name, _) in enumerate(criteria)
        }
        # delta value used as objective in step 1
        # in step 2 used as a free value obained in step 1
        self._delta: float = delta

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

    @property
    def delta(self) -> float:
        return self._delta

    @delta.setter
    def delta(self, delta):
        self._delta = delta

    def save_to_file(self, filename: str):
        # data section
        data_section_lines = ['#Data']
        data_section_lines.append(
            ','.join([f'{criterion_name}[{criterion_type}]' for criterion_name, criterion_type in self.criteria])
        )
        for alternative, alternative_values in (self.alternatives, self.matrix):
            values = ','.join([str(value) for value in alternative_values])
            line = f'{alternative}, {values}'
            data_section_lines.append(line)
        
        # parameters section


class RORDataset(Dataset):
    def __init__(
            self,
            alternatives: List[str],
            data: any,
            criteria: List[Tuple[str, str]],
            # HACK: without quotes we would need to import those 2 clases
            # but then we will get circular import,
            # this is still better than no type hints
            preference_relations: List["PreferenceRelation"] = None,
            intensity_relations: List["PreferenceIntensityRelation"] = None,
            eps: float = None):
        super().__init__(alternatives, data, criteria, eps=eps)
        self._preference_relations: List["PreferenceRelation"] = \
            preference_relations if preference_relations is not None else []
        self._intensity_relations: List["PreferenceIntensityRelation"] = \
            intensity_relations if intensity_relations is not None else []

    @property
    def preferenceRelations(self) -> List["PreferenceRelation"]:
        return self._preference_relations

    @property
    def intensityRelations(self) -> List["PreferenceIntensityRelation"]:
        return self._intensity_relations
