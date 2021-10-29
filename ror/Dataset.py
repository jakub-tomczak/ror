from __future__ import annotations
import logging
from ror.Constraint import ConstraintVariable
import numpy as np
from typing import List, Tuple
from ror.loader_utils import DATA_SECTION, PARAMETERS_SECTION, PARAMETERS_VALUE_SEPARATOR, PREFERENCES_SECTION, VALID_SEPARATORS, RORParameter
from ror.dataset_constants import DEFAULT_EPS, DEFAULT_M, CRITERION_TYPES
from ror.RORParameters import RORParameters
import os


class Dataset:
    def reverse_cost_type_criteria(data: any, criteria: List[Tuple[str, str]]):
        assert type(data) is np.ndarray, "Data must be a numpy array"
        # reverse values in the cost type criteria - assume that all criteria are of a gain type
        for index, (criterion_name, criterion_type) in enumerate(criteria):
            if criterion_type == CRITERION_TYPES["cost"]:
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
        self._eps = eps if eps is not None else DEFAULT_EPS
        self._M = DEFAULT_M
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

    def _prepare_data_for_saving(self, parameters: RORParameters) -> Tuple[List[str], List[str]]:
        # data section
        data_section_lines = [DATA_SECTION]
        header = ['alternative id']
        header.extend([f'{criterion_name}[{criterion_type}]' for criterion_name,
                                      criterion_type in self.criteria])
        data_section_lines.append(
            VALID_SEPARATORS[0].join(header)
        )
        for alternative, alternative_values in zip(self.alternatives, self.matrix):
            values = [alternative]
            values.append(VALID_SEPARATORS[0].join(
                [str(value) for value in alternative_values]))
            data_section_lines.append(VALID_SEPARATORS[0].join(values))

        # parameters section
        parameters_section = [PARAMETERS_SECTION]
        if parameters is not None:
            for parameter in RORParameter:
                parameter_value = parameters.get_parameter(parameter)
                parameters_section.append(f'{parameter.value}{PARAMETERS_VALUE_SEPARATOR}{parameter_value}')

        return (data_section_lines, parameters_section)

    def _save_data(self, filename: str, data: List[str]):
        if os.path.exists(filename):
            msg = f'File {filename} already exists. Saving dataset skipped.'
            logging.error(msg)
            raise Exception(msg)
        try:
            data_str: str = os.linesep.join(data)
            with open(filename, 'w') as file:
                file.write(data_str)
                    
        except Exception as e:
            logging.error(f'Failed to save file: {e}')
            raise e
    
    def save_to_file(self, filename: str, parameters: RORParameters):
        data_section, preferences_section = self._prepare_data_for_saving(parameters)
        data_section.extend(preferences_section)
        self._save_data(filename, data_section)

    def deep_copy(self) -> Dataset:
        import copy
        return copy.deepcopy(self)


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
        Dataset.__init__(self, alternatives, data, criteria, eps=eps)
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

    def add_preference_relation(self, relation: "PreferenceRelation"):
        if relation not in self._preference_relations:
            self._preference_relations.append(relation)

    def add_intensity_relation(self, relation: "PreferenceIntensityRelation"):
        if relation not in self._intensity_relations:
            self._intensity_relations.append(relation)

    def remove_preference_relation(self, relation: "PreferenceRelation"):
        if relation in self._preference_relations:
            self._preference_relations.remove(relation)

    def remove_intensity_relation(self, relation: "PreferenceIntensityRelation"):
        if relation in self._intensity_relations:
            self._intensity_relations.remove(relation)

    def __prepare_preferences_data_for_saving(self) -> List[str]:
        relations = [PREFERENCES_SECTION]
        sep = VALID_SEPARATORS[0]
        for preference in self._preference_relations:
            relations.append(
                f'{preference.alternative_1}{sep}{preference.alternative_2}{sep}{preference.relation.name}')

        for relation in self._intensity_relations:
            relations.append(sep.join(
                [
                    relation.alternative_1,
                    relation.alternative_2,
                    relation.alternative_3,
                    relation.alternative_4,
                    relation.relation.name
                ]
            ))
        return relations

    def save_to_file(self, filename: str, parameters: RORParameters):
        data, preferences = super()._prepare_data_for_saving(parameters)
        data.extend(self.__prepare_preferences_data_for_saving())
        data.extend(preferences)

        self._save_data(filename, data)
