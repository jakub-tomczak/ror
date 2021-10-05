import os
from ror.Relation import PREFERENCE_NAME_TO_RELATION
from ror.PreferenceRelations import PreferenceIntensityRelation, PreferenceRelation
from typing import Dict, List, Tuple, DefaultDict
from ror.Dataset import Dataset, RORDataset
from collections import defaultdict
import numpy as np
from enum import Enum


class AvailableParameters(Enum):
    EPS = 'eps'
    INITIAL_ALPHA = 'initial_alpha'


class LoaderResult:
    def __init__(self, ror_dataset: RORDataset, parameters: Dict[AvailableParameters, float]) -> None:
        self._ror_dataset: RORDataset = ror_dataset
        self._parameters: Dict[AvailableParameters, float] = parameters

    @property
    def dataset(self) -> RORDataset:
        return self._ror_dataset

    @property
    def parameters(self) -> Dict[AvailableParameters, float]:
        return self._parameters


def get_default_parameter_value(parameter: AvailableParameters) -> float:
    if parameter == AvailableParameters.EPS:
        return Dataset.DEFAULT_EPS
    elif parameter == AvailableParameters.INITIAL_ALPHA:
        return 0.0
    else:
        return None


def read_txt_by_section(filename: str) -> DefaultDict[str, List[str]]:
    '''
    Reads file section by section.
    Section must start with #. Section template: #Name_of_the_section
    Returns: dictionary, keys are the names of sections (with hash sign),
    values are list of lines from that section.
    Returns None if file doesn't exist or no section was found.
    '''
    if not os.path.exists(filename):
        raise DatasetReaderException(f"file {filename} doesn't exist")
    current_section = None
    sections_data: DefaultDict[str, List[str]] = defaultdict(list)
    with open(filename, 'r') as file:
        for line in file:
            if len(line) < 1:
                # skip empty lines
                continue
            line_no_whitespaces = line.strip()
            if line_no_whitespaces.startswith("#"):
                current_section = line_no_whitespaces
            elif current_section is not None:
                sections_data[current_section].append(line_no_whitespaces)
            else:
                print('Warning: every line with no section defined is skipped.')
    if len(sections_data) < 1:
        return None

    return sections_data


def parse_criterion(criterion: str) -> Tuple[str, str]:
    data = criterion.strip().split('[')

    if len(data) != 2 or len(data[0]) < 1 or len(data[1]) < 1:
        raise DatasetReaderException(f"Failed to parse criterion {criterion}")
    criterion_type = data[1][0]
    if criterion_type not in Dataset.CRITERION_TYPES.values():
        raise DatasetReaderException(
            f"Invalid criterion type: {criterion_type}, expected values: {criterion_type.values()}")
    return (data[0], criterion_type)


def parse_data_section(sectioned_data: List[str], column_separator: str) -> Tuple[List[str], List[Tuple[str, str]], List[List[float]]]:
    '''
    Parse data section.
    '''

    header_data: List[str] = sectioned_data[0].split(column_separator)
    expected_number_of_columns = len(header_data)

    # skip first column - this should be id
    parsed_criteria = [parse_criterion(criterion)
                       for criterion in header_data[1:]]
    # filter out invalid criteria
    criteria = list(
        filter(lambda criterion: criterion is not None, parsed_criteria))
    # header should have id column as the first one, so subtract 1
    if len(criteria) != expected_number_of_columns - 1:
        raise DatasetReaderException(
            "Failed to read dataset from txt file: failed to parse all criteria.")
    # rest of the lines in the data list should have only alternatives data
    alternatives_data = [line.split(column_separator)
                         for line in sectioned_data[1:]]
    alternatives_data = map(
        lambda x: (x[0], x[1:]) if len(x) > 1 else (x[0]),
        alternatives_data
    )
    alternatives: List[str] = []
    values: List[float] = []
    for alternative in alternatives_data:
        if len(alternative) < 2:
            raise DatasetReaderException(
                f"Failed to read dataset from txt file: failed to parse alternative {alternative}")
        if alternative[0] in alternatives:
            raise DatasetReaderException(
                f"Failed to read dataset from txt file: alternative {alternative[0]} already loaded")
        alternatives.append(alternative[0])

        if len(alternative[1]) != expected_number_of_columns - 1:
            raise DatasetReaderException(
                f"Failed to read dataset from txt file: expected {expected_number_of_columns} values, got {len(alternative[1])}")
        try:
            alternative_values = [float(number.strip())
                                  for number in alternative[1]]
            values.append(alternative_values)
        except:
            raise DatasetReaderException(
                f"Failed to read dataset from txt file: failed to parse line with numbers: {alternative[1]}")
    return (alternatives, criteria, values)


def parse_preferences_section(sectioned_data: List[str], alternatives: List[str], separator: str) -> Tuple[List[PreferenceRelation], List[PreferenceIntensityRelation]]:
    preference_relations = []
    preference_intensities = []

    def check_if_alternatives_exists(alternative: List[str], alternatives_list: List[str]) -> bool:
        return len(set(alternative) - set(alternatives_list)) == 0

    for line in sectioned_data:
        if len(line) < 1:
            continue
        splited = line.split(separator)
        if len(splited) == 3:
            # preference relation
            alternative_1, alternative_2 = [
                alternative.strip() for alternative in splited[:2]]
            relation_name = splited[2].strip()
            if not check_if_alternatives_exists([alternative_1, alternative_2], alternatives):
                raise DatasetReaderException(
                    f"Failed to read dataset from txt file: one of alternatives in the relation {line} doesn't exist")

            if relation_name not in PREFERENCE_NAME_TO_RELATION:
                raise DatasetReaderException(
                    f"Failed to read dataset from txt file: relation name '{relation_name}' is not supported")

            preference_relations.append(PreferenceRelation(
                alternative_1,
                alternative_2,
                PREFERENCE_NAME_TO_RELATION[relation_name]
            ))
        elif len(splited) == 5:
            # preference intensity relation
            alternative_1, alternative_2, alternative_3, alternative_4 = [
                alternative.strip() for alternative in splited[:4]]
            relation_name = splited[4].strip()
            if not check_if_alternatives_exists([alternative_1, alternative_2, alternative_3, alternative_4], alternatives):
                raise DatasetReaderException(
                    f"Failed to read dataset from txt file: one of alternatives in the relation {line} doesn't exist")

            if relation_name not in PREFERENCE_NAME_TO_RELATION:
                raise DatasetReaderException(
                    f"Failed to read dataset from txt file: relation name '{relation_name}' is not supported")

            preference_intensities.append(PreferenceIntensityRelation(
                alternative_1,
                alternative_2,
                alternative_3,
                alternative_4,
                PREFERENCE_NAME_TO_RELATION[relation_name]
            ))
        else:
            raise DatasetReaderException(
                f"Failed to read dataset from txt file: Invalid number of arguments for preference in line {line}")

    return (preference_relations, preference_intensities)


def parse_parameters_section(sectioned_data: List[str]) -> Dict[AvailableParameters, float]:
    parameter_value_separator = '='
    parameters: Dict[AvailableParameters, float] = dict()
    for line in sectioned_data:
        splited = line.split(parameter_value_separator)
        if len(splited) != 2:
            raise DatasetReaderException(
                f"Failed to read dataset from txt file: failed to parse parameter from line: {line}. Parameter should be in format 'key=value'")
        key, value = splited
        parsed_value: float = None
        try:
            parsed_value = float(value)
        except:
            raise DatasetReaderException(
                f"Failed to read dataset from txt file: failed to parse parameter value: {value}. All parameter values should be of float type")
        # check whether parameter has a valid value
        for parameter in AvailableParameters:
            if key == parameter.value:
                parameters[parameter] = parsed_value
    # set default values to parameters that were not in the file
    parameters_with_no_value = set(AvailableParameters) - set(parameters.keys())
    for parameter in parameters_with_no_value:
        default_value = get_default_parameter_value(parameter)
        print(
            f'Info: Parameter {parameter.name} is not present in the dataset, adding it with default value: {default_value}')
        parameters[parameter] = default_value

    return parameters


def read_dataset_from_txt(filename: str) -> LoaderResult:
    DATA_SECTION = "#Data"
    PREFERENCES_SECTION = "#Preferences"
    PARAMETERS_SECTION = "#Parameters"

    section_data = read_txt_by_section(filename)
    if section_data is None:
        raise DatasetReaderException("Failed to read dataset from txt file.")
    if DATA_SECTION not in section_data:
        raise DatasetReaderException(
            "Failed to read dataset from txt file: no data section in the file.")

    sectioned_data = section_data[DATA_SECTION]
    # detect column separator by looking at header
    # valid separators are , and ;
    # , has precedence over ;
    valid_separators = [',', ';']
    separator = None
    for sep in valid_separators:
        if sectioned_data[0].find(sep):
            separator = sep
            break
    if separator is None:
        raise DatasetReaderException(
            f'No column separator detected. Valid separators are: {" or ".join(valid_separators)}')

    if len(sectioned_data) < 2:
        raise DatasetReaderException(
            "Failed to read dataset: expected at least 2 lines: first with header, rest with alternatives data.")

    result = parse_data_section(section_data[DATA_SECTION], separator)
    alternatives, criteria, values = result

    result = parse_preferences_section(
        section_data[PREFERENCES_SECTION], alternatives, separator)
    preference_relations, preferences_intensities = result

    parameters = parse_parameters_section(
        section_data[PARAMETERS_SECTION]
    )

    numpy_values = np.array(values)
    dataset = RORDataset(
        alternatives=alternatives,
        data=numpy_values,
        criteria=criteria,
        preference_relations=preference_relations,
        intensity_relations=preferences_intensities,
        eps=parameters[AvailableParameters.EPS]
    )
    return LoaderResult(dataset, parameters)


class DatasetReaderException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
