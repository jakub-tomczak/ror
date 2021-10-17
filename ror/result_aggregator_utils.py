from typing import Dict, List, Tuple, Union
from ror.OptimizationResult import AlternativeOptimizedValue
import numpy as np
from ror.alpha import AlphaValue

BIG_NUMBER = 10e10


class ResultAlternative:
    def __init__(self, alternative_name: str, alternative_values: List[float], aggregated_value: float, rank_positon: int):
        self._alternative_name: str = alternative_name
        self._alternative_values = alternative_values
        self._optimizaiton_results: List[AlternativeOptimizedValue] = []
        self._aggregated_value = aggregated_value
        self._rank_position = rank_positon
        self._rank_sum = 0.0


class RankItem:
    def __init__(self, alternative: str, value: float) -> None:
        self._alternative: str = alternative
        self._value: float = value

    @property
    def value(self):
        return self._value

    @property
    def alternative(self):
        return self._alternative

    def __repr__(self):
        return f'<RankItem: alternative: {self._alternative}, value: {self._value}>'

    def __eq__(self, o: object) -> bool:
        if type(o) is RankItem:
            return o._value == self._value and o._alternative == self._alternative
        return False


class Rank:
    def __init__(self, rank: List[List[RankItem]], img_filename: str, alpha_value: AlphaValue) -> None:
        self.__rank: List[List[RankItem]] = rank
        self.__img_filename: str = img_filename
        self.__alpha_value: AlphaValue = alpha_value

    @property
    def image_filename(self) -> str:
        return self.__img_filename

    @property
    def alpha_value(self) -> AlphaValue:
        return self.__alpha_value
    
    @property
    def rank(self) -> List[List[RankItem]]:
        return self.__rank

def values_equal_with_epsilon(first_alternative_value, second_alternative_value, epsilon: float) -> bool:
    return abs(first_alternative_value - second_alternative_value) < epsilon


def group_equal_alternatives_in_ranking(rank: List[RankItem], eps: float) -> List[List[RankItem]]:
    if len(rank) < 1:
        return []
    new_rank: List[List[RankItem]] = [[rank[0]]]
    for index in range(1, len(rank)):
        if values_equal_with_epsilon(rank[index].value, rank[index-1].value, eps):
            # add another item at the same place in the new rank
            new_rank[len(new_rank)-1].append(rank[index])
        else:
            # add another item at the next position in the rank
            new_rank.append([rank[index]])
    return new_rank


def from_alternatives_and_values_to_rank(alternatives: List[str], values: List[float]) -> List[RankItem]:
    return [RankItem(alternative, value) for alternative, value in zip(alternatives, values)]


def from_rank_to_alternatives(rank: List[List[RankItem]]) -> List[List[str]]:
    return [[item.alternative for item in items] for items in rank]

def get_position_in_rank(alternative_name: str, rank: Union[Rank, List[List[RankItem]]]) -> int:
    position = 1
    iterable = rank if type(rank) is list else rank.rank
    for rank_item in iterable:
        for item in rank_item:
            if item.alternative == alternative_name:
                return position
        position += 1
    raise Exception(f'Alternative {alternative_name} is not in rank with alpha value {rank.alpha_value}')

def validate_aggregator_arguments(data: Dict[str, List[float]], eps: float):
    assert eps > 0.0, 'Epsilon value must be higher than 0'
    for key in data:
        assert len(data[key]) == 3, 'Each alternative must have 3 values'


'''
Creates ranks from alternatives and their values
Flat rank is a rank with no items at the same position.
If two alternatives have the same position then they are on different places in the list anyway.
i.e.
alternatives: [b01, b02, b03]
values: [0.1, 0.4, 0.1]
results in
[(b01, 0.1), (b03, 0.1), (b02, 0.4)]
'''


def create_flat_r_q_s_ranks(data: Dict[str, List[float]]) -> Tuple[List[RankItem], List[RankItem], List[RankItem]]:
    '''
    Returns tuplce of flat ranks - ranks with one item per index for 3 ranks.
    Each item will be a list of alternatives, one per index.
    '''
    # columns - data per alpha value
    # rows - data per alternative
    values = np.array([data_values for data_values in data.values()])
    alternatives = np.array(list(data.keys()))

    # sort in R
    R_values = values[:, 0]
    Q_values = values[:, 1]
    S_values = values[:, 2]
    # sort descending
    R_sorted = np.sort(R_values)
    R_sorted_args = np.argsort(R_values)

    ranking_list_R = alternatives[R_sorted_args]
    Q_sorted = np.sort(Q_values)
    ranking_list_Q = alternatives[np.argsort(Q_values)]
    S_sorted = np.sort(S_values)
    ranking_list_S = alternatives[np.argsort(S_values)]
    flat_r_rank = from_alternatives_and_values_to_rank(
        ranking_list_R, R_sorted)
    flat_q_rank = from_alternatives_and_values_to_rank(
        ranking_list_Q, Q_sorted)
    flat_s_rank = from_alternatives_and_values_to_rank(
        ranking_list_S, S_sorted)
    return flat_r_rank, flat_q_rank, flat_s_rank

def create_flat_ranks(data: Dict[str, List[float]]) -> List[List[RankItem]]:
    '''
    Returns list of flat ranks - ranks with one item per index.
    For 3 alpha values a list of 3 items will be returned.
    Each item will be a list of alternatives, one per index.
    '''
    # columns - data per alpha value
    # rows - data per alternative
    values = np.array([data_values for data_values in data.values()]).T
    alternatives = np.array(list(data.keys()))

    # sort descending
    flat_ranks = []
    for value in values:
        sorted_values = np.sort(value)
        sorted_args = np.argsort(value)
        ranking_list = alternatives[sorted_args]
        flat_ranks.append(from_alternatives_and_values_to_rank(ranking_list, sorted_values))
    return flat_ranks
