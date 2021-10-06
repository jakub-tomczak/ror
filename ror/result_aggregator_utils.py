from typing import Dict, List, Tuple
from ror.OptimizationResult import AlternativeOptimizedValue
import numpy as np

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


def create_flat_ranks(data: Dict[str, List[float]]) -> Tuple[List[RankItem], List[RankItem], List[RankItem]]:
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
