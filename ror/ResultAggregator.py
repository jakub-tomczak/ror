from ror.graphviz_helper import draw_rank
from typing import Dict, List
from ror.OptimizationResult import AlternativeOptimizedValue
import numpy as np
import datetime


class ResultAlternative:
    def __init__(self, alternative_name: str, alternative_values: List[float], aggregated_value: float, rank_positon: int):
        self._alternative_name: str = alternative_name
        self._alternative_values = alternative_values
        self._optimizaiton_results: List[AlternativeOptimizedValue] = []
        self._aggregated_value = aggregated_value
        self._rank_position = rank_positon
        self._rank_sum = 0.0


def aggregate_result_default(data: Dict[str, List[float]], mapping) -> List[ResultAlternative]:
    assert 'S' in mapping
    assert 'R' in mapping
    assert 'Q' in mapping
    for key in data:
        assert len(data[key]) == 3, 'Each alternative must have 3 values'

    # columns - data per alpha value
    # rows - data per alternative
    values = np.array([data_values for data_values in data.values()])
    alternatives = np.array(list(data.keys()))

    # sort in R
    R_values = values[:, 0]
    Q_values = values[:, 1]
    S_values = values[:, 2]
    R_sorted = np.sort(R_values)
    R_sorted_args = np.argsort(R_values)
    data_sorted = values[R_sorted_args]
    print('sorted args', R_sorted_args.shape, alternatives.shape)
    alternatives_sorted = alternatives[R_sorted_args]

    ranking_list_R = alternatives_sorted
    ranking_list_Q = alternatives[np.argsort(Q_values)]
    ranking_list_S = alternatives[np.argsort(S_values)]

    now = datetime.datetime.now()
    date_time = now.strftime("%H-%M-%S_%Y-%m-%d")

    draw_rank(ranking_list_R, f'{date_time}_rank_R')
    draw_rank(ranking_list_Q, f'{date_time}rank_Q')
    draw_rank(ranking_list_S, f'{date_time}rank_S')
