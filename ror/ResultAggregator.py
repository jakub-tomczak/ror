import logging
from collections import defaultdict
from ror.graphviz_helper import draw_rank
from typing import Dict, List, Set, Tuple
from ror.OptimizationResult import AlternativeOptimizedValue
import numpy as np
import datetime

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


def validate_aggregator_arguments(data: Dict[str, List[float]], mapping: Dict[str, str], eps: float):
    assert eps > 0.0, 'Epsilon value must be higher than 0'
    '''
    Mapping is a dictionary containing letter with ranking [Q, R, S] as key and their description as values.
    '''
    assert 'S' in mapping
    assert 'R' in mapping
    assert 'Q' in mapping
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


def aggregate_result_default(data: Dict[str, List[float]], mapping: Dict[str, str], eps: float) -> List[List[RankItem]]:
    validate_aggregator_arguments(data, mapping, eps)
    flat_r_rank, flat_q_rank, flat_s_rank = create_flat_ranks(data)

    '''
    Creates a mapping: alternative -> position
    '''
    def create_rank_positions(rank: List[List[RankItem]]) -> Dict[str, int]:
        positions: Dict[str, int] = dict()
        position_index = 1
        for rank_items in rank:
            for item in rank_items:
                positions[item.alternative] = position_index
            position_index += 1
        return positions

    r_rank = group_equal_alternatives_in_ranking(flat_r_rank, eps)
    q_rank = group_equal_alternatives_in_ranking(flat_q_rank, eps)
    q_rank_positions = create_rank_positions(q_rank)
    s_rank = group_equal_alternatives_in_ranking(flat_s_rank, eps)
    s_rank_positions = create_rank_positions(s_rank)

    logging.debug('flat r rank')
    logging.debug(flat_r_rank)

    final_rank: List[List[RankItem]] = []
    alternatives_checked: Set[str] = set()

    logging.debug('*'*100)
    # now we check whether there are indifferent alternatives
    # alternative a_i and a_j are indifferent if:
    # 1. If a_i < a_j in r_rank and a_i > a_j in q_rank and a_i > a_j in s_rank (position changes in q and s ranks)
    for alternative_index in range(len(flat_r_rank)-1):
        current_alternative = flat_r_rank[alternative_index]
        if current_alternative.alternative in alternatives_checked:
            logging.debug(
                f'skipping alternative {current_alternative.alternative} - already in final rank')
            continue

        logging.debug(f'checked alternatives {alternatives_checked}')
        alternatives_checked.add(current_alternative.alternative)
        logging.debug(
            f'checking alternative {current_alternative.alternative}, r rank position: {alternative_index+1}')
        final_rank.append([current_alternative])
        # get rank position of the current alternative in q and s rank to compare it with the position
        # of the next alternative in q and s rank
        q_rank_current_alternative_position = q_rank_positions[current_alternative.alternative]
        s_rank_current_alternative_position = s_rank_positions[current_alternative.alternative]
        logging.debug(
            f'q rank: {q_rank_current_alternative_position}, s rank {s_rank_current_alternative_position}')
        for next_alternative_index in range(alternative_index+1, len(flat_r_rank)):
            next_alternative = flat_r_rank[next_alternative_index]
            if next_alternative.alternative in alternatives_checked:
                logging.debug(
                    f'skipping next alternative {next_alternative.alternative} - already in final rank')
                continue
            q_rank_next_alternative_position = q_rank_positions[next_alternative.alternative]
            s_rank_next_alternative_position = s_rank_positions[next_alternative.alternative]
            logging.debug(
                f'next alternative: {next_alternative.alternative}, r rank {next_alternative_index+1}')
            logging.debug(
                f'q rank: {q_rank_next_alternative_position}, s rank {s_rank_next_alternative_position}')
            # r_rank was sorted so current_alternative_position <= r_rank_next_alternative_position must be always true
            # which means that next_alternative must have higher position in the rank (worst) than current_alternative
            if q_rank_current_alternative_position > q_rank_next_alternative_position and s_rank_current_alternative_position > s_rank_next_alternative_position:
                logging.debug(
                    f'alternative {next_alternative} is indifferent to {current_alternative}')
                logging.debug(
                    f'adding alternative {next_alternative.alternative} to the alternatives {final_rank[len(final_rank)-1]}')
                # add next_alternative at the same place in the final rank as the current alternative
                final_rank[len(final_rank)-1].append(next_alternative)
                alternatives_checked.add(next_alternative.alternative)
            else:
                logging.debug(
                    f'alternative {next_alternative} is worst than alternative {current_alternative}')
                logging.debug('-'*30)
            # else: next alternative is not added to the final rank yet - it will be added
            # in the next iteration of the outer for loop

    now = datetime.datetime.now()
    date_time = now.strftime("%H-%M-%S_%Y-%m-%d")

    draw_rank(from_rank_to_alternatives(r_rank), f'default_{date_time}_rank_R')
    draw_rank(from_rank_to_alternatives(q_rank), f'default_{date_time}_rank_Q')
    draw_rank(from_rank_to_alternatives(s_rank), f'default_{date_time}_rank_S')
    draw_rank(from_rank_to_alternatives(final_rank),
              f'default_{date_time}_final_rank')

    return final_rank


'''
Function that aggregates results from ranks: R, Q and S by adding weights to ranks.
Weights must be greater or equal 0.0
Weight > 1.0 increases importance of the rank (lowers value)
Weight < 1.0 decreases importance of the rank (increases value)
Weight == 1.0 doesn't change the importance of the rank
'''


def weighted_results_aggregator(data: Dict[str, List[float]], mapping: Dict[str, str], weights: Dict[str, float], eps: float) -> List[List[RankItem]]:
    validate_aggregator_arguments(data, mapping, eps)
    assert 'S' in weights, 'weights dict must contain weight for S rank'
    assert 'R' in weights, 'weights dict must contain weight for R rank'
    assert 'Q' in weights, 'weights dict must contain weight for Q rank'

    assert all([weight >= 0.0 for weight in weights.values()]
               ), 'All weights must be greater or equal 0.0'

    # divide values by weights - alternative value is the distance to the ideal alternative
    # so we need to divide instead of multiplying
    weighted_data: Dict[str, List[float]] = {}
    for alternative in data:
        r_value, q_value, s_value = data[alternative]
        weighted_data[alternative] = [
            r_value / weights['R'] if weights['R'] > 0.0 else BIG_NUMBER,
            q_value / weights['Q'] if weights['Q'] > 0.0 else BIG_NUMBER,
            s_value / weights['S'] if weights['S'] > 0.0 else BIG_NUMBER
        ]

    flat_r_rank, flat_q_rank, flat_s_rank = create_flat_ranks(weighted_data)

    values_per_alternative: Dict[str, float] = defaultdict(lambda: 0)
    for r_item, q_item, s_item in zip(flat_r_rank, flat_q_rank, flat_s_rank):
        values_per_alternative[r_item.alternative] += r_item.value
        values_per_alternative[q_item.alternative] += q_item.value
        values_per_alternative[s_item.alternative] += s_item.value

    sorted_final_rank = sorted(
        values_per_alternative.items(), key=lambda alternative: alternative[1])
    # wrap sorted final rank into RankItem
    final_rank = [RankItem(alternative, value)
                  for alternative, value in sorted_final_rank]
    # place same results into same positions
    final_rank = group_equal_alternatives_in_ranking(final_rank, eps)

    # draw positions
    # create intermediate ranks for drawing
    r_rank = group_equal_alternatives_in_ranking(flat_r_rank, eps)
    q_rank = group_equal_alternatives_in_ranking(flat_q_rank, eps)
    s_rank = group_equal_alternatives_in_ranking(flat_s_rank, eps)
    now = datetime.datetime.now()
    date_time = now.strftime("%H-%M-%S_%Y-%m-%d")
    draw_rank(from_rank_to_alternatives(r_rank),
              f'weighted_{date_time}_rank_R')
    draw_rank(from_rank_to_alternatives(q_rank),
              f'weighted_{date_time}_rank_Q')
    draw_rank(from_rank_to_alternatives(s_rank),
              f'weighted_{date_time}_rank_S')
    draw_rank(from_rank_to_alternatives(final_rank),
              f'weighted_{date_time}_final_rank')

    # return result
    return final_rank
