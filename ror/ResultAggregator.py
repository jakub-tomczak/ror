import logging
from collections import defaultdict
from ror.RORResult import RORResult
from ror.alpha import AlphaValues
from ror.graphviz_helper import draw_rank
from typing import Dict, List, Set
import datetime
from ror.result_aggregator_utils import BIG_NUMBER, RankItem, create_flat_ranks,\
    from_rank_to_alternatives, group_equal_alternatives_in_ranking, validate_aggregator_arguments


def aggregate_result_default(ror_result: RORResult, alpha_values: AlphaValues, eps: float) -> RORResult:
    data: Dict[str, List[float]] = ror_result.get_results_dict(alpha_values)
    validate_aggregator_arguments(data, eps)
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

    ror_result.final_rank = final_rank
    return ror_result


'''
Function that aggregates results from ranks: R, Q and S by adding weights to ranks.
Weights must be greater or equal 0.0
Weight > 1.0 increases importance of the rank (lowers value)
Weight < 1.0 decreases importance of the rank (increases value)
Weight == 1.0 doesn't change the importance of the rank
'''


def weighted_results_aggregator(ror_result: RORResult, alpha_values: AlphaValues, weights: Dict[str, float], eps: float) -> RORResult:
    data = ror_result.get_results_dict(alpha_values)
    validate_aggregator_arguments(data, eps)
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
    ror_result.final_rank = final_rank
    return ror_result
