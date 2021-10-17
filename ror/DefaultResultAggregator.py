from io import StringIO
from typing import Dict, List, Set
from ror.RORParameters import RORParameters
from ror.RORResult import RORResult
from ror.ResultAggregator import AbstractResultAggregator
from ror.alpha import AlphaValues
from ror.loader_utils import RORParameter
from ror.result_aggregator_utils import Rank, RankItem, create_flat_r_q_s_ranks, get_position_in_rank, group_equal_alternatives_in_ranking, validate_aggregator_arguments
import logging


class DefaultResultAggregator(AbstractResultAggregator):
    def __init__(self) -> None:
        super().__init__()

    def alternatives_are_indifferent(self, better_alternative_in_rank_r: str, worst_alternative_in_rank_r: str, q_rank: Rank, s_rank: Rank) -> bool:
        better_alt_q_rank_position = get_position_in_rank(
            better_alternative_in_rank_r, q_rank)
        better_alt_s_rank_position = get_position_in_rank(
            better_alternative_in_rank_r, s_rank)
        worst_alt_q_rank_position = get_position_in_rank(
            worst_alternative_in_rank_r, q_rank)
        worst_alt_s_rank_position = get_position_in_rank(
            worst_alternative_in_rank_r, s_rank)
        # alterntives are indifferent if
        # better alternative (better in rank r)
        # is worst in q and s ranks than alternative worst (worst in rank r)
        return better_alt_q_rank_position > worst_alt_q_rank_position and better_alt_s_rank_position > worst_alt_s_rank_position

    def explain_result(self, alternative_1: str, alternative_2: str) -> str:
        super().explain_result(alternative_1, alternative_2)

        explanation = StringIO()
        # get numerical results
        result = self._ror_result.get_result_table()
        alternatives = set(result.index)
        assert alternative_1 in alternatives, f'There are no results for alternative {alternative_1}'
        assert alternative_2 in alternatives, f'There are no results for alternative {alternative_2}'
        # get ranks
        r_rank = self._ror_result.get_intermediate_rank('alpha_0.5')
        q_rank = self._ror_result.get_intermediate_rank('alpha_0.0')
        s_rank = self._ror_result.get_intermediate_rank('alpha_1.0')
        final_rank = self._ror_result.final_rank
        # get positions
        # final rank positions
        for alt in [alternative_1, alternative_2]:
            pos = get_position_in_rank(alt, final_rank)
            value = result[alt, 'alpha_sum']
            explanation.write(
                f'Alternative {alt} is at position {pos} in the final rank with total distance of {value}.')

        r_rank_alt_1_position = get_position_in_rank(alternative_1, r_rank)
        r_rank_alt_2_position = get_position_in_rank(alternative_2, r_rank)
        if r_rank_alt_1_position > r_rank_alt_2_position:
            explanation.write(
                f'Alternative {alternative_1} is lower in rank R than alternative {alternative_2}')
        elif r_rank_alt_1_position < r_rank_alt_2_position:
            explanation.write(
                f'Alternative {alternative_1} is higher in rank R than alternative {alternative_2}')
        else:
            explanation.write(
                f'Alternatives {alternative_1} and {alternative_2} are at the same position in rank R')

        # better alternative is lower in rank = has position with lower value
        better_alternative = alternative_1 if r_rank_alt_1_position < r_rank_alt_2_position else alternative_2
        worst_alternative = alternative_1 if better_alternative == alternative_2 else alternative_2
        if self.alternatives_are_indifferent(better_alternative, worst_alternative, q_rank, s_rank):
            explanation.write(
                'Set alternatives as indifferent in the final rank,')
            explanation.write(
                f'because alternative {better_alternative} position in rank q and s are reversed in relation to alternative {worst_alternative}')
        else:
            explanation.write(
                f'Position of alternative {better_alternative} is not reversed in both q and s rank in relation to alternative {worst_alternative}')
        final_rank_alt_1_position = get_position_in_rank(
            alternative_1, final_rank)
        final_rank_alt_2_position = get_position_in_rank(
            alternative_2, final_rank)

        if better_alternative == alternative_1:
            explanation.write(
                f'Alternative {alternative_1} that was better in rank R is at position {final_rank_alt_1_position} in the final rank.')
            explanation.write(
                f'Alternative {alternative_2} that was worst in rank R is at position {final_rank_alt_2_position} in the final rank.')
        else:
            explanation.write(
                f'Alternative {alternative_1} that was worst in rank R is at position {final_rank_alt_1_position} in the final rank.')
            explanation.write(
                f'Alternative {alternative_2} that was better in rank R is at position {final_rank_alt_2_position} in the final rank.')

        return explanation.getvalue()

    def aggregate_results(self, result: RORResult, parameters: RORParameters, *args, **kwargs) -> RORResult:
        super().aggregate_results(result, parameters, *args, **kwargs)
        alpha_values = AlphaValues.from_list(
            parameters.get_parameter(RORParameter.ALPHA_VALUES))
        eps = parameters.get_parameter(RORParameter.EPS)

        data: Dict[str, List[float]] = result.get_results_dict(alpha_values)
        print('data', data)
        validate_aggregator_arguments(data, eps)
        flat_r_rank, flat_q_rank, flat_s_rank = create_flat_r_q_s_ranks(data)

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
                if self.alternatives_are_indifferent(current_alternative.alternative, next_alternative.alternative, q_rank, s_rank):
                    # if q_rank_current_alternative_position > q_rank_next_alternative_position and s_rank_current_alternative_position > s_rank_next_alternative_position:
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

        rank_names = ['alpha_0.5', 'alpha_0.0', 'alpha_1.0']
        ranks = [r_rank, q_rank, s_rank]
        filename = [f'default_rank_R', f'default_rank_Q', f'default_rank_S']
        # get dir for all ranks because dir contains datetime so must be one for all
        dir = self.get_dir_for_rank_image()
        for name, rank, filename in zip(rank_names, ranks, filename):
            alpha_value = alpha_values[name]
            assert alpha_value is not None, f'Rank name {name} is not present in alpha_values provided'
            image_filename = self.draw_rank(rank, dir, filename)
            result.add_intermediate_rank(
                name, Rank(rank, image_filename, alpha_value))
        final_rank_img_path = self.draw_rank(final_rank, dir, f'default_final_rank')

        result.final_rank = Rank(final_rank, final_rank_img_path, 'final rank')
        return result

    def help(self) -> str:
        return """
Method requires 3 ranks: Q with alpha 0.0, R with alpha 0.5 and S with alpha 1.0.
Having 3 ranks it prioritizes rank R. When constructing final rank it takes a position
for all pairs of alternatives ak, aj taking consecutive positions in R,
keep the order between them in the  final ranking if their order is not reversed in S and Q;
otherwise consider them as indifferent, i.e., put them in the same position in the final ranking.
"""
