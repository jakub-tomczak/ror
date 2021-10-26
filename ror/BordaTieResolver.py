from typing import DefaultDict, Dict, List
from ror.RORParameters import RORParameter, RORParameters
from ror.RORResult import RORResult
from ror.AbstractTieResolver import AbstractTieResolver
from ror.result_aggregator_utils import Rank, RankItem, SimpleRank, group_equal_alternatives_in_ranking
import logging
import numpy as np


class BordaTieResolver(AbstractTieResolver):
    def __init__(self) -> None:
        super().__init__("BordaTieResolver")

    def resolve_rank(self, rank: SimpleRank, result: RORResult, parameters: RORParameters) -> SimpleRank:
        super().resolve_rank(rank, result, parameters)
        # get sum of results
        # calculate mean position (across all ranks)
        # create final rank - if positions are equal then the one with
        # we assume that the last alternative gets 1, the best gets len(alternatives)
        eps = parameters.get_parameter(RORParameter.EPS)
        data = result.get_result_table()
        numpy_alternatives: np.ndarray = np.array(list(data.index))
        number_of_alternatives = len(numpy_alternatives)
        logging.debug(f'Borda resolver, results {result.get_result_table()}')
        # get name of all columns with ranks, beside last one - with sum
        columns_with_ranks: List[str] = list(
            set(data.columns) - set(['alpha_sum']))
        # go through each rank (per each alpha value)
        # sort values to get positions for borda voting
        alternative_to_mean_position: Dict[str, float] = DefaultDict(lambda: 0.0)
        for column_name in columns_with_ranks:
            sorted_indices = np.argsort(data[column_name])
            sorted_alternatives = numpy_alternatives[sorted_indices]
            logging.debug(
                f'Sorted alternatives for rank {column_name} is {sorted_alternatives}')
            # go through all alternatives, sorted by value for a specific alpha value
            for index, alternative in enumerate(sorted_alternatives):
                alternative_to_mean_position[alternative] += (
                    number_of_alternatives - index)
        for alternative in alternative_to_mean_position:
            alternative_to_mean_position[alternative] /= len(columns_with_ranks)

        sorted_alpha_sum = np.sort(data['alpha_sum'])
        sorted_alpha_sum_args = np.argsort(data['alpha_sum'])
        sorted_alpha_sum_alternatives = numpy_alternatives[sorted_alpha_sum_args]
        initial_final_rank = [
            RankItem(alternative, value)
            for value, alternative in zip(sorted_alpha_sum, sorted_alpha_sum_alternatives)
        ]
        final_rank_with_ties = group_equal_alternatives_in_ranking(
            initial_final_rank, eps)
        logging.debug(f'Final rank with ties {final_rank_with_ties}')
        final_rank_with_borda: List[RankItem] = []

        for items_at_same_position in final_rank_with_ties:
            if len(items_at_same_position) > 1:
                # gets borda's value for all alternatives from this position
                items_borda_value = [
                    (r_item.alternative,
                     alternative_to_mean_position[r_item.alternative])
                    for r_item in items_at_same_position
                ]
                # sort all items from the same position by mean position from all ranks using borda voting
                sorted_by_mean_position = sorted(
                    items_borda_value, key=lambda item: item[1])
                logging.debug(
                    f'items at the same position: {items_at_same_position}')
                for alternative, mean_position in sorted_by_mean_position:
                    logging.debug(
                        f'Alternative: {alternative}, mean position: {mean_position}')
                    final_rank_with_borda.append(
                        RankItem(alternative, items_at_same_position[0].value))
            else:
                # only one item at this position, add it to the new final rank
                final_rank_with_borda.append(items_at_same_position[0])
        logging.debug(f'Final rank without ties {final_rank_with_borda}')
        
        # use wrapped final rank - to have consistent constructor for Rank object that assumes rank with ties
        # therefore list of lists of RankItem is required
        wrapped_borda_final_rank: List[List[RankItem]] = [
            [rank_item] for rank_item in final_rank_with_borda]
        return wrapped_borda_final_rank

    def help(self) -> str:
        return """
Borda resolver uses Borda voting to decide what should be the better alternative
in case of a tie.
It starts from calculating a sum of distances from all alpha values for each alternative.
Then, a final rank is produced. If any alternatives are at the same place then wins the one
that had lower mean position over all other ranks.
"""
