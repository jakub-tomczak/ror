from typing import DefaultDict, Dict, List
from ror.RORParameters import RORParameter, RORParameters
from ror.RORResult import RORResult
from ror.AbstractTieResolver import AbstractTieResolver
from ror.result_aggregator_utils import RankItem, SimpleRank
import logging
import numpy as np


class CopelandTieResolver(AbstractTieResolver):
    def __init__(self) -> None:
        super().__init__("CopelandTieResolver")

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
        logging.debug(f'Copeland tie resolver, results {result.get_result_table()}')
        # get name of all columns with ranks, beside last one - with sum
        columns_with_ranks: List[str] = list(
            set(data.columns) - set(['alpha_sum']))
        # each rank is one voter
        # votes for each alterantive are in the row
        # if alternative from row i has value 1 in the column j
        # it means that alternative i is preffered in some rank to alternative j.
        # the best alternative should have the biggest sum
        votes = np.zeros(shape=(number_of_alternatives, number_of_alternatives))
        for column_name in columns_with_ranks:
            logging.debug(f'Checking column {column_name}')
            for row_idx, row_alternative_name in enumerate(numpy_alternatives):
                # run only over columns that index is greater than row index - less calculations
                for col_idx, column_alternative_name in enumerate(numpy_alternatives[row_idx:]):
                    row_alternative_value = data.loc[row_alternative_name, column_name]
                    column_alternative_value = data.loc[column_alternative_name, column_name]
                    # if in this rank alternative from row is preferred than the alternative from col
                    # then row alternative's value is increased by one (one vote)
                    # if alternative from column is preferred by alternative from row
                    # then alternative from column gets one point.
                    # Otherwise (alternatives' values are equal, with eps precision)
                    # both alternatives get 0.5
                    if row_alternative_value + eps > column_alternative_value:
                        logging.debug(f'Alternative in row {row_alternative_name} has greater value than alternative in column {column_alternative_name}')
                        votes[row_idx, col_idx] += 1
                    elif row_alternative_value < column_alternative_value + eps:
                        logging.debug(f'Alternative in row {row_alternative_name} has lower value than alternative in column {column_alternative_name}')
                        votes[col_idx, row_idx] += 1
                    else:
                        logging.debug(f'Alternative in row {row_alternative_name} has same value as alternative in column {column_alternative_name}')
                        votes[row_idx, col_idx] += 0.5
                        votes[col_idx, row_idx] += 0.5

        # aggregate votes - calculate
        per_alternative_votes_sum = np.zeros(shape=(number_of_alternatives))
        for alternative_idx in range(len(numpy_alternatives)):
            per_alternative_votes_sum[alternative_idx] = np.sum(votes[alternative_idx, :])
        
        final_rank = np.sort(per_alternative_votes_sum)
        final_rank_alternatives_indices = np.argsort(per_alternative_votes_sum)
        final_rank_alternatives = numpy_alternatives[final_rank_alternatives_indices]

        # produce List[RankItem]
        final_rank_items = [
            RankItem(alternative, value)
            for alternative, value in zip(final_rank_alternatives, final_rank)
        ]
        # helper structure for deciding which alternative from 2 has better value
        alternative_to_final_rank: Dict[str, float] = {
            item.alternative: item.value
            for item in final_rank_items
        }

        final_rank_with_copeland: List[RankItem] = []
        for items_at_same_position in rank:
            if len(items_at_same_position) > 1:
                # gets copeland's value for all alternatives from this position
                items_copeland_value = [
                    (r_item.alternative, alternative_to_final_rank[r_item.alternative])
                    for r_item in items_at_same_position
                ]
                # sort all items from the same position using value obtained in copeland voting earlier
                # reverse sorting order because the best alternative has the biggest value
                sorted_by_copeland_sum = sorted(items_copeland_value, key=lambda item: item[1], reverse=True)
                logging.debug(f'items at the same position: {items_at_same_position}')
                for alternative, copeland_sum in sorted_by_copeland_sum:
                    logging.debug(f'Alternative: {alternative}, obtained sum: {copeland_sum}')
                    # items_at_same_position[0].value == give ayny value - all items at this position had same value
                    final_rank_with_copeland.append(RankItem(alternative, items_at_same_position[0].value))
            else:
                # only one item at this position, add it to the new final rank
                final_rank_with_copeland.append(items_at_same_position[0])

        # wrap into list - produce List[List[RankItem]]
        return [[item] for item in final_rank_items]

    def help(self) -> str:
        return """
Copeland tie resolver produces matrix that maps results from
intermediate ranks into votes.
Rows and columns are represented by alternatives.
For each rank:
    if alternative in row has same value as alternative in column then each alternative recives 0.5
    if alternative in row has greater value than alternative in column then alternative in row receives 1
    if alternative in row has lower value than alternative in column then column in row receives 0
    alternative for itself receives 0
"""
