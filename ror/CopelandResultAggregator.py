from typing import List
from ror import RORModel
from ror.RORParameters import RORParameter, RORParameters
from ror.RORResult import RORResult
from ror.ResultAggregator import AbstractResultAggregator
from ror.alpha import AlphaValue, AlphaValues
from ror.result_aggregator_utils import Rank, RankItem, SimpleRank, create_flat_ranks, group_equal_alternatives_in_ranking
import logging
import numpy as np


class CopelandResultAggregator(AbstractResultAggregator):
    def __init__(self) -> None:
        super().__init__("CopelandResultAggregator")

    def aggregate_results(self, result: RORResult, parameters: RORParameters, *args, **kwargs) -> RORResult:
        super().aggregate_results(result, parameters, *args, **kwargs)
        # get sum of results
        # calculate mean position (across all ranks)
        # create final rank - if positions are equal then the one with
        # we assume that the last alternative gets 1, the best gets len(alternatives)
        number_of_ranks = parameters.get_parameter(
            RORParameter.NUMBER_OF_ALPHA_VALUES)
        eps = parameters.get_parameter(RORParameter.EPS)
        data = result.get_result_table()
        numpy_alternatives: np.ndarray = np.array(list(data.index))
        number_of_alternatives = len(numpy_alternatives)
        logging.info(f'Borda resolver, results {result.get_result_table()}')
        # get name of all columns with ranks, beside last one - with sum
        columns_with_ranks: List[str] = list(
            set(data.columns) - set(['alpha_sum']))
        assert len(columns_with_ranks) == number_of_ranks,\
            'Invalid number of columns in the result or number of ranks'
        # each rank is one voter
        votes = np.zeros(shape=(number_of_alternatives, number_of_alternatives))
        for column_name in columns_with_ranks:
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
                    if row_idx == col_idx:
                        # leave 0 value for same alternative
                        continue
                    elif abs(row_alternative_value - column_alternative_value) <= eps:
                        votes[row_idx, col_idx] += 0.5
                        votes[col_idx, row_idx] += 0.5
                    elif row_alternative_value > column_alternative_value:
                        votes[row_idx, col_idx] += 1
                    elif row_alternative_value < column_alternative_value:
                        votes[col_idx, row_idx] += 1

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
        wrapped_copeland_final_rank: List[List[RankItem]] = [[rank_item] for rank_item in final_rank_items]

        # produce rank images
        alpha_values = self.get_alpha_values(result.model, parameters)
        results_per_alternative = result.get_results_dict(alpha_values)
        ranks = create_flat_ranks(results_per_alternative)
        
        # generate rank images
        dir = self.get_dir_for_rank_image()
        for alpha_value, intermediate_flat_rank in zip(alpha_values.values, ranks):
            # create intermediate ranks for drawing
            grouped_rank = group_equal_alternatives_in_ranking(
                intermediate_flat_rank, eps)
            name = f'alpha_{alpha_value}'
            image_filename = self.draw_rank(grouped_rank, dir, f'copeland_{name}')
            result.add_intermediate_rank(
                name,
                Rank(intermediate_flat_rank, image_filename, AlphaValue.from_value(alpha_value))
            )
        
        final_rank_image_filename = self.draw_rank(wrapped_copeland_final_rank, dir, 'copeland_final_rank')

        result.final_rank = Rank(
            wrapped_copeland_final_rank,
            img_filename=final_rank_image_filename,
        )
        return result


    def get_alpha_values(self, model: RORModel, parameters: RORParameters) -> AlphaValues:
        number_of_alpha_values = parameters.get_parameter(RORParameter.NUMBER_OF_ALPHA_VALUES)
        return AlphaValues.from_list(np.linspace(0.0, 1.0, number_of_alpha_values))
        

    def help(self) -> str:
        return """
Copeland result aggregator produces matrix that maps results from
intermediate ranks into votes.
Rows and columns are represented by alternatives.
For each rank:
    if alternative in row has same value as alternative in column then each alternative recives 0.5
    if alternative in row has greater value than alternative in column then alternative in row receives 1
    if alternative in row has lower value than alternative in column then column in row receives 0
    alternative for itself receives 0
"""
