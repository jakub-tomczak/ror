from typing import List
from ror import RORModel
from ror.CopelandVoter import CopelandVoter
from ror.RORParameters import RORParameter, RORParameters
from ror.RORResult import RORResult
from ror.ResultAggregator import AbstractResultAggregator
from ror.alpha import AlphaValue, AlphaValues
from ror.result_aggregator_utils import Rank, RankItem, create_flat_ranks, group_equal_alternatives_in_ranking
import logging
import numpy as np


class CopelandResultAggregator(AbstractResultAggregator):
    def __init__(self) -> None:
        self.__copeland_voter: CopelandVoter = CopelandVoter()
        super().__init__("CopelandResultAggregator")
    
    @property
    def voter(self) -> CopelandVoter:
        return self.__copeland_voter

    def aggregate_results(self, result: RORResult, parameters: RORParameters) -> RORResult:
        super().aggregate_results(result, parameters)
        # get sum of results
        # calculate mean position (across all ranks)
        # create final rank - if positions are equal then the one with
        # we assume that the last alternative gets 1, the best gets len(alternatives)
        number_of_ranks = parameters.get_parameter(
            RORParameter.NUMBER_OF_ALPHA_VALUES)
        eps = parameters.get_parameter(RORParameter.EPS)
        data = result.get_result_table()
        numpy_alternatives: np.ndarray = np.array(list(data.index))
        logging.debug(f'Borda resolver, results {result.get_result_table()}')
        # get name of all columns with ranks, beside last one - with sum
        columns_with_ranks: List[str] = list(
            set(data.columns) - set(['alpha_sum']))
        assert len(columns_with_ranks) == number_of_ranks,\
            'Invalid number of columns in the result or number of ranks'
        per_alternative_votes_mean = self.__copeland_voter.vote(data, columns_with_ranks, eps)

        final_rank = np.sort(per_alternative_votes_mean)
        final_rank_alternatives_indices = np.argsort(per_alternative_votes_mean)
        final_rank_alternatives = numpy_alternatives[final_rank_alternatives_indices]

        # produce List[RankItem]
        final_rank_items = [
            RankItem(alternative, value)
            for alternative, value in zip(final_rank_alternatives, final_rank)
        ]
        aggregated_copeland_final_rank = group_equal_alternatives_in_ranking(final_rank_items, eps)

        # produce rank images
        alpha_values = self.get_alpha_values(result.model, parameters)
        results_per_alternative = result.get_results_dict(alpha_values)
        ranks = create_flat_ranks(results_per_alternative)
        
        # generate rank images
        dir = result.output_dir
        for alpha_value, intermediate_flat_rank in zip(alpha_values.values, ranks):
            # create intermediate ranks for drawing
            grouped_rank = group_equal_alternatives_in_ranking(
                intermediate_flat_rank, eps)
            name = f'alpha_{round(alpha_value, 4)}'
            image_filename = self.draw_rank(grouped_rank, dir, f'copeland_{name}')
            result.add_intermediate_rank(
                name,
                Rank(intermediate_flat_rank, image_filename, AlphaValue.from_value(alpha_value))
            )
        
        final_rank_image_filename = self.draw_rank(aggregated_copeland_final_rank, dir, 'copeland_final_rank')

        result.final_rank = Rank(
            aggregated_copeland_final_rank,
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
