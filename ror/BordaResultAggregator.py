import logging
from typing import List
from ror.BordaVoter import BordaVoter
from ror.ResultAggregator import AbstractResultAggregator
from ror.RORModel import RORModel
from ror.RORResult import RORResult
from ror.RORParameters import RORParameters
from ror.alpha import AlphaValue, AlphaValues
from ror.loader_utils import RORParameter
from ror.result_aggregator_utils import Rank, RankItem, create_flat_ranks, group_equal_alternatives_in_ranking
import numpy as np
import pandas as pd


class BordaResultAggregator(AbstractResultAggregator):
    def __init__(self) -> None:
        self.__number_of_ranks: int = 3
        self.__voter: BordaVoter = BordaVoter()
        super().__init__('BordaResultAggregator')

    @property
    def voter(self) -> BordaVoter:
        return self.__voter

    def aggregate_results(self, result: RORResult, parameters: RORParameters) -> RORResult:
        super().aggregate_results(result, parameters)
        # get sum of results
        # calculate mean position (across all ranks)
        # create final rank - if positions are equal then the one with 
        # we assume that the last alternative gets 1, the best gets len(alternatives)
        eps = parameters.get_parameter(RORParameter.EPS)
        data = result.get_result_table()
        numpy_alternatives: np.ndarray = np.array(list(data.index))
        number_of_alternatives = len(numpy_alternatives)
        alpha_values = self.get_alpha_values(result.model, parameters)
        logging.debug(f'Borda aggregator, results {result.get_result_table()}')
        # get name of all columns with ranks, beside last one - with sum
        columns_with_ranks: List[str] = list(set(data.columns) - set(['alpha_sum']))
        
        alternative_to_mean_position = self.__voter.vote(data, number_of_alternatives, columns_with_ranks, numpy_alternatives)

        # transform alternative to the list of rank items
        final_rank: List[RankItem] = [
            RankItem(alternative, value)
            for alternative, value in alternative_to_mean_position.items()
        ]
        # sort final rank by the mean position, in case of a tie any alternative is takes as the better one
        # reverse sorting as the better alternative should have bigger value
        sorted_final_rank = sorted(final_rank, key=lambda item: item.value, reverse=True)
        logging.debug(f'Final rank {sorted_final_rank}')

        results_per_alternative = result.get_results_dict(alpha_values)
        ranks = create_flat_ranks(results_per_alternative)
        # generate rank images
        dir = result.output_dir
        for alpha_value, intermediate_flat_rank in zip(alpha_values.values, ranks):
            # create intermediate ranks for drawing
            grouped_rank = group_equal_alternatives_in_ranking(
                intermediate_flat_rank, eps)
            name = f'alpha_{round(alpha_value, 4)}'
            image_filename = self.draw_rank(grouped_rank, dir, f'borda_{name}')
            result.add_intermediate_rank(
                name,
                Rank(intermediate_flat_rank, image_filename, AlphaValue.from_value(alpha_value))
            )
        
        borda_final_rank = group_equal_alternatives_in_ranking(sorted_final_rank, eps)
        final_rank_image_filename = self.draw_rank(borda_final_rank, dir, 'borda_final_rank')

        result.final_rank = Rank(
            borda_final_rank,
            img_filename=final_rank_image_filename,
        )
        return result
        
    def get_alpha_values(self, model: RORModel, parameters: RORParameters) -> AlphaValues:
        number_of_alpha_values = parameters.get_parameter(RORParameter.NUMBER_OF_ALPHA_VALUES)
        return AlphaValues.from_list(np.linspace(0.0, 1.0, number_of_alpha_values))

    def explain_result(self, alternative_1: str, alternative_2: str) -> str:
        assertion_error_msg = 'Results were not yet aggregated. Please run aggregate_results method before calling explain_result'
        assert self.__alternative_to_mean_position is not None, assertion_error_msg
        assert self.__number_of_ranks is not None, assertion_error_msg
        return super().explain_result(alternative_1, alternative_2)

    def help(self) -> str:
        return """
Borda aggregator uses Borda voting to decide what should be the better alternative
in case of draw.
It starts from calculating a sum of distances from all alpha values for each alternative.
Then, each rank votes for each alternative. The best alternative in the rank
gets n points.
Then we calculate mean score per alternative over all ranks.
The best score is n and the worst is 1.
An alternative with the best score wins.
If there is more than one alternative in the the rank with the same score then there they are indifferent
and the order in which they are placed in the final rank is not specified,
i.e. alternatives with same scores (a1, a2, a3) can be placed in the final rank
in one of the 6 permutations:
(a1, a2, a3), (a1, a3, a2), (a2, a1, a3), (a3, a1, a2), (a2, a3, a1), (a3, a2, a1)
"""