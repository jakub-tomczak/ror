from collections import defaultdict
import logging
from typing import Dict, List
from ror.ResultAggregator import AbstractResultAggregator
from ror.RORModel import RORModel
from ror.RORResult import RORResult
from ror.RORParameters import RORParameters
from ror.alpha import AlphaValue, AlphaValues
from ror.loader_utils import RORParameter
from ror.result_aggregator_utils import Rank, RankItem, create_flat_ranks, group_equal_alternatives_in_ranking
import numpy as np


class BordaResultAggregator(AbstractResultAggregator):
    def __init__(self) -> None:
        self.__number_of_ranks: int = 3
        self.__alternative_to_mean_position: Dict[str, float] = None
        super().__init__('BordaResultAggregator')

    def aggregate_results(self, result: RORResult, parameters: RORParameters, *args, **kwargs) -> RORResult:
        # get sum of results
        # calculate mean position (across all ranks)
        # create final rank - if positions are equal then the one with 
        # we assume that the last alternative gets 1, the best gets len(alternatives)
        number_of_ranks = parameters.get_parameter(RORParameter.NUMBER_OF_ALPHA_VALUES)
        eps = parameters.get_parameter(RORParameter.EPS)
        data = result.get_result_table()
        numpy_alternatives: np.ndarray = np.array(list(data.index))
        number_of_alternatives = len(numpy_alternatives)
        alpha_values = self.get_alpha_values(result.model, parameters)
        logging.info(f'Borda aggregator, results {result.get_result_table()}')
        # get name of all columns with ranks, beside last one - with sum
        columns_with_ranks: List[str] = list(set(data.columns) - set(['alpha_sum']))
        assert len(columns_with_ranks) == number_of_ranks,\
            'Invalid number of columns in the result or number of ranks'
        # go through each rank (per each alpha value)
        # sort values to get positions for borda voting
        alternative_to_mean_position: Dict[str, float] = defaultdict(lambda: 0.0)
        for column_name in columns_with_ranks:
            sorted_indices = np.argsort(data[column_name])
            sorted_alternatives = numpy_alternatives[sorted_indices]
            logging.info(f'Sorted alternatives for rank {column_name} is {sorted_alternatives}')
            # go through all alternatives, sorted by value for a specific alpha value
            for index, alternative in enumerate(sorted_alternatives):
                alternative_to_mean_position[alternative] += (number_of_alternatives - index)
        for alternative in alternative_to_mean_position:
            alternative_to_mean_position[alternative] /= number_of_ranks
        
        sorted_alpha_sum = np.sort(data['alpha_sum'])
        sorted_alpha_sum_args = np.argsort(data['alpha_sum'])
        sorted_alpha_sum_alternatives = numpy_alternatives[sorted_alpha_sum_args]
        initial_final_rank = [
            RankItem(alternative, value)
            for value, alternative in zip(sorted_alpha_sum, sorted_alpha_sum_alternatives)
        ]
        final_rank_with_ties = group_equal_alternatives_in_ranking(initial_final_rank, eps)
        logging.info(f'Final rank with ties {final_rank_with_ties}')
        final_rank_with_borda: List[RankItem] = []

        for items_at_same_position in final_rank_with_ties:
            if len(items_at_same_position) > 1:
                # gets borda's value for all alternatives from this position
                items_borda_value = [
                    (r_item.alternative, alternative_to_mean_position[r_item.alternative])
                    for r_item in items_at_same_position
                ]
                # sort all items from the same position by mean position from all ranks using borda voting
                sorted_by_mean_position = sorted(items_borda_value, key=lambda item: item[1])
                logging.info(f'items at the same position: {items_at_same_position}')
                for alternative, mean_position in sorted_by_mean_position:
                    logging.info(f'Alternative: {alternative}, mean position: {mean_position}')
                    final_rank_with_borda.append(RankItem(alternative, items_at_same_position[0].value))
            else:
                # only one item at this position, add it to the new final rank
                final_rank_with_borda.append(items_at_same_position[0])
        logging.info(f'Final rank without ties {final_rank_with_borda}')
        
        for rank in result.intermediate_ranks:
            logging.info(rank.rank)
        results_per_alternative = result.get_results_dict(alpha_values)
        logging.info('results per alternative', results_per_alternative)
        ranks = create_flat_ranks(results_per_alternative)
        
        # generate rank images
        dir = self.get_dir_for_rank_image()
        for alpha_value, intermediate_flat_rank in zip(alpha_values.values, ranks):
            # create intermediate ranks for drawing
            grouped_rank = group_equal_alternatives_in_ranking(
                intermediate_flat_rank, eps)
            name = f'alpha_{alpha_value}'
            image_filename = self.draw_rank(grouped_rank, dir, f'borda_{name}')
            result.add_intermediate_rank(
                name,
                Rank(intermediate_flat_rank, image_filename, AlphaValue.from_value(alpha_value))
            )
        
        # use wrapped final rank - to have consistent constructor for Rank object that assumes rank with ties
        # therefore list of lists of RankItem is required
        wrapped_borda_final_rank: List[List[RankItem]] = [[rank_item] for rank_item in final_rank_with_borda]
        final_rank_image_filename = self.draw_rank(wrapped_borda_final_rank, dir, 'borda_final_rank')
        
        # replace old final rank with some alternatives at the same position with a new rank
        # with alternatives from the same position ranked using borda voting
        result.final_rank = Rank(
            wrapped_borda_final_rank,
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
Then, a final rank is produced. If any alternatives are 
        """