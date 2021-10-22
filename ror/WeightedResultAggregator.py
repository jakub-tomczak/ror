from io import StringIO
from collections import defaultdict
from typing import Dict, List
from ror.RORParameters import RORParameters
from ror.RORResult import RORResult
from ror.ResultAggregator import AbstractResultAggregator
from ror.alpha import AlphaValue, AlphaValues
from ror.loader_utils import RORParameter
from ror.result_aggregator_utils import BIG_NUMBER, Rank, RankItem, create_flat_ranks, get_position_in_rank, group_equal_alternatives_in_ranking, validate_aggregator_arguments, values_equal_with_epsilon


class WeightedResultAggregator(AbstractResultAggregator):
    def __init__(self) -> None:
        super().__init__('WeightedResultAggregator')
        self.weighted_data: Dict[str, List[float]] = dict()
        self.alpha_values: AlphaValues = None

    def aggregate_results(self, result: RORResult, parameters: RORParameters, *args, **kwargs) -> RORResult:
        super().aggregate_results(result, parameters, *args, **kwargs)

        weights_parameter = parameters.get_parameter(
            RORParameter.ALPHA_WEIGHTS)
        assert type(
            weights_parameter) is list, 'Weights must be provided as a list with values >= 0 that corresponds to alpha values'
        self.alpha_values = AlphaValues.from_list(
            parameters.get_parameter(RORParameter.ALPHA_VALUES))
        assert len(weights_parameter) == len(
            self.alpha_values.values), 'Number of weights must correspond to the number of alpha values'
        weights = {
            f'alpha_{alpha_value}': weight
            for alpha_value, weight in zip(self.alpha_values.values, weights_parameter)
        }
        assert all([weight >= 0.0 for weight in weights.values()]
                   ), 'All weights must be greater or equal 0.0'
        eps = parameters.get_parameter(RORParameter.EPS)
        data = result.get_results_dict(self.alpha_values)
        # divide values by weights - alternative value is the distance to the ideal alternative
        # so we need to divide instead of multiplying
        for alternative in data:
            alternative_data = data[alternative]
            self.weighted_data[alternative] = [
                value / weight if weight > 0 else BIG_NUMBER for value, weight in zip(alternative_data, weights.values())
            ]
        flat_ranks = create_flat_ranks(self.weighted_data)

        values_per_alternative: Dict[str, float] = defaultdict(lambda: 0)
        for rank in flat_ranks:
            for rank_item in rank:
                values_per_alternative[rank_item.alternative] += rank_item.value

        # sort by alternative's value
        sorted_final_rank = sorted(
            values_per_alternative.items(), key=lambda alternative: alternative[1])
        # wrap sorted final rank into RankItem
        final_rank = [RankItem(alternative, value)
                      for alternative, value in sorted_final_rank]
        # place same results into same positions
        final_rank = group_equal_alternatives_in_ranking(final_rank, eps)

        # draw positions
        # get dir for all ranks because dir contains datetime so must be one for all
        dir = self.get_dir_for_rank_image()
        for alpha_value, intermediate_flat_rank in zip(self.alpha_values.values, flat_ranks):
            # create intermediate ranks for drawing
            grouped_rank = group_equal_alternatives_in_ranking(
                intermediate_flat_rank, eps)
            name = f'alpha_{alpha_value}'
            image_filename = self.draw_rank(grouped_rank, dir, f'weighted_{name}')
            result.add_intermediate_rank(
                name, Rank(rank, image_filename, AlphaValue.from_value(alpha_value)))

        final_rank_image_filename = self.draw_rank(final_rank, dir, 'weighted_final_rank')
        final_rank_object = Rank(
            final_rank,
            final_rank_image_filename,
            'final_rank'
        )
        # return result
        result.final_rank = final_rank_object
        return result

    def explain_result(self, alternative_1: str, alternative_2: str) -> str:
        assert alternative_1 in self.weighted_data, f'No results for alternative {alternative_1} found'
        assert alternative_2 in self.weighted_data, f'No results for alternative {alternative_2} found'
        alternative_1_weights = self.weighted_data[alternative_1]
        alternative_2_weights = self.weighted_data[alternative_2]

        precision = 3
        def rounded(number):
            return round(number, precision)

        explanation = StringIO()
        explanation.write(
            f'First alternative {alternative_1} has the following results for alpha values:\n')
        alternative_1_sum: float = 0
        for alpha_value, result in zip(self.alpha_values.values, alternative_1_weights):
            explanation.write(f'Alpha {alpha_value}: {rounded(result)}\n')
            alternative_1_sum += result
        explanation.write(f'Sum is {rounded(alternative_1_sum)}\n')

        explanation.write(
            f'Second alternative {alternative_2} has the following results for alpha values:\n')
        alternative_2_sum: float = 0
        for alpha_value, result in zip(self.alpha_values.values, alternative_2_weights):
            explanation.write(f'Alpha {alpha_value}: {rounded(result)}\n')
            alternative_2_sum += result
        explanation.write(f'Sum is {rounded(alternative_2_sum)}\n')

        final_rank = self._ror_result.final_rank
        final_rank_alt_1_position = get_position_in_rank(
            alternative_1, final_rank)
        final_rank_alt_2_position = get_position_in_rank(
            alternative_2, final_rank)

        if alternative_1_sum > alternative_2_sum:
            explanation.write(
                f'First alternative {alternative_1} has bigger distance (sum)\n')
            explanation.write(
                f'therefore it is on the lower position in the rank ({final_rank_alt_1_position})\n')
            explanation.write(
                f'than the second alternative {alternative_2} (position {final_rank_alt_2_position}\n')
        elif alternative_1_sum > alternative_2_sum:
            explanation.write(
                f'First alternative {alternative_1} has lower distance (sum)\n')
            explanation.write(
                f'therefore it is on the higher position in the rank ({final_rank_alt_1_position})\n')
            explanation.write(
                f'than the second alternative {alternative_2} (position {final_rank_alt_2_position}\n')
        else:
            explanation.write(
                f'First {alternative_1} and second {alternative_2} alternative have same distance (sum)\n')
            explanation.write(
                f'with the precision of eps value {self._ror_parameters.get_parameter(RORParameter.EPS)}\n')
            explanation.write(
                f'therefore they are on the same position {final_rank_alt_1_position} in the final rank\n')

    def help(self) -> str:
        return '''
Function that aggregates results from ranks: R, Q and S by adding weights to ranks.
Weights must be greater or equal 0.0
Weight > 1.0 increases importance of the rank (lowers value)
Weight < 1.0 decreases importance of the rank (increases value)
Weight == 1.0 doesn't change the importance of the rank
'''
