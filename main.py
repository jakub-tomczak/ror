from ror.DefaultResultAggregator import DefaultResultAggregator
from ror.data_loader import read_dataset_from_txt
import logging
from ror.loader_utils import RORParameter
from ror.ror_solver import solve_model

loading_result = read_dataset_from_txt("problems/buses.txt")
data = loading_result.dataset
parameters = loading_result.parameters
chosen_aggregation_method = parameters.get_parameter(RORParameter.RESULTS_AGGREGATOR)

result = solve_model(data, parameters, chosen_aggregation_method)

logging.info(f'Final rank is {result.final_rank}')
