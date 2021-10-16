from ror.data_loader import read_dataset_from_txt
import logging
from ror.ror_solver import get_available_aggregators, solve_model

loading_result = read_dataset_from_txt("problems/buses.txt")
data = loading_result.dataset
parameters = loading_result.parameters
aggregation_methods = get_available_aggregators()

result = solve_model(data, parameters, aggregation_methods['Default aggregator'])

logging.info(f'Final rank is {result.final_rank}')
