from ror.data_loader import read_dataset_from_txt
import logging
from ror.ror_solver import solve_model

loading_result = read_dataset_from_txt("problems/buses.txt")
data = loading_result.dataset
parameters = loading_result.parameters

result = solve_model(data, parameters, None)

logging.info(f'Final rank is {result.final_rank}')
