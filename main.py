from ror.data_loader import read_dataset_from_txt
import logging
from ror.latex_exporter import export_latex
from ror.loader_utils import RORParameter
from ror.ror_solver import solve_model

loading_result = read_dataset_from_txt("problems/buses.txt")
data = loading_result.dataset
parameters = loading_result.parameters
chosen_aggregation_method = parameters.get_parameter(RORParameter.RESULTS_AGGREGATOR)
print('chosen method', chosen_aggregation_method)

result = solve_model(data, parameters, chosen_aggregation_method)

export_latex(result.model, 'first model')

logging.info(f'Final rank is {result.final_rank}')
