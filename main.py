from ror.data_loader import read_dataset_from_txt
import logging
from ror.latex_exporter import export_latex, export_latex_pdf
from ror.loader_utils import RORParameter
from ror.ror_solver import solve_model

loading_result = read_dataset_from_txt("problems/buses_small.txt")
data = loading_result.dataset
parameters = loading_result.parameters
chosen_aggregation_method = parameters.get_parameter(RORParameter.RESULTS_AGGREGATOR)

result = solve_model(
    data,
    parameters,
    result_aggregator_name=chosen_aggregation_method,
    save_all_data=True
)

try:
    filename = export_latex_pdf(result.model, 'first model')
    logging.info(f'Exported initial model\'s constraints to PDF latex file, path to exported file {filename}')
except:
    logging.info('Failed to export initial model\'s constraints to PDF latex file. Check if have pdflatex installed on your system.')
    filename = export_latex(result.model, 'first model')
    logging.info(f'Exported file to tex format, path to exported file {filename}')

logging.info(f'Final rank is {result.final_rank.rank_to_string()}')
