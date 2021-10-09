from collections import namedtuple
import logging
from typing import Callable, Dict
from ror.Constraint import ConstraintVariable, ConstraintVariablesSet
from ror.Dataset import RORDataset
from ror.RORModel import RORModel
from ror.RORResult import RORResult
from ror.alpha import AlphaValue, AlphaValues
from ror.data_loader import LoaderResult
from ror.loader_utils import AvailableParameters
from ror.d_function import d
from ror.ResultAggregator import aggregate_result_default


class ProcessingCallbackData:
    def __init__(self, progress: float, status: str):
        self.progress: float = progress
        self.status: str = status

def solve_model(loaderResult: LoaderResult) -> RORResult:
    return solve_model(loaderResult.dataset, loaderResult.parameters)


def solve_model(data: RORDataset, parameters: Dict[AvailableParameters, float], progress_callback: Callable[[ProcessingCallbackData], None] = None, values_for_alpha: AlphaValues = None) -> RORResult:
    alpha_values = AlphaValues(
        [
            AlphaValue(0.0, 'Q'),
            AlphaValue(0.5, 'R'),
            AlphaValue(1.0, 'S')
        ]
    ) if values_for_alpha is None else values_for_alpha
    # models to solve is the number of all models that needs to be solved by the solver
    # used to calculate the total progress of calculations
    models_to_solve = 1 + len(data.alternatives) * len(alpha_values.values) + 1
    models_solved = 0

    # step 1
    logging.info('Starting step 1')
    model = RORModel(
        data,
        parameters[AvailableParameters.INITIAL_ALPHA],
        f"ROR Model, step 1, with alpha {parameters[AvailableParameters.INITIAL_ALPHA]}"
    )

    model.target = ConstraintVariablesSet([
        ConstraintVariable("delta", 1.0)
    ])
    result = model.solve()
    models_solved += 1
    progress_callback(ProcessingCallbackData(models_solved / models_to_solve, 'Step 1'))
    logging.info(f"Solved step 1, delta value is {result.objective_value}")

    logging.info('Starting step 2')
    # assign delta value to the data
    data.delta = result.objective_value

    ror_result = RORResult()
    ror_result.alpha_values = alpha_values
    for alternative in data.alternatives:
        for alpha in alpha_values.values:
            model = RORModel(
                data, alpha, f"ROR Model, step 2, with alpha {alpha}, alternative {alternative}")
            model.target = d(alternative, alpha, data)
            result = model.solve()
            assert result is not None, 'Failed to optimize the problem. Model is infeasible'

            models_solved += 1
            progress_callback(ProcessingCallbackData(models_solved / models_to_solve, f'Step 2, alternative: {alternative}, alpha {alpha}.'))

            ror_result.add_result(alternative, alpha, result.objective_value)
            logging.info(
                f"alternative {alternative}, objective value {result.objective_value}")

    progress_callback(ProcessingCallbackData(models_solved / models_to_solve, f'Aggregating results.'))
    final_result = aggregate_result_default(ror_result, alpha_values, data.eps)
    models_solved += 1
    progress_callback(ProcessingCallbackData(models_solved / models_to_solve, f'Calculations done.'))
    return final_result
