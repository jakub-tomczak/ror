from collections import namedtuple
import logging
from typing import Callable, Dict
from ror.Constraint import ConstraintVariable, ConstraintVariablesSet
from ror.Dataset import RORDataset
from ror.RORModel import RORModel
from ror.RORParameters import RORParameters
from ror.RORResult import RORResult
from ror.alpha import AlphaValue, AlphaValues
from ror.data_loader import LoaderResult
from ror.loader_utils import RORParameter
from ror.d_function import d
from ror.ResultAggregator import AbstractResultAggregator
from ror.DefaultResultAggregator import DefaultResultAggregator
from ror.WeightedResultAggregator import WeightedResultAggregator


class ProcessingCallbackData:
    def __init__(self, progress: float, status: str):
        self.progress: float = progress
        self.status: str = status

def solve_model(loaderResult: LoaderResult) -> RORResult:
    return solve_model(loaderResult.dataset, loaderResult.parameters)

AVAILABLE_AGGREGATORS: Dict[str, AbstractResultAggregator] = {
    DefaultResultAggregator.__name__: DefaultResultAggregator,
    WeightedResultAggregator.__name__: WeightedResultAggregator
}

def solve_model(
        data: RORDataset,
        parameters: RORParameters,
        aggregation_method: str,
        *aggregation_method_args,
        progress_callback: Callable[[ProcessingCallbackData], None] = None,
        **aggregation_method_kwargs,
    ) -> RORResult:
    assert aggregation_method in AVAILABLE_AGGREGATORS, f'Invalid aggregator method name {aggregation_method}, available: [{", ".join(AVAILABLE_AGGREGATORS.keys())}]'
    alpha_values = AlphaValues.from_list(parameters.get_parameter(RORParameter.ALPHA_VALUES))
    # models to solve is the number of all models that needs to be solved by the solver
    # used to calculate the total progress of calculations
    models_to_solve = 1 + len(data.alternatives) * len(alpha_values.values) + 2
    models_solved = 0
    def report_progress(models_solved: int, description: str):
        models_solved += 1
        if progress_callback is not None:
            progress_callback(ProcessingCallbackData(models_solved / models_to_solve, description))
        return models_solved

    # step 1
    logging.info('Starting step 1')
    model = RORModel(
        data,
        parameters[RORParameter.INITIAL_ALPHA],
        f"ROR Model, step 1, with alpha {parameters[RORParameter.INITIAL_ALPHA]}"
    )

    model.target = ConstraintVariablesSet([
        ConstraintVariable("delta", 1.0)
    ])
    result = model.solve()
    models_solved = report_progress(models_solved, 'Step 1')
    logging.info(f"Solved step 1, delta value is {result.objective_value}")

    logging.info('Starting step 2')
    # assign delta value to the data
    data.delta = result.objective_value

    ror_result = RORResult()
    ror_result.alpha_values = alpha_values
    for alternative in data.alternatives:
        for alpha in alpha_values.values:
            tmp_model = RORModel(
                data, alpha, f"ROR Model, step 2, with alpha {alpha}, alternative {alternative}")
            tmp_model.target = d(alternative, alpha, data)
            result = tmp_model.solve()
            assert result is not None, 'Failed to optimize the problem. Model is infeasible'

            models_solved = report_progress(models_solved, f'Step 2, alternative: {alternative}, alpha {alpha}.')
            
            ror_result.add_result(alternative, alpha, result.objective_value)
            logging.info(
                f"alternative {alternative}, objective value {result.objective_value}")

    models_solved = report_progress(models_solved, f'Aggregating results.')
    # create new instance of aggregator
    aggregator = AVAILABLE_AGGREGATORS[aggregation_method]()
    final_result = aggregator.aggregate_results(
        ror_result,
        parameters,
        *aggregation_method_args,
        **aggregation_method_kwargs
    )
    final_result.model = model
    models_solved = report_progress(models_solved, 'Calculations done.')
    return final_result
