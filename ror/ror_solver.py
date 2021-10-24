from collections import namedtuple
import logging
from typing import Callable, Dict
from ror.BordaResultAggregator import BordaResultAggregator
from ror.Constraint import ConstraintVariable, ConstraintVariablesSet
from ror.CopelandResultAggregator import CopelandResultAggregator
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
from ror.AbstractTieResolver import AbstractTieResolver
from ror.BordaTieResolver import BordaTieResolver
from ror.CopelandTieResolver import CopelandTieResolver
from ror.NoTieResolver import NoTieResolver


class ProcessingCallbackData:
    def __init__(self, progress: float, status: str):
        self.progress: float = progress
        self.status: str = status

def solve_model(loaderResult: LoaderResult) -> RORResult:
    return solve_model(loaderResult.dataset, loaderResult.parameters)

AVAILABLE_AGGREGATORS: Dict[str, AbstractResultAggregator] = {
    result_aggregator.name: result_aggregator
    for result_aggregator
    in [
        DefaultResultAggregator(),
        WeightedResultAggregator(),
        BordaResultAggregator(),
        CopelandResultAggregator()
    ]
}

TIE_RESOLVERS: Dict[str, AbstractTieResolver] = {
    resolver.name: resolver
    for resolver
    in [
        NoTieResolver(),
        BordaTieResolver(),
        CopelandTieResolver()
    ]
}

def solve_model(
        data: RORDataset,
        parameters: RORParameters,
        aggregation_method: str,
        *aggregation_method_args,
        progress_callback: Callable[[ProcessingCallbackData], None] = None,
        tie_resolver_name: str = None,
        **aggregation_method_kwargs,
    ) -> RORResult:
    assert aggregation_method in AVAILABLE_AGGREGATORS, f'Invalid aggregator method name {aggregation_method}, available: [{", ".join(AVAILABLE_AGGREGATORS.keys())}]'
    tie_resolver: AbstractTieResolver = None
    if tie_resolver_name is not None:
        assert tie_resolver_name in TIE_RESOLVERS,\
            f'Invalid tie resolver name {tie_resolver_name}, available: [{", ".join(TIE_RESOLVERS.keys())}]'
        tie_resolver = TIE_RESOLVERS[tie_resolver_name]
    else:
        tie_resolver = CopelandTieResolver()
    logging.info(f'Using rank resolver: {tie_resolver.name}')

    initial_model = RORModel(
        data,
        parameters[RORParameter.INITIAL_ALPHA],
        f"ROR Model, step 1, with alpha {parameters[RORParameter.INITIAL_ALPHA]}"
    )

    initial_model.target = ConstraintVariablesSet([
        ConstraintVariable("delta", 1.0)
    ])
    aggregator = AVAILABLE_AGGREGATORS[aggregation_method]
    aggregator.set_tie_resolver(tie_resolver)
    # get alpha values depending on the result aggregator
    alpha_values = aggregator.get_alpha_values(initial_model, parameters)

    # for raporting progress:
    # Calculate number of steps to be solved.
    # It consists of:
    # 1. solving 1st model (initial model that verifies whether model is feasible)
    # 2. solving all models (depends on the alpha value)
    # 3. generating images for each rank
    # 4. aggregating results
    steps_to_solve = 1 + len(data.alternatives) * len(alpha_values.values) + 2
    steps_solved = 0

    # inner function for reporting calculations progress
    def report_progress(models_solved: int, description: str):
        models_solved += 1
        if progress_callback is not None:
            progress_callback(ProcessingCallbackData(models_solved / steps_to_solve, description))
        return models_solved

    # step 1
    logging.info('Starting step 1')
    
    result = initial_model.solve()
    steps_solved = report_progress(steps_solved, 'Step 1')
    logging.info(f"Solved step 1, delta value is {result.objective_value}")

    logging.info('Starting step 2')
    # assign delta value to the data
    data.delta = result.objective_value

    ror_result = RORResult()
    precision = parameters.get_parameter(RORParameter.PRECISION)
    # assign model here - this can be used later in result aggregator
    ror_result.model = initial_model
    ror_result.alpha_values = alpha_values
    for alternative in data.alternatives:
        for alpha in alpha_values.values:
            tmp_model = RORModel(
                data, alpha, f"ROR Model, step 2, with alpha {alpha}, alternative {alternative}")
            tmp_model.target = d(alternative, alpha, data)
            result = tmp_model.solve()
            assert result is not None, 'Failed to optimize the problem. Model is infeasible'

            steps_solved = report_progress(steps_solved, f'Step 2, alternative: {alternative}, alpha {round(alpha, precision)}.')
            
            ror_result.add_result(alternative, alpha, result.objective_value)
            logging.info(
                f"alternative {alternative}, objective value {result.objective_value}")

    steps_solved = report_progress(steps_solved, f'Aggregating results.')
    final_result: RORResult = aggregator.aggregate_results(
        ror_result,
        parameters,
        *aggregation_method_args,
        **aggregation_method_kwargs
    )
    final_result.results_aggregator = aggregator
    steps_solved = report_progress(steps_solved, 'Calculations done.')
    return final_result
