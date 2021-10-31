import logging
from typing import Callable, Dict
from ror.BordaResultAggregator import BordaResultAggregator
from ror.Constraint import ConstraintVariable, ConstraintVariablesSet
from ror.CopelandResultAggregator import CopelandResultAggregator
from ror.Dataset import RORDataset
from ror.RORModel import RORModel
from ror.RORParameters import RORParameters
from ror.RORResult import RORResult
from ror.constraints_constants import ConstraintsName
from ror.data_loader import LoaderResult
from ror.inner_maximization_constraints import create_inner_maximization_constraint_for_alternative
from ror.loader_utils import RORParameter
from ror.d_function import d
from ror.ResultAggregator import AbstractResultAggregator
from ror.DefaultResultAggregator import DefaultResultAggregator
from ror.WeightedResultAggregator import WeightedResultAggregator
from ror.AbstractTieResolver import AbstractTieResolver
from ror.BordaTieResolver import BordaTieResolver
from ror.CopelandTieResolver import CopelandTieResolver
from ror.NoTieResolver import NoTieResolver
from copy import deepcopy

from ror.AbstractSolver import AbstractSolver
from ror.GurobiSolver import GurobiSolver


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
        progress_callback: Callable[[ProcessingCallbackData], None] = None,
        # user can provide either his ResultAggregator or provide a name of the existing one
        # aggregator object has precedence over aggregator name
        result_aggregator: AbstractResultAggregator = None,
        result_aggregator_name: str = None,
        # user can provide either his TieResolver or provide a name of the existing one
        # tie_resolver object has precedence over tie_resolver name
        tie_resolver: AbstractTieResolver = None,
        tie_resolver_name: str = None,
        # if False then only images with ranks are saved,
        # otherwise all data (images, distances and voting data) is saved
        save_all_data: bool = False,
        solver: AbstractSolver = None
    ) -> RORResult:
    _aggregator: AbstractResultAggregator = None
    def validate_aggregator_name(name: str):
        assert name in AVAILABLE_AGGREGATORS,\
            f'Invalid aggregator method name {name}, available: [{", ".join(AVAILABLE_AGGREGATORS.keys())}]'
    if result_aggregator is not None:
        # try result aggregator based on object
        logging.info('Trying to get result aggregator from provided object')
        assert isinstance(result_aggregator, AbstractResultAggregator), 'Provided result aggregator must inherit from AbstractResultAggregator'
        _aggregator = result_aggregator
    elif result_aggregator_name is not None and result_aggregator_name != '':
        # try result aggregator based on provided name
        logging.info('Trying to get result aggregator from provided result aggregator name')
        validate_aggregator_name(result_aggregator_name)
        _aggregator = deepcopy(AVAILABLE_AGGREGATORS[result_aggregator_name])
    else:
        logging.info('Trying to get result aggregator from parameters')
        # as the last resort try to get ResultAggregator from parameters
        _name = parameters.get_parameter(RORParameter.RESULTS_AGGREGATOR)
        validate_aggregator_name(_name)
        _aggregator = deepcopy(AVAILABLE_AGGREGATORS[_name])
    assert _aggregator is not None, 'Aggregator must not be None'
    logging.info(f'Using result aggregator: {_aggregator.name}')

    _tie_resolver: AbstractTieResolver = None
    def validate_tie_resolver_name(name: str):
        assert name in TIE_RESOLVERS,\
            f'Invalid tie resolver name {name}, available: [{", ".join(TIE_RESOLVERS.keys())}]'
    if tie_resolver is not None:
        # try to get tie resolver from provided object
        logging.info('Trying to get tie resolver from provided object')
        assert isinstance(tie_resolver, AbstractTieResolver), 'PProvided tie resolver must inherit from AbstractTieResolver'
        _tie_resolver = tie_resolver
    elif tie_resolver_name is not None and tie_resolver_name != '':
        logging.info('Trying to get tie resolver from provided tie resolver name')
        validate_tie_resolver_name(tie_resolver_name)
        _tie_resolver = deepcopy(TIE_RESOLVERS[tie_resolver_name])
    else:
        # try to get tie resovler from parameters
        logging.info('Trying to get tie resolver from provided parameters')
        _tie_resolver_name = parameters.get_parameter(RORParameter.TIE_RESOLVER)
        validate_tie_resolver_name(_tie_resolver_name)
        _tie_resolver = deepcopy(TIE_RESOLVERS[_tie_resolver_name])
    logging.info(f'Using rank resolver: {_tie_resolver.name}')

    if solver is None:
        solver = GurobiSolver()

    initial_model = RORModel(
        data,
        parameters[RORParameter.INITIAL_ALPHA],
        f"ROR Model, step 1, with alpha {parameters[RORParameter.INITIAL_ALPHA]}"
    )
    initial_model.solver = solver

    initial_model.target = ConstraintVariablesSet([
        ConstraintVariable("delta", 1.0)
    ])
    _aggregator.set_tie_resolver(_tie_resolver)
    # get alpha values depending on the result aggregator
    alpha_values = _aggregator.get_alpha_values(initial_model, parameters)

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
    # calculate minimum distance from alternative a_{j}
    for alternative in data.alternatives:
        for alpha in alpha_values.values:
            tmp_model = RORModel(
                data, alpha, f"ROR Model, step 2, with alpha {alpha}, alternative {alternative}")
            tmp_model.solver = solver
            # In addition, the constraints (j) to (m) are defined on extended set A^{R} + a_{j}.
            tmp_model.add_constraints(
                create_inner_maximization_constraint_for_alternative(data, alternative),
                ConstraintsName.INNER_MAXIMIZATION.value
            )
            tmp_model.target = d(alternative, alpha, data)
            # uncomment 2 lines below to export pdf for each model
            # from ror.latex_exporter import export_latex, export_latex_pdf
            # export_latex_pdf(result.model, f'model, alternative {alternative}, alpha {alpha}')
            result = tmp_model.solve()
            assert result is not None, 'Failed to optimize the problem. Model is infeasible'

            steps_solved = report_progress(steps_solved, f'Step 2, alternative: {alternative}, alpha {round(alpha, precision)}.')
            
            ror_result.add_result(alternative, alpha, result.objective_value)
            logging.info(
                f"alternative {alternative}, objective value {result.objective_value}")

    steps_solved = report_progress(steps_solved, f'Aggregating results.')
    final_result: RORResult = _aggregator.aggregate_results(
        ror_result,
        parameters
    )
    final_result.results_aggregator = _aggregator
    if save_all_data:
        final_result.save_result_to_csv('distances.csv', directory = final_result.output_dir)
        final_result.save_tie_resolvers_data()
        parameters.save_to_json('parameters.json', directory = final_result.output_dir)
    steps_solved = report_progress(steps_solved, 'Calculations done.')
    return final_result
