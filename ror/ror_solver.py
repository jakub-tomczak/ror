import logging
from typing import Dict
from ror.Constraint import ConstraintVariable, ConstraintVariablesSet
from ror.Dataset import RORDataset
from ror.RORModel import RORModel
from ror.RORResult import RORResult
from ror.alpha import AlphaValue, AlphaValues
from ror.data_loader import LoaderResult
from ror.loader_utils import AvailableParameters
from ror.d_function import d
from ror.ResultAggregator import aggregate_result_default


def solve_model(loaderResult: LoaderResult) -> RORResult:
    return solve_model(loaderResult.dataset, loaderResult.parameters)


def solve_model(data: RORDataset, parameters: Dict[AvailableParameters, float]) -> RORResult:
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
    logging.info(f"Solved step 1, delta value is {result.objective_value}")

    logging.info('Starting step 2')
    # assign delta value to the data
    data.delta = result.objective_value

    alpha_values = AlphaValues(
        [
            AlphaValue(0.0, 'Q'),
            AlphaValue(0.5, 'R'),
            AlphaValue(1.0, 'S')
        ]
    )

    ror_result = RORResult()
    for alternative in data.alternatives:
        for alpha in alpha_values.values:
            model = RORModel(
                data, alpha, f"ROR Model, step 2, with alpha {alpha}, alternative {alternative}")
            model.target = d(alternative, alpha, data)
            result = model.solve()
            if result is None:
                logging.error('Failed to optimize the problem')
                exit(1)

            ror_result.add_result(alternative, alpha, result.objective_value)
            logging.info(
                f"alternative {alternative}, objective value {result.objective_value}")

    return aggregate_result_default(ror_result, alpha_values, data.eps)
