from typing import Dict, List
from ror.alpha import AlphaValues, AlphaValue
from ror.RORResult import RORResult

DEFAULT_MAPPING = AlphaValues(
    [
        AlphaValue(0.0, 'Q'),
        AlphaValue(0.5, 'R'),
        AlphaValue(1.0, 'S')
    ]
)


def create_ror_result(data: Dict[str, List[float]]) -> RORResult:
    result = RORResult()
    alpha_values = ['0.0', '0.5', '1.0']
    for alternative, alternative_values in data.items():
        for alpha_value, alternative_value in zip(alpha_values, alternative_values):
            result.add_result(alternative, alpha_value, alternative_value)
    return result
