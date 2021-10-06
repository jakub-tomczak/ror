from collections import defaultdict
from typing import DefaultDict, Dict, List, Union
import pandas as pd
from ror.result_aggregator_utils import RankItem
from ror.alpha import AlphaValues


class RORResult:
    def alpha_value_key_generator(alpha: Union[str, float]): return f"alpha_{alpha}"

    def __init__(self) -> None:
        # {alternative: {'0.0': 0.345 }, ('0.5', 0.564)...}
        self.__optimization_results: Dict[str, Dict[str, float]] = DefaultDict(
            lambda: defaultdict(lambda: 1.0))
        # final rank after aggregation with one of the available methods
        self.__final_rank: List[List[RankItem]] = []
        # ranks for different alpha values - those ranks are used for aggregation
        self.__intermediate_ranks: Dict[List[List[RankItem]]] = defaultdict(lambda: [
        ])

    def add_result(self, alternative: str, alpha_value: str, result: float):
        self.__optimization_results[alternative][str(alpha_value)] = result

    def add_intermediate_rank(self, alpha: str, rank: List[List[RankItem]]):
        self.__intermediate_ranks[alpha] = rank

    def get_result_table(self) -> pd.DataFrame:
        columns: Dict[str, List[Union[float, str]]] = defaultdict(lambda: [])
        index_column_name = 'id'
        for alternative in self.__optimization_results:
            columns[index_column_name].append(alternative)
            for alpha_value, result in self.__optimization_results[alternative].items():
                columns[RORResult.alpha_value_key_generator(alpha_value)].append(result)
        all_data = pd.DataFrame(columns)
        all_data.set_index(index_column_name, inplace=True)
        # get all columns with alpha values
        alpha_columns = set(all_data.columns) - set(index_column_name)
        # create series with sum of all alphas
        sum_per_alternative_series = all_data[alpha_columns].sum(axis=1)
        sum_per_alternative_series.name = "alpha_sum"
        return pd.concat([all_data, sum_per_alternative_series], axis=1)

    def get_results_dict(self, alpha_values: AlphaValues) -> Dict[str, List[float]]:
        result: Dict[str, List[float]] = defaultdict(lambda: [])
        # get the first key to get all alpha_values
        # we need to get all alpha values to have one order
        alpha_values_keys = list(map(str, alpha_values.values))
        for alternative, alternative_values in self.__optimization_results.items():
            result[alternative].extend([
                alternative_values[alpha_key]
                for alpha_key in alpha_values_keys
            ])
        return result

    '''
    Returns an intermediate rank for a specific alpha
    Returns None if there is no rank for the specified alpha
    '''

    def get_intermediate_rank(self, alpha: str) -> List[List[RankItem]]:
        if alpha in self.__intermediate_ranks:
            return self.__intermediate_ranks[alpha]
        return None

    @property
    def final_rank(self) -> List[List[RankItem]]:
        return self.__final_rank

    @final_rank.setter
    def final_rank(self, final_rank: List[List[RankItem]]):
        self.__final_rank = final_rank
