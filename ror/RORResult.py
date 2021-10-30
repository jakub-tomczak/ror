from collections import defaultdict
import logging
from typing import DefaultDict, Dict, List, Union
import pandas as pd
from ror.BordaTieResolver import BordaTieResolver
from ror.CopelandTieResolver import CopelandTieResolver
from ror.RORModel import RORModel
from ror.RORParameters import RORParameters
from ror.loader_utils import RORParameter
from ror.result_aggregator_utils import Rank
from ror.alpha import AlphaValues
from ror.CalculationsException import CalculationsException
from ror.datetime_utils import get_date_time
import os


class RORResult:
    def alpha_value_key_generator(alpha: Union[str, float]): return f"alpha_{alpha}"

    def __init__(self) -> None:
        # {alternative: {'0.0': 0.345 }, ('0.5', 0.564)...}
        self.__optimization_results: Dict[str, Dict[str, float]] = DefaultDict(
            lambda: defaultdict(lambda: 1.0))
        # final rank after aggregation with one of the available methods
        self.__final_rank: Rank = None
        # ranks for different alpha values - those ranks are used for aggregation
        self.__intermediate_ranks: Dict[str, Rank] = dict()
        self.__alpha_values: AlphaValues = None
        self.model: RORModel = None
        self.__parameters: RORParameters = None
        self.__aggregator: 'AbstractResultAggregator' = None
        self.__output_dir: str = self.get_dir_for_rank_image()

    def add_result(self, alternative: str, alpha_value: str, result: float):
        self.__optimization_results[alternative][str(alpha_value)] = result

    def add_intermediate_rank(self, name: str, rank: Rank):
        self.__intermediate_ranks[name] = rank

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
        '''
        Returns a maping alternative -> results for alternative
        the number of results corresponds to the number of alpha values
        i.e. in case of 3 alpha values
        'a1': [0.3, 4.5, 1.2]
        '''
        result: Dict[str, List[float]] = defaultdict(lambda: [])
        # get the first key to get all alpha_values
        # we need to get all alpha values to have one order
        alpha_values_keys = list(map(str, alpha_values.values))
        # print('optimization results', self.__optimization_results.items())
        for alternative, alternative_values in self.__optimization_results.items():
            result[alternative].extend([
                alternative_values[alpha_key]
                for alpha_key in alpha_values_keys
            ])
        return result

    def get_intermediate_rank(self, name: str) -> Rank:
        if name in self.__intermediate_ranks:
            return self.__intermediate_ranks[name]
        return None

    def save_result_to_csv(self, filename: str, directory: str = None) -> str:
        try:
            result = self.get_result_table()
            if directory is not None:
                filename = os.path.join(directory, filename)
            result.to_csv(filename, sep=';')
            logging.info(f'Saved calculated distances to {filename}')
        except Exception as e:
            logging.error(f'Failed to save to csv file, cause: {e}')
            raise e
        return filename
    
    def save_result_to_latex(self, filename: str, directory: str = None) -> str:
        try:
            result = self.get_result_table()
            precision = self.__parameters.get_parameter(RORParameter.PRECISION)
            if directory is not None:
                filename = os.path.join(directory, filename)
            result.to_latex(filename, float_format=f"%.{precision}f")
            logging.info(f'Saved calculated distances to {filename}')
        except Exception as e:
            logging.error(f'Failed to save to latex file, cause: {e}')
            raise e
        return filename

    def save_tie_resolvers_data(self, directory: str = None) -> List[str]:
        tie_resolver = self.results_aggregator.tie_resolver
        dir = directory if directory is not None else self.output_dir
        if isinstance(tie_resolver, BordaTieResolver):
            borda_voter = tie_resolver.voter
            return borda_voter.save_voting_data(dir)
        elif isinstance(tie_resolver, CopelandTieResolver):
            copeland_voter = tie_resolver.voter
            return copeland_voter.save_voting_data(dir)
        else:
            logging.info('No tie resolver data available.')
            return None
            

    def get_dir_for_rank_image(self) -> str:
        current_dir = os.path.abspath(os.path.curdir)
        output_dir = os.path.join(current_dir, 'ror_distance_output', get_date_time())
        if not os.path.exists(output_dir):
            # try to create output dir
            try:
                os.makedirs(output_dir)
            except Exception as e:
                msg = f'Failed to create output dir: {output_dir}, cause: {e}'
                logging.error(msg)
                raise CalculationsException(e)
        return output_dir

    @property
    def results_aggregator(self) -> 'AbstractResultAggregator':
        return self.__aggregator
    
    @results_aggregator.setter
    def results_aggregator(self, value: 'AbstractResultAggregator'):
        self.__aggregator = value

    @property
    def intermediate_ranks(self) -> List[Rank]:
        return [rank for rank in self.__intermediate_ranks.values()]

    @property
    def final_rank(self) -> Rank:
        return self.__final_rank

    @final_rank.setter
    def final_rank(self, final_rank: Rank):
        self.__final_rank = final_rank

    @property
    def alpha_values(self) -> AlphaValues:
        return self.__alpha_values

    @alpha_values.setter
    def alpha_values(self, alpha_values: AlphaValues):
        self.__alpha_values = alpha_values

    @property
    def parameters(self) -> RORParameters:
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters: RORParameters):
        self.__parameters = parameters

    @property
    def output_dir(self) -> str:
        return self.__output_dir