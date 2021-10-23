from abc import abstractmethod
from typing import List
from ror.RORModel import RORModel
from ror.RORParameters import RORParameters
from ror.RORResult import RORResult
from ror.alpha import AlphaValues
from ror.datetime_utils import get_date_time
from ror.graphviz_helper import draw_rank
from ror.rank.AbstractTieResolver import AbstractTieResolver
from ror.result_aggregator_utils import RankItem, from_rank_to_alternatives
import os


class AbstractResultAggregator:
    '''
    Common class for all aggregators.
    Defines a common method for aggregating results.
    '''

    def __init__(self, name: str) -> None:
        self._ror_result: RORResult = None
        self._ror_parameters: RORParameters = None
        self._name: str = name
        self._tie_resolver: AbstractTieResolver = None

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def aggregate_results(self, result: RORResult, parameters: RORParameters, *args, **kwargs) -> RORResult:
        '''
        Common method for aggregating results from calculations.
        It should aggregate different values obtained from solving model with different target
        and different alpha.
        '''
        self._ror_result = result
        self._ror_parameters = parameters
        self._ror_result.parameters = parameters
        pass

    @abstractmethod
    def explain_result(self, alternative_1: str, alternative_2: str) -> str:
        '''
        Common method for explaining position of alternatives in the rank
        Takes 2 alternatives and outputs a string with an explanation why alternatives
        were ranked in this way.
        '''
        pass

    @abstractmethod
    def help(self) -> str:
        '''
        Method that returns a string that explains how an aggregation method works.
        '''
        pass

    @abstractmethod
    def get_alpha_values(self, model: RORModel, parameters: RORParameters) -> AlphaValues:
        '''
        Method that returns alpha values for result aggregator.
        Result aggregator should prepare here all alpha values for which
        model should be solved.
        '''
        pass

    def draw_rank(self, rank: List[List[RankItem]], dir: str, rank_name: str) -> str:
        return draw_rank(from_rank_to_alternatives(rank), dir, rank_name)

    def set_tie_resolver(self, tie_resolver: AbstractTieResolver):
        self._tie_resolver = tie_resolver

    def get_dir_for_rank_image(self) -> str:
        current_dir = os.path.abspath(os.path.curdir)
        return os.path.join(current_dir, 'output', get_date_time())
