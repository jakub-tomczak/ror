from abc import abstractmethod
from typing import List
from ror.RORParameters import RORParameters
from ror.result_aggregator_utils import SimpleRank

class AbstractTieResolver:
    def __init__(self, name: str) -> None:
        self._ror_result: 'RORResult' = None
        self._ror_parameters: RORParameters = None
        self._name: str = name

    @abstractmethod
    def resolve_rank(self, rank: SimpleRank, result: 'RORResult', parameters: RORParameters) -> SimpleRank:
        assert result is not None, 'ROR result must not be None'
        assert parameters is not None, 'ROR parameters must not be None'
        self._ror_parameters = parameters
        self._ror_result = result

    @abstractmethod
    def help(self) -> str:
        pass

    @property
    def name(self) -> str:
        return self._name