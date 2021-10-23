from ror.RORParameters import RORParameters
from ror.RORResult import RORResult
from ror.rank.AbstractTieResolver import AbstractTieResolver
from ror.result_aggregator_utils import Rank


class NoTieResolver(AbstractTieResolver):
    def __init__(self) -> None:
        super().__init__('NoTieResolver')

    def resolve_rank(self, rank: Rank, result: RORResult, parameters: RORParameters) -> Rank:
        super().resolve_rank(rank, result, parameters)
        return rank
    
    def help(self) -> str:
        return 'This resolver does nothing. It just returns same rank as was provided as an input.'