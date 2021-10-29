from collections import defaultdict
from typing import Dict, List
from ror.ResultAggregator import VotesPerRank
import pandas as pd
import logging
import numpy as np


class BordaVoter():
    def __init__(self) -> None:
        self.__votes_per_rank: VotesPerRank = defaultdict(lambda: dict())
        # alternative_1 -> mean_votes
        self.__alternative_to_mean_position: Dict[str, float] = None

    @property
    def votes_per_rank(self) -> pd.DataFrame:
        return pd.DataFrame(self.__votes_per_rank)

    @property
    def alternative_to_mean_position(self) -> Dict[str, float]:
        return self.__alternative_to_mean_position

    def vote(self, data: pd.DataFrame, number_of_alternatives: int, columns_with_ranks: List[str], numpy_alternatives: np.ndarray) -> Dict[str, float]:
        # go through each rank (per each alpha value)
        # sort values to get positions for borda voting
        alternative_to_mean_position: Dict[str, float] = defaultdict(lambda: 0.0)
        for column_name in columns_with_ranks:
            sorted_indices = np.argsort(data[column_name])
            sorted_alternatives = numpy_alternatives[sorted_indices]
            logging.debug(
                f'Sorted alternatives for rank {column_name} is {sorted_alternatives}')
            # go through all alternatives, sorted by value for a specific alpha value
            for index, alternative in enumerate(sorted_alternatives):
                if alternative not in self.__votes_per_rank[column_name]:
                    self.__votes_per_rank[column_name][alternative] = (number_of_alternatives - index)
                else:
                    raise Exception(f'Invalid operation: rank {column_name} already voted for alternative {alternative}')
                alternative_to_mean_position[alternative] += (
                    number_of_alternatives - index)
        for alternative in alternative_to_mean_position:
            alternative_to_mean_position[alternative] /= len(columns_with_ranks)
        self.__alternative_to_mean_position = alternative_to_mean_position
        return alternative_to_mean_position