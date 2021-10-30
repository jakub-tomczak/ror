from collections import defaultdict
from typing import Dict, List
from ror.types import VotesPerRank
import pandas as pd
import logging
import os
import numpy as np


class BordaVoter():
    def __init__(self) -> None:
        self.__votes_per_rank: VotesPerRank = defaultdict(lambda: dict())
        # alternative_1 -> mean_votes
        self.__alternative_to_mean_votes: Dict[str, float] = None

    @property
    def votes_per_rank(self) -> pd.DataFrame:
        return pd.DataFrame(self.__votes_per_rank)

    @property
    def alternative_to_mean_votes(self) -> Dict[str, float]:
        return self.__alternative_to_mean_votes

    def save_voting_data(self, directory: str) -> List[str]:
        votes_per_rank_file = os.path.join(directory, 'votes_per_rank.csv')
        self.votes_per_rank.to_csv(votes_per_rank_file, sep=';')
        logging.info(f'Saved votes per rank from Borda voting to {votes_per_rank_file}')
        alternative_to_mean_votes_file = os.path.join(directory, 'mean_votes_per_alternative.csv')
        indices = list(self.alternative_to_mean_votes.keys())
        data = list(self.alternative_to_mean_votes.values())
        headers = ['mean votes']
        data = pd.DataFrame(
            data=data,
            index=indices,
            columns=headers)
        data.to_csv(alternative_to_mean_votes_file, sep=';')
        logging.info(f'Saved mean votes from Borda voting to {alternative_to_mean_votes_file}')
        return [
            votes_per_rank_file,
            alternative_to_mean_votes_file
        ]

    def vote(self, data: pd.DataFrame, number_of_alternatives: int, columns_with_ranks: List[str], numpy_alternatives: np.ndarray) -> Dict[str, float]:
        # go through each rank (per each alpha value)
        # sort values to get positions for borda voting
        alternative_to_mean_votes: Dict[str, float] = defaultdict(lambda: 0.0)
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
                alternative_to_mean_votes[alternative] += (
                    number_of_alternatives - index)
        for alternative in alternative_to_mean_votes:
            alternative_to_mean_votes[alternative] /= len(columns_with_ranks)
        self.__alternative_to_mean_votes = alternative_to_mean_votes
        return alternative_to_mean_votes