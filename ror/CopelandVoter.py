from typing import List, Tuple
import numpy as np
import pandas as pd
import logging

class CopelandVoter():
    def __init__(self) -> None:
        self.__voting_matrix: np.ndarray = None
        self.__voting_sum: List[Tuple[str, float]] = []

    @property
    def voting_matrix(self) -> np.ndarray:
        return self.__voting_matrix

    @property
    def voting_sum(self) -> List[Tuple[str, float]]:
        return self.__voting_sum

    def vote(self, data: pd.DataFrame, columns_with_ranks: List[str], eps: float) -> np.array:
        numpy_alternatives: np.ndarray = np.array(list(data.index))
        number_of_alternatives = len(numpy_alternatives)
        votes = np.zeros(shape=(number_of_alternatives, number_of_alternatives))
        # reset results
        self.__voting_sum = []
        for column_name in columns_with_ranks:
            for row_idx, row_alternative_name in enumerate(numpy_alternatives):
                # run only over columns that index is greater than row index - less calculations
                for col_idx, column_alternative_name in zip(range(row_idx+1,number_of_alternatives), numpy_alternatives[row_idx+1:]):
                    row_alternative_value = data.loc[row_alternative_name, column_name]
                    column_alternative_value = data.loc[column_alternative_name, column_name]
                    # if in this rank alternative from row is preferred than the alternative from col
                    # then row alternative's value is increased by one (one vote)
                    # if alternative from column is preferred by alternative from row
                    # then alternative from column gets one point.
                    # Otherwise (alternatives' values are equal, with eps precision)
                    # both alternatives get 0.5
                    if row_alternative_value > column_alternative_value + eps:
                        logging.debug(f'Alternative in row {row_alternative_name} has greater value than alternative in column {column_alternative_name}')
                        votes[row_idx, col_idx] += 1
                    elif row_alternative_value + eps < column_alternative_value:
                        logging.debug(f'Alternative in row {row_alternative_name} has lower value than alternative in column {column_alternative_name}')
                        votes[col_idx, row_idx] += 1
                    else:
                        logging.debug(f'Alternative in row {row_alternative_name} has same value as alternative in column {column_alternative_name}')
                        votes[row_idx, col_idx] += 0.5
                        votes[col_idx, row_idx] += 0.5

        self.__voting_matrix = votes
        # aggregate votes - calculate
        per_alternative_votes_mean = np.zeros(shape=(number_of_alternatives))
        for alternative_idx in range(len(numpy_alternatives)):
            per_alternative_votes_mean[alternative_idx] = np.sum(votes[alternative_idx, :]) / (len(columns_with_ranks) * (number_of_alternatives-1))
        for alternative, mean_votes in zip(numpy_alternatives, per_alternative_votes_mean):
            self.__voting_sum.append((alternative, mean_votes))
        return per_alternative_votes_mean