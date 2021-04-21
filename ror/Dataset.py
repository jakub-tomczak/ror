import numpy as np
from typing import List, Tuple
import pandas as pd
import os


criterion_types = {
    "gain": "g",
    "cost": "c"
}

class Dataset:
    def __init__(self, alternatives: List[str], data: any, criteria: List[Tuple[str, str]]):
        assert type(data) is np.ndarray, "Data must be a numpy array"
        assert len(alternatives) == data.shape[0],\
            "Number of alternatives labels doesn't match the number of data rows"
        assert len(criteria) == data.shape[1],\
            "Number of criteria doesn't match the number of data columns"

        self.alternatives = alternatives
        self.data = data
        self.criteria = criteria

def read_dataset_from_txt(filename: str):
    if not os.path.exists(filename):
        print(f"file {filename} doesn't exist")
        return None
    
    data = pd.read_csv(filename, sep=',')

    def parse_criterion(criterion: str) -> Tuple[str, str]:
        data = criterion.strip().split('[')

        if len(data) != 2 or len(data[0]) < 1 or len(data[1]) < 1:
            print(f"Failed to parse criterion {criterion}")
            return ('', '')
        criterion_type = data[1][0]
        if criterion_type not in criterion_types.values():
            print(f"Invalid criterion type: {criterion_type}, expected values: {criterion_type.values()}")
            return ('', '')
        return (data[0], criterion_type)

    alternatives = data.iloc[:, 0]
    values = data.iloc[:, 1:].to_numpy()
    # skip first column - this should be id
    criteria = [parse_criterion(criterion) for criterion in data.columns[1:]]

    return Dataset(
        alternatives=alternatives,
        data=values,
        criteria=criteria
    )


