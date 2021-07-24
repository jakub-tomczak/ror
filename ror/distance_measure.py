from ror.Dataset import Dataset
from ror.Constraint import Constraint
import numpy as np


def function_d_constraint(alpha: float, alternative: str, dataset: Dataset):
    '''
        Calculates alpha*L_1 + (1-alpha)*L_inf
        alpha * sum by criterion(1-u_criterion(alternative)) + (1-alpha)*lambda(alternative)
    '''
    assert 0 <= alpha <= 1, f"Alpha must be in range <0, 1>, provided {alpha}"
    
    data = dataset.get_data_for_alternative(alternative)
    l1_variables = -1 * alpha * data # this is from free variable
    l1_free = alpha * len(dataset.criteria)
    l_inf = (1-alpha) * data
    
    variable_names = [f"{alternative}_{criterion}" for criterion in dataset.criteria]

    return [l1_variables, l_inf]