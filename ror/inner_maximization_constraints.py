from ror.Dataset import Dataset


def create_inner_maximization_constraints(data: Dataset):
    assert data is not None, "dataset must not be none"