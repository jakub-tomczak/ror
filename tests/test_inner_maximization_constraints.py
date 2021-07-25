from ror.inner_maximization_constraints import create_inner_maximization_constraints
import unittest
from ror.Dataset import read_dataset_from_txt


class TestInnerMaximization(unittest.TestCase):
    def test_creating_inner_maximization_constraints_for_none_dataset(self):
        with self.assertRaises(AssertionError):
            create_inner_maximization_constraints(None)

    def test_creating_inner_maximization_constraints_for_none_dataset(self):
        data = read_dataset_from_txt("tests/datasets/example.txt")
        inner_max_constraints = create_inner_maximization_constraints(data)

        
