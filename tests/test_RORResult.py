import unittest
from tests.helpers.test_ror_result_helpers import DEFAULT_MAPPING, create_ror_result


class TestRORResult(unittest.TestCase):
    def test_creating_result_table(self):
        data = {
            # alternative: [r_value, q_value, s_value]
            'a1': [1.0, 2.0, 3.0],
            'a2': [0.0, 1.0, 2.0],
            'a3': [2.0, 1.0, 2.0]
        }
        ror_result = create_ror_result(data)

        table = ror_result.get_result_table()
        self.assertEqual(len(table.columns), 4)
        self.assertSetEqual(set(table.columns), set(["alpha_0.0", "alpha_0.5", "alpha_1.0", "alpha_sum"]))
        self.assertEqual(table.index.name, 'id')

        self.assertAlmostEqual(table.loc['a1', 'alpha_0.0'], 1.0)
        self.assertAlmostEqual(table.loc['a1', 'alpha_0.5'], 2.0)
        self.assertAlmostEqual(table.loc['a1', 'alpha_1.0'], 3.0)
        self.assertAlmostEqual(table.loc['a1', 'alpha_sum'], 6.0)

        self.assertSetEqual(set(table.index.values), set(['a1', 'a2', 'a3']))
        self.assertEqual(table.shape[0], 3)

    def test_creating_result_dict(self):
        data = {
            # alternative: [r_value, q_value, s_value]
            'a1': [1.0, 2.0, 3.0],
            'a2': [0.0, 1.0, 2.0],
            'a3': [2.0, 1.0, 2.0]
        }
        ror_result = create_ror_result(data)
        result_dict = ror_result.get_results_dict(DEFAULT_MAPPING)
        self.assertEqual(len(result_dict), 3)
        self.assertSetEqual(set(result_dict.keys()), set(['a1', 'a2', 'a3']))
        self.assertListEqual(result_dict['a1'], [1.0, 2.0, 3.0])
