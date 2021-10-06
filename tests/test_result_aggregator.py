from functools import reduce
from ror.result_aggregator_utils import BIG_NUMBER, RankItem, group_equal_alternatives_in_ranking
from ror.ResultAggregator import weighted_results_aggregator
import unittest
from tests.helpers.test_ror_result_helpers import DEFAULT_MAPPING, create_ror_result


class TestResultAggregator(unittest.TestCase):
    def test_groupping_alternatives_in_rank(self):
        rank = ['a1', 'a2', 'a3']
        rank_values = [0.0, 0.0, 1.0]
        rank = [RankItem(item, value)
                for item, value in zip(rank, rank_values)]
        new_rank = group_equal_alternatives_in_ranking(rank, 0.1)

        self.assertEqual(len(new_rank), 2)
        self.assertListEqual(
            [RankItem('a1', 0.0), RankItem('a2', 0.0)], new_rank[0])
        self.assertListEqual([RankItem('a3', 1.0)], new_rank[1])

    def test_groupping_alternatives_in_rank_1(self):
        rank = ['a1', 'a2', 'a3']
        rank_values = [0.0, 0.5, 1.0]
        rank = [RankItem(item, value)
                for item, value in zip(rank, rank_values)]
        new_rank = group_equal_alternatives_in_ranking(rank, 0.1)

        self.assertEqual(len(new_rank), 3)
        self.assertListEqual([RankItem('a1', 0.0)], new_rank[0])
        self.assertListEqual([RankItem('a2', 0.5)], new_rank[1])
        self.assertListEqual([RankItem('a3', 1.0)], new_rank[2])

    def test_groupping_alternatives_in_rank_2(self):
        rank = []
        rank_values = []
        rank = [RankItem(item, value)
                for item, value in zip(rank, rank_values)]
        new_rank = group_equal_alternatives_in_ranking(rank, 0.1)

        self.assertEqual(len(new_rank), 0)

    def test_weighted_results_aggregator(self):
        data = {
            # alternative: [r_value, q_value, s_value]
            'a1': [1.0, 2.0, 3.0],
            'a2': [0.0, 1.0, 2.0],
            'a3': [2.0, 1.0, 2.0]
        }
        ror_result = create_ror_result(data)
        weights = {
            'R': 1.0,
            'Q': 2.0,
            'S': 1.0
        }

        result = weighted_results_aggregator(ror_result, DEFAULT_MAPPING, weights, eps=1e-9)
        final_rank = result.final_rank
        # get all alternative names in the final rank
        alternatives_in_final_rank = list(map(lambda rank_item: rank_item.alternative, reduce(
            list.__add__, (list(items) for items in final_rank))))
        self.assertTrue(all(
            [alternative in alternatives_in_final_rank for alternative in ['a1', 'a2', 'a3']]))

        for rank_items in final_rank:
            self.assertEqual(len(rank_items), 1)
        self.assertEqual(final_rank[0][0].alternative, 'a2')
        self.assertAlmostEqual(final_rank[0][0].value, 2.5)
        self.assertEqual(final_rank[1][0].alternative, 'a3')
        self.assertAlmostEqual(final_rank[1][0].value, 4.5)
        self.assertEqual(final_rank[2][0].alternative, 'a1')
        self.assertAlmostEqual(final_rank[2][0].value, 5.0)

    def test_weighted_results_aggregator_with_zero_weights(self):
        data = {
            # alternative: [r_value, q_value, s_value]
            'a1': [1.0, 2.0, 3.0],
            'a2': [0.0, 1.0, 2.0],
            'a3': [2.0, 1.0, 2.0]
        }
        ror_result = create_ror_result(data)
        weights = {
            'R': 0.0,
            'Q': 2.0,
            'S': 0.0
        }

        result = weighted_results_aggregator(
            ror_result, DEFAULT_MAPPING, weights, eps=1e-9)
        final_rank = result.final_rank
        # get all alternative names in the final rank
        alternatives_in_final_rank = list(map(lambda rank_item: rank_item.alternative, reduce(
            list.__add__, (list(items) for items in final_rank))))
        self.assertTrue(all(
            [alternative in alternatives_in_final_rank for alternative in ['a1', 'a2', 'a3']]))

        # there are 2 items on the first place
        self.assertEqual(len(final_rank[0]), 2)
        first_place_alternatives = set(
            item.alternative for item in final_rank[0])
        self.assertSetEqual(first_place_alternatives, set(['a2', 'a3']))
        self.assertAlmostEqual(final_rank[0][0].value, 2*BIG_NUMBER + 0.5)
        self.assertAlmostEqual(final_rank[0][0].value, final_rank[0][1].value)
        self.assertEqual(final_rank[1][0].alternative, 'a1')
        self.assertAlmostEqual(final_rank[1][0].value, 2*BIG_NUMBER+1.0)
