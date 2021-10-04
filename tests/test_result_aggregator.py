from ror.ResultAggregator import RankItem, group_equal_alternatives_in_ranking
import unittest


class TestResultAggregator(unittest.TestCase):
    def test_groupping_alternatives_in_rank(self):
        rank = ['a1', 'a2', 'a3']
        rank_values = [0.0, 0.0, 1.0]
        rank = [RankItem(item, value) for item, value in zip(rank, rank_values)]
        new_rank = group_equal_alternatives_in_ranking(rank, 0.1)

        self.assertEqual(len(new_rank), 2)
        self.assertListEqual([RankItem('a1', 0.0), RankItem('a2', 0.0)], new_rank[0])
        self.assertListEqual([RankItem('a3', 1.0)], new_rank[1])
    
    def test_groupping_alternatives_in_rank_1(self):
        rank = ['a1', 'a2', 'a3']
        rank_values = [0.0, 0.5, 1.0]
        rank = [RankItem(item, value) for item, value in zip(rank, rank_values)]
        new_rank = group_equal_alternatives_in_ranking(rank, 0.1)

        self.assertEqual(len(new_rank), 3)
        self.assertListEqual([RankItem('a1', 0.0)], new_rank[0])
        self.assertListEqual([RankItem('a2', 0.5)], new_rank[1])
        self.assertListEqual([RankItem('a3', 1.0)], new_rank[2])

    def test_groupping_alternatives_in_rank_2(self):
        rank = []
        rank_values = []
        rank = [RankItem(item, value) for item, value in zip(rank, rank_values)]
        new_rank = group_equal_alternatives_in_ranking(rank, 0.1)

        self.assertEqual(len(new_rank), 0)
