import unittest

from recoxplainer.config import cfg
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.data_reader.user_item_dict import UserItemDict


class UserItemMatrixTest(unittest.TestCase):

    def setUp(self) -> None:
        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()

    def test_user_item_matrix(self):
        user_dict = UserItemDict(self.data._dataset_df)

        x = self.data._dataset_df.userId[0]
        y = self.data._dataset_df.itemId[0]
        v = self.data._dataset_df.rating[0]
        self.assertEqual(user_dict[x][y], v)
