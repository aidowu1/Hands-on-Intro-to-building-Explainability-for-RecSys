import unittest as ut
from inspect import stack
from os import chdir
import pandas as pd
import numpy as np
from typing import List, Tuple

import Code.DataAccessLayer.Src.DataProvider as dp
from Code.Constants import PROJECT_ROOT_PATH

class TestDataProvider(ut.TestCase):
    """
    Test suit for the DataProvider    
    """

    def setUp(self):
        """
        Setup fixture for the unit testing
        """
        chdir(PROJECT_ROOT_PATH)
        from Code.Configs.Src import ConfigProvider as cp
        self.config = cp.cfg

    def tearDown(self):
        """
        Tear down fixture for unit testing
        """
        pass

    def test_DataReader_Constructor_Is_Valid(self):
        """
        Test the validity of the 'DataReader' constructor
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        data_reader = dp.DataReader(**self.config.testdata)
        self.assertIsNotNone(data_reader, msg=error_msg)

    def test_DataReader_Dataset_Getter_Property_Is_Valid(self):
        """
        Test the validity of the 'DataReader' getter property
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        data_reader = dp.DataReader(**self.config.testdata)
        data_df = data_reader.dataset
        n_rows = data_df.shape[0]
        n_cols = data_df.shape[1]
        self.assertIsNotNone(data_reader, msg=error_msg)
        self.assertIsNotNone(data_df, msg=error_msg)
        self.assertTrue(isinstance(data_df, pd.DataFrame), msg=error_msg)
        self.assertTrue(n_rows > 0, msg=error_msg)
        self.assertTrue(n_cols > 0, msg=error_msg)
        print(f"Dataset with {n_rows} rows and {n_cols} columns has been read successfully!!")

    def test_DataReader_Make_User_And_Item_Ids_Consecutive_Is_Valid(self):
        """
        Test the validity of the 'DataReader' making the user and item ids consecutive
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        data_reader = dp.DataReader(**self.config.testdata)
        original_data_df = data_reader.dataset.copy()
        data_reader.makeConsecutiveIdsInDataset()
        new_data_df = data_reader.dataset.copy()
        original_check_status, new_check_status = self.isDataConsecutive(data_reader)
        self.assertIsNotNone(data_reader, msg=error_msg)
        self.assertIsNotNone(original_data_df, msg=error_msg)
        self.assertIsNotNone(new_data_df, msg=error_msg)
        self.assertTrue(original_check_status is False, msg=error_msg)
        self.assertTrue(new_check_status is True, msg=error_msg)

    def test_DataReader_Binarize_Interaction_Matrix_Rating_Is_Valid(self):
        """
        Test the validity of the 'DataReader' capability to binarize the interaction matrix rating/scores
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        threshold = 3
        data_reader = dp.DataReader(**self.config.testdata)
        before_data_df = data_reader.dataset.copy()
        before_ratings = before_data_df.rating.values
        data_reader.binarize(binary_threshold=threshold)
        after_data_df = data_reader.dataset.copy()
        after_ratings = after_data_df.rating.values
        is_rating_not_binarised = not np.all(before_ratings <= 1)
        is_rating_binarised = np.all(after_ratings <= 1)
        self.assertIsNotNone(data_reader, msg=error_msg)
        self.assertTrue(is_rating_not_binarised, msg=error_msg)
        self.assertTrue(is_rating_binarised, msg=error_msg)


    def isDataConsecutive(self,
                          data_reader: dp.DataReader) -> Tuple[bool, bool]:
        """
        Check if user and item id lists of the interaction dataset are consecutive
        :param data_reader: Interaction dataset data reader
        :return: Boolean flag to indicate if the user/item are consecutive
        """
        def consecutiveIdSequenceCheck(ids: List[int]) -> bool:
            """
            Check if a sequence of ids are consecutive
            :param ids: Sequence of ids
            :return: Boolean flag to indicate if the sequence is consecutive
            """
            return sorted(ids) == ids

        def consecutiveItemUserCheck(items_ids: List[int], users_ids: List[int]) -> bool:
            """
            Check if item and users ids are consecutive
            :param items_ids: Items ids
            :param users_ids: Users ids
            :return: Boolean flag to indicate if the user/item are consecutive
            """
            items_check = consecutiveIdSequenceCheck(items_ids)
            users_check = consecutiveIdSequenceCheck(users_ids)
            status = items_check and users_check
            return status

        original_items = data_reader.original_item_id_df.item_id.tolist()
        original_users = data_reader.original_user_id_df.user_id.tolist()
        new_items = data_reader.new_item_id_df.itemId.tolist()
        new_users = data_reader.new_user_id_df.userId.tolist()
        original_check_status = consecutiveItemUserCheck(original_items, original_users)
        new_check_status = consecutiveItemUserCheck(new_items, new_users)
        return original_check_status, new_check_status




    
if __name__ == '__main__':
    ut.main()

