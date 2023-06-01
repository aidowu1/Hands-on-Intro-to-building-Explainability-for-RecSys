import random
import unittest as ut
import inspect
import os
from pprint import pprint
import pathlib
import pandas as pd
import numpy as np
from pprint import pprint
from typing import Tuple, Dict, List
import pickle

working_dir = pathlib.Path(os.getcwd()).parent.parent
os.chdir(working_dir)

from Code.DataAccessLayer.Src.MovielensDataset import MovielenDataset
import Code.DataAccessLayer.Src.Constants as c
import Code.DataAccessLayer.Src.MovieLensDataPreProcessing as md
from Code.Model.Src.FactorizationMachineModel import FMModel
from Code.Utils.Src.Logging import Logger
from Code.Utils.Src.Utils import Helpers
from Code.Constants import LINE_DIVIDER

class MyTestCase(ut.TestCase):
    """
    Test suit for the "MovielensDataset" component
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        self.logger = Logger.getLogger()
        self.logger.info(f"Working folder is: {working_dir}")
        self.one_hot_encoder_pickle_path = "C:/temp/one_hot_encoder.pl"

    def test_MovielensDataset_Constructor_Is_Valid(self):
        """
        Test the validity of 'MovielensDataset' constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        dataset = MovielenDataset()
        self.assertIsNotNone(dataset, msg=error_msg)
        self._displayDataframeSample(dataset.user_item_interaction_df, "user_item_interaction_df")
        self._displayDataframeSample(dataset.user_item_interaction_modified_df, "user_item_interaction_modified_df")
        self._displayDataframeSample(dataset.items_side_info_df, "items_side_info_df")
        self._displayDataframeSample(dataset.items_side_info_modified_df, "items_side_info_modified_df")
        self._displayDataframeSample(dataset.users_side_info_df, "users_side_info_df")
        self._displayDataframeSample(dataset.users_side_info_modified_df, "users_side_info_modified_df")

    def test_MovielensDataset_Partition_Dataset_Is_Valid(self):
        """
        Test the validity of 'MovielensDataset' component when
        used to partition the dataset into train and test fractions
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        dataset = MovielenDataset()
        dataset.splitDataset()
        self.assertIsNotNone(dataset.train_users_items_df, msg=error_msg)
        self.assertIsNotNone(dataset.test_users_items_df, msg=error_msg)
        self.assertIsNotNone(dataset.train_rating, msg=error_msg)
        self.assertIsNotNone(dataset.test_rating, msg=error_msg)

    def test_MovielensDataset_OnehotEconding_Transform_For_Training_Is_Valid(self):
        """
        Test the validity of 'MovielensDataset' component when the one-hot encoding transform is invoked
        after the partitioning of the datset into 'train' and 'test' fractions
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        dataset = MovielenDataset()
        dataset.splitDataset()
        X_train, y_train, one_hot_encoder, ohe_columns = md.Movielens100KPreprocessor.transformFMModelItemSideDataset(
            user_item_interaction_df=dataset.train_users_items_df,
            item_side_df=dataset.items_side_info_df,
            is_consolidate_genre_values=c.IS_CONSOLIDATE_GENRE_VALUES,
            is_get_label_from_interaction_data=False
        )
        Helpers.writePickleToFile(one_hot_encoder, self.one_hot_encoder_pickle_path)
        self.logger.info(f"One-hot encoder is serialised at this path: {self.one_hot_encoder_pickle_path}")
        if not y_train:
            y_train = dataset.train_rating
        self.assertIsNotNone(X_train, msg=error_msg)
        self.assertIsNotNone(y_train, msg=error_msg)
        self.assertIsNotNone(one_hot_encoder, msg=error_msg)
        self.assertIsNotNone(ohe_columns, msg=error_msg)
        self._displayNumpyArray(X_train.toarray(), "X_train")
        self._displayNumpyArray(y_train, "y_train")
        self._displayNumpyArray(ohe_columns, "ohe_columns")

    def test_MovielensDataset_OnehotEconding_Transform_For_Testing_Is_Valid(self):
        """
        Test the validity of 'MovielensDataset' component when the one-hot encoding transform is invoked
        after the partitioning of the dataset into 'train' and 'test' fractions
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        dataset = MovielenDataset()
        dataset.splitDataset()
        one_hot_encoder = Helpers.readPickleFromFile(self.one_hot_encoder_pickle_path)
        self.logger.info(f"One-hot encoder object has successfully de-serialised to this path: {self.one_hot_encoder_pickle_path}")
        X_test, y_test, _, ohe_columns = md.Movielens100KPreprocessor.transformFMModelItemSideDataset(
            user_item_interaction_df=dataset.test_users_items_df,
            item_side_df=dataset.items_side_info_df,
            is_consolidate_genre_values=c.IS_CONSOLIDATE_GENRE_VALUES,
            is_get_label_from_interaction_data=False,
            one_hot_encoder=one_hot_encoder
        )
        if not y_test:
            y_test = dataset.train_rating
        self.assertIsNotNone(X_test, msg=error_msg)
        self.assertIsNotNone(y_test, msg=error_msg)
        self.assertIsNotNone(one_hot_encoder, msg=error_msg)
        self.assertIsNotNone(ohe_columns, msg=error_msg)
        self._displayNumpyArray(X_test.toarray(), "X_test")
        self._displayNumpyArray(y_test, "y_test")
        self._displayNumpyArray(np.array(ohe_columns), "ohe_columns")

    def test_MovielensDataset_Persist_RS_LIME_Library_Dataset_Is_Valid(self):
        """
        Test the validity of 'MovielensDataset' component tp persist RS-LIME dataset
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        rs_lime_data_folder = "ThirdParty/lime_rs/Data"
        dataset = MovielenDataset()
        dataset.splitDataset()
        training_df = dataset.train_users_items_df.copy()
        training_df.rename(columns={"user id": "user_id", "item id": "item_id"}, inplace=True)
        training_df.to_csv(f"{rs_lime_data_folder}/training.csv", index=False)
        test_df = dataset.test_users_items_df.copy()
        test_df.rename(columns={"user id": "user_id", "item id": "item_id"}, inplace=True)
        test_df.to_csv(f"{rs_lime_data_folder}/test.csv", index=False)
        item_info_wide = dataset.items_side_info_modified_df.copy()
        item_info_wide.rename(columns={"item id": "item_id"}, inplace=True)
        item_info_wide.to_csv(f"{rs_lime_data_folder}/item_features.csv", index=False)
        y_train = np.array(dataset.train_rating)
        np.save(f"{rs_lime_data_folder}/train_rating", y_train)
        y_test = np.array(dataset.test_rating)
        np.save(f"{rs_lime_data_folder}/test_rating", y_test)
        print(f"Successfully persisted dataset for RS-LIME in the folder: {rs_lime_data_folder}")

    def _displayDataframeSample(self, df: pd.DataFrame, title: str = ""):
        """
        Helper used to display a dataframe dataset sample
        :param df: Dataset
        :param title: Title of dataset
        """
        print(f"Sample of the '{title}' dataset:")
        print(Helpers.tableize(df.head(5)))
        print(LINE_DIVIDER)
        print("\n\n")

    def _displayNumpyArray(self, data: np.ndarray, title: str = ""):
        """
        Helper used to display a numpy array dataset sample
        :param data: Dataset
        :param title: Title of dataset
        """
        print(f"Sample of the '{title}' dataset:")
        pprint(data[:5])
        print(LINE_DIVIDER)
        print("\n\n")


if __name__ == '__main__':
    ut.main()
