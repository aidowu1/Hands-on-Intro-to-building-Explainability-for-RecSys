import pandas as pd
import numpy as np
from typing import Tuple
import math
from typing import Union, List

import Code.DataAccessLayer.Src.MovieLensDataPreProcessing as md
from Code.DataAccessLayer.Src import Constants as c

class MovielenDataset(object):
    """
    Movielens dataset component
    This component serves the following items:
        => Training user-item interaction data (train_users_items_df)
        => Training item side data (train_items_df)
        => Training user side data (train_users_df)
        => Test user-item interaction data (test_users_items_df)
        => Test item side data (test_items_df)
        => Test user side data (test_users_df)
    """
    def __init__(self, is_consolidate_genre_values: bool = c.IS_CONSOLIDATE_GENRE_VALUES):
        """
        Constructor
        :param: is_consolidate_genre_values: Flag to indicate the consolidation of genre values
        """
        self.pre_processor = md.Movielens100KPreprocessor(is_consolidate_genre_values=is_consolidate_genre_values)
        self.user_item_interaction_df = self.pre_processor.rating_df[c.MOVIELENS_USER_ITEM_INTERACTION_WITH_RATING_COLUMNS]
        self.rating = self.pre_processor.rating_df.rating.values
        self.items_side_info_df = self.pre_processor.preProcessItemGenre()
        self.users_side_info_df = self.pre_processor.user_df
        self.train_users_items_df = None
        self.test_users_items_df = None
        self.train_rating = None
        self.test_rating = None

        self.user_item_interaction_modified_df = None
        self.items_side_info_modified_df = None
        self.users_side_info_modified_df = None
        self._item_id_df = None
        self._user_id_df = None
        self.original_user_id_df = None
        self.original_item_id_df = None
        self.new_user_id_df = None
        self.new_item_id_df = None
        self.train_split_indices = None
        self.test_split_indices = None
        self._makeUserItemInteractionConsecutive()
        self._makeItemSideInfoConsecutive()
        self._makeUserSideInfoConsecutive()

    def _splitDataset(self,
                      test_fraction: int = c.MOVIELENS_TEST_DATA_SPLIT_FRACTION
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the user-item interaction dataset into train/test factions
        :param test_fraction: Split fraction for the test partition
        """
        pass

    def _makeUserItemInteractionConsecutive(self):
        """
        Make the user and item ids in the user-item interaction dataset consecutive
        """
        user_item_interaction_clone_df = self.user_item_interaction_df.rename({
            c.MOVIELENS_USER_ID_COLUMN_KEY: c.MOVIELENS_USER_ID_SEQUENTIAL_COLUMN_KEY,
            c.MOVIELENS_ITEM_ID_COLUMN_KEY: c.MOVIELENS_ITEM_ID_SEQUENTIAL_COLUMN_KEY
        }, axis=1)

        self._user_id_df = user_item_interaction_clone_df[[c.MOVIELENS_USER_ID_SEQUENTIAL_COLUMN_KEY]].drop_duplicates().reindex()
        num_user = len(self._user_id_df)
        self._user_id_df[c.MOVIELENS_USER_ID_COLUMN_KEY] = np.arange(num_user)

        self.user_item_interaction_modified_df = pd.merge(
            user_item_interaction_clone_df, self._user_id_df,
            on=[c.MOVIELENS_USER_ID_SEQUENTIAL_COLUMN_KEY], how='left')

        self._item_id_df = user_item_interaction_clone_df[[c.MOVIELENS_ITEM_ID_SEQUENTIAL_COLUMN_KEY]].drop_duplicates()
        num_item = len(self._item_id_df)
        self._item_id_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY] = np.arange(num_item)

        self.user_item_interaction_modified_df = pd.merge(
            self.user_item_interaction_modified_df, self._item_id_df,
            on=[c.MOVIELENS_ITEM_ID_SEQUENTIAL_COLUMN_KEY], how='left')

        self.original_user_id_df = self._user_id_df.set_index(c.MOVIELENS_USER_ID_COLUMN_KEY)
        self.original_item_id_df = self._item_id_df.set_index(c.MOVIELENS_ITEM_ID_COLUMN_KEY)
        self.new_user_id_df = self._user_id_df.set_index(c.MOVIELENS_USER_ID_SEQUENTIAL_COLUMN_KEY)
        self.new_item_id_df = self._item_id_df.set_index(c.MOVIELENS_ITEM_ID_SEQUENTIAL_COLUMN_KEY)

        self.user_item_interaction_modified_df = self.user_item_interaction_modified_df[
            list(self.user_item_interaction_df.columns)]

        self.user_item_interaction_modified_df[c.MOVIELENS_USER_ID_COLUMN_KEY] = \
            [int(i) for i in self.user_item_interaction_modified_df[c.MOVIELENS_USER_ID_COLUMN_KEY]]
        self.user_item_interaction_modified_df[[c.MOVIELENS_ITEM_ID_COLUMN_KEY]] = \
            [int(i) for i in self.user_item_interaction_modified_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY]]

    def _makeItemSideInfoConsecutive(self):
        """
        Make the item ids for the item side dataset consecutive
        """
        items_side_info_df_clone_df = self.items_side_info_df.rename({
            c.MOVIELENS_ITEM_ID_COLUMN_KEY: c.MOVIELENS_ITEM_ID_SEQUENTIAL_COLUMN_KEY
        }, axis=1)

        self.items_side_info_modified_df = pd.merge(
            items_side_info_df_clone_df, self._item_id_df,
            on=[c.MOVIELENS_ITEM_ID_SEQUENTIAL_COLUMN_KEY], how='right')
        self.items_side_info_modified_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY] = \
            self.items_side_info_modified_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY].astype('int')

        self.items_side_info_modified_df = self.items_side_info_modified_df[
            list(self.items_side_info_df.columns)]

        self.items_side_info_modified_df[[c.MOVIELENS_ITEM_ID_COLUMN_KEY]] = \
            [int(i) for i in self.items_side_info_modified_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY]]

    def _makeUserSideInfoConsecutive(self):
        """
        Make the user ids for the user side dataset consecutive
        """
        users_side_info_df_clone_df = self.users_side_info_df.rename({
            c.MOVIELENS_USER_ID_COLUMN_KEY: c.MOVIELENS_USER_ID_SEQUENTIAL_COLUMN_KEY
        }, axis=1)

        self.users_side_info_modified_df = pd.merge(
            users_side_info_df_clone_df, self._user_id_df,
            on=[c.MOVIELENS_USER_ID_SEQUENTIAL_COLUMN_KEY], how='right')
        self.users_side_info_modified_df[c.MOVIELENS_USER_ID_COLUMN_KEY] = \
            self.users_side_info_modified_df[c.MOVIELENS_USER_ID_COLUMN_KEY].astype('int')

        self.users_side_info_modified_df = self.users_side_info_modified_df[
            list(self.users_side_info_df.columns)]

        self.users_side_info_modified_df[[c.MOVIELENS_USER_ID_COLUMN_KEY]] = \
            [int(i) for i in self.users_side_info_modified_df[c.MOVIELENS_USER_ID_COLUMN_KEY]]

    def _shuffleDataset(self, test_data_fraction: float):
        """
        Shuffles the user-item interaction dataset
        :param rating_df: Rating dataset
        :return: Indices for training, testing and validation datasets
        """
        n_all_rows = self.user_item_interaction_modified_df.shape[0]
        n_train_rows = math.floor((1 - test_data_fraction) * n_all_rows)
        row_indices = list(range(n_all_rows))
        np.random.shuffle(row_indices)
        self.train_split_indices = row_indices[:n_train_rows]
        self.test_split_indices = row_indices[n_train_rows: n_all_rows]

    def splitDataset(self, test_data_fraction: float = c.MOVIELENS_TEST_DATA_SPLIT_FRACTION):
        """
        Split rating dataset into train, test and validation fractions
        """
        self._shuffleDataset(test_data_fraction)
        self.train_users_items_df = self.user_item_interaction_modified_df.iloc[self.train_split_indices]
        self.test_users_items_df = self.user_item_interaction_modified_df.iloc[self.test_split_indices]
        self.train_users_items_df.reset_index(drop=True, inplace=True)
        self.test_users_items_df.reset_index(drop=True, inplace=True)
        self.train_rating = self.rating[self.train_split_indices]
        self.test_rating = self.rating[self.test_split_indices]

    def getOriginalItemId(self, new_item_id) -> Union[int, List]:
        """
        Getter property to get the original item ID
        :param new_item_id: Original item id
        :return: Returns the original item id or list of item ids
        """
        if isinstance(new_item_id, int):
            return self.original_item_id_df.loc[new_item_id].item_id
        return list(self.original_item_id_df.loc[new_item_id].item_id)

    def getOriginalUserId(self, new_user_id) -> Union[int, List]:
        """
        Getter property to get the original user ID
        :param new_user_id: Original user id
        :return: Returns the original user id or list of user ids
        """
        if isinstance(new_user_id, int):
            return self.original_user_id_df.loc[new_user_id].user_id
        return list(self.original_user_id_df.loc[new_user_id].user_id)




