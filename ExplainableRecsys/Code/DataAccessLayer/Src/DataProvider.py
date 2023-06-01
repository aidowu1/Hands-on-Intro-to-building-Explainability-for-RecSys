import numpy as np
import pandas as pd
from typing import List, Optional, Union

class DataReader:
    """
    Data provider component used to read dataset for the Recsys computation
    """

    def __init__(self,
                 filepath_or_buffer: str,
                 sep: str,
                 column_names: list,
                 skiprows: int = 0):
        """
        Constructor
        :param filepath_or_buffer: File/buffer path of dataset
        :param sep: CSV separator
        :param column_names: Names of dataset columns (selected columns of interest)
        :param skiprows: Flag to indicate the rows to skip
        """
        self.filepath_or_buffer = filepath_or_buffer
        self.sep = sep
        self.names = column_names
        self.skiprows = skiprows

        self._dataset_df = None
        self._num_user = None
        self._num_item = None

    @property
    def dataset(self) -> pd.DataFrame:
        """
        Dataset getter property
        :return: Dataset
        """
        if self._dataset_df is None:
            self._dataset_df = pd.read_csv(filepath_or_buffer=self.filepath_or_buffer,
                                           sep=self.sep,
                                           names=self.names,
                                           skiprows=self.skiprows,
                                           engine='python')
            self._num_item = int(self._dataset_df[['itemId']].nunique())
            self._num_user = int(self._dataset_df[['userId']].nunique())

        return self._dataset_df

    @dataset.setter
    def dataset(self, new_data):
        """
        Dataset setter property
        :param new_data: New dataset
        """
        self._dataset_df = new_data

    def makeConsecutiveIdsInDataset(self):
        """
        Makes the user and item ids consecutively structured within the dataset
        """
        # TODO: create mapping function
        dataset_df = self.dataset.rename({
            "userId": "user_id",
            "itemId": "item_id"
        }, axis=1)

        user_id_df = dataset_df[['user_id']].drop_duplicates().reindex()
        num_user = len(user_id_df)

        user_id_df['userId'] = np.arange(num_user)
        self._dataset_df = pd.merge(
            dataset_df, user_id_df,
            on=['user_id'], how='left')

        item_id_df = dataset_df[['item_id']].drop_duplicates()
        num_item = len(item_id_df)
        item_id_df['itemId'] = np.arange(num_item)

        self._dataset_df = pd.merge(
            self._dataset_df, item_id_df,
            on=['item_id'], how='left')

        self.original_user_id_df = user_id_df.set_index('userId')
        self.original_item_id_df = item_id_df.set_index('itemId')
        self.new_user_id_df = user_id_df.set_index('user_id')
        self.new_item_id_df = item_id_df.set_index('item_id')

        self._dataset_df = self.dataset[
            ['userId', 'itemId', 'rating', 'timestamp']
        ]

        self._dataset_df.userId = [int(i) for i in self._dataset_df.userId]
        self._dataset_df.itemId = [int(i) for i in self._dataset_df.itemId]

    def binarize(self, binary_threshold=1):
        """
        Binarize the rating/score into 0 or 1, for implicit feedback
        :param binary_threshold: Binary threshold
        """
        dataset_clone_df = self.dataset.copy()
        self._dataset_df.loc[dataset_clone_df['rating'] > binary_threshold, "rating"] = 1
        self._dataset_df.loc[dataset_clone_df['rating'] <= binary_threshold, "rating"] = 0


    @property
    def num_user(self) -> int:
        """
        Getter property for the number of users
        """
        return self._num_user

    @property
    def num_item(self) -> int:
        """
        Getter property for the number of items
        """
        return self._num_item

    def getOriginalUserId(self, new_user_id) -> Union[int, List]:
        """
        Getter property to get the original user ID
        :param new_user_id: Original user id
        :return: Returns the original user id or list of user ids
        """
        if isinstance(new_user_id, int):
            return self.original_user_id_df.loc[new_user_id].user_id
        return list(self.original_user_id_df.loc[new_user_id].user_id)

    def getOriginalItemId(self, new_item_id) -> Union[int, List]:
        """
        Getter property to get the original item ID
        :param new_item_id: Original item id
        :return: Returns the original item id or list of item ids
        """
        if isinstance(new_item_id, int):
            return self.original_item_id_df.loc[new_item_id].item_id
        return list(self.original_item_id_df.loc[new_item_id].item_id)

    def getNewUserId(self, original_user_id) -> Union[int, List]:
        """
        Getter property to get the new user ID
        :param original_user_id: Original item id
        :return: Returns the new user id or list of users ids
        """
        if isinstance(original_user_id, int):
            return self.new_user_id_df.loc[original_user_id].userId
        return list(self.new_user_id_df.loc[original_user_id].userId)

    def getNewItemId(self, original_item_id) -> Union[int, List]:
        """
        Getter property to get the item user ID
        :param original_item_id: Original item id
        :return: Returns the new item id or list of item ids
        """
        if isinstance(original_item_id, int):
            return self.new_item_id_df.loc[original_item_id].itemId
        return list(self.new_item_id_df.loc[original_item_id].itemId)

    @property
    def dataset_df(self) ->pd.DataFrame:
        """
        Getter property for dataset table
        :return: Dataset table
        """
        return self._dataset_df


