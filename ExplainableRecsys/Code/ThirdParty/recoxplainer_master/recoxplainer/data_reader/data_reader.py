import numpy as np
import pandas as pd


class DataReader:

    def __init__(self,
                 filepath_or_buffer: str,
                 sep: str,
                 names: list,
                 skiprows: int = 0):
        self.filepath_or_buffer = filepath_or_buffer
        self.sep = sep
        self.names = names
        self.skiprows = skiprows

        self._dataset = None
        self._num_user = None
        self._num_item = None
        self.dataset

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = pd.read_csv(filepath_or_buffer=self.filepath_or_buffer,
                                        sep=self.sep,
                                        names=self.names,
                                        skiprows=self.skiprows,
                                        engine='python')
            self._num_item = int(self._dataset[['itemId']].nunique())
            self._num_user = int(self._dataset[['userId']].nunique())

        self._dataset_df = self._dataset.copy()
        return self._dataset

    @dataset.setter
    def dataset(self, new_data):
        self._dataset = new_data

    def make_consecutive_ids_in_dataset(self):
        # TODO: create mapping function
        dataset = self.dataset.rename({
            "userId": "user_id",
            "itemId": "item_id"
        }, axis=1)

        user_id = dataset[['user_id']].drop_duplicates().reindex()
        num_user = len(user_id)

        user_id['userId'] = np.arange(num_user)
        self._dataset = pd.merge(
            dataset, user_id,
            on=['user_id'], how='left')

        item_id = dataset[['item_id']].drop_duplicates()
        num_item = len(item_id)
        item_id['itemId'] = np.arange(num_item)

        self._dataset = pd.merge(
            self._dataset, item_id,
            on=['item_id'], how='left')

        self.original_user_id = user_id.set_index('userId')
        self.original_item_id = item_id.set_index('itemId')
        self.new_user_id = user_id.set_index('user_id')
        self.new_item_id = item_id.set_index('item_id')

        self._dataset = self.dataset[
            ['userId', 'itemId', 'rating', 'timestamp']
        ]

        self._dataset.userId = [int(i) for i in self._dataset.userId]
        self._dataset.itemId = [int(i) for i in self._dataset.itemId]

    def binarize(self, binary_threshold=1):
        """binarize into 0 or 1, imlicit feedback"""

        self._dataset[self._dataset['rating'] > binary_threshold].rating = 1
        self._dataset[self._dataset['rating'] <= binary_threshold].rating = 0

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_item

    def get_original_user_id(self, u):
        if isinstance(u, int):
            return self.original_user_id.loc[u].user_id

        return list(self.original_user_id.loc[u].user_id)

    def get_original_item_id(self, i):
        if isinstance(i, int):
            return self.original_item_id.loc[i].item_id

        return list(self.original_item_id.loc[i].item_id)

    def get_new_user_id(self, u):
        if isinstance(u, int):
            return self.new_user_id.loc[u].userId

        return list(self.new_user_id.loc[u].userId)

    def get_new_item_id(self, i):
        if isinstance(i, int):
            return self.new_item_id.loc[i].itemId

        return list(self.new_item_id.loc[i].itemId)
