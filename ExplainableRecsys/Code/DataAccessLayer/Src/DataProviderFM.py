import pandas as pd
import numpy as np

from Code.DataAccessLayer.Src.MovielensDataset import MovielenDataset
import Code.DataAccessLayer.Src.Constants as c

class DataReader(object):
    """
    DataReader component (proxy object) of the DataReader used in the RecoXplainer library
    """
    def __init__(
            self,
            movielens_metadata: MovielenDataset
    ):
        """
        Constructor
        :param movielens_metadata: Full Movielens metadata
        """
        self.dataset_df = movielens_metadata.user_item_interaction_modified_df.copy()
        self.dataset_df[c.MOVIELENS_RATING_COLUMN_KEY] = movielens_metadata.rating.tolist()
        self.dataset = self.dataset_df.copy()
        self.test_df = movielens_metadata.test_users_items_df
        self.train_df = movielens_metadata.train_users_items_df
        self.test_df[c.MOVIELENS_RATING_COLUMN_KEY] = movielens_metadata.test_rating.tolist()
        self.train_df[c.MOVIELENS_RATING_COLUMN_KEY] = movielens_metadata.train_rating.tolist()
        self.num_item = int(self.dataset_df[[c.MOVIELENS_ITEM_ID_COLUMN_KEY]].nunique())
        self.num_user = int(self.dataset_df[[c.MOVIELENS_USER_ID_COLUMN_KEY]].nunique())

