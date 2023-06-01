import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from typing import List, Tuple, Any, Dict
from tqdm.auto import tqdm
from myfm import MyFMRegressor

from Code.DataAccessLayer.Src.Constants import (MOVIELENS_ITEM_ID_COLUMN_KEY, MOVIELENS_USER_ID_COLUMN_KEY)
import Code.Model.Src.Constants as c
import Code.DataAccessLayer.Src.Constants as c2
import Code.DataAccessLayer.Src.MovieLensDataPreProcessing as md
from Code.DataAccessLayer.Src.MovielensDataset import MovielenDataset
from Code.Utils.Src.Logging import Logger




class Model(ABC):
    def __init__(self,
                 training_data: Tuple[csr_matrix, np.ndarray],
                 fit_parameters: Dict[str, Any]):
        """
        Factorization Machine constructor
        :param training_data: Training data i.e. a sequence features and labels
        :param fit_parameters: Fit parameters
        """
        self.X_train = training_data[0]
        self.y_train = training_data[1]
        self.fit_parameters = fit_parameters
        self.logger = Logger.getLogger()

    @abstractmethod
    def train(self):
        """
        Invokes the FM model training
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, df):
        """
        Invokes the FM prediction
        """
        raise NotImplementedError

    @abstractmethod
    def recommend(self,
                  user_id: int,
                  user_item_interaction_df: pd.DataFrame,
                  items_side_info_df: pd.DataFrame,
                  top_k: int=10,
                  is_filter_interacted_items: bool = True
                  ):
        """
        Invokes the FM the top K recommendations for a specific user
        :param user_id: User ID
        :param user_item_interaction_df: User-item interaction dataset
        :param items_side_info_df: Item side info dataset
        :param top_k: Top K number of recommendations
        :param is_filter_interacted_items: Flag to indicate the filtering of the historical data
        :return: Top k recommendations
        """
        raise NotImplementedError

    @staticmethod
    def getMovielensUserItems(
                               user_id: int,
                               sample_user_item_interaction_df: pd.DataFrame,
                               is_filter_interacted_items: bool = False
                            ) -> pd.DataFrame:
        """
        Gets the candidates required for recommendation
        :param user_id: User ID
        :param sample_user_item_interaction_df: User-item interaction dataset
        :param is_filter_interacted_items: Flag to indicate the filtering of user interacted items
        :return: Returns items interacted by a specified user
        """
        all_items = set(sample_user_item_interaction_df[MOVIELENS_ITEM_ID_COLUMN_KEY].values.tolist())
        filter_1 = sample_user_item_interaction_df[MOVIELENS_USER_ID_COLUMN_KEY] == user_id
        filter_results_df = sample_user_item_interaction_df[filter_1]
        user_items = set(filter_results_df[MOVIELENS_ITEM_ID_COLUMN_KEY].values.tolist())
        candidate_items = all_items
        if is_filter_interacted_items:
            candidate_items = all_items - user_items
        n_candidate_items = len(candidate_items)
        sample_user_item_interaction_df = pd.DataFrame()
        sample_user_item_interaction_df[MOVIELENS_ITEM_ID_COLUMN_KEY] = list(candidate_items)
        sample_user_item_interaction_df[MOVIELENS_USER_ID_COLUMN_KEY] = [user_id] * n_candidate_items
        return sample_user_item_interaction_df


class FMModel(Model):
    def __init__(self,
                 training_data: Tuple[csr_matrix, np.ndarray],
                 fit_parameters: Dict[str, Any]):
        """
        Factorization Machine constructor
        :param training_data: Training data
        :param fit_parameters: Fit parameters
        """
        super().__init__(training_data, fit_parameters)
        self._one_hot_encoder = None
        self._one_hot_columns = None
        self.fm = MyFMRegressor(
            rank=self.fit_parameters[c.FM_PARAMS_RANK_KEY],
            random_seed=self.fit_parameters[c.FM_PARAMS_SEED_KEY])

    def train(self):
        """
        Trains the FM model
        """
        self.fm.fit(
            self.X_train,
            self.y_train,
            n_iter=self.fit_parameters[c.FM_PARAM_N_ITER_KEY],
            n_kept_samples=self.fit_parameters[c.FM_PARAMS_N_KEPT_SAMPLES_KEY]
        )

    def predict(self, X_test: csr_matrix) -> np.ndarray:
        """
        Uses the FM model to make rating (label) predictions
        :param X_test: Test feature dataset
        :return: The predicted labels (ratings)
        """
        all_predictions = []
        # Partition into chunks to avoid memory errors
        partitions = np.array_split(X_test, c.FM_PREDICT_PARTITION_SIZE)
        for partition in partitions:
            # get predictions
            preds = self.fm.predict(partition)
            all_predictions.extend(preds.round(3))
        all_preditions_array = np.asarray(all_predictions)
        return all_preditions_array

    def recommend(self,
                  user_ids: List[int],
                  user_item_interaction_df: pd.DataFrame,
                  items_side_info_df: pd.DataFrame,
                  top_k: int = 10,
                  is_filter_interacted_items: bool = False
                  ):
        """
        Logic used to make k-top recommendations based on the predicted ratings
        :param user_ids: User IDs
        :param user_item_interaction_df: User-item interaction dataset
        :param items_side_info_df: Item side info dataset
        :param top_k: Top K number of recommendations
        :param is_filter_interacted_items: Flag to indicate the filtering of user interacted items
        :return: return: Recommended items per user
        """
        self.logger.info(f"Making recommendations for users: {user_ids}")
        df_list = list()
        with tqdm(total=len(user_ids)) as progress:
            for user_id in user_ids:
                sample_user_item_interaction_df = self.getMovielensUserItems(
                    user_id,
                    user_item_interaction_df,
                    is_filter_interacted_items=is_filter_interacted_items)
                X_train, ohe_feature_columns = md.Movielens100KPreprocessor.transformFMModelItemSideDatasetV2(
                    user_item_interaction_df=sample_user_item_interaction_df,
                    item_side_df=items_side_info_df,
                    is_consolidate_genre_values=False,
                    ohe_feature_columns=self._one_hot_columns
                )
                X_array_train = X_train.toarray()
                predictions = self.predict(X_array_train)
                sample_user_item_interaction_df[c2.MOVIELENS_RATING_PREDICTION_COLUMN_KEY] = predictions

                df_list.append(sample_user_item_interaction_df)
                progress.update(1)
                progress.set_postfix({"Current user ID": user_id})
        recommendation_df = pd.concat(df_list)

        # Sort by uid and predictions
        # result.sort_values(by=[c2.MOVIELENS_USER_ID_COLUMN_KEY
        #     , c2.MOVIELENS_RATING_PREDICTION_COLUMN_KEY], inplace=True, ascending=[True, False])
        # return result.groupby(c2.MOVIELENS_USER_ID_COLUMN_KEY).head(top_k)
        ranked_recommendations = self.rankRecommendations(recommendation_df, top_k)
        return ranked_recommendations

    def rankRecommendations(
            self,
            recommendations_df: pd.DataFrame,
            top_k: int) -> pd.DataFrame:
        """
        Ranks the recommendations in descending order based on the computed rating
        :param recommendations_df: Input recommendations
        :return: Ranked recommendations
        """
        ranked_recommendations_df = recommendations_df.copy()
        ranked_recommendations_df[c2.MOVIELENS_RANK_COLUMN_KEY] = (recommendations_df.
            groupby(c2.MOVIELENS_USER_ID_COLUMN_KEY)
            [c2.MOVIELENS_RATING_PREDICTION_COLUMN_KEY]).rank(method='first', ascending=False)
        ranked_recommendations_df.sort_values([c2.MOVIELENS_USER_ID_COLUMN_KEY,
                                               c2.MOVIELENS_RANK_COLUMN_KEY], inplace=True)

        filter_1 = ranked_recommendations_df[c2.MOVIELENS_RANK_COLUMN_KEY] <= float(top_k)
        ranked_recommendations_df = ranked_recommendations_df[filter_1]
        ranked_recommendations_df = ranked_recommendations_df[c2.MOVIELENS_RECOMMENDATION_COLUMNS]
        return ranked_recommendations_df


    @property
    def one_hot_encoder(self):
        """
        One-hot encoder getter property
        """
        return self._one_hot_encoder

    @one_hot_encoder.setter
    def one_hot_encoder(self, value):
        """
        One-hot setter property
        """
        self._one_hot_encoder = value

    @property
    def one_hot_columns(self):
        """
        One-hot columns getter property
        """
        return self._one_hot_columns

    @one_hot_columns.setter
    def one_hot_columns(self, value):
        """
        One-hot columns setter property
        """
        self._one_hot_columns = value
