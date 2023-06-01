import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from typing import List, Tuple, Dict, Any, Optional
from scipy.sparse.csr import csr_matrix

import Code.DataAccessLayer.Src.Constants as c


class Movielens100KPreprocessor(object):
    """
    Component used to pre-process Movielens 100K dataset
    """
    def __init__(self, is_consolidate_genre_values : bool = True):
        """
        Constructor
        :param: is_consolidate_genre_values: Flag to indicate the consolidation of genre values
        """
        self.is_consolidate_genre_values = is_consolidate_genre_values
        self.rating_df = pd.read_csv(c.MOVIELENS_RATING_PATH,
                                sep=c.MOVIELENS_RATING_FILE_DELIMITER,
                                header=None)
        self.rating_df.columns = c.MOVIELENS_RATING_COLUMNS
        self.user_df = pd.read_csv(c.MOVIELENS_USER_PATH,
                    sep=c.MOVIELENS_USER_FILE_DELIMITER,
                    encoding=c.MOVIELENS_USERS_FILE_ENCODING,
                    header=None)
        self.user_df.columns = c.MOVIELENS_USER_COLUMNS
        self.item_df = pd.read_csv(c.MOVIELENS_ITEM_PATH,
                    sep=c.MOVIELENS_USER_FILE_DELIMITER,
                    encoding=c.MOVIELENS_USERS_FILE_ENCODING,
                    header=None)
        self.item_df.columns = c.MOVIELENS_ITEM_COLUMNS
        self.genres_df = pd.read_csv(c.MOVIELENS_GENRE_PATH,
                                sep=c.MOVIELENS_USER_FILE_DELIMITER,
                                header=None,
                                usecols=[c.MOVIELENS_GENRE_USE_COLUMN_INDEX]
                                )[c.MOVIELENS_GENRE_USE_COLUMN_INDEX].tolist()

    def joinRatingUserItemDatasets(self,
                                   user_item_interaction_df : pd.DataFrame) -> pd.DataFrame:
        """
        Consolidates the Rating, User (with it attributes) and Item (with its attributes) datasets
        The steps for doing the join will include:
            - Step 1: Read the Item data into an in-memory dataframe
            - Step 2: Create the Rating/User dataframe by joining Rating dataframe to User dataframe using the UserId foreign key
            - Step 3: Join Rating/User dataframe to Item dataframe using the ItemId foreign key

        :param user_item_interaction_df: user-item interaction data
        :return: Consolidated Rating, User and Item datasets
        """
        item_df = self.preProcessItemGenre()
        rating_user_df = user_item_interaction_df.merge(self.user_df,
                                              how="right",
                                              on=c.MOVIELENS_USER_ID_COLUMN_KEY)
        rating_user_item_df = rating_user_df.merge(item_df,
                                                   how="right",
                                                   on=c.MOVIELENS_ITEM_ID_COLUMN_KEY
                                                   )
        return rating_user_item_df

    def preProcessItemGenre(self) -> pd.DataFrame:
        """
        Pre-processes the Item genre values
        :return: Pre-processed item genre values
        """
        def consolidateGenreValues(row: pd.Series) -> str:
            """
            Consolidates the item genre values
            :param row: Table row
            :return: genre values
            """
            genre_values = ""
            for genre in c.MOVIELENS_GENRE_COLUMNS:
                if row[genre] == 1:
                    genre_values += f"{genre}|"
            genre_values = genre_values.rstrip(c.MOVIELENS_USER_FILE_DELIMITER)
            return genre_values

        item_df = self.item_df.copy()
        if self.is_consolidate_genre_values:
            item_df["genre"] = item_df.apply(lambda x: consolidateGenreValues(x), axis=1)
            item_columns = list(set(list(item_df.columns)).difference(set(c.MOVIELENS_GENRE_COLUMNS)))
            item_df = item_df[item_columns]
        return item_df

    def transformFMModelTrainData(self,
                                  user_item_interaction_df : pd.DataFrame
                                  ) -> Tuple[csr_matrix, np.ndarray, OneHotEncoder]:
        """
        Transforms the Movielens training data for FM model
        :param user_item_interaction_df: User-item interaction dataset
        :return: FM model feature and label data
        """
        rating_user_item_df = self.joinRatingUserItemDatasets(
            user_item_interaction_df=user_item_interaction_df
        )
        if self.is_consolidate_genre_values:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_2
        else:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_3
        X, y, ohe, ohe_columns = self.computeOneHotEncoding(rating_user_item_df, feature_columns)
        return X, y, ohe

    @staticmethod
    def transformFMModelTestData(
                                one_hot_encoder: Optional[OneHotEncoder],
                                problem_dataset_df: pd.DataFrame,
                                user_item_interaction_df: pd.DataFrame,
                                is_consolidate_genre_values: bool = False
                                ) -> Tuple[csr_matrix, np.ndarray]:
        """
        Transforms the Movielens training data for FM model
        :param one_hot_encoder: One-hot encoder
        :param problem_dataset_df: Consolidated problem super dataset
        :param user_item_interaction_df: User-item interaction dataset (pertaining to the current user_id)
        :param is_consolidate_genre_values: Flag to indicate the consolidation of genre values
        :return: FM model feature and label data
        """
        test_dataset_df = problem_dataset_df.drop(labels=c.MOVIELENS_USER_ID_COLUMN_KEY, axis=1)
        test_dataset_df.drop_duplicates(c.MOVIELENS_ITEM_ID_COLUMN_KEY, inplace=True)
        rating_user_item_df = pd.merge(
            user_item_interaction_df,
            test_dataset_df,
            on=c.MOVIELENS_ITEM_ID_COLUMN_KEY,
            left_index=True,
            how="left"
        )
        if is_consolidate_genre_values:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_2
        else:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_3
        X, y, ohe, ohe_columns = Movielens100KPreprocessor.computeOneHotEncoding(rating_user_item_df,
                                                                    feature_columns,
                                                                    one_hot_encoder=one_hot_encoder)
        return X, y

    @staticmethod
    def transformFMModelItemSideDataset(
                                  user_item_interaction_df: pd.DataFrame,
                                  item_side_df: pd.DataFrame,
                                  is_consolidate_genre_values: bool,
                                  is_get_label_from_interaction_data: bool = False,
                                  one_hot_encoder: OneHotEncoder = None
                                  ) -> Tuple[csr_matrix, np.ndarray, OneHotEncoder, List[str]]:
        """
        Transforms the Movielens training data for FM model
        :param user_item_interaction_df: User-item interaction dataset
        :param item_side_df:Item side dataset
        :param is_consolidate_genre_values: Flag to indicate the consolidation of genre values
        :param is_get_label_from_interaction_data: Flag to indicate if the label should be extracted
               from the interaction dataset
        :param one_hot_encoder: One-hot encoder
        :return: FM model feature and label data
        """
        user_item_with_side_info_df = user_item_interaction_df.merge(
            item_side_df,
            how="left",
            on=c.MOVIELENS_ITEM_ID_COLUMN_KEY)

        if is_consolidate_genre_values:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_2
        else:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_3
        X, y, ohe, ohe_columns = Movielens100KPreprocessor.computeOneHotEncoding(
            user_item_with_side_info_df,
            feature_columns,
            one_hot_encoder=one_hot_encoder,
            is_get_label=is_get_label_from_interaction_data
        )
        return X, y, ohe, ohe_columns

    @staticmethod
    def transformFMModelItemSideDatasetV2(
            user_item_interaction_df: pd.DataFrame,
            item_side_df: pd.DataFrame,
            ohe_feature_columns: Optional[List[str]],
            is_consolidate_genre_values: bool
    ) -> Tuple[csr_matrix, List[str]]:
        """
        Transforms the Movielens training data for FM model
        :param user_item_interaction_df: User-item interaction dataset
        :param item_side_df:Item side dataset
        :param ohe_feature_columns: Feature columns
        :param is_consolidate_genre_values: Flag to indicate the consolidation of genre values
        :return: FM model feature and label data
        """
        user_item_with_side_info_df = user_item_interaction_df.merge(
            item_side_df,
            how="left",
            on=c.MOVIELENS_ITEM_ID_COLUMN_KEY)
        user_item_with_side_info_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY] = \
            user_item_with_side_info_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY].astype(str)

        user_item_with_side_info_df[c.MOVIELENS_USER_ID_COLUMN_KEY] = \
            user_item_with_side_info_df[c.MOVIELENS_USER_ID_COLUMN_KEY].astype(str)

        if is_consolidate_genre_values:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_2
        else:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_3
        new_user_item_with_side_info_df = user_item_with_side_info_df[feature_columns]
        X, ohe_columns = Movielens100KPreprocessor.convertToPyFmFormat(
            new_user_item_with_side_info_df,
            ohe_feature_columns)
        return X, ohe_columns

    @staticmethod
    def computeOneHotEncoding(
                              interaction_data_with_side_info_df: pd.DataFrame,
                              feature_columns: List[str],
                              one_hot_encoder: OneHotEncoder = None,
                              is_get_label: bool = True,
                              is_consolidate_genre_values: bool = False
    ) -> Tuple[csr_matrix, Optional[np.ndarray], OneHotEncoder, List[str]]:
        """
        Computes the One-hot-encoding transformation of the interaction data
        :param interaction_data_with_side_info_df: interation matrix with side info
        :param feature_columns: Feature columns
        :param one_hot_encoder: One-hot encoder
        :param is_get_label: Flag to indicate the return of the label values of the dataset
        :param is_consolidate_genre_values: Flag to indicate the consolidation of genre values
        :return: Encoded dataset and encoder model
        """
        dataset = interaction_data_with_side_info_df[feature_columns]
        X, one_hot_encoder = Movielens100KPreprocessor.calculateOneHotEncoding(dataset,
                                                                               one_hot_encoder)
        ohe_columns = Movielens100KPreprocessor.getOneHotEncodingColumns(one_hot_encoder, is_consolidate_genre_values)
        if is_get_label:
            y = interaction_data_with_side_info_df[c.MOVIELENS_RATING_COLUMN_KEY]
            return X, y, one_hot_encoder, ohe_columns
        else:
            return X, None, one_hot_encoder, ohe_columns

    @staticmethod
    def calculateOneHotEncoding(
            dataset,
            one_hot_encoder: OneHotEncoder = None
    ) -> Tuple[csr_matrix, OneHotEncoder]:
        """
        Calculate One-hot encoding
        :param dataset: Dataset
        :param one_hot_encoder: One-hot encoder
        :return:
        """
        if one_hot_encoder:
            X = one_hot_encoder.transform(dataset)
        else:
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
            X = one_hot_encoder.fit_transform(dataset)
        return X, one_hot_encoder

    @staticmethod
    def convertToPyFmFormat(df: pd.DataFrame, columns: List[str]=None
                            ) -> Tuple[csr_matrix, List[str]]:
        """
        Converts the data to FM sparse format
        :param df: Dataset
        :param columns: Columns
        """
        df_ohe = pd.get_dummies(df)
        if columns is not None:
            df_ohe = df_ohe.reindex(columns=columns)
            df_ohe = df_ohe.fillna(0)
        data_sparse = csr_matrix(df_ohe.astype(np.float64))
        data_sparse = data_sparse.astype(np.float64)
        return data_sparse, list(df_ohe.columns)

    @staticmethod
    def getOneHotEncodingColumns(
            one_hot_encoder: OneHotEncoder,
            is_consolidate_genre_values: bool = False
    ) -> List[str]:
        """
        Gets the one-hot encoder columns
        :param: is_consolidate_genre_values: Flag to indicate the return of the label values of the dataset
        :return: List of the one-hot-encoding columns
        """
        if is_consolidate_genre_values:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_2
        else:
            feature_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_3
        one_hot_encoder_columns = one_hot_encoder.get_feature_names(feature_columns)
        return one_hot_encoder_columns

    @staticmethod
    def correctFeatureColumns(
            feature_columns: List[str]
    ) -> List[str]:
        """
        Corrects the list of required feature columns for doing one-hot encoding i.e. it
        ensures that the list include the item and user id columns
        :param feature_columns: Feature columns
        :return: Corrected list of features
        """
        user_item_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_0
        is_user_item_cols_exist = set(user_item_columns).issubset(set(feature_columns))
        new_feature_columns = [s for s in feature_columns]
        if not is_user_item_cols_exist:
            new_feature_columns = list(set(feature_columns + user_item_columns))
        return new_feature_columns




