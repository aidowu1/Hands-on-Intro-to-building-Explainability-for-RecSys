import unittest as ut
from inspect import stack
import os
import pathlib
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Any, Tuple

working_dir = pathlib.Path(os.getcwd()).parent.parent
os.chdir(working_dir)

import Code.DataAccessLayer.Src.MovieLensDataPreProcessing as md
from Code.Model.Src.FactorizationMachineModel import FMModel
from Code.Utils.Src.Utils import Helpers
import Code.DataAccessLayer.Src.Constants as c

class TestMovielens100KPreprocessor(ut.TestCase):
    """
    Test suit for the "Movielens100KPreprocessor"
    """
    def setUp(self) -> None:
        """
        Setup fixture for the unit testing
        """
        print(f"Working folder is: {working_dir}")
        self.pre_processor = md.Movielens100KPreprocessor(is_consolidate_genre_values=True)

    def tearDown(self) -> None:
        """
        Teardown fixture for the unit test
        """
        pass

    def test_Movielens100KPreprocessor_Constructor_Is_Valid(self):
        """
        Test the validity of "Movielens100KPreprocessor" constructor
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        self.assertIsNotNone(self.pre_processor, msg=error_msg)
        print("Sample of the rating dataset:")
        print(Helpers.tableize(self.pre_processor.rating_df.head(10)))
        print(f"\n\nSample of the user dataset:")
        print(Helpers.tableize(self.pre_processor.user_df.head(10)))
        print(f"\n\nSample of the item dataset:")
        print(Helpers.tableize(self.pre_processor.item_df.head(10)))


    # def test_Movielens100KPreprocessor_Consolidate_Dataset_Is_Valid(self):
    #     """
    #     Test the validity of the "Movielens100KPreprocessor" consolidated feature-aware dataset
    #     """
    #     error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
    #     self.assertIsNotNone(self.pre_processor, msg=error_msg)
    #     consolidated_dataset_df = self.pre_processor.createFeatureAwareDataset()
    #     print(f"\n\nSample of the consolidated Movielens dataset:")
    #     print(Helpers.tableize(consolidated_dataset_df.head(10)))

    def test_Movielens100KPreprocessor_Item_Genre_Is_Valid(self):
        """
        Test the validity of the "Movielens100KPreprocessor" consolidated feature-aware dataset
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        item_df = self.pre_processor.preProcessItemGenre()
        self.assertIsNotNone(item_df, msg=error_msg)
        print(f"\n\nSample of the Movielens items dataset (with consolidated genre):")
        print(Helpers.tableize(item_df.head(5)))

    def test_Movielens100KPreprocessor_Join_Rating_User_And_Item_Data_Is_Valid(self):
        """
        Test the validity of join Movielens Rating, User and Item datasets.
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        self.user_item_interaction_df = self.pre_processor.rating_df
        rating_user_item_df = self.pre_processor.joinRatingUserItemDatasets(self.user_item_interaction_df)
        self.assertIsNotNone(rating_user_item_df, msg=error_msg)
        print(f"\n\nSample of the Movielens items dataset (with consolidated genre):")
        print(Helpers.tableize(rating_user_item_df.head(5)))

    def test_Movielens100KPreprocessor_Feature_OnehotEncoding_With_Consolidated_Genre_Is_Valid(self):
        """
        Test the validity of join Movielens datasets OHE (with consolidated genre data)
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        self.user_item_interaction_df = self.pre_processor.rating_df
        X_train, y_train, one_hot_encoder = self.pre_processor.transformFMModelTrainData(
            user_item_interaction_df=self.user_item_interaction_df)
        self.assertIsNotNone(X_train, msg=error_msg)
        self.assertIsNotNone(y_train, msg=error_msg)
        self.assertIsNotNone(one_hot_encoder, msg=error_msg)
        X_train_matrix = X_train.toarray()
        cols = [f"{x}" for x in range(X_train_matrix.shape[1])]
        X_train_matrix_df = pd.DataFrame(data=X_train_matrix, columns=cols)

        print(f"Sample OHE feature matrix:")
        print(f"{Helpers.tableize(X_train_matrix_df)}")

    def test_Movielens100KPreprocessor_Feature_OnehotEncoding_Of_Train_Without_Consolidated_Genre_Is_Valid(self):
        """
        Test the validity of test Movielens datasets OHE (without consolidated genre data)
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        self.pre_processor = md.Movielens100KPreprocessor(is_consolidate_genre_values=False)
        self.user_item_interaction_df = self.pre_processor.rating_df
        X_train, y_train, one_hot_encoder = self.pre_processor.transformFMModelTrainData(
            user_item_interaction_df=self.user_item_interaction_df)
        X_train_matrix = X_train.toarray()
        cols = [f"{x}" for x in range(X_train_matrix.shape[1])]
        X_train_matrix_df = pd.DataFrame(data=X_train_matrix, columns=cols)
        self.assertIsNotNone(X_train, msg=error_msg)
        print(f"Sample OHE feature matrix:")
        print(f"{Helpers.tableize(X_train_matrix_df)}")

    def test_Movielens100KPreprocessor_Feature_OnehotEncoding_Of_Test_Without_Consolidated_Genre_Is_Valid(self):
        """
        Test the validity of test Movielens datasets OHE (with consolidated genre data)
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        self.pre_processor = md.Movielens100KPreprocessor(is_consolidate_genre_values=False)
        X_test, _ = self._getTestData()
        X_train_matrix = X_test.toarray()
        cols = [f"{x}" for x in range(X_train_matrix.shape[1])]
        X_train_matrix_df = pd.DataFrame(data=X_train_matrix, columns=cols)
        self.assertIsNotNone(X_test, msg=error_msg)
        print(f"Sample OHE feature matrix:")
        print(f"{Helpers.tableize(X_train_matrix_df)}")

    def test_Movielens100KPreprocessor_Convert_Dataset_To_PyFm_Format_Is_Valid(self):
        """
        Test the validity of converting the user-item interaction dataset to sparse mYFM format
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        n_samples = 200
        user_item_interaction_df = self.pre_processor.rating_df[c.MOVIELENS_USER_ITEM_INTERACTION_COLUMNS].head(n_samples)
        user_item_interaction_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY] = user_item_interaction_df[
            c.MOVIELENS_ITEM_ID_COLUMN_KEY].astype(str)
        user_item_interaction_df[c.MOVIELENS_USER_ID_COLUMN_KEY] = user_item_interaction_df[
            c.MOVIELENS_USER_ID_COLUMN_KEY].astype(str)
        self.assertIsNotNone(user_item_interaction_df, msg=error_msg)
        user_item_interaction_ohe = self.pre_processor.convertToPyFmFormat(user_item_interaction_df)
        self.assertIsNotNone(user_item_interaction_ohe)

    def test_Movielens100KPreprocessor_OneHotEncoding_Of_Dataset_Is_Valid(self):
        """
        Test the validity of converting the user-item interaction dataset to sparse mYFM format
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        n_samples = 200
        user_item_interaction_train_df = self.pre_processor.rating_df[
            c.MOVIELENS_USER_ITEM_INTERACTION_COLUMNS].head(n_samples)
        user_item_interaction_test_df = self.pre_processor.rating_df[
            c.MOVIELENS_USER_ITEM_INTERACTION_COLUMNS].tail(n_samples)
        self.assertIsNotNone(user_item_interaction_train_df, msg=error_msg)
        self.assertIsNotNone(user_item_interaction_test_df, msg=error_msg)

        user_item_interaction_train_ohe, one_hot_encoder = self.pre_processor.calculateOneHotEncoding(
            user_item_interaction_train_df)
        self.assertIsNotNone(user_item_interaction_train_ohe)

        user_item_interaction_test_ohe, _ = self.pre_processor.calculateOneHotEncoding(
            user_item_interaction_test_df, one_hot_encoder)
        self.assertIsNotNone(user_item_interaction_test_ohe)

    def _getTestData(self, user_id: int = 10) -> Tuple[csr_matrix, np.ndarray]:
        """
        Gets the FM model test data for a specified user ID
        :param user_id: User ID
        :return: FM model feature and label test data
        """
        one_hot_encoder = None
        all_user_item_interaction_df = self.pre_processor.rating_df
        problem_dataset_df = self.pre_processor.joinRatingUserItemDatasets(all_user_item_interaction_df)
        sample_user_item_interaction_df = FMModel.getMovielensUserItems(user_id, all_user_item_interaction_df)
        X_test, y_test = self.pre_processor.transformFMModelTestData(
            one_hot_encoder=one_hot_encoder,
            problem_dataset_df=problem_dataset_df,
            user_item_interaction_df=sample_user_item_interaction_df,
            is_consolidate_genre_values=self.pre_processor.is_consolidate_genre_values
        )
        return X_test, y_test

    def _trainFMModelOnMovielensDataset(self, X_train, y_train) -> Dict[str, Any]:
        """
        Trains the Movielens data on FM model
        :param X_train: Training feature dataset
        :param y_train: Labels
        :return: Results of the training
        """
        # fm = myfm.MyFMRegressor(rank=FM_RANK, random_seed=42)
        # fm.fit(X_train, y_train, n_iter=200, n_kept_samples=200)
        # prediction = fm.predict(X_test)
        # rmse = ((y_test - prediction) ** 2).mean() ** .5
        # mae = np.abs(y_test - prediction).mean()
        # print(f'rmse={rmse}, mae={mae}')
        pass


if __name__ == '__main__':
    ut.main()

