import unittest as ut
import inspect
import os
from pprint import pprint
import pathlib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM

working_dir = pathlib.Path(os.getcwd()).parent.parent
os.chdir(working_dir)

from Code.Utils.Src.Utils import Helpers
from Code.Model.Src.Constants import (FM_MOVILENS_SAMPLE_DATA_PATH,
                                      FM_MOVIELENS_COLUMNS,
                                      FM_MOVIELENS_RATING_COLUMN,
                                      )
import Code.DataAccessLayer.Src.MovieLensDataPreProcessing as md
from Code.Model.Src.FactorizationMachineModel import FMModel
from Code.DataAccessLayer.Src.MovielensDataset import MovielenDataset
import Code.Model.Src.Constants as c
import Code.DataAccessLayer.Src.Constants as c2
from Code.Utils.Src.Logging import Logger

class TestFactorizationMachineModel(ut.TestCase):
    """
    Test suit for the "FactorizationMachineModel"
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        self.logger = Logger.getLogger()
        self.logger.info(f"Working folder is: {working_dir}")
        self.data_df = pd.read_csv(FM_MOVILENS_SAMPLE_DATA_PATH)
        self.test_data_fraction = c2.MOVIELENS_TEST_DATA_SPLIT_FRACTION
        self.pre_processor = md.Movielens100KPreprocessor(is_consolidate_genre_values=False)
        self.dataset = MovielenDataset()
        self.dataset.splitDataset(self.test_data_fraction)
        self.model_parameters = {
            c.FM_PARAMS_RANK_KEY: c.FM_PARAMS_RANK_VALUE,
            c.FM_PARAMS_SEED_KEY: c.FM_PARAMS_RANK_VALUE,
            c.FM_PARAM_N_ITER_KEY: c.FM_PARAM_N_ITER_VALUE,
            c.FM_PARAMS_N_KEPT_SAMPLES_KEY: c.FM_PARAMS_N_KEPT_SAMPLES_VALUE
        }


    def test_FactorizationMachineModel_Constructor_Using_OneHotencoder_Is_Valid(self):
        """
        Test the validity of the "FactorizationMachineModel" constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        self.assertIsNotNone(self.dataset.user_item_interaction_modified_df, msg=error_msg)
        X_train, y_train, one_hot_encoder, _ = md.Movielens100KPreprocessor.transformFMModelItemSideDataset(
                                             user_item_interaction_df=self.dataset.train_users_items_df,
                                             item_side_df=self.dataset.items_side_info_modified_df,
                                             is_consolidate_genre_values=False,
                                             is_get_label_from_interaction_data=True,
                                             one_hot_encoder=None
                                            )
        fm_model = FMModel(
            training_data=(X_train, y_train),
            fit_parameters=self.model_parameters)
        self.assertIsNotNone(fm_model, msg=error_msg)
        print("Sample of the dataset:")
        print(Helpers.tableize(self.dataset.train_users_items_df.head(10)))
        print("\n\n")

    def test_FactorizationMachineModel_Constructor_Using_Get_Dummies_Encoding_Is_Valid(self):
        """
        Test the validity of the "FactorizationMachineModel" constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        self.assertIsNotNone(self.dataset.user_item_interaction_modified_df, msg=error_msg)
        feature_columns = None
        X_train, ohe_columns = md.Movielens100KPreprocessor.transformFMModelItemSideDatasetV2(
                                             user_item_interaction_df=self.dataset.train_users_items_df,
                                             item_side_df=self.dataset.items_side_info_modified_df,
                                             ohe_feature_columns=feature_columns,
                                             is_consolidate_genre_values=False
                                             )
        y_train = self.dataset.train_rating
        fm_model = FMModel(
            training_data=(X_train, y_train),
            fit_parameters=self.model_parameters)
        self.assertIsNotNone(fm_model, msg=error_msg)
        print("Sample of the dataset:")
        print(Helpers.tableize(self.dataset.train_users_items_df.head(10)))
        print("\n\n")
        print("One-hot encoding feature columns are:")
        print(f"{ohe_columns}")

    def test_FactorizationMachineModel_Integration_Testing_For_MyFM_Training_Is_Valid(self):
        """
        Test the validity of "FactorizationMachineModel" integration testing
        for 'MyFM' FM model training (with in-sample prediction)
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        perf_results = self._runMyFMTrainingIntegrationTest()
        self.assertIsNotNone(perf_results, msg=error_msg)

    def test_FactorizationMachineModel_Integration_Testing_For_MyFM_Inference_Is_Valid(self):
        """
        Test the validity of "FactorizationMachineModel" integration testing
        for 'MyFM' FM model inference (with out-of-sample test data)
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        perf_results = self._runMyFMInferenceIntegrationTest()
        self.assertIsNotNone(perf_results, msg=error_msg)

    def test_FactorizationMachineModel_Integration_Testing_For_MyFM_Recommendations_For_Sample_Users_Is_Valid(self):
        """
        Test the validity of "FactorizationMachineModel" integration testing for 'MyFM' recommendations for
        sample users
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        user_ids = [x for x in range(10) if x % 2 == 0 and x != 0]
        recommendations = self._runMyFMRecommendationsForSampleUsersIntegrationTest(user_ids=user_ids)
        self.assertIsNotNone(recommendations, msg=error_msg)
        print(f"Test performance metrics for users: {user_ids} are:")
        pprint(recommendations)

    def test_FactorizationMachineModel_Integration_Testing_For_DeepFM_Prediction_Is_Valid(self):
        """
        Test the validity of "FactorizationMachineModel" integration testing for 'DeepFM'
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        model, predicted_ratings = self._runDeepFMPredictionIntegrationTest()
        self.assertIsNotNone(model, msg=error_msg)
        self.assertIsNotNone(predicted_ratings, msg=error_msg)

    def _runMyFMTrainingIntegrationTest(self) -> Dict[str, float]:
        """
        Core logic used to run the "FactorizationMachineModel" integration test
        for 'MyFM' FM model training with in-sample prediction
        :return: Model and the predicted ratings
        """
        ohe_feature_columns = None
        X_all_train, one_hot_columns = md.Movielens100KPreprocessor.transformFMModelItemSideDatasetV2(
            user_item_interaction_df=self.dataset.user_item_interaction_modified_df,
            item_side_df=self.dataset.items_side_info_modified_df,
            ohe_feature_columns=ohe_feature_columns,
            is_consolidate_genre_values=False
        )
        y_train = self.dataset.rating
        X_test = X_all_train[self.dataset.test_split_indices]
        y_test = y_train[self.dataset.test_split_indices]
        fm_model = FMModel(
            training_data=(X_all_train, y_train),
            fit_parameters=self.model_parameters)
        fm_model.train()
        fm_model.one_hot_encoder = None
        fm_model.one_hot_columns = one_hot_columns
        model_output_file = Helpers.getLatestModelOutputFile()
        self.logger.info("Model train is complete")
        self.logger.info(f"Model is serialized on disk in the path: {model_output_file}")
        self.logger.info(f"With one-hot encoding columns: {one_hot_columns}")
        Helpers.writePickleModelDump(fm_model)
        perf_results = self._computeFMPrediction(X_test, y_test, fm_model)
        return perf_results

    def _runMyFMInferenceIntegrationTest(self) -> Dict[str, float]:
        """
        Core logic used to run the "FactorizationMachineModel" integration test
        for 'MyFM' FM model inference (with out-of-sample test data)
        :return: Model and the predicted ratings
        """
        self.logger.info("FM Model inference is starting..")
        fm_model = Helpers.readPickleModelDump()
        one_hot_encoder = fm_model.one_hot_encoder
        one_hot_columns = fm_model.one_hot_columns
        X_test, ohe_feature_columns = md.Movielens100KPreprocessor.transformFMModelItemSideDatasetV2(
            user_item_interaction_df=self.dataset.test_users_items_df,
            item_side_df=self.dataset.items_side_info_modified_df,
            is_consolidate_genre_values=False,
            ohe_feature_columns=one_hot_columns
        )
        y_test = self.dataset.test_rating
        perf_results = self._computeFMPrediction(X_test, y_test, fm_model)
        self.logger.info("FM Model inference is complete..")
        return perf_results

    def _computeFMPrediction(
            self,
            X_test: csr_matrix,
            y_test: np.ndarray,
            fm_model: FMModel
    ) -> Dict[str, float]:
        """
        Computes the FM prediction for in-sample and out-of-sample datasets
        :param X_test: Feature dataset
        :param y_test: Label dataset
        :param fm_model: FM model
        :return:
        """
        X_test_array = X_test.toarray()
        prediction = fm_model.predict(X_test_array)
        rmse = round(((y_test - prediction) ** 2).mean() ** .5, 4)
        mae = np.round(np.abs(y_test - prediction).mean(), 4)
        perf_results = {
            "rmse": rmse,
            "mae": mae
        }
        print("Training performance metrics are:")
        print(f"rmse={rmse}\tmae={mae}")
        return perf_results

    def _runMyFMRecommendationsForSampleUsersIntegrationTest(self, user_ids: List[int]) -> Dict[str, float]:
        """
        Core logic used to run the "FactorizationMachineModel" recommendations integration test for 'MyFM'
        :param user_ids: List of users
        :return: Model and the predicted ratings
        """
        fm_model: FMModel = Helpers.readPickleModelDump()
        recomendations = fm_model.recommend(
            user_ids=user_ids,
            user_item_interaction_df=self.dataset.train_users_items_df,
            items_side_info_df=self.dataset.items_side_info_modified_df
        )
        return recomendations

    def _runDeepFMPredictionIntegrationTest(self) -> Tuple[DeepFM, np.ndarray]:
        """
        Core logic used to run the "FactorizationMachineModel" integration test for 'DeepFM'
        :return: Model and the predicted ratings
        """
        data = self.data_df
        sparse_features = FM_MOVIELENS_COLUMNS
        target = FM_MOVIELENS_RATING_COLUMN

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])

        # 2.count #unique features for each sparse field
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                  for feat in sparse_features]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train, test = train_test_split(data, test_size=self.test_data_fraction)
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
        model.compile("adam", "mse", metrics=['mse'], )
        history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=2,
                            validation_split=0.2)
        pred_ans = model.predict(test_model_input, batch_size=256)
        print("test MSE", round(mean_squared_error(
            test[target].values, pred_ans), 4))
        return model, pred_ans

    def _computeFMTrainingCycle(self, dataset, n_training_test_samples
                                ) -> Dict[str, float]:
        """
        Computes the FM model training cycle
        :param dataset: Preprocessed Movielens dataset
        :param n_training_test_samples: Number of training test samples
        """
        X_train, y_train, one_hot_encoder, ohe_columns = md.Movielens100KPreprocessor.transformFMModelItemSideDataset(
            user_item_interaction_df=dataset.train_users_items_df,
            item_side_df=dataset.items_side_info_df,
            is_consolidate_genre_values=c2.IS_CONSOLIDATE_GENRE_VALUES,
            is_get_label_from_interaction_data=False
        )
        if not y_train:
            y_train = dataset.train_rating
        X_test = X_train[n_training_test_samples].toarray()
        y_test = y_train[n_training_test_samples]
        fm_model = FMModel(
            training_data=(X_train, y_train),
            fit_parameters=self.model_parameters)
        fm_model.train()
        fm_model.one_hot_encoder = one_hot_encoder
        fm_model.one_hot_columns = ohe_columns
        model_output_file = Helpers.getLatestModelOutputFile()
        self.logger.info("Model train is complete")
        self.logger.info(f"Model is serialized on disk in the path: {model_output_file}")
        Helpers.writePickleModelDump(fm_model)
        prediction = fm_model.predict(X_test)
        rmse = round(((y_test - prediction) ** 2).mean() ** .5, 4)
        mae = np.round(np.abs(y_test - prediction).mean(), 4)
        perf_results = {
            "rmse": rmse,
            "mae": mae
        }
        print("Training performance metrics are:")
        print(f"rmse={rmse}\tmae={mae}")
        return perf_results

if __name__ == "__main__":
    ut.main()

