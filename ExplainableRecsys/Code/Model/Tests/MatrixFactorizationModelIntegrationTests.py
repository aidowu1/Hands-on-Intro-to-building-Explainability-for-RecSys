import pprint
import unittest as ut
import inspect
import os
import pathlib
import pandas as pd
from typing import Optional, Tuple, Any, Dict
from copy import deepcopy

working_dir = pathlib.Path(os.getcwd()).parent.parent
os.chdir(working_dir)

from Code.ThirdParty.recoxplainer_master.recoxplainer.config import cfg
from Code.ThirdParty.recoxplainer_master.recoxplainer.recommender import Recommender
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import Evaluator
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.post_hoc_association_rules import ARPostHocExplainer
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.post_hoc_knn import KNNPostHocExplainer
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import ExplanationEvaluator
from Code.Model.Src.MatrixFactorizationModel import MFModel
from Code.ThirdParty.recoxplainer_master.recoxplainer.data_reader import data_reader
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import splitter
from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.DataAccessLayer.Src.DataSplitter import Splitter
import Code.Model.Src.Constants as c
from Code.Utils.Src.Utils import Helpers
class TestMatrixFactorizationModel(ut.TestCase):
    """
    Integration test suit for custom Matrix Factorization (MF) Model
    This component tests the following features:
        - Construction of the MF Model i.e. that is used to compute the recommendations
        - Training of the MF Model
        - Recommendation of items i.e. ranking using the trained MF model
        - Evaluation of the performance metrics of the MF Model using:
            - Hit ratio
            - Normalized Discounted Cumulative Gain (NDCG)
        - Post-hoc explanation of the MF recommendations using AR explainer (with performance metrics)
        - Post-hoc explanation of the MF recommendations using KNN explainer (with performance metrics)
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")
        self.metadata = DataReader(**cfg.ml100k)
        self.learning_rate = 0.01
        self.weight_decay = 0.001
        self.latent_dim = 100
        self.epochs = 50
        self.batch_size = 128
        self.split_fraction = 0.1

    def test_MFModel_Constructor_Is_Valid(self):
        """
        Test the validity of MFModel constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        model = MFModel(learning_rate=self.learning_rate,
                        weight_decay=self.weight_decay,
                        latent_dim=self.latent_dim,
                        epochs=self.epochs,
                        batch_size=self.batch_size
                        )
        self.assertIsNotNone(model, msg=error_msg)

    def test_MFModel_Training_Is_Valid(self):
        """
        Test the validity of MFModel training
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        model = MFModel(learning_rate=self.learning_rate,
                        weight_decay=self.weight_decay,
                        latent_dim=self.latent_dim,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        split_fraction=self.split_fraction
                        )
        self.assertIsNotNone(model, msg=error_msg)
        self.metadata.makeConsecutiveIdsInDataset()
        splitter = Splitter()
        train_metadata, test_df = splitter.splitLeaveNOut(self.metadata, frac=self.split_fraction)
        status = model.fit(train_metadata)
        self.assertTrue(status, msg=error_msg)
        Helpers.writePickleModelDump(model, c.MF_MODEL_CACHE_PATH)
        model_output_file = Helpers.getLatestModelOutputFile(c.MF_MODEL_CACHE_PATH)
        if os.path.exists(model_output_file):
            cached_model = Helpers.readPickleFromFile(model_output_file)
            self.assertIsNotNone(cached_model, msg=error_msg)

    def test_MFModel_Compute_Recommendations_Is_Valid(self):
        """
        Test the validity of MFModel computed recommendations
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        cached_model = self.getLatestMFRecommendationModelFromCache()
        train_metadata = cached_model.dataset_metadata
        self.assertIsNotNone(cached_model, msg=error_msg)
        recommender = Recommender(dataset_metadata=train_metadata, model=cached_model)
        self.assertIsNotNone(recommender, msg=error_msg)
        recommendations_df = recommender.recommend_all()
        self.assertIsNotNone(recommendations_df, msg=error_msg)
        print("Sample recommendations are:")
        print(Helpers.tableize(recommendations_df.head()))
        self.writeRecommendationsToCache(recommendations_df)

    def test_MFModel_Compute_Recommendations_Evaluations_Is_Valid(self):
        """
        Test the validity of MFModel computed recommendation evaluations
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        cached_model = self.getLatestMFRecommendationModelFromCache()
        train_metadata = cached_model.dataset_metadata
        _, test_df = self.getTrainTestDataset(train_metadata)
        recommendations_df = self.readRecommendationsFromCache()
        metrics = self.evaluateRecommender(
            recommendations_df=recommendations_df,
            test_df=test_df
        )
        self.assertIsNotNone(metrics, msg=error_msg)
        print(f"Recommendation metrics for test data:")
        pprint.pprint(metrics)

    def test_MFModel_Compute_AR_Explainability_Is_Valid(self):
        """
        Test the validity of MFModel with AR explainability
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        cached_model = self.getLatestMFRecommendationModelFromCache()
        train_metadata = cached_model.dataset_metadata
        _, test_df = self.getTrainTestDataset(train_metadata)
        recommendations_df = self.readRecommendationsFromCache()
        explanations_df, metrics = self.computeARExplanation(cached_model, recommendations_df, train_metadata)
        self.assertIsNotNone(explanations_df, msg=error_msg)
        self.assertTrue(explanations_df.shape[0] > 0, msg=error_msg)
        print("Sample recommendation explanations are:")
        print(Helpers.tableize(explanations_df.head()))
        print(f"AR Post-hoc Explanation performance metrics are:")
        pprint.pprint(metrics)


    def test_MFModel_Compute_KNN_Explainability_Is_Valid(self):
        """
        Test the validity of MFModel with KNN explainability
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        cached_model = self.getLatestMFRecommendationModelFromCache()
        train_metadata = cached_model.dataset_metadata
        _, test_df = self.getTrainTestDataset(train_metadata)
        recommendations_df = self.readRecommendationsFromCache()
        explanations_df, metrics = self.computeKNNExplanation(cached_model, recommendations_df, train_metadata)
        self.assertIsNotNone(explanations_df, msg=error_msg)
        self.assertTrue(explanations_df.shape[0] > 0, msg=error_msg)
        print("Sample recommendation explanations are:")
        print(Helpers.tableize(explanations_df.head()))
        print(f"KNN Post-hoc Explanation performance metrics are:")
        pprint.pprint(metrics)

    def test_MFModel_Integration_Test_Is_Valid(self):
        """
        Test the validity of MFModel training
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        cached_model = self.getLatestMFRecommendationModelFromCache()
        train_metadata = cached_model.dataset_metadata
        _, test_df = self.getTrainTestDataset(train_metadata)
        recommendations_df = self.readRecommendationsFromCache()
        recommendation_metrics = self.evaluateRecommender(
            recommendations_df=recommendations_df,
            test_df=test_df
        )
        self.assertIsNotNone(recommendation_metrics, msg=error_msg)
        print(f"Recommendation metrics for test data:")
        pprint.pprint(recommendation_metrics)
        self.runARExplainationTest(cached_model, error_msg, recommendations_df, train_metadata)
        self.runKNNExplainationTest(cached_model, error_msg, recommendations_df, train_metadata)

    def runKNNExplainationTest(
            self,
            cached_model: MFModel,
            error_msg: str,
            recommendations_df: pd.DataFrame,
            train_metadata: DataReader):
        """
        Executes the KNN Explanation tests
        :param cached_model: Recsys Model
        :param error_msg: Error message
        :param recommendations_df: Computed recommendations
        :param train_metadata: Training metadata
        """
        explanations_df, knn_explain_metrics = self.computeKNNExplanation(cached_model, recommendations_df,
                                                                          train_metadata)
        self.assertIsNotNone(explanations_df, msg=error_msg)
        self.assertTrue(explanations_df.shape[0] > 0, msg=error_msg)
        print("Sample recommendation explanations are:")
        print(Helpers.tableize(explanations_df.head()))
        print(f"KNN Post-hoc Explanation performance metrics are:")
        pprint.pprint(knn_explain_metrics)

    def runARExplainationTest(
            self,
            cached_model: MFModel,
            error_msg: str,
            recommendations_df: pd.DataFrame,
            train_metadata: DataReader):
        """
        Executes the KNN Explanation tests
        :param cached_model: Recsys Model
        :param error_msg: Error message
        :param recommendations_df: Computed recommendations
        :param train_metadata: Training metadata
        """
        explanations_df, ar_explain_metrics = self.computeARExplanation(cached_model, recommendations_df,
                                                                        train_metadata)
        self.assertIsNotNone(explanations_df, msg=error_msg)
        self.assertTrue(explanations_df.shape[0] > 0, msg=error_msg)
        print("Sample recommendation explanations are:")
        print(Helpers.tableize(explanations_df.head()))
        print(f"AR Explanation performance metrics are:")
        pprint.pprint(ar_explain_metrics)

    def writeRecommendationsToCache(self, recommendations_df: pd.DataFrame):
        """
        Writes (caches) recommendations to disc
        :param recommendations_df: Recommendations
        """
        if not os.path.exists(c.MF_RECOMMENDATIONS_PATH):
            os.makedirs(c.MF_RECOMMENDATIONS_PATH)
        recommendations_path = f"{c.MF_RECOMMENDATIONS_PATH}/{c.MF_TRAIN_RECOMMENDATIONS_FILENAME}"
        recommendations_df.to_csv(recommendations_path, index=False)

    def readRecommendationsFromCache(self) -> Optional[pd.DataFrame]:
        """
        Reads recommendations from disc (cache)
        :return: Recommendations
        """
        if os.path.exists(c.MF_RECOMMENDATIONS_PATH):
            recommendations_path = f"{c.MF_RECOMMENDATIONS_PATH}/{c.MF_TRAIN_RECOMMENDATIONS_FILENAME}"
            return pd.read_csv(recommendations_path)
        else:
            return None

    def evaluateRecommender(self, recommendations_df, test_df):
        """
        Evaluate the performance of the trained recommender
        :param recommendations_df: Recommendations
        :param test_df: Test dataset
        :return: Evaluation metric
        """
        eval = Evaluator(test_df)
        hit_ratio = eval.cal_hit_ratio(recommendations_df)
        ndcg = eval.cal_ndcg(recommendations_df)
        metrics = {
            "hit_ratio": hit_ratio,
            "ndcg": ndcg
        }
        return metrics

    def computeARExplanation(
            self,
            recommendation_model: MFModel,
            recommendations: pd.DataFrame,
            train_dataset: DataReader
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the Association Rules explanability of the recommendation model
        :param recommendation_model: Recommendation model
        :param recommendations: Computed recommendations
        :param train_dataset: Training dataset
        :return: Explanations
        """
        ar_explainer = ARPostHocExplainer(recommendation_model, recommendations, train_dataset)
        return self.computePostHocExplainer(ar_explainer, train_dataset)

    def computeKNNExplanation(
            self,
            recommendation_model: MFModel,
            recommendations: pd.DataFrame,
            train_dataset: DataReader,
            nearest_k: int = 10
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the K-Nearest Neighbors explanability of the recommendation model
        :param recommendation_model: Recommendation model
        :param recommendations: Computed recommendations
        :param train_dataset: Training dataset
        :param nearest_k: Number of k nearest neighbours
        :return: Explanations
        """
        knn_explainer = KNNPostHocExplainer(recommendation_model, recommendations, train_dataset, nearest_k)
        return self.computePostHocExplainer(knn_explainer, train_dataset)

    def computePostHocExplainer(self, knn_explainer, train_dataset):
        """
        Computes post-hoc explainers of the recommendation model
        :param knn_explainer: Explainer
        :param train_dataset: Training dataset
        :return: Explanations
        """
        explanations_df = knn_explainer.explain_recommendations()
        explanations_df["explanations_as_list"] = explanations_df.explanations.apply(lambda x: list(x))
        explanations_df["n_explanations"] = explanations_df["explanations_as_list"].apply(
            lambda x: len(x))
        filter_1 = explanations_df["n_explanations"] > 0
        explainer_evaluator = ExplanationEvaluator(train_dataset.num_user)
        explanation_metrics = {
            "fidelity": explainer_evaluator.model_fidelity(explanations_df)
        }
        return explanations_df[filter_1], explanation_metrics

    def getTrainTestDataset(self, metadata: DataReader) -> Tuple[DataReader, pd.DataFrame]:
        """
        Gets the train-test split datasets
        """
        metadata.makeConsecutiveIdsInDataset()
        splitter = Splitter()
        train, test_df = splitter.splitLeaveNOut(metadata, frac=self.split_fraction)
        return train, test_df

    def getLatestMFRecommendationModelFromCache(self) -> Any:
        """
        Gets the latest MF model from the cache
        :return: Cached model
        """
        model_output_file = Helpers.getLatestModelOutputFile(c.MF_MODEL_CACHE_PATH)
        cached_model = None
        if os.path.exists(model_output_file):
            print(f"Current model is in the cache file: {model_output_file}")
            cached_model = Helpers.readPickleFromFile(model_output_file)
        return cached_model


if __name__ == '__main__':
    ut.main()
