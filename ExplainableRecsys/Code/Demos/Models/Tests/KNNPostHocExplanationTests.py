import unittest as ut
import inspect
import os
import pathlib
from pprint import pprint

working_dir = pathlib.Path(os.getcwd()).parent.parent.parent
os.chdir(working_dir)

from Code.Demos.Models.Src.KNNPostHocExplanation import KNNPostHocExplainerDemo
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Utils import Helpers
import Code.Model.Src.Constants as c

class KNNPostHocExplainerDemoTest(ut.TestCase):
    """
    Integration test suit for AR Post hoc recommendation
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")

    def test_KNNPostHocExplainerDemo_Constructor_Is_Valid(self):
        """
        Test validity of 'KNNPostHocExplainerDemo' construction
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        knn_post_hoc_explainer_demo = KNNPostHocExplainerDemo(ExplanationType.knn_explainer)
        self.assertIsNotNone(knn_post_hoc_explainer_demo, error_msg)

    def test_KNNPostHocExplainerDemo_Process_Dataset_Is_Valid(self):
        """
        Test validity of 'KNNPostHocExplainerDemo' Preprocessing of the Movielens dataset.
        Steps include:
            - Make the data consecutive
            - Split the data into train-test partitions
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        knn_post_hoc_explainer_demo = KNNPostHocExplainerDemo(ExplanationType.knn_explainer)
        self.assertIsNotNone(knn_post_hoc_explainer_demo, error_msg)
        print("Dataset before making item/user IDs consecutive:")
        print(Helpers.tableize(knn_post_hoc_explainer_demo.metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        train_metadata, test_df = knn_post_hoc_explainer_demo.preprocessMovielensDataset()
        self.assertIsNotNone(train_metadata, error_msg)
        print("Dataset after making item/user IDs consecutive:")
        print(Helpers.tableize(train_metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(test_df, error_msg)
        print(Helpers.tableize(test_df.head(10)))
        Helpers.createConsoleDivider()
        Helpers.writePickleToFile(knn_post_hoc_explainer_demo, c.MF_BEST_KNN_TRAIN_RECOMMENDER_PATH)

    def test_KNNPostHocExplainerDemo_Model_Training_Is_Valid(self):
        """
        Test validity of 'KNNPostHocExplainerDemo' MF model training.
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        knn_post_hoc_explainer_demo = Helpers.readPickleFromFile(c.MF_BEST_KNN_TRAIN_RECOMMENDER_PATH)
        status = knn_post_hoc_explainer_demo.fit(knn_post_hoc_explainer_demo.train_metadata_clone)
        self.assertTrue(status, msg=error_msg)
        Helpers.writePickleToFile(knn_post_hoc_explainer_demo, c.MF_BEST_KNN_TRAIN_RECOMMENDER_PATH)

    def test_KNNPostHocExplainer_Generate_Recommendations_Is_Valid(self):
        """
        Test validity of 'KNNPostHocExplainer' generate recommendations with MF model
        (including recommendation metrics)
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        knn_post_hoc_explainer_demo = Helpers.readPickleFromFile(c.MF_BEST_KNN_TRAIN_RECOMMENDER_PATH)
        knn_post_hoc_explainer_demo.computeRecommendations(
            knn_post_hoc_explainer_demo.model,
            knn_post_hoc_explainer_demo.train_metadata_clone,
            knn_post_hoc_explainer_demo.test_data_clone_df)
        Helpers.writePickleToFile(knn_post_hoc_explainer_demo, c.MF_BEST_KNN_TRAIN_RECOMMENDER_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(
            knn_post_hoc_explainer_demo.recommendations_df,
            c.MF_BEST_KNN_RECOMMENDATIONS_PATH)
        self.assertIsNotNone(knn_post_hoc_explainer_demo.recommendations_df, error_msg)
        print("Sample of generated recommendations:")
        print(Helpers.tableize(knn_post_hoc_explainer_demo.recommendations_df.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(knn_post_hoc_explainer_demo.recommendation_metrics, error_msg)
        print("Recommendation metrics:")
        pprint(knn_post_hoc_explainer_demo.recommendation_metrics)

    def test_KNNPostHocExplainer_Generate_Explanations_Is_Valid(self):
        """
        Test validity of 'KNNPostHocExplainer' generate explanations with AR post hoc model
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        knn_post_hoc_explainer_demo = Helpers.readPickleFromFile(c.MF_BEST_KNN_TRAIN_RECOMMENDER_PATH)
        knn_post_hoc_explainer_demo.computeExplanations(
            knn_post_hoc_explainer_demo.train_metadata_clone,
            ExplanationType.knn_explainer)
        self.assertIsNotNone(knn_post_hoc_explainer_demo.explanations_df, error_msg)
        self.assertIsNotNone(knn_post_hoc_explainer_demo.explanation_metrics, error_msg)
        print("Sample of generated explanations:")
        print(Helpers.tableize(knn_post_hoc_explainer_demo.explanations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Explanation metrics:")
        pprint(knn_post_hoc_explainer_demo.explanation_metrics)
        Helpers.writePickleToFile(knn_post_hoc_explainer_demo, c.MF_BEST_KNN_EXPLANATION_MODEL_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(
            knn_post_hoc_explainer_demo.explanations_df,
            c.MF_BEST_KNN_EXPLANATIONS_PATH)

    def test_KNNPostHocExplainer_Integration_Based_On_Cached_Models_Is_Valid(self):
        """
        Test validity of 'PostHocExplainer' full integration of ALS model steps
        - based on cached/serialized recommendation and explanation models
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        knn_best_recommender = Helpers.readPickleFromFile(c.MF_BEST_KNN_TRAIN_RECOMMENDER_PATH)
        self.assertIsNotNone(knn_best_recommender.recommendations_df, error_msg)
        self.assertIsNotNone(knn_best_recommender.recommendation_metrics, error_msg)

        knn_best_explainer = Helpers.readPickleFromFile(c.MF_BEST_KNN_EXPLANATION_MODEL_PATH)
        self.assertIsNotNone(knn_best_explainer.explanations_df, error_msg)
        self.assertIsNotNone(knn_best_explainer.explanation_metrics, error_msg)

        ModelBasedExplainerDemo.reportDemoResults(
            knn_best_explainer.explanation_metrics,
            knn_best_explainer.explanations_df,
            knn_best_recommender.recommendation_metrics,
            knn_best_recommender.recommendations_df)
