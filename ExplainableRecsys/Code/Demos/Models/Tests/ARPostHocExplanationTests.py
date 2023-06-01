import unittest as ut
import inspect
import os
import pathlib
from pprint import pprint

working_dir = pathlib.Path(os.getcwd()).parent.parent.parent
os.chdir(working_dir)

from Code.Demos.Models.Src.ARPostHocExplanation import ARPostHocExplainerDemo
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Utils import Helpers
import Code.Model.Src.Constants as c

class ARPostHocExplainerDemoTest(ut.TestCase):
    """
    Integration test suit for AR Post hoc recommendation
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")

    def test_ARPostHocExplainerDemo_Constructor_Is_Valid(self):
        """
        Test validity of 'ARPostHocExplainerDemo' construction
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ar_post_hoc_explainer_demo = ARPostHocExplainerDemo(ExplanationType.ar_explainer)
        self.assertIsNotNone(ar_post_hoc_explainer_demo, error_msg)

    def test_ARPostHocExplainerDemo_Process_Dataset_Is_Valid(self):
        """
        Test validity of 'ARPostHocExplainerDemo' Preprocessing of the Movielens dataset.
        Steps include:
            - Make the data consecutive
            - Split the data into train-test partitions
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ar_post_hoc_explainer_demo = ARPostHocExplainerDemo(ExplanationType.ar_explainer)
        self.assertIsNotNone(ar_post_hoc_explainer_demo, error_msg)
        print("Dataset before making item/user IDs consecutive:")
        print(Helpers.tableize(ar_post_hoc_explainer_demo.metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        train_metadata, test_df = ar_post_hoc_explainer_demo.preprocessMovielensDataset()
        self.assertIsNotNone(train_metadata, error_msg)
        print("Dataset after making item/user IDs consecutive:")
        print(Helpers.tableize(train_metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(test_df, error_msg)
        print(Helpers.tableize(test_df.head(10)))
        Helpers.createConsoleDivider()
        Helpers.writePickleToFile(ar_post_hoc_explainer_demo, c.MF_BEST_AR_TRAIN_RECOMMENDER_PATH)

    def test_ARPostHocExplainerDemo_Model_Training_Is_Valid(self):
        """
        Test validity of 'ARPostHocExplainerDemo' MF model training.
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ar_post_hoc_explainer_demo = Helpers.readPickleFromFile(c.MF_BEST_AR_TRAIN_RECOMMENDER_PATH)
        status = ar_post_hoc_explainer_demo.fit(ar_post_hoc_explainer_demo.train_metadata_clone)
        self.assertTrue(status, msg=error_msg)
        Helpers.writePickleToFile(ar_post_hoc_explainer_demo, c.MF_BEST_AR_TRAIN_RECOMMENDER_PATH)

    def test_ARPostHocExplainer_Generate_Recommendations_Is_Valid(self):
        """
        Test validity of 'ARPostHocExplainer' generate recommendations with MF model
        (including recommendation metrics)
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ar_post_hoc_explainer_demo = Helpers.readPickleFromFile(c.MF_BEST_AR_TRAIN_RECOMMENDER_PATH)
        ar_post_hoc_explainer_demo.computeRecommendations(
            ar_post_hoc_explainer_demo.model,
            ar_post_hoc_explainer_demo.train_metadata_clone,
            ar_post_hoc_explainer_demo.test_data_clone_df)
        Helpers.writePickleToFile(ar_post_hoc_explainer_demo, c.MF_BEST_AR_TRAIN_RECOMMENDER_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(
            ar_post_hoc_explainer_demo.recommendations_df,
            c.MF_BEST_AR_RECOMMENDATIONS_PATH)
        self.assertIsNotNone(ar_post_hoc_explainer_demo.recommendations_df, error_msg)
        print("Sample of generated recommendations:")
        print(Helpers.tableize(ar_post_hoc_explainer_demo.recommendations_df.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(ar_post_hoc_explainer_demo.recommendation_metrics, error_msg)
        print("Recommendation metrics:")
        pprint(ar_post_hoc_explainer_demo.recommendation_metrics)

    def test_ARPostHocExplainer_Generate_Explanations_Is_Valid(self):
        """
        Test validity of 'ARPostHocExplainer' generate explanations with AR post hoc model
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ar_post_hoc_explainer_demo = Helpers.readPickleFromFile(c.MF_BEST_AR_TRAIN_RECOMMENDER_PATH)
        ar_post_hoc_explainer_demo.computeExplanations(
            ar_post_hoc_explainer_demo.train_metadata_clone,
            ExplanationType.ar_explainer)
        self.assertIsNotNone(ar_post_hoc_explainer_demo.explanations_df, error_msg)
        self.assertIsNotNone(ar_post_hoc_explainer_demo.explanation_metrics, error_msg)
        print("Sample of generated explanations:")
        print(Helpers.tableize(ar_post_hoc_explainer_demo.explanations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Explanation metrics:")
        pprint(ar_post_hoc_explainer_demo.explanation_metrics)
        Helpers.writePickleToFile(ar_post_hoc_explainer_demo, c.MF_BEST_AR_EXPLANATION_MODEL_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(
            ar_post_hoc_explainer_demo.explanations_df,
            c.MF_BEST_AR_EXPLANATIONS_PATH)

    def test_ARPostHocExplainer_Integration_Based_On_Cached_Models_Is_Valid(self):
        """
        Test validity of 'PostHocExplainer' full integration of ALS model steps
        - based on cached/serialized recommendation and explanation models
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        ar_best_recommender = Helpers.readPickleFromFile(c.MF_BEST_AR_TRAIN_RECOMMENDER_PATH)
        self.assertIsNotNone(ar_best_recommender.recommendations_df, error_msg)
        self.assertIsNotNone(ar_best_recommender.recommendation_metrics, error_msg)

        ar_best_explainer = Helpers.readPickleFromFile(c.MF_BEST_AR_EXPLANATION_MODEL_PATH)
        self.assertIsNotNone(ar_best_explainer.explanations_df, error_msg)
        self.assertIsNotNone(ar_best_explainer.explanation_metrics, error_msg)

        ModelBasedExplainerDemo.reportDemoResults(
            ar_best_explainer.explanation_metrics,
            ar_best_explainer.explanations_df,
            ar_best_recommender.recommendation_metrics,
            ar_best_recommender.recommendations_df)
