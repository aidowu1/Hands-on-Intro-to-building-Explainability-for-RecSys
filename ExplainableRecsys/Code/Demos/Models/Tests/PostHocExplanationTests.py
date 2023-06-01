import unittest as ut
import inspect
import os
import pathlib
from pprint import pprint

working_dir = pathlib.Path(os.getcwd()).parent.parent.parent
os.chdir(working_dir)

from Code.Demos.Models.Src.PostHocExplanation import PostHocExplainerDemo
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Utils import Helpers
import Code.Model.Src.Constants as c

class PostHocExplainerDemoTest(ut.TestCase):
    """
    Integration test suit for Post hoc recommendation
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")

    def test_PostHocExplainerDemo_Constructor_Is_Valid(self):
        """
        Test validity of 'PostHocExplainerDemo' construction
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        post_hoc_explainer_demo = PostHocExplainerDemo(ExplanationType.posthoc_noexplainer)
        self.assertIsNotNone(post_hoc_explainer_demo, error_msg)

    def test_PostHocExplainerDemo_Process_Dataset_Is_Valid(self):
        """
        Test validity of 'PostHocExplainerDemo' Preprocessing of the Movielens dataset.
        Steps include:
            - Make the data consecutive
            - Split the data into train-test partitions
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        post_hoc_explainer_demo = PostHocExplainerDemo(ExplanationType.posthoc_noexplainer)
        self.assertIsNotNone(post_hoc_explainer_demo, error_msg)
        print("Dataset before making item/user IDs consecutive:")
        print(Helpers.tableize(post_hoc_explainer_demo.metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        train_metadata, test_df = post_hoc_explainer_demo.preprocessMovielensDataset()
        self.assertIsNotNone(train_metadata, error_msg)
        print("Dataset after making item/user IDs consecutive:")
        print(Helpers.tableize(train_metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(test_df, error_msg)
        print(Helpers.tableize(test_df.head(10)))
        Helpers.createConsoleDivider()
        Helpers.writePickleToFile(post_hoc_explainer_demo, c.MF_BEST_TRAIN_RECOMMENDER_PATH)

    def test_PostHocExplainerDemo_Model_Training_Is_Valid(self):
        """
        Test validity of 'PostHocExplainerDemo' MF model training.
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        post_hoc_explainer_demo = Helpers.readPickleFromFile(c.MF_BEST_TRAIN_RECOMMENDER_PATH)
        status = post_hoc_explainer_demo.fit(post_hoc_explainer_demo.train_metadata_clone)
        self.assertTrue(status, msg=error_msg)
        Helpers.writePickleToFile(post_hoc_explainer_demo, c.MF_BEST_TRAIN_RECOMMENDER_PATH)

    def test_PostHocExplainer_Generate_Recommendations_Is_Valid(self):
        """
        Test validity of 'PostHocExplainer' generate recommendations with MF model
        (including recommendation metrics)
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        post_hoc_explainer_demo = Helpers.readPickleFromFile(c.MF_BEST_TRAIN_RECOMMENDER_PATH)
        post_hoc_explainer_demo.computeRecommendations(
            post_hoc_explainer_demo.model,
            post_hoc_explainer_demo.train_metadata_clone,
            post_hoc_explainer_demo.test_data_clone_df)
        Helpers.writePickleToFile(post_hoc_explainer_demo, c.MF_BEST_TRAIN_RECOMMENDER_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(
            post_hoc_explainer_demo.recommendations_df,
            c.MF_BEST_RECOMMENDATIONS_PATH)
        self.assertIsNotNone(post_hoc_explainer_demo.recommendations_df, error_msg)
        print("Sample of generated recommendations:")
        print(Helpers.tableize(post_hoc_explainer_demo.recommendations_df.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(post_hoc_explainer_demo.recommendation_metrics, error_msg)
        print("Recommendation metrics:")
        pprint(post_hoc_explainer_demo.recommendation_metrics)

    def test_PostHocExplainer_Integration_Based_On_Cached_Models_Is_Valid(self):
        """
        Test validity of 'PostHocExplainer' full integration of ALS model steps
        - based on cached/serialized recommendation and explanation models
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        mf_best_recommender = Helpers.readPickleFromFile(c.MF_BEST_TRAIN_RECOMMENDER_PATH)
        self.assertIsNotNone(mf_best_recommender.recommendations_df, error_msg)
        self.assertIsNotNone(mf_best_recommender.recommendation_metrics, error_msg)
        mf_best_explanations_df = None
        mf_best_explanation_metrics = None
        ModelBasedExplainerDemo.reportDemoResults(
            mf_best_explanation_metrics,
            mf_best_explanations_df,
            mf_best_recommender.recommendation_metrics,
            mf_best_recommender.recommendations_df)
