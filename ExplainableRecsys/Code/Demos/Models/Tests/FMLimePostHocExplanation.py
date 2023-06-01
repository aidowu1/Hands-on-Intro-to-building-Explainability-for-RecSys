import unittest as ut
import inspect
import os
import pathlib
from pprint import pprint

working_dir = pathlib.Path(os.getcwd()).parent.parent.parent
os.chdir(working_dir)

from Code.Demos.Models.Src.FMLimePostHocExplanation import FMLimePostHocExplainerDemo
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Utils import Helpers
import Code.Model.Src.Constants as c

class FMLimePostHocExplainerDemoTest(ut.TestCase):
    """
    Integration test suit for AR Post hoc recommendation
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")

    def test_FMLimePostHocExplainerDemo_Constructor_Is_Valid(self):
        """
        Test validity of 'FMLimePostHocExplainerDemo' construction
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        fm_lime_post_hoc_explainer_demo = FMLimePostHocExplainerDemo(ExplanationType.fmlime_explainer)
        self.assertIsNotNone(fm_lime_post_hoc_explainer_demo, error_msg)

    def test_FMLimePostHocExplainerDemo_Process_Dataset_Is_Valid(self):
        """
        Test validity of 'FMLimePostHocExplainerDemo' Preprocessing of the Movielens dataset.
        Steps include:
            - Make the data consecutive
            - Split the data into train-test partitions
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        fm_lime_post_hoc_explainer_demo = FMLimePostHocExplainerDemo(ExplanationType.fmlime_explainer)
        self.assertIsNotNone(fm_lime_post_hoc_explainer_demo, error_msg)
        train_metadata, test_df = fm_lime_post_hoc_explainer_demo.preprocessMovielensDataset()
        self.assertIsNotNone(train_metadata, error_msg)
        print("Dataset after making item/user IDs consecutive:")
        print(Helpers.tableize(train_metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(test_df, error_msg)
        print("Sample Test data:")
        print(Helpers.tableize(test_df.head(10)))
        Helpers.createConsoleDivider()
        Helpers.writePickleToFile(fm_lime_post_hoc_explainer_demo, c.FM_BEST_TRAIN_RECOMMENDER_PATH)

    def test_FMLimePostHocExplainerDemo_Model_Training_Is_Valid(self):
        """
        Test validity of 'FMLimePostHocExplainerDemo' MF model training.
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        fm_lime_post_hoc_explainer_demo = Helpers.readPickleFromFile(c.FM_BEST_TRAIN_RECOMMENDER_PATH)
        status = fm_lime_post_hoc_explainer_demo.fit(fm_lime_post_hoc_explainer_demo.train_metadata_clone)
        self.assertTrue(status, msg=error_msg)
        Helpers.writePickleToFile(fm_lime_post_hoc_explainer_demo, c.FM_BEST_TRAIN_RECOMMENDER_PATH)

    def test_FMLimePostHocExplainer_Generate_Recommendations_Is_Valid(self):
        """
        Test validity of 'FMLimePostHocExplainer' generate recommendations with MF model
        (including recommendation metrics)
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        fm_lime_post_hoc_explainer_demo = Helpers.readPickleFromFile(c.FM_BEST_TRAIN_RECOMMENDER_PATH)
        fm_lime_post_hoc_explainer_demo.computeRecommendations(
            fm_lime_post_hoc_explainer_demo.model,
            fm_lime_post_hoc_explainer_demo.train_metadata_clone,
            fm_lime_post_hoc_explainer_demo.test_data_clone_df)
        Helpers.writePickleToFile(fm_lime_post_hoc_explainer_demo, c.FM_BEST_TRAIN_RECOMMENDER_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(
            fm_lime_post_hoc_explainer_demo.recommendations_df,
            c.FM_BEST_RECOMMENDATIONS_PATH)
        self.assertIsNotNone(fm_lime_post_hoc_explainer_demo.recommendations_df, error_msg)
        print("Sample of generated recommendations:")
        print(Helpers.tableize(fm_lime_post_hoc_explainer_demo.recommendations_df.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(fm_lime_post_hoc_explainer_demo.recommendation_metrics, error_msg)
        print("Recommendation metrics:")
        pprint(fm_lime_post_hoc_explainer_demo.recommendation_metrics)

    def test_FMLimePostHocExplainer_Generate_Explanations_Is_Valid(self):
        """
        Test validity of 'FMLimePostHocExplainer' generate explanations with AR post hoc model
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        fm_lime_post_hoc_explainer_demo = Helpers.readPickleFromFile(c.FM_BEST_TRAIN_RECOMMENDER_PATH)
        fm_lime_post_hoc_explainer_demo.computeExplanations(
            fm_lime_post_hoc_explainer_demo.train_metadata_clone,
            ExplanationType.fmlime_explainer)
        self.assertIsNotNone(fm_lime_post_hoc_explainer_demo.explanations_df, error_msg)
        self.assertIsNotNone(fm_lime_post_hoc_explainer_demo.explanation_metrics, error_msg)
        print("Sample of generated explanations:")
        print(Helpers.tableize(fm_lime_post_hoc_explainer_demo.explanations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Explanation metrics:")
        pprint(fm_lime_post_hoc_explainer_demo.explanation_metrics)
        Helpers.writePickleToFile(fm_lime_post_hoc_explainer_demo, c.FM_BEST_LIME_EXPLANATION_MODEL_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(
            fm_lime_post_hoc_explainer_demo.explanations_df,
            c.FM_BEST_EXPLANATIONS_PATH)

    def test_FMLimePostHocExplainer_Integration_Based_On_Cached_Models_Is_Valid(self):
        """
        Test validity of 'PostHocExplainer' full integration of FM-Lime model steps
        - based on cached/serialized recommendation and explanation models
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        fm_best_recommender = Helpers.readPickleFromFile(c.FM_BEST_TRAIN_RECOMMENDER_PATH)
        self.assertIsNotNone(fm_best_recommender.recommendations_df, error_msg)
        self.assertIsNotNone(fm_best_recommender.recommendation_metrics, error_msg)

        fm_best_explainer = Helpers.readPickleFromFile(c.FM_BEST_LIME_EXPLANATION_MODEL_PATH)
        self.assertIsNotNone(fm_best_explainer.explanations_df, error_msg)
        self.assertIsNotNone(fm_best_explainer.explanation_metrics, error_msg)

        ModelBasedExplainerDemo.reportDemoResults(
            fm_best_explainer.explanation_metrics,
            fm_best_explainer.explanations_df,
            fm_best_recommender.recommendation_metrics,
            fm_best_recommender.recommendations_df)
