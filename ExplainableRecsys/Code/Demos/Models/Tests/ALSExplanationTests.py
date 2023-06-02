import unittest as ut
import inspect
import os
import pathlib
from pprint import pprint
import pandas as pd
from typing import Optional, Tuple, Any, Dict
from copy import deepcopy
import random

working_dir = pathlib.Path(os.getcwd()).parent.parent.parent
os.chdir(working_dir)

from Code.Demos.Models.Src.ALSExplanation import ALSExplainerDemo
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Utils import Helpers
import Code.Model.Src.Constants as c

class ALSExplainerDemoTest(ut.TestCase):
    """
    Integration test suit for 'ALS' model-based recommendation and explanations
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")

    def test_ALSExplainerDemo_Constructor_Is_Valid(self):
        """
        Test validity of 'ALSExplainerDemo' construction
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        als_explainer_demo = ALSExplainerDemo(ExplanationType.als_explainer)
        self.assertIsNotNone(als_explainer_demo, error_msg)

    def test_ALSExplainerDemo_Process_Dataset_Is_Valid(self):
        """
        Test validity of 'ALSExplainerDemo' Preprocessing of the Movielens dataset.
        Steps include:
            - Make the data consecutive
            - Binarise the data i.e. these models use implicit feedback
            - Split the data into train-test partitions
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        als_explainer_demo = ALSExplainerDemo(ExplanationType.als_explainer)
        self.assertIsNotNone(als_explainer_demo, error_msg)
        print("Dataset before making item/user IDs consecutive:")
        print(Helpers.tableize(als_explainer_demo.metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        train_metadata, test_df = als_explainer_demo.preprocessMovielensDataset()
        self.assertIsNotNone(train_metadata, error_msg)
        print("Dataset after making item/user IDs consecutive:")
        print(Helpers.tableize(train_metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(test_df, error_msg)
        print(Helpers.tableize(test_df.head(10)))
        Helpers.createConsoleDivider()
        Helpers.writePickleToFile(als_explainer_demo, c.ALS_BEST_TRAIN_RECOMMENDER_PATH)

    def test_ALSExplainerDemo_Model_Training_Is_Valid(self):
        """
        Test validity of 'ALSExplainerDemo' ALS model training.
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        als_explainer_demo = Helpers.readPickleFromFile(c.ALS_BEST_TRAIN_RECOMMENDER_PATH)
        status = als_explainer_demo.fit(als_explainer_demo.train_metadata_clone)
        self.assertTrue(status, msg=error_msg)
        Helpers.writePickleToFile(als_explainer_demo, c.ALS_BEST_TRAIN_RECOMMENDER_PATH)

    def test_ALSExplainerDemo_Generate_Recommendations_Is_Valid(self):
        """
        Test validity of 'ALSExplainerDemo' generate recommendations with ALS model
        (including recommendation metrics)
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        als_explainer_demo = Helpers.readPickleFromFile(c.ALS_BEST_TRAIN_RECOMMENDER_PATH)
        als_explainer_demo.computeRecommendations(
            als_explainer_demo.model,
            als_explainer_demo.train_metadata_clone,
            als_explainer_demo.test_data_clone_df)
        Helpers.writePickleToFile(als_explainer_demo, c.ALS_BEST_TRAIN_RECOMMENDER_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(als_explainer_demo.recommendations_df, c.ALS_BEST_RECOMMENDATIONS_PATH)
        self.assertIsNotNone(als_explainer_demo.recommendations_df, error_msg)
        print("Sample of generated recommendations:")
        print(Helpers.tableize(als_explainer_demo.recommendations_df.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(als_explainer_demo.recommendation_metrics, error_msg)
        print("Recommendation metrics:")
        pprint(als_explainer_demo.recommendation_metrics)

    def test_ALSExplainerDemo_Generate_Explanations_Is_Valid(self):
        """
        Test validity of 'ALSExplainerDemo' generate explanations with ALS model
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        als_explainer_demo = Helpers.readPickleFromFile(c.ALS_BEST_TRAIN_RECOMMENDER_PATH)
        als_explainer_demo.computeExplanations(
            als_explainer_demo.train_metadata_clone,
            ExplanationType.als_explainer)
        self.assertIsNotNone(als_explainer_demo.explanations_df, error_msg)
        self.assertIsNotNone(als_explainer_demo.explanation_metrics, error_msg)
        print("Sample of generated explanations:")
        print(Helpers.tableize(als_explainer_demo.explanations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Explanation metrics:")
        pprint(als_explainer_demo.explanation_metrics)
        Helpers.writePickleToFile(als_explainer_demo, c.ALS_BEST_EXPLANATION_MODEL_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(als_explainer_demo.explanations_df, c.ALS_BEST_EXPLANATIONS_PATH)

    def test_ALSExplainerDemo_Integration_Based_On_Cached_Models_Is_Valid(self):
        """
        Test validity of 'ALSExplainerDemo' full integration of ALS model steps
        - based on cached/serialized recommendation and explanation models
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        als_best_recommender = Helpers.readPickleFromFile(c.ALS_BEST_TRAIN_RECOMMENDER_PATH)
        self.assertIsNotNone(als_best_recommender.recommendations_df, error_msg)
        self.assertIsNotNone(als_best_recommender.recommendation_metrics, error_msg)

        als_best_explainer = Helpers.readPickleFromFile(c.ALS_BEST_EXPLANATION_MODEL_PATH)
        best_explanations_df = Helpers.prettifyExplanations(als_best_explainer.explanations_df)

        self.assertIsNotNone(als_best_explainer.explanations_df, error_msg)
        self.assertIsNotNone(als_best_explainer.explanation_metrics, error_msg)
        ModelBasedExplainerDemo.reportDemoResults(
            als_best_explainer.explanation_metrics,
            best_explanations_df,
            als_best_recommender.recommendation_metrics,
            als_best_recommender.recommendations_df)

    def test_ALSExplainerDemo_Full_Integration_Is_Valid(self):
        """
        Test validity of 'ALSExplainerDemo' full integration of ALS model steps
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        als_explainer_demo = ALSExplainerDemo(ExplanationType.als_explainer)
        self.assertIsNotNone(als_explainer_demo, error_msg)
        results = als_explainer_demo.runDemo(n_recommendation_explanations=c.N_RECOMMENDATION_SAMPLES_FOR_EXPLANATION)
        self.assertIsNotNone(results, error_msg)
        recommendations_df = results["recommendations_df"]
        recommendation_metrics = results["recommendation_metrics"]
        explanations_df = results["explanations_df"]
        explanation_metrics = results["explanation_metrics"]
        ModelBasedExplainerDemo.reportDemoResults(
            explanation_metrics,
            explanations_df,
            recommendation_metrics,
            recommendations_df)

    def reportDemoResults(
            self,
            explanation_metrics,
            explanations_df,
            recommendation_metrics,
            recommendations_df):
        """
        Reports the demo results
        :param explanation_metrics: Explanation metrics
        :param explanations_df: Explanations
        :param recommendation_metrics: Recommendation metrics
        :param recommendations_df: recommendations
        """
        print("Sample of generated recommendations:")
        print(Helpers.tableize(recommendations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Recommendation metrics:")
        pprint(recommendation_metrics)
        Helpers.createConsoleDivider()
        print("Sample of generated explanations:")
        print(Helpers.tableize(explanations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Explanation metrics:")
        pprint(explanation_metrics)
        Helpers.createConsoleDivider()


if __name__ == '__main__':
    ut.main()
