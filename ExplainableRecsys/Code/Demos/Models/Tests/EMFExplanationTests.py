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

from Code.Demos.Models.Src.EMFExplanation import EMFExplainerDemo
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Utils import Helpers
import Code.Model.Src.Constants as c

class EMFExplainerDemoTest(ut.TestCase):
    """
    Integration test suit for 'EMF' model-based recommendation and explanations
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")

    def test_EMFExplainerDemo_Constructor_Is_Valid(self):
        """
        Test validity of 'EMFExplainerDemo' construction
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        emf_explainer_demo = EMFExplainerDemo(ExplanationType.emf_explainer)
        self.assertIsNotNone(emf_explainer_demo, error_msg)

    def test_EMFExplainerDemo_Process_Dataset_Is_Valid(self):
        """
        Test validity of 'EMFExplainerDemo' Preprocessing of the Movielens dataset.
        Steps include:
            - Make the data consecutive
            - Binarise the data i.e. these models use implicit feedback
            - Split the data into train-test partitions
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        emf_explainer_demo = EMFExplainerDemo(ExplanationType.emf_explainer)
        self.assertIsNotNone(emf_explainer_demo, error_msg)
        print("Dataset before making item/user IDs consecutive:")
        print(Helpers.tableize(emf_explainer_demo.metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        train_metadata, test_df = emf_explainer_demo.preprocessMovielensDataset()
        self.assertIsNotNone(train_metadata, error_msg)
        print("Dataset after making item/user IDs consecutive:")
        print(Helpers.tableize(train_metadata.dataset.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(test_df, error_msg)
        print(Helpers.tableize(test_df.head(10)))
        Helpers.createConsoleDivider()
        Helpers.writePickleToFile(emf_explainer_demo, c.EMF_BEST_TRAIN_RECOMMENDER_PATH)

    def test_EMFExplainerDemo_Model_Training_Is_Valid(self):
        """
        Test validity of 'EMFExplainerDemo' EMF model training.
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        emf_explainer_demo = Helpers.readPickleFromFile(c.EMF_BEST_TRAIN_RECOMMENDER_PATH)
        self.assertIsNotNone(emf_explainer_demo, error_msg)
        status = emf_explainer_demo.fit(emf_explainer_demo.train_metadata_clone)
        self.assertTrue(status, msg=error_msg)
        Helpers.writePickleToFile(emf_explainer_demo, c.EMF_BEST_TRAIN_RECOMMENDER_PATH)

    def test_EMFExplainerDemo_Generate_Recommendations_Is_Valid(self):
        """
        Test validity of 'EMFExplainerDemo' generate recommendations with EMF model
        (including recommendation metrics)
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        emf_explainer_demo = Helpers.readPickleFromFile(c.EMF_BEST_TRAIN_RECOMMENDER_PATH)
        emf_explainer_demo.computeRecommendations(
            emf_explainer_demo.model,
            emf_explainer_demo.train_metadata_clone,
            emf_explainer_demo.test_data_clone_df)
        Helpers.writePickleToFile(emf_explainer_demo, c.EMF_BEST_TRAIN_RECOMMENDER_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(emf_explainer_demo.recommendations_df, c.EMF_BEST_RECOMMENDATIONS_PATH)
        self.assertIsNotNone(emf_explainer_demo.recommendations_df, error_msg)
        print("Sample of generated recommendations:")
        print(Helpers.tableize(emf_explainer_demo.recommendations_df.head(10)))
        Helpers.createConsoleDivider()
        self.assertIsNotNone(emf_explainer_demo.recommendation_metrics, error_msg)
        print("Recommendation metrics:")
        pprint(emf_explainer_demo.recommendation_metrics)

    def test_EMFExplainerDemo_Generate_Explanations_Is_Valid(self):
        """
        Test validity of 'EMFExplainerDemo' generate explanations with EMF model
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        emf_explainer_demo = Helpers.readPickleFromFile(c.EMF_BEST_TRAIN_RECOMMENDER_PATH)
        emf_explainer_demo.computeExplanations(
            emf_explainer_demo.metadata,
            ExplanationType.emf_explainer)
        self.assertIsNotNone(emf_explainer_demo.explanations_df, error_msg)
        self.assertIsNotNone(emf_explainer_demo.explanation_metrics, error_msg)
        emf_explainer_demo.explanations_df.expl = emf_explainer_demo.explanations_df.expl.round(4)
        print("Sample of generated explanations:")
        print(Helpers.tableize(emf_explainer_demo.explanations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Explanation metrics:")
        pprint(emf_explainer_demo.explanation_metrics)
        Helpers.writePickleToFile(emf_explainer_demo, c.EMF_BEST_EXPLANATION_MODEL_PATH)
        Helpers.writeRecommendationsOrExplanationsToCache(
            emf_explainer_demo.explanations_df,
            c.EMF_BEST_EXPLANATIONS_PATH)

    def test_EMFExplainerDemo_Integration_Based_On_Cached_Models_Is_Valid(self):
        """
        Test validity of 'EMFExplainerDemo' full integration of EMF model steps
        - based on cached/serialized recommendation and explanation models
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        emf_best_recommender = Helpers.readPickleFromFile(c.EMF_BEST_TRAIN_RECOMMENDER_PATH)
        self.assertIsNotNone(emf_best_recommender.recommendations_df, error_msg)
        self.assertIsNotNone(emf_best_recommender.recommendation_metrics, error_msg)

        emf_best_explainer = Helpers.readPickleFromFile(c.EMF_BEST_EXPLANATION_MODEL_PATH)
        self.assertIsNotNone(emf_best_explainer.explanations_df, error_msg)
        self.assertIsNotNone(emf_best_explainer.explanation_metrics, error_msg)
        ModelBasedExplainerDemo.reportDemoResults(
            emf_best_explainer.explanation_metrics,
            emf_best_explainer.explanations_df,
            emf_best_recommender.recommendation_metrics,
            emf_best_recommender.recommendations_df)





if __name__ == '__main__':
    ut.main()
