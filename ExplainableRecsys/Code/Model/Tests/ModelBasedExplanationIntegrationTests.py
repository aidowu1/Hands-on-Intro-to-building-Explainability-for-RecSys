import pprint
import unittest as ut
import inspect
import os
import pathlib
import pandas as pd
from typing import Optional, Tuple, Any, Dict
from copy import deepcopy
import random

working_dir = pathlib.Path(os.getcwd()).parent.parent
os.chdir(working_dir)

from Code.ThirdParty.recoxplainer_master.recoxplainer.config import cfg
from Code.ThirdParty.recoxplainer_master.recoxplainer.models import ALS, EMFModel
from Code.ThirdParty.recoxplainer_master.recoxplainer.recommender import Recommender
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import Evaluator
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.post_hoc_association_rules import ARPostHocExplainer
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.post_hoc_knn import KNNPostHocExplainer
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.explainer import Explainer
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.model_based_als_explain import ALSExplainer
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.model_based_emf import EMFExplainer
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import ExplanationEvaluator
from Code.Model.Src.MatrixFactorizationModel import MFModel
from Code.ThirdParty.recoxplainer_master.recoxplainer.data_reader import data_reader
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import splitter
from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.DataAccessLayer.Src.DataSplitter import Splitter
import Code.DataAccessLayer.Src.Constants as c
from Code.Utils.Src.Utils import Helpers
from Code.Utils.Src.Enums import ExplanationType

class TestModelBasedExplainers(ut.TestCase):
    """
    Integration test suit for model-based explanations
    This component tests the following features:
        - Training of Models used to compute the recommendations using model-based explanation approaches, namely:
            - Alternating Least Square (ALS)
            - Explainable Matrix Factorization (EMF)
        - Recommendation of items i.e. ranking using the trained models
        - Evaluation of the performance metrics of the models:
            - Hit ratio
            - Normalized Discounted Cumulative Gain (NDCG)
        - ALS-Explain model-based explanation of the MF recommendations using AR explainer (with performance metrics)
        - EMF-Explain model-based explanation of the MF recommendations using KNN explainer (with performance metrics)
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")
        self.metadata = DataReader(**cfg.ml100k)
        self.split_fraction = 0.1

    def test_ALS_Model_Recommendation_And_Explanation_Is_Valid(self):
        """
        Test the validity of ALS model recommendation and explanation
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        model = ALS(**cfg.model.als)
        self.testModelBasedExplainers(model, error_msg, ExplanationType.als_explainer)

    def test_EMF_Model_Recommendation_And_Explanation_Is_Valid(self):
        """
        Test the validity of EMF model recommendation and explanation
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        model = EMFModel(**cfg.model.emf)
        self.testModelBasedExplainers(model, error_msg, ExplanationType.emf_explainer)

    def testModelBasedExplainers(
            self,
            model,
            error_msg: str,
            explanation_type) -> Dict[str, Any]:
        """
        Runs model-based explainers
        :param model: Recommendation model
        :param explanation_type: Explanation type
        :return: results
        """
        train_metadata, test_df = self.preprocessMovielensDataset()
        self.assertIsNotNone(model, msg=error_msg)
        model.fit(train_metadata)
        recommender = Recommender(train_metadata, model)
        self.assertIsNotNone(recommender, msg=error_msg)
        recommendations_df = recommender.recommend_all()
        self.assertIsNotNone(recommendations_df, msg=error_msg)
        recommendation_metrics = self.evaluateRecommender(recommendations_df, test_df)
        self.assertIsNotNone(recommendation_metrics, msg=error_msg)
        explanations_df, explain_metrics = self.computeExplanations(
            model,
            recommendations_df,
            train_metadata,
            explanation_type)
        self.assertIsNotNone(explanations_df, msg=error_msg)
        self.assertIsNotNone(explain_metrics, msg=error_msg)
        return {
            "recommendations_df": recommendations_df,
            "recommendation_metrics": recommendation_metrics,
            "explanations_df": explanations_df,
            "explain_metrics": explain_metrics
        }

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

    def computeExplanations(
            self,
            model: Any,
            recommendations_df: pd.DataFrame,
            train_metadata: DataReader,
            explanation_type: ExplanationType
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Compute the recommendation explanations
        :param model: Ecommendation model
        :param recommendations_df: Recommendations
        :param train_metadata: Training metadata
        :param explanation_type: Explanation type
        :return: explanations
        """
        if explanation_type is ExplanationType.als_explainer:
            explanations_df, explanation_metrics = self.computeALSExplainerMetrics(
                train_metadata,
                model,
                recommendations_df)
        elif explanation_type is ExplanationType.emf_explainer:
            explanations_df, explanation_metrics = self.computeEMFExplainerMetrics(
                train_metadata,
                model,
                recommendations_df
            )
        else:
            explanations_df, explanation_metrics = self.computeALSExplainerMetrics(
                train_metadata,
                model,
                recommendations_df)

        return explanations_df, explanation_metrics

    def computeALSExplainerMetrics(
            self,
            train_metadata: DataReader,
            model: Any,
            recommendations_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the ALS Model Explainer performance metrics
        :param train_metadata: Training metadata
        :param model: Model
        :param recommendations_df: Recommendations
        :return: Explainer metrics
        """
        explainer = ALSExplainer(model, recommendations_df, train_metadata)
        explanations_df = explainer.explain_recommendations()
        explainer_evaluator = ExplanationEvaluator(train_metadata.num_user)
        explanation_metrics = {
            "fidelity": explainer_evaluator.model_fidelity(explanations_df)
        }
        return explanations_df, explanation_metrics

    def computeEMFExplainerMetrics(
            self,
            train_metadata: DataReader,
            model: Any,
            recommendations_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the EMF Model Explainer performance metrics
        :param train_metadata: Training metadata
        :param model: Model
        :param recommendations_df: Recommendations
        :return: Explainer metrics
        """
        explainer = EMFExplainer(model, recommendations_df, train_metadata)
        explanations_df = explainer.explain_recommendations()
        explainer_evaluator = ExplanationEvaluator(train_metadata.num_user)
        explanation_metrics = {
            "fidelity": explainer_evaluator.model_fidelity(explanations_df),
            "mean-explainable-precision": explainer_evaluator.mean_explaianable_precision(explanations_df, model.explainability_matrix)
        }
        return explanations_df, explanation_metrics

    def preprocessMovielensDataset(self) -> Tuple[DataReader, pd.DataFrame]:
        """
        Preprocesses the Movielens dataset`
        """
        self.metadata.makeConsecutiveIdsInDataset()
        self.metadata.binarize(binary_threshold=1)
        splitter = Splitter()
        train_metadata, test_df = splitter.splitLeaveNOut(self.metadata, frac=self.split_fraction)
        return train_metadata, test_df

    def generateALSExplanationAppGui(
            self,
            train_metadata: DataReader,
            explanations_df: pd.DataFrame) -> Any:
        """
        Generates ALS Explanation App GUI (Dash Plotly)
        :param train_metadata: Training metadata
        :param explanations_df: Explanations
        :return: Dash Plotly GUI app
        """
        sample_users = random.sample(set(train_metadata.dataset.userId), 10)
        sample_explanations = explanations_df[explanations_df.userId.isin(sample_users)]

    def getMovielensContent(self) -> pd.DataFrame:
        """
        Gets the Movielens content - Movie Titles

        """
        content_df = pd.read_csv(
            c.MOVIELENS_ITEM_PATH,
            sep='|',
            encoding="ISO-8859-1",
            skiprows=0,
            engine='python',
            header=None)
        content_df = content_df.set_index(0)[[1]]
        content_df.columns = ['movie']
        return content_df




if __name__ == '__main__':
    ut.main()
