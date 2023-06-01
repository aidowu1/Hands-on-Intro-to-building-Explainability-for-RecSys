import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any, Dict, Union
from abc import ABC, abstractmethod
from pprint import pprint

from Code.ThirdParty.recoxplainer_master.recoxplainer.config import cfg
from Code.ThirdParty.recoxplainer_master.recoxplainer.models import ALS, EMFModel
from Code.ThirdParty.recoxplainer_master.recoxplainer.recommender import Recommender
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import Evaluator
from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.DataAccessLayer.Src.DataSplitter import Splitter
import Code.DataAccessLayer.Src.Constants as c
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Logging import Logger
from Code.Utils.Src.Utils import Helpers
from Code.Demos.Models.Src.PostHocExplanation import PostHocExplainerDemo
from Code.Model.Src.MatrixFactorizationModel import MFModel
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.post_hoc_knn import KNNPostHocExplainer


class KNNPostHocExplainerDemo(PostHocExplainerDemo):
    """
    KNN Post Hoc Explainer demo
    """
    def __init__(
            self,
            explanation_type: ExplanationType = ExplanationType.knn_explainer,
            is_implicit_feedback=False,
            nearest_k=10
    ):
        """
        Constructor
        :param explanation_type: Explanation type
        :param is_implicit_feedback: Flag to indicate the use of implicit rating feedback
        """
        super().__init__(explanation_type, is_implicit_feedback)
        self.logger.info(f"Start of '{explanation_type.name}' post-hoc RecSys explanation demo")
        self.nearest_k = nearest_k

    def computeExplainerMetrics(
            self,
            train_metadata: DataReader,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the AR post hoc Explainer performance metrics
        :param train_metadata: Training metadata
        :return: Explainer metrics
        """
        explanations_df, explanation_metrics = self.computeKNNExplanation(
            self.model,
            self.recommendations_df,
            self.train_metadata_clone)
        explanations_df.explanation_score = self.transformExplanationScore(explanations_df)
        return explanations_df, explanation_metrics

    def computeKNNExplanation(
            self,
            recommendation_model: MFModel,
            recommendations: pd.DataFrame,
            train_dataset: DataReader
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the KNN explanability of the recommendation model
        :param recommendation_model: Recommendation model
        :param recommendations: Computed recommendations
        :param train_dataset: Training dataset
        :return: Explanations
        """
        ar_explainer = KNNPostHocExplainer(
            recommendation_model,
            recommendations,
            train_dataset,
            knn=self.nearest_k
        )
        return self.computePostHocExplainer(ar_explainer, train_dataset)

    def transformExplanationScore(self, explanations_df: pd.DataFrame):
        """
        Transforms the explanation score series (per row) as a list
        :param explanations_df: Explanations
        :return: Transformed confidence metrics
        """
        result = [[round(y, 4) for y in x] for x in explanations_df.explanation_score.tolist() if len(x) > 0]
        return result