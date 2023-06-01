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
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo
from Code.Model.Src.MatrixFactorizationModel import MFModel
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import ExplanationEvaluator

class PostHocExplainerDemo(ModelBasedExplainerDemo):
    """
    Post Hoc Explainer demo
    """
    def __init__(
            self,
            explanation_type: ExplanationType = ExplanationType.posthoc_noexplainer,
            is_implicit_feedback=False
    ):
        """
        Constructor
        :param explanation_type: Explanation type
        :param is_implicit_feedback: Flag to indicate the use of implicit rating feedback
        """
        self.learning_rate = None
        self.weight_decay = None
        self.latent_dim = None
        self.epochs = None
        self.batch_size = None
        super().__init__(explanation_type, is_implicit_feedback)
        self.logger.info(f"Start of '{explanation_type.name}' post-hoc RecSys explanation demo")

    def fit(self, train_metadata: DataReader) -> bool:
        """
        Fit invocation used to train the recommendation model
        :param train_metadata: Train metadata
        :return: Status
        """
        self.learning_rate = 0.01
        self.weight_decay = 0.001
        self.latent_dim = 100
        self.epochs = 150
        self.batch_size = 128
        self.model = MFModel(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            latent_dim=self.latent_dim,
            epochs=self.epochs,
            batch_size=self.batch_size,
            split_fraction=self.split_fraction
        )
        return self.model.fit(train_metadata)

    def computeExplainerMetrics(
            self,
            train_metadata: DataReader,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the ALS Model Explainer performance metrics
        No explanation is required for this component so explanations are empty and there are no metrics
        :param train_metadata: Training metadata
        :return: Explainer metrics
        """
        explanations_df = pd.DataFrame()
        explanation_metrics = {
        }
        return explanations_df, explanation_metrics

    def computePostHocExplainer(self, explainer, train_dataset):
        """
        Computes post-hoc explainers of the recommendation model
        :param explainer: Explainer
        :param train_dataset: Training dataset
        :return: Explanations
        """
        explanations_df = explainer.explain_recommendations()
        explanations_df["explanations_as_list"] = explanations_df.explanations.apply(lambda x: list(x))
        explanations_df["n_explanations"] = explanations_df["explanations_as_list"].apply(
            lambda x: len(x))
        filter_1 = explanations_df["n_explanations"] > 0
        explainer_evaluator = ExplanationEvaluator(train_dataset.num_user)
        explanation_metrics = {
            "fidelity": explainer_evaluator.model_fidelity(explanations_df)
        }
        return explanations_df[filter_1], explanation_metrics

