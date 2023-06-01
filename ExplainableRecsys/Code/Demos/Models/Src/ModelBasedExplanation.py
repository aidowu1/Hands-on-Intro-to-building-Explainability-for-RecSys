import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any, Dict, Union
from abc import ABC, abstractmethod
from pprint import pprint

from Code.ThirdParty.recoxplainer_master.recoxplainer.config import cfg
from Code.ThirdParty.recoxplainer_master.recoxplainer.models import ALS, EMFModel
from Code.Model.Src.FactorizationMachineModel import FMModel
from Code.ThirdParty.recoxplainer_master.recoxplainer.recommender import Recommender
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import Evaluator
from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.DataAccessLayer.Src.DataSplitter import Splitter
import Code.DataAccessLayer.Src.Constants as c
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Logging import Logger
from Code.Utils.Src.Utils import Helpers

class ModelBasedExplainerDemo(ABC):
    """
    Model-based explainer i.e., recommendation model explanations are done intrinsically in the 'black-box'
    """
    def __init__(
            self,
            explanation_type: ExplanationType,
            is_implicit_feedback=True
    ):
        """
        Constructor
        :param explanation_type: Explanation type
        :param is_implicit_feedback: Flag to indicate the use of implicit rating feedback
        """
        self.logger = Logger.getLogger()
        self.explanation_type = explanation_type
        if (self.explanation_type is ExplanationType.knn_explainer or
            self.explanation_type is ExplanationType.als_explainer or
            self.explanation_type is ExplanationType.ar_explainer or
            self.explanation_type is ExplanationType.emf_explainer
        ):
            self.metadata = DataReader(**cfg.ml100k)
        self.split_fraction = 0.1
        self.is_implicit_feedback = is_implicit_feedback
        self.rating_binarize_threshold = 1
        self.model = None
        self.recommender = None
        self.recommendations_df = None
        self.recommendation_metrics = None
        self.explanations_df = None
        self.explanation_metrics = None
        self.train_metadata_clone = None
        self.test_data_clone_df = None
        self.train_data_clone_df = None
        self.train_performance_metrics = None
        self.test_performance_metrics = None
        super().__init__()

    def preprocessMovielensDataset(self) -> Tuple[DataReader, pd.DataFrame]:
        """
        Preprocesses the Movielens dataset`
        """
        self.metadata.makeConsecutiveIdsInDataset()
        if self.is_implicit_feedback:
            self.metadata.binarize(binary_threshold=self.rating_binarize_threshold)
        splitter = Splitter()
        self.train_metadata_clone, self.test_data_clone_df = splitter.splitLeaveNOut(self.metadata, frac=self.split_fraction)
        return self.train_metadata_clone, self.test_data_clone_df

    @abstractmethod
    def fit(self, train_metadata: DataReader) -> bool:
        """
        Fit invocation used to train the recommendation model
        :param train_metadata: Train metadata
        """
        pass

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
            "hit_ratio": round(hit_ratio, 4),
            "ndcg": round(ndcg, 4)
        }
        return metrics

    def computeRecommendations(
            self,
            model: Union[ALS, EMFModel, FMModel],
            train_metadata: DataReader,
            test_dataset_df: pd.DataFrame):
        """
        Computes the recommendations
        """
        self.recommender = Recommender(train_metadata, model)
        self.recommendations_df = self.recommender.recommend_all()
        self.recommendation_metrics = self.evaluateRecommender(self.recommendations_df, test_dataset_df)

    def computeExplanations(
            self,
            train_metadata: DataReader,
            explanation_type: ExplanationType
    ) -> None:
        """
        Compute the recommendation explanations
        :param train_metadata: Training metadata
        :param explanation_type: Explanation type
        :return: explanations
        """
        self.getExplanationText(explanation_type)
        self.explanations_df, self.explanation_metrics = self.computeExplainerMetrics(train_metadata)

    def getExplanationText(self, explanation_type):
        """
        Gets the text of the specified Explanation Type
        :param explanation_type: Explanation type
        """
        explanation_text = [x.upper() for x in explanation_type.name.split("_")]
        self.logger.info(f"Explanations for {explanation_text[0]} {explanation_text[1].lower()}")

    def runDemo(self):
        """
        Runs the entire demo
        Produces results, namely:
            - recommendations
            - recommendation metrics
            - explanations
            - explanation metrics
        """
        train_metadata, test_df = self.preprocessMovielensDataset()
        self.fit(train_metadata)
        self.computeRecommendations(self.model, train_metadata, test_df)
        self.computeExplanations(train_metadata, self.explanation_type)
        return {
            "recommendations_df": self.recommendations_df,
            "recommendation_metrics": self.recommendation_metrics,
            "explanations_df": self.explanations_df,
            "explanation_metrics": self.explanation_metrics
        }

    @abstractmethod
    def computeExplainerMetrics(
            self,
            train_metadata: DataReader
        ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Compute Model-based Explanation and metrics
        :param train_metadata: Training metadata
        :return: Explanation and its metrics
        """
        pass

    @staticmethod
    def reportDemoResults(
            explanation_metrics: Dict[str, float],
            explanations_df: pd.DataFrame,
            recommendation_metrics: Dict[str, float],
            recommendations_df: pd.DataFrame,
            is_report_explanations: bool=True
    ):
        """
        Reports the demo results
        :param explanation_metrics: Explanation metrics
        :param explanations_df: Explanations
        :param recommendation_metrics: Recommendation metrics
        :param recommendations_df: recommendations
        :param is_report_explanations: Flag to indicate the reporting of explanations
        """
        print("Sample of generated recommendations:")
        print(Helpers.tableize(recommendations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Recommendation metrics:")
        pprint(recommendation_metrics)
        Helpers.createConsoleDivider()
        if is_report_explanations:
            print("Sample of generated explanations:")
            if explanations_df is not None:
                print(Helpers.tableize(explanations_df.head(10)))
            else:
                print("No explanations computed!!")
            Helpers.createConsoleDivider()
        print("Explanation metrics:")
        if explanation_metrics is not None:
            pprint(explanation_metrics)
        else:
            print("No explanations metrics computed!!")
        Helpers.createConsoleDivider()





