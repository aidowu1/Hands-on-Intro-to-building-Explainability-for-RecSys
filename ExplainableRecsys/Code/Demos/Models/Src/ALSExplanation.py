import pandas as pd
from typing import Tuple, Dict

from Code.ThirdParty.recoxplainer_master.recoxplainer.config import cfg
from Code.ThirdParty.recoxplainer_master.recoxplainer.models import ALS
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import ExplanationEvaluator
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.model_based_als_explain import ALSExplainer
from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.Utils.Src.Enums import ExplanationType
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo


class ALSExplainerDemo(ModelBasedExplainerDemo):
    """
    ALS Explainer demo
    """
    def __init__(
            self,
            explanation_type: ExplanationType = ExplanationType.als_explainer,
            is_implicit_feedback: bool = True
    ):
        """
        Constructor

        """
        super().__init__(explanation_type, is_implicit_feedback)
        self.logger.info(f"Start of '{explanation_type.name}' model-based RecSys explanation demo")

    def fit(self, train_metadata: DataReader) -> bool:
        """
        Fit invocation used to train the recommendation model
        :param train_metadata: Train metadata
        :return: Status
        """
        self.model = ALS(**cfg.model.als)
        return self.model.fit(train_metadata)

    def computeExplainerMetrics(
            self,
            train_metadata: DataReader,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the ALS Model Explainer performance metrics
        :param train_metadata: Training metadata
        :return: Explainer metrics
        """
        explainer = ALSExplainer(self.model, self.recommendations_df, train_metadata)
        explanations_df = explainer.explain_recommendations()
        explainer_evaluator = ExplanationEvaluator(train_metadata.num_user)
        explanation_metrics = {
            "fidelity": round(explainer_evaluator.model_fidelity(explanations_df), 4)
        }
        return explanations_df, explanation_metrics



