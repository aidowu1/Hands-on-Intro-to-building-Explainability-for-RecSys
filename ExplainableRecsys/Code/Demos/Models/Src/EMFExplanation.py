import pandas as pd
from typing import Tuple, Dict

from Code.ThirdParty.recoxplainer_master.recoxplainer.config import cfg
from Code.ThirdParty.recoxplainer_master.recoxplainer.models import EMFModel
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import ExplanationEvaluator
from Code.ThirdParty.recoxplainer_master.recoxplainer.explain.model_based_emf import EMFExplainer
from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.Utils.Src.Enums import ExplanationType
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo


class EMFExplainerDemo(ModelBasedExplainerDemo):
    """
    EMF Explainer demo
    """
    def __init__(
            self,
            explanation_type: ExplanationType = ExplanationType.emf_explainer,
            is_implicit_feedback: bool = False
    ):
        """
        Constructor
        :param explanation_type: Explanation type
        :param is_implicit_feedback: Flag to indicate the use of implicit rating feedback
        """
        super().__init__(explanation_type, is_implicit_feedback)
        self.logger.info(f"Start of {explanation_type.name} model-based RecSys explanation demo")

    def fit(self, train_metadata: DataReader) -> bool:
        """
        Fit invocation used to train the recommendation model
        :param train_metadata: Train metadata
        :return: Status
        """
        self.model = EMFModel(**cfg.model.emf)
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
        explainer = EMFExplainer(self.model, self.recommendations_df, train_metadata)
        explanations_df = explainer.explain_recommendations()
        explainer_evaluator = ExplanationEvaluator(train_metadata.num_user)
        explanation_metrics = {
            "fidelity": round(explainer_evaluator.model_fidelity(explanations_df), 4),
            "mean-explainable-precision": round(explainer_evaluator.mean_explaianable_precision(
                explanations_df,
                self.model.explainability_matrix), 4)
        }
        return explanations_df, explanation_metrics

