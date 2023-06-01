import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any, Dict, Union
from abc import ABC, abstractmethod
from pprint import pprint
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm

from Code.ThirdParty.recoxplainer_master.recoxplainer.config import cfg
from Code.ThirdParty.recoxplainer_master.recoxplainer.models import ALS, EMFModel
from Code.ThirdParty.recoxplainer_master.recoxplainer.recommender import Recommender
from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import Evaluator
from Code.DataAccessLayer.Src.DataProviderFM import DataReader
from Code.DataAccessLayer.Src.DataSplitter import Splitter
import Code.DataAccessLayer.Src.Constants as c
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Logging import Logger
from Code.Utils.Src.Utils import Helpers
from Code.Demos.Models.Src.PostHocExplanation import PostHocExplainerDemo
from Code.Model.Src.MatrixFactorizationModel import MFModel
import Code.DataAccessLayer.Src.MovieLensDataPreProcessing as md

from Code.Model.Src.FactorizationMachineModel import FMModel
from Code.Explainer.Src.LimeExplainer import LimeRSExplainer
from Code.Explainer.Src.LimeExplainerReporting import LimeFMExplainerReporter
from Code.DataAccessLayer.Src.MovielensDataset import MovielenDataset
import Code.Model.Src.Constants as c
import Code.DataAccessLayer.Src.Constants as c2
from Code.Utils.Src.Utils import Helpers

class FMLimePostHocExplainerDemo(PostHocExplainerDemo):
    """
    FM-Lime Post Hoc Explainer demo
    """
    def __init__(
            self,
            explanation_type: ExplanationType = ExplanationType.fmlime_explainer,
            is_implicit_feedback=False
    ):
        """
        Constructor
        :param explanation_type: Explanation type
        :param is_implicit_feedback: Flag to indicate the use of implicit rating feedback
        """
        self.metadata = None
        self.model_parameters = None
        self.feature_map = None
        self.problem_type = None
        self.feature_selection = None
        self.class_names = np.array(['rec'])
        self.seed = 100
        self.instance_df = None
        self.entity = None
        self.n_explainer_perturbation_samples = None
        self.explainer_labels = None
        self.one_hot_columns = None
        self.dataset = None
        self.item_columns = None
        self.sample_instances_user_item_df = None
        super().__init__(explanation_type, is_implicit_feedback)
        self.logger.info(f"Start of '{explanation_type.name}' post-hoc RecSys explanation demo")

    def _FMModelinitilaizer(self) -> None:
        """
        FM model initialization
        """
        self.model_parameters = {
            c.FM_PARAMS_RANK_KEY: c.FM_PARAMS_RANK_VALUE,
            c.FM_PARAMS_SEED_KEY: c.FM_PARAMS_RANK_VALUE,
            c.FM_PARAM_N_ITER_KEY: c.FM_PARAM_N_ITER_VALUE,
            c.FM_PARAMS_N_KEPT_SAMPLES_KEY: c.FM_PARAMS_N_KEPT_SAMPLES_VALUE
        }
        self.split_fraction = c2.MOVIELENS_TEST_DATA_SPLIT_FRACTION

    def _limeExplainerInitializer(self):
        """
        LIME model initialization
        """
        self.feature_map = {i: self.model.one_hot_columns[i]
                            for i in range(len(list(self.model.one_hot_columns)))}
        self.problem_type = 'regression'
        self.feature_selection = "auto"
        self.class_names = np.array(['rec'])
        self.seed = 100
        self.instance_df = pd.DataFrame({
            c2.MOVIELENS_USER_ID_COLUMN_KEY: [1],
            c2.MOVIELENS_ITEM_ID_COLUMN_KEY: [5]
        })
        self.entity = "item"
        self.n_explainer_perturbation_samples = 1000
        self.explainer_labels = [0]

    def preprocessMovielensDataset(self) -> Tuple[DataReader, pd.DataFrame]:
        """
        Preprocesses the Movielens dataset`
        """
        self.dataset = MovielenDataset()
        self.item_columns = c2.MOVIELENS_GENRE_COLUMNS
        self.dataset.splitDataset()
        self.train_metadata_clone = DataReader(self.dataset)
        self.test_data_clone_df = self.train_metadata_clone.test_df
        self.train_data_clone_df = self.train_metadata_clone.train_df
        return self.train_metadata_clone, self.test_data_clone_df

    def fit(self, train_metadata: DataReader) -> bool:
        """
        Fit invocation used to train the recommendation model
        :param train_metadata: Train metadata
        """
        self.logger.info("Starting the training cycle of the FM model")
        self._FMModelinitilaizer()
        ohe_feature_columns = None
        X_all_train, one_hot_columns = md.Movielens100KPreprocessor.transformFMModelItemSideDatasetV2(
            user_item_interaction_df=self.dataset.user_item_interaction_modified_df,
            item_side_df=self.dataset.items_side_info_modified_df,
            ohe_feature_columns=ohe_feature_columns,
            is_consolidate_genre_values=False
        )
        y_all_train = self.dataset.rating
        X_train = X_all_train[self.dataset.train_split_indices]
        y_train = y_all_train[self.dataset.train_split_indices]
        X_test = X_all_train[self.dataset.test_split_indices]
        y_test = y_all_train[self.dataset.test_split_indices]
        self.model = FMModel(
            training_data=(X_train, y_train),
            fit_parameters=self.model_parameters)
        self.model.train()
        self.train_performance_metrics = self.computeFMMetrics(X_train, y_train)
        self.test_performance_metrics = self.computeFMMetrics(X_test, y_test)
        self.logger.info("FM Train performance metrics:")
        self.logger.info(self.train_performance_metrics)
        self.logger.info("FM Test performance metrics:")
        self.logger.info(self.test_performance_metrics)
        self.model.one_hot_encoder = None
        self.model.one_hot_columns = one_hot_columns
        self.logger.info("Model train is complete")
        self.logger.info(f"With one-hot encoding columns: {one_hot_columns}")
        return True

    def computeFMMetrics(self, X: csr_matrix, y: np.ndarray) -> Dict[str, float]:
        """
        Computes the performance metrics of the FM model
        :param X: Features
        :param y: Label
        :return: FM metrics
        """
        X_array = X.toarray()
        prediction = self.model.predict(X_array)
        rmse = round(((y - prediction) ** 2).mean() ** .5, 4)
        mae = np.round(np.abs(y - prediction).mean(), 4)
        perf_results = {
            "rmse": rmse,
            "mae": mae
        }
        return perf_results

    def computeRecommendations(
            self,
            model: Union[ALS, EMFModel, FMModel],
            train_metadata: DataReader,
            test_dataset_df: pd.DataFrame):
        """
        Computes the recommendations
        """
        user_ids = list(set(test_dataset_df[c2.MOVIELENS_USER_ID_COLUMN_KEY].tolist()))
        self.recommendations_df = self.model.recommend(
            user_ids=user_ids,
            user_item_interaction_df=self.dataset.train_users_items_df,
            items_side_info_df=self.dataset.items_side_info_modified_df,
            is_filter_interacted_items=True
        )
        self.recommendation_metrics = self.evaluateRecommender(self.recommendations_df, test_dataset_df)

    def computeExplainerMetrics(
            self,
            train_metadata: DataReader,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the FM-Lime post hoc Explainer performance metrics
        :param train_metadata: Training metadata (but not used as training data is obtained internally from the
                               FM-Lime model)
        :return: Explainer metrics
        """
        self.computeFMLimeExplainability()
        explanations_df, explanation_metrics = self.computeFMLimeExplainabilityMetrics(self.recommendations_df, explanations_df)
        return explanations_df, explanation_metrics

    def computeFMLimeExplainability(
            self
    ) -> None:
        """
        Computes the FM Lime expalianability
        """
        self._limeExplainerInitializer()
        explainer = LimeRSExplainer(dataset=self.dataset,
                                    feature_names=self.model.one_hot_columns,
                                    feature_map=self.feature_map,
                                    feature_selection=self.feature_selection,
                                    mode=self.problem_type,
                                    class_names=self.class_names,
                                    random_state=self.seed
                                    )
        n_instances = c.N_LOCAL_SAMPLES
        if n_instances:
            sample_instances_df = explainer.training_df.sample(n_instances)
        else:
            sample_instances_df = explainer.training_df.copy()
            n_instances = sample_instances_df.shape[0]
        self.sample_instances_user_item_df = sample_instances_df[[c2.MOVIELENS_USER_ID_COLUMN_KEY,
                                                                  c2.MOVIELENS_ITEM_ID_COLUMN_KEY]]
        sample_local_instances = LimeRSExplainer.createSampleLocalInstances(sample_instances_df)
        all_instances_explanations = []
        with tqdm(total=n_instances) as progress:
            for i in range(n_instances):
                instance_df = sample_local_instances[i]
                explanations = explainer.explainInstance(instance=instance_df,
                                                         rec_model=self.model,
                                                         neighborhood_entity=self.entity,
                                                         labels=self.explainer_labels,
                                                         num_samples=self.n_explainer_perturbation_samples
                                                         )
                report_df, reports_json = LimeFMExplainerReporter.reportExplanationResults(
                    instance=instance_df,
                    explanation_details=explanations,
                    explainer=explainer)

                explanation_report_df = pd.DataFrame(reports_json)
                explanation_report_df.fillna(0, inplace=True)
                all_instances_explanations.append(explanation_report_df)
                progress.update(1)
                progress.set_postfix({"Current instance": i})
        self.explanations_df = pd.concat(all_instances_explanations)
        self.explanations_df.fillna(0, inplace=True)


    def computeFMLimeExplainabilityMetrics(
            self,
            recommendation_df: pd.DataFrame,
            explanations_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the explainability metric of the FM-Lime model
        :param recommendation_df: Recommendations
        :param explanations_df: Explanations
        """
        user_id_column = c2.MOVIELENS_USER_ID_COLUMN_KEY
        item_id_column = c2.MOVIELENS_ITEM_ID_COLUMN_KEY
        recommedation_filter = (recommendation_df[user_id_column].
            isin(self.sample_instances_user_item_df[user_id_column])
            & recommendation_df[item_id_column].
            isin(self.sample_instances_user_item_df[item_id_column])
        )
        explanation_filter = explanations_df.R2_Score > c.R2_SCORE_THRESHOLD
        sampled_recommendations_df = recommendation_df[recommedation_filter]
        n_recommended_users = sampled_recommendations_df[user_id_column].nunique()
        valid_explanations_df = explanations_df[explanation_filter]
        self.explanations_df = valid_explanations_df.copy()
        fidelity = sum(valid_explanations_df.groupby(user_id_column)[item_id_column].count()) / n_recommended_users
        fidelity = round(fidelity, 4)
        return self.explanations_df, {"fidelity": fidelity}
