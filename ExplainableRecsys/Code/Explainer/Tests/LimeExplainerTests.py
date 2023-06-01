import unittest as ut
import inspect
import os
from pprint import pprint
import pathlib
import pandas as pd
import numpy as np
import json
from lime import explanation
from typing import List, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

working_dir = pathlib.Path(os.getcwd()).parent.parent
os.chdir(working_dir)

from Code.Model.Src.FactorizationMachineModel import FMModel
from Code.Explainer.Src.LimeExplainer import LimeRSExplainer
from Code.Explainer.Src.LimeExplainerReporting import LimeFMExplainerReporter
from Code.Demos.Visualization.ExplainabilityVisualization import ExplainerVisualizer
from Code.DataAccessLayer.Src.MovielensDataset import MovielenDataset
import Code.Model.Src.Constants as c
import Code.DataAccessLayer.Src.Constants as c2
from Code.Utils.Src.Logging import Logger
from Code.Utils.Src.Utils import Helpers

class TestLimeRSExplainer(ut.TestCase):
    """
     Test suit for the Lime Explainer for the FM recommendation model
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        self.logger = Logger.getLogger()
        self.logger.info(f"Working folder is: {working_dir}")
        self.dataset = MovielenDataset()
        self.item_columns = c2.MOVIELENS_GENRE_COLUMNS
        self.dataset.splitDataset()
        self.fm_model: FMModel = Helpers.readPickleModelDump()
        self.feature_map = {i: self.fm_model.one_hot_columns[i]
                             for i in range(len(list(self.fm_model.one_hot_columns)))}
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

    def test_LimeRSExplainer_Constructor_Is_Valid(self):
        """
        Test the validity of the Lime Explainer for the recommendation FM model
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        explainer = LimeRSExplainer(dataset=self.dataset,
                                    feature_names=self.fm_model.one_hot_columns,
                                    feature_map=self.feature_map,
                                    feature_selection=self.feature_selection,
                                    mode=self.problem_type,
                                    class_names=self.class_names,
                                    random_state=self.seed
                                    )
        self.assertIsNotNone(explainer, msg=error_msg)

    def test_LimeRSExplainer_Generate_Neighborhood_Values_Is_Valid(self):
        """
        Test the validity of the Lime Explainer generating neighborhood values of a local instance
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        explainer = LimeRSExplainer(dataset=self.dataset,
                                    feature_names=self.fm_model.one_hot_columns,
                                    feature_map=self.feature_map,
                                    feature_selection=self.feature_selection,
                                    mode=self.problem_type,
                                    class_names=self.class_names,
                                    random_state=self.seed
                                    )
        self.assertIsNotNone(explainer, msg=error_msg)
        neighborhoods_df = explainer.generateNeighborhood(
            instance=self.instance_df,
            entity=self.entity,
            num_samples=self.n_explainer_perturbation_samples
        )
        self.assertIsNotNone(neighborhoods_df, msg=error_msg)
        print("Sample of the local instance neighborhood dataset:")
        print(Helpers.tableize(neighborhoods_df.head(10)))

    def test_LimeRSExplainer_Explain_An_Instance_Is_Valid(self):
        """
        Test the validity of the Lime Explainer explaining a local instance
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        explainer = LimeRSExplainer(dataset=self.dataset,
                                    feature_names=self.fm_model.one_hot_columns,
                                    feature_map=self.feature_map,
                                    feature_selection=self.feature_selection,
                                    mode=self.problem_type,
                                    class_names=self.class_names,
                                    random_state=self.seed
                                    )
        self.assertIsNotNone(explainer, msg=error_msg)
        explanations = explainer.explainInstance(instance=self.instance_df,
                                                 rec_model=self.fm_model,
                                                 neighborhood_entity=self.entity,
                                                 labels=self.explainer_labels,
                                                 num_samples=self.n_explainer_perturbation_samples
                                                 )
        self.assertIsNotNone(explanations, msg=error_msg)
        report_df, reports_json = LimeFMExplainerReporter.reportExplanationResults(
            instance=self.instance_df,
            explanation_details=explanations,
            explainer=explainer)
        self.assertIsNotNone(report_df, msg=error_msg)
        print("Sample reported explanation results are:")
        print(Helpers.tableize(report_df))
        ExplainerVisualizer.visualizeLimeFMExplainability(report_df)

    def test_LimeRSExplainer_Explain_List_Of_Instances_Are_Valid(self):
        """
        Test the validity of the Lime Explainer explaining a list of local instances
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        explainer = LimeRSExplainer(dataset=self.dataset,
                                    feature_names=self.fm_model.one_hot_columns,
                                    feature_map=self.feature_map,
                                    feature_selection=self.feature_selection,
                                    mode=self.problem_type,
                                    class_names=self.class_names,
                                    random_state=self.seed
                                    )
        self.assertIsNotNone(explainer, msg=error_msg)
        n_instances = 50
        sample_instances_df = self.dataset.user_item_interaction_modified_df.head(n_instances)
        sample_local_instances = LimeRSExplainer.createSampleLocalInstances(sample_instances_df)
        all_instances_explanations = []
        with tqdm(total=n_instances) as progress:
            for i in range(n_instances):
                instance_df = sample_local_instances[i]
                explanations = explainer.explainInstance(instance=instance_df,
                                                         rec_model=self.fm_model,
                                                         neighborhood_entity=self.entity,
                                                         labels=self.explainer_labels,
                                                         num_samples=self.n_explainer_perturbation_samples
                                                         )
                self.assertIsNotNone(explanations, msg=error_msg)
                report_df, reports_json = LimeFMExplainerReporter.reportExplanationResults(
                    instance=instance_df,
                    explanation_details=explanations,
                    explainer=explainer)

                explanation_report_df = pd.DataFrame(reports_json)
                explanation_report_df.fillna(0, inplace=True)
                all_instances_explanations.append(explanation_report_df)
                progress.update(1)
                progress.set_postfix({"Current instance": i})
        explanations_df = pd.concat(all_instances_explanations)
        explanations_df.fillna(0, inplace=True)
        self.assertIsNotNone(explanations_df, msg=error_msg)
        print("Sample reported explanation results are:")
        print(Helpers.tableize(explanations_df))


    @staticmethod
    def visualizeLimeFMExplainability(explanations_df, title="Movielens Lime Explainability"):
        """
        Visualization of FM model LIME explainability
        :param explainer: Explainer
        """
        lime_feature_coefs = json.loads(explanations_df.explanations.iloc[0])
        feature_names = list(lime_feature_coefs.keys())
        feature_names.reverse()
        coef_values = list(lime_feature_coefs.values())
        coef_values.reverse()
        pos = np.arange(len(coef_values)) + .5
        colors = ['green' if x > 0 else 'red' for x in coef_values]
        plt.barh(pos, coef_values, align='center', color=colors)
        plt.yticks(pos, feature_names, rotation=45)
        plt.title(title)
        plt.show()

    @staticmethod
    def extractFeatures(
            explanation_all_ids,
            feature_type,
            feature_map) -> Dict[int, float]:
        """
        Extracts features from the computed explanation object
        :param explanation_all_ids: Explanation IDs
        :param feature_type: Feature type
        :param feature_map: Feature map
        :return: Extracted features from the explanation
        """
        filtered_dict = dict()
        if feature_type == "features":
            for tup in explanation_all_ids:
                if not (feature_map[tup[0]].startswith(c2.MOVIELENS_USER_ID_COLUMN_KEY) or
                        feature_map[tup[0]].startswith(c2.MOVIELENS_ITEM_ID_COLUMN_KEY)):
                    filtered_dict[feature_map[tup[0]]] = round(tup[1], 3)

        elif feature_type == "item":
            top_features = 500
            for tup in explanation_all_ids:
                if feature_map[tup[0]].startswith(c2.MOVIELENS_USER_ID_COLUMN_KEY) and len(filtered_dict) <= top_features:
                    filtered_dict[feature_map[tup[0]]] = round(tup[1], 3)
        return filtered_dict

    @staticmethod
    def reportExplanationResults(
            instance: pd.DataFrame,
            explanation: explanation.Explanation,
            explainer: LimeRSExplainer,
            feature_type: str = 'features') -> pd.DataFrame:
        """
        Reports the computed explanation results
        :param instance: Local instance (dataset point/row) that is being explained
        :param explanation: Computed explanation object
        :param explainer: Explainer component
        :param feature_type: Feature type
        :return: Report of the computed explanation
        """
        filtered_features = TestLimeRSExplainer.extractFeatures(explanation.local_exp[0],
                                                                feature_type=feature_type,
                                                                feature_map=explainer.feature_map)
        #
        explanation_str = json.dumps(filtered_features)
        explanation_results_df = pd.DataFrame({c2.MOVIELENS_USER_ID_COLUMN_KEY: [instance[c2.MOVIELENS_USER_ID_COLUMN_KEY]],
                                               c2.MOVIELENS_ITEM_ID_COLUMN_KEY: [instance[c2.MOVIELENS_ITEM_ID_COLUMN_KEY]],
                                  'explanations': [explanation_str],
                                  'local_prediction': [round(explanation.local_pred[0], 3)]})
        return explanation_results_df


if __name__ == '__main__':
    ut.main()
