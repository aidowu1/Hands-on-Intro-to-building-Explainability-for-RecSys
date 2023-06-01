import pandas as pd
from lime import explanation
from typing import List, Dict, Any, Tuple
import json

from Code.Explainer.Src.LimeExplainer import LimeRSExplainer
import Code.DataAccessLayer.Src.Constants as c2



class LimeFMExplainerReporter(object):
    """
    Lime FM Explainer reporter used to report the computed explanations from
    the FM Lime model
    """
    @staticmethod
    def reportExplanationResults(
            instance: pd.DataFrame,
            explanation_details: explanation.Explanation,
            explainer: LimeRSExplainer,
            feature_type: str = 'features') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Reports the computed explanation results
        :param instance: Local instance (dataset point/row) that is being explained
        :param explanation_details: Computed explanation object
        :param explainer: Explainer component
        :param feature_type: Feature type
        :return: Report of the computed explanation
        """
        filtered_features = LimeFMExplainerReporter.extractFeatures(explanation_details.local_exp[0],
                                                                    feature_type=feature_type,
                                                                    feature_map=explainer.feature_map)
        explanation_str = json.dumps(filtered_features)
        explanation_results_df = LimeFMExplainerReporter.createAsDataFrame(explanation_details, explanation_str, instance)
        explanation_results_json = LimeFMExplainerReporter.createAsJsonObject(explanation_details, filtered_features, instance)
        return explanation_results_df, explanation_results_json

    @staticmethod
    def createAsDataFrame(
            explanation_object: explanation.Explanation,
            explanation_str: str,
            instance: pd.DataFrame) -> pd.DataFrame:
        """
        Creates explanation result as a dataframe table
        :param explanation_object: Explanation
        :param explanation_str: Explanation labels
        :param instance: Local instance being explained
        :return: Explanation result
        """
        explanation_results_df = pd.DataFrame(
            {c2.MOVIELENS_USER_ID_COLUMN_KEY: [instance[c2.MOVIELENS_USER_ID_COLUMN_KEY]],
             c2.MOVIELENS_ITEM_ID_COLUMN_KEY: [instance[c2.MOVIELENS_ITEM_ID_COLUMN_KEY]],
             'explanations': [explanation_str],
             'local_prediction': [round(explanation_object.local_pred[0], 3)],
             'predicted_value': [round(explanation_object.predicted_value, 3)],
             'intercept': [round(explanation_object.intercept[0], 3)],
             'R2_Score': [round(explanation_object.score, 3)]
             })
        return explanation_results_df

    @staticmethod
    def createAsJsonObject(
            explanation_object: explanation.Explanation,
            explanation_json: Dict[int, float],
            instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Creates explanation result as a dataframe table
        :param explanation_object: Explanation
        :param explanation_json: Explanation labels
        :param instance: Local instance being explained
        :return: Explanation result
        """
        explanation_results_json = {
            c2.MOVIELENS_USER_ID_COLUMN_KEY: [instance[c2.MOVIELENS_USER_ID_COLUMN_KEY].tolist()[0]],
            c2.MOVIELENS_ITEM_ID_COLUMN_KEY: [instance[c2.MOVIELENS_ITEM_ID_COLUMN_KEY].tolist()[0]],
            'explanations': {k: [v] for k, v in explanation_json.items()},
            'local_prediction': round(explanation_object.local_pred[0], 3),
            'predicted_value': round(explanation_object.predicted_value, 3),
            'R2_Score': round(explanation_object.score, 3),
            'intercept': round(explanation_object.intercept[0], 3)
        }
        new_explanation_results = {}
        for k, v in explanation_results_json.items():
            if k is 'explanations':
                for k2, v2 in explanation_results_json['explanations'].items():
                    new_explanation_results[k2] = v2
            else:
                new_explanation_results[k] = v
        return new_explanation_results

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
                if feature_map[tup[0]].startswith(c2.MOVIELENS_USER_ID_COLUMN_KEY) and len(
                        filtered_dict) <= top_features:
                    filtered_dict[feature_map[tup[0]]] = round(tup[1], 3)
        return filtered_dict
