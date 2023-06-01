import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from typing import List, Tuple

import Code.DataAccessLayer.Src.Constants as c

class ExplainerVisualizer(object):
    """
    Explainer Visualizer component i.e. used to visualize the recommendation of model explainability
    """

    @staticmethod
    def visualizeLimeFMExplainability(
            explanations_df: pd.DataFrame,
            title="Movielens Lime Explainability"):
        """
        Visualization of FM model LIME explainability
        :param explanations_df: Explanations
        :param title: Title
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
    def visualizeLimeFMExplainabilityPerUserId(
            explanations_df: pd.DataFrame,
            user_id: int,
            item_id: int,
            movie_title: str="Star Wars",
            title="Movielens Lime Explainability"):
        """
        Visualization of FM model LIME explainability
        :param explanations_df: Explanations
        :param user_id: User ID
        :param item_id: Item ID
        :param movie_title: Movie title
        :param title: Title
        """
        filter_1 = explanations_df[c.MOVIELENS_USER_ID_COLUMN_KEY] == user_id
        filter_2 = explanations_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY] == item_id
        filtered_explanation_df = explanations_df[(filter_1 & filter_2)]
        genre_side_info_explanations_df = filtered_explanation_df[c.MOVIELENS_GENRE_COLUMNS]
        explanations_as_dict = genre_side_info_explanations_df.to_dict('records')[0]
        feature_names = list(explanations_as_dict.keys())
        coef_values = list(explanations_as_dict.values())
        new_feature_names, new_coef_values = ExplainerVisualizer.filterNonZeroCoefficents(feature_names, coef_values)
        pos = np.arange(len(new_coef_values)) + .5
        colors = ['green' if x > 0 else 'red' for x in new_coef_values]
        plt.barh(pos, new_coef_values, align='center', color=colors)
        plt.yticks(pos, new_feature_names, rotation=45)
        new_title = f"{title} for user ID={user_id} and Movie title: {movie_title}"
        plt.title(new_title)
        plt.show()

    @staticmethod
    def filterNonZeroCoefficents(
            feature_names: List[str],
            coefficient_values: List[float]
    ) -> Tuple[List[str], List[float]]:
        """
        Filters the non-zero coefficients
        :param feature_names: Feature names
        :param coefficient_values: Coefficient values
        :return: Modified feature_names and coefficient_values
        """
        coefficient_values_array = np.array(coefficient_values)
        feature_names_array = np.array(feature_names)
        non_zero_index = np.where(np.array(coefficient_values) != 0.0)[0]
        new_coefficient_values = coefficient_values_array[non_zero_index].tolist()
        new_feature_names = feature_names_array[non_zero_index].tolist()
        return new_feature_names, new_coefficient_values