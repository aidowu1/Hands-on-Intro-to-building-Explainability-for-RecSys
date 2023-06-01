import pandas as pd
from typing import List
from tqdm import tqdm

from Code.Recommender.Src.Constants import (ASSOCIATION_RULE_USER_COL,
                                            ASSOCIATION_RULE_ANTECEDENTS_COL,
                                            ASSOCIATION_RULE_COL_RENAME_MAP,
                                            RATING_MATRIX_ITEM_ID_COL
                                            )

class AssociationRulesRecommender(object):

    def __init__(self, dataset_metadata, model, top_n: int = 10):
        self._dataset = dataset_metadata.dataset
        self._rules_df = model.rules
        self._metric = model.metric

    def computeRecommendationsFromRules(self,
                                        user_id: int,
                                        rated_items_per_user: List[int],
                                        top_n: int = 10
                                        ) -> pd.DataFrame:
        """
        Gets recommendations from association rules
        i.e. for each rated item by the user (antecedent) recommended items (consequent) are computed
        :param user_id: User ID
        :param rated_items_per_user: previously rated_items by the user
        :param top_n: Top N recommendations
        :return: Recommendations
        """
        recommendations = []
        for item in rated_items_per_user:
            recommendations.append(self.computeRecommendationsPerItemFromRules(user_id=user_id, rated_item=item))
        recommendations_df = pd.concat(recommendations)
        sorted_recommendations_df = recommendations_df.sort_values(self._metric, ascending=False)
        sorted_recommendations_df.drop([self._metric, ASSOCIATION_RULE_ANTECEDENTS_COL], axis=1, inplace=True)
        sorted_recommendations_df = sorted_recommendations_df.assign(row_number=range(len(recommendations_df)))
        sorted_recommendations_df.rename(columns=ASSOCIATION_RULE_COL_RENAME_MAP, inplace=True)
        return sorted_recommendations_df.head(top_n)

    def computeRecommendationsPerItemFromRules(self,
                                               user_id: int,
                                               rated_item: int
                                               ) -> pd.DataFrame:
        """
        Computes the association rules based recommendation per item
        :param user_id: User ID
        :param rated_item: Previously rated item by the user
        :return: Recommendations
        """
        self._rules_df[ASSOCIATION_RULE_USER_COL] = user_id
        filter_1 = self._rules_df.antecedents == rated_item
        recommendations_df = self._rules_df[filter_1]
        return recommendations_df

    def getAllRecomendations(self, top_n: int = 10) -> pd.DataFrame:
        """
        Computes the top n recommendations for each user ID
        :param top_n: Top N recommendations per user ID
        """
        ratings = self._dataset.groupby('userId')
        recommendations_df = pd.DataFrame({'userId': [], 'itemId': [], 'rank': []})

        with tqdm(total=self._dataset['userId'].nunique(), desc="Recommending for users: ") as pbar:
            for user_id, user_ratings in ratings:
                rated_items_per_user = user_ratings[RATING_MATRIX_ITEM_ID_COL]
                recommendations_df = recommendations_df \
                    .append(self.computeRecommendationsFromRules(user_id, rated_items_per_user, top_n=top_n))
                pbar.update()
        return recommendations_df


