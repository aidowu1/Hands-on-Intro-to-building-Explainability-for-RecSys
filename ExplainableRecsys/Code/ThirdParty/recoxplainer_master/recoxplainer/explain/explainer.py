from tqdm.auto import tqdm
from typing import Tuple, Set, List


class Explainer:
    def __init__(self, model, recommendations, data):
        self.model = model
        self.recommendations = recommendations
        self.dataset = data.dataset
        self.num_items = data.num_item
        self.num_users = data.num_user
        self.users = self.dataset.groupby(by='userId')

    def explain_recommendations(self):

        explanations = []
        explanation_score = []

        with tqdm(total=self.recommendations.shape[0], desc="Computing explanations: ") as pbar:

            for _, row in self.recommendations.iterrows():
                explanation_per_user, explanation_score_per_user = self.explain_recommendation_to_user(
                                                                        int(row.userId),
                                                                        int(row.itemId))
                explanations.append(explanation_per_user)
                explanation_score.append(explanation_score_per_user)
                pbar.update()

        self.recommendations['explanations'] = explanations
        self.recommendations['explanation_score'] = explanation_score
        return self.recommendations

    def get_user_items(self, user_id):
        """
        Items Ids rated by a user.
        :param user_id: the user
        :return: list
        """
        return self.users.get_group(user_id).itemId.values

    def explain_recommendation_to_user(self, user_id: int, item_id: int) -> Tuple[Set[int], float]:
        pass