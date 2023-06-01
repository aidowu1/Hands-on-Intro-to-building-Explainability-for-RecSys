import unittest as ut
import inspect
import os
import pathlib

working_dir = pathlib.Path(os.getcwd()).parent.parent
os.chdir(working_dir)

from Code.ThirdParty.recoxplainer_master.recoxplainer.config import cfg
from Code.Recommender.Src.AssociationRulesRecommender import AssociationRulesRecommender
from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.Model.Src.AssociationRules import AssociationRulesModel
from Code.Recommender.Src.Constants import RATING_MATRIX_USER_ID_COL, RATING_MATRIX_ITEM_ID_COL


class TestAssociationRulesRecommender(ut.TestCase):
    """
    Integration test suit for Association Rules Recommender
    """
    def test_AssociationRulesRecommender_Constructor_Is_Valid(self):
        """
        Test the validity of AssociationRulesRecommender constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        data = DataReader(**cfg.ml100k)
        model = AssociationRulesModel()
        recommender = AssociationRulesRecommender(dataset_metadata=data, model=model)
        self.assertIsNotNone(recommender,msg=error_msg)

    def test_AssociationRulesRecommender_computeRecommendationsPerItemFromRules_Is_Valid(self):
        """
        Test the validity of AssociationRulesRecommender compute recommendation per item from
        association rules
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        data = DataReader(**cfg.ml100k)
        model = AssociationRulesModel()
        model.fit(data)
        recommender = AssociationRulesRecommender(dataset_metadata=data, model=model)
        self.assertIsNotNone(recommender, msg=error_msg)
        recommendations_df = recommender.computeRecommendationsPerItemFromRules(user_id=1, rated_item=1)
        self.assertIsNotNone(recommendations_df, msg=error_msg)
        print(f"recommendations_df:\n\n{recommendations_df.head()}")

    def test_AssociationRulesRecommender_computeRecommendationsFromRules_Is_Valid(self):
        """
        Test the validity of AssociationRulesRecommender compute recommendation for all items from
        association rules
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        data = DataReader(**cfg.ml100k)
        data_df = data.dataset
        user_id = 1
        filter_1 = data_df[RATING_MATRIX_USER_ID_COL] == user_id
        item_ids = data_df[filter_1][RATING_MATRIX_ITEM_ID_COL].tolist()
        model = AssociationRulesModel()
        model.fit(data)
        recommender = AssociationRulesRecommender(dataset_metadata=data, model=model)
        self.assertIsNotNone(recommender, msg=error_msg)
        recommendations_df = recommender.computeRecommendationsFromRules(
            user_id=user_id, rated_items_per_user=item_ids)
        self.assertIsNotNone(recommendations_df, msg=error_msg)
        print(f"recommendations_df:\n\n{recommendations_df.head()}")

    def test_AssociationRulesRecommender_getAllRecomendations_Is_Valid(self):
        """
        Test the validity of AssociationRulesRecommender compute all recommendation for all users via
        association rules
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        data = DataReader(**cfg.ml100k)
        model = AssociationRulesModel()
        model.fit(data)
        recommender = AssociationRulesRecommender(dataset_metadata=data, model=model)
        self.assertIsNotNone(recommender, msg=error_msg)
        recommendations_df = recommender.getAllRecomendations(top_n=10)
        self.assertIsNotNone(recommendations_df, msg=error_msg)
        print(f"recommendations_df:\n\n{recommendations_df.head()}")



if __name__ == '__main__':
    ut.main()
