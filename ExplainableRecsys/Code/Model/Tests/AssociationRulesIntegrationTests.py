import unittest as ut
import inspect
import os
import pathlib

working_dir = pathlib.Path(os.getcwd()).parent.parent
os.chdir(working_dir)

from Code.ThirdParty.recoxplainer_master.recoxplainer.config import cfg
from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.Model.Src.AssociationRules import AssociationRulesModel

class MyTestCase(ut.TestCase):
    """
    Integration test suit for Association Rules Model
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")

    def test_AssociationRules_Constructor_Is_Valid(self):
        """
        Test the validity of the AssociationRules constructor
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        model = AssociationRulesModel()
        self.assertIsNotNone(model, msg=error_msg)

    def test_AssociationRules_Fit_Is_Valid(self):
        """
        Test the validity of the AssociationRules fit() method
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        data = DataReader(**cfg.ml100k)
        model = AssociationRulesModel()
        self.assertIsNotNone(data, msg=error_msg)
        self.assertIsNotNone(model, msg=error_msg)
        model.fit(data)
        self.assertIsNotNone(model.rules, msg=error_msg)





if __name__ == '__main__':
    ut.main()
