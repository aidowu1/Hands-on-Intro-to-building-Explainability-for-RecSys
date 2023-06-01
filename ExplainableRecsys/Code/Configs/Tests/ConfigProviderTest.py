import unittest as ut 
from inspect import stack
from os import getcwd, chdir
from pprint import pprint

from Code.Constants import PROJECT_ROOT_PATH

chdir(PROJECT_ROOT_PATH)

class TestConfigProvider(ut.TestCase):
    """
    Test suit for Configuration Provider
    """
    def setUp(self) -> None:
        """
        Setup test fixture
        """
        from Code.Configs.Src import ConfigProvider as cp
        self.config = cp.cfg
        print(f"Current directory: {getcwd()}")

    def test_ConfigProvider_Get_Test_Data_Configs_Is_Valid(self):
        """
        Test the validity of the ConfigProvider getting 'testdata' configs
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        test_data_config = self.config.testdata
        self.assertIsNotNone(test_data_config, msg=error_msg)
        self.assertTrue(isinstance(test_data_config, dict), msg=error_msg)
        print("Test data configs: ")
        pprint(test_data_config)

    def test_ConfigProvider_Get_100K_Movielens_Data_Configs_Is_Valid(self):
        """
        Test the validity of the ConfigProvider getting 'ml100k' configs
        """
        error_msg = f"Invalid tests: Error testing function: {stack()[0][3]}()"
        ml_data_config = self.config.ml100k
        self.assertIsNotNone(ml_data_config, msg=error_msg)
        self.assertTrue(isinstance(ml_data_config, dict), msg=error_msg)
        print("Movielens 100K data configs: ")
        pprint(ml_data_config)
        

    if __name__ == "__main__":
        ut.main()

