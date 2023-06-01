import unittest as ut
import inspect
import os
import pathlib
import pandas as pd

working_dir = pathlib.Path(os.getcwd()).parent.parent
os.chdir(working_dir)

from Code.Utils.Src.Utils import Helpers

class TestHelpers(ut.TestCase):
    """
    Unit test suit for 'Helpers' component
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")

    def test_Helpers_Tableize_Is_Valid(self):
        """
        Test the validity of tablelizing a pandas dataframe for visualizing
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        iterator_value = range(10)
        table_df = pd.DataFrame(
            {
                "col1": list(iterator_value),
                "col2": [f"value_{x}" for x in iterator_value]
            }
        )
        table_str = Helpers.tableize(table_df)
        self.assertIsNotNone(table_str, msg=error_msg)
        print("table_df visualization is:")
        print(table_str)

    def test_Helpers_Create_New_Model_Output_File_Is_Valid(self):
        """
        Test the validity of creating a new model output file
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        new_model_output_file = Helpers.createNewModelOutputFile()
        self.assertIsNotNone(new_model_output_file, msg=error_msg)
        print(f"New model output file is: {new_model_output_file}")

    def test_Helpers_Create_GMT_Timestamp_Is_Valid(self):
        """
        Test the validity of creating GMT timestamp
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        timestamp = Helpers.createGmtTimestamp()
        self.assertIsNotNone(timestamp, msg=error_msg)
        print(f"New model output file is: {timestamp}")

    def test_Helpers_Write_Pickle_Model_Dump_Is_Valid(self):
        """
        Test the validity of serializing an object to a pickle
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        mock_model = {"test": "test2"}
        Helpers.writePickleModelDump(mock_model)
        self.assertTrue(True, msg=error_msg)

    def test_Helpers_Read_Pickle_Model_Dump_Is_Valid(self):
        """
        Test the validity of deserializing an object to a pickle
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        mock_model = Helpers.readPickleModelDump()
        self.assertIsNotNone(mock_model, msg=error_msg)
        print(f"Mock model: {mock_model}")

    def test_Helpers_Get_Latest_Model_Output_File_Is_Valid(self):
        """
        Test the validity of getting the latest model output file
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        model_output_file = Helpers.getLatestModelOutputFile()
        self.assertIsNotNone(model_output_file, msg=error_msg)
        print(f"Latest model output file: {model_output_file}")



if __name__ == '__main__':
    ut.main()
