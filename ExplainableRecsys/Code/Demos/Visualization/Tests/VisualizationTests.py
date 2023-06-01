import unittest as ut
import inspect
import os
import pathlib
import pandas as pd
from typing import Optional, Tuple, Any, Dict

working_dir = pathlib.Path(os.getcwd()).parent.parent.parent
os.chdir(working_dir)
print(f"Working folder is: {working_dir}")

from Code.Demos.Visualization.Src.ExplainabilityVisualization import ExplainerVisualizer
from Code.Demos.Models.Src.ModelBasedExplanation import ModelBasedExplainerDemo
from Code.Utils.Src.Enums import ExplanationType
from Code.Utils.Src.Utils import Helpers
import Code.Model.Src.Constants as c


class ExplainerVisualizerTest(ut.TestCase):
    """
    Integration test suit for 'Explainer Visualization
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        """
        print(f"Working folder is: {working_dir}")

    def test_ExplainerVisualizer_Visualize_LimeFM_Explainability_Per_UserId_Is_valid(self):
        """
        Test the validity of the FM-Lime explanation visualization
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        fm_best_explainer = Helpers.readPickleFromFile(c.FM_BEST_LIME_EXPLANATION_MODEL_PATH)
        self.assertIsNotNone(fm_best_explainer.explanations_df, error_msg)
        user_id, item_id = 0, 365
        ExplainerVisualizer.visualizeLimeFMExplainabilityPerUserId(
            fm_best_explainer.explanations_df,
            user_id=user_id,
            item_id=item_id
        )


if __name__ == '__main__':
    ut.main()
