import numpy as np
import pandas as pd
import glob
import os
import pickle
import calendar
import time
from typing import Optional, Any
from pprint import pprint

import Code.DataAccessLayer.Src.Constants as c
import Code.Utils.Src.Logging as lo

logger = lo.Logger.getLogger()

class Helpers(object):
    """
    Helper utility functions
    """

    @staticmethod
    def tableize(df) -> Optional[str]:
        """
        Tabulate the dataframe
        :param df: Input dataframe
        :return: Output pretty display of dataframe
        """
        if not isinstance(df, pd.DataFrame):
            return None
        df_columns = df.columns.tolist()
        max_len_in_lst = lambda lst: len(sorted(lst, reverse=True, key=len)[0])
        align_center = lambda st, sz: "{0}{1}{0}".format(" " * (1 + (sz - len(st)) // 2), st)[:sz] if len(
            st) < sz else st
        align_right = lambda st, sz: "{0}{1} ".format(" " * (sz - len(st) - 1), st) if len(st) < sz else st
        max_col_len = max_len_in_lst(df_columns)
        max_val_len_for_col = dict(
            [(col, max_len_in_lst(df.iloc[:, idx].astype('str'))) for idx, col in enumerate(df_columns)])
        col_sizes = dict([(col, 2 + max(max_val_len_for_col.get(col, 0), max_col_len)) for col in df_columns])
        build_hline = lambda row: '+'.join(['-' * col_sizes[col] for col in row]).join(['+', '+'])
        build_data = lambda row, align: "|".join(
            [align(str(val), col_sizes[df_columns[idx]]) for idx, val in enumerate(row)]).join(['|', '|'])
        hline = build_hline(df_columns)
        out = [hline, build_data(df_columns, align_center), hline]
        for _, row in df.iterrows():
            out.append(build_data(row.tolist(), align_right))
        out.append(hline)
        return "\n".join(out)

    @staticmethod
    def writePickleModelDump(model: Any, model_folder_path: str = c.MOVIELENS_100K_MODEL_OUTPUT_PATH):
        """
        Writes/serializes the recommendation model to disc
        :param model_folder_path: Model folder path
        :return: None
        """
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # save
        new_model_output_file_path = Helpers.createNewModelOutputFile(model_folder_path)
        Helpers.writePickleToFile(model, new_model_output_file_path)

    @staticmethod
    def writePickleToFile(object: Any, pickle_output_file_path: str):
        """
        Writes a serialiseable  object to file as pickle blob
        :param object: Object being pickled
        :param pickle_output_file_path: Pickle file path
        """
        with open(os.path.join(pickle_output_file_path), "wb") as output:
            pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def writeRecommendationsOrExplanationsToCache(
            recommendations_df: pd.DataFrame,
            recommendations_path: str
    ):
        """
        Writes (caches) recommendations to disc
        :param recommendations_df: Recommendations
        """
        recommendations_df.to_csv(recommendations_path, index=False)

    @staticmethod
    def readPickleModelDump() -> Any:
        """
        Reads the pickled recommendation model
        :return: Deserialized recommendation model
        """
        output_model_path = Helpers.getLatestModelOutputFile()
        return Helpers.readPickleFromFile(output_model_path)

    @staticmethod
    def readPickleFromFile(output_model_path: str):
        """
        Writes a serializeable  object to file as pickle blob
        :param output_model_path: Pickle file path
        """
        with open(output_model_path, "rb") as input_file:
            rec_model = pickle.load(input_file)
        return rec_model

    @staticmethod
    def createGmtTimestamp() -> int:
        """
        Create a timestamp based on the current GMT format datetime
        :return: Timestamp
        """
        # Current GMT time in a tuple format
        current_GMT = time.gmtime()

        # ts stores timestamp
        timestamp = calendar.timegm(current_GMT)
        return timestamp

    @staticmethod
    def getLatestModelOutputFile(model_folder_path: str = c.MOVIELENS_100K_MODEL_OUTPUT_PATH):
        """
        Get the latest Model output file.
        If one exist it will return it.
        If it does not exist, it will create a new one
        :param model_folder_path: Path of the models sub-folders
        :return: Model output filename
        """
        if not os.path.exists(model_folder_path):
            error_msg = f"The folder: {model_folder_path} does not exist.\t" \
                        f"Missing Model output directory!!"
            logger.error(error_msg)
            raise Exception(error_msg)
        model_files = glob.glob(f"{model_folder_path}/*")
        if len(model_files) > 0:
            latest_model_file = max(model_files, key=os.path.getctime)
        else:
            latest_model_file = Helpers.createNewModelOutputFile(model_folder_path)
        return latest_model_file

    @staticmethod
    def createNewModelOutputFile(model_folder_path: str = c.MOVIELENS_100K_MODEL_OUTPUT_PATH) -> str:
        """
        Creates a new recommendation model output file
        :return: New model output filename
        """
        timestamp = Helpers.createGmtTimestamp()
        latest_model_file = f"{model_folder_path}/model_{timestamp}.pl"
        return latest_model_file

    @staticmethod
    def createConsoleDivider():
        """
        Creates a divider for console display of results
        """
        print("==========" * 5)
        print("\n\n")

    @staticmethod
    def prettifyExplanations(
            explanations_df: pd.DataFrame,
            top_k: int = 10
    ) -> pd.DataFrame:
        """
        Make the presentation of the computed explanations better
        :param explanations_df: Explanations
        :param top_k: Top K explanations per item
        :return: Improved explanations
        """
        improved_explanations_df = explanations_df.copy()
        explanations = improved_explanations_df.explanations.tolist()
        improved_explanations = [list(zip(x['item'].values, x['contribution'].values.round(3)))[:top_k]
                                 for x in explanations]
        improved_explanations_df["explanations"] = improved_explanations
        return improved_explanations_df

    @staticmethod
    def reportDemoResults(
            self,
            explanation_metrics,
            explanations_df,
            recommendation_metrics,
            recommendations_df):
        """
        Reports the demo results
        :param explanation_metrics: Explanation metrics
        :param explanations_df: Explanations
        :param recommendation_metrics: Recommendation metrics
        :param recommendations_df: recommendations
        """
        print("Sample of generated recommendations:")
        print(Helpers.tableize(recommendations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Recommendation metrics:")
        pprint(recommendation_metrics)
        Helpers.createConsoleDivider()
        print("Sample of generated explanations:")
        print(Helpers.tableize(explanations_df.head(10)))
        Helpers.createConsoleDivider()
        print("Explanation metrics:")
        pprint(explanation_metrics)
        Helpers.createConsoleDivider()






