import os
import pathlib
from _datetime import datetime

PROJECT_ROOT_PATH = pathlib.Path(__file__).parent.parent
CONFIG_FILE_PATH = "./Code/Configs/Config.yml"
TEST_DATA_PATH = "./Datasets/fake_data.csv"
MOVIELENS_100K = "./Datasets/ml-100k/ratings.dat"

# Logging configs
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - : %(message)s in %(pathname)s:%(lineno)d'
CURRENT_DATE = datetime.today().strftime("%d%b%Y")
COMPAIRA_LOG_FILE = f"ExplainableRecsys_{CURRENT_DATE}.log"
MOVIELENS_100K_LOGGING_PATH = f"Logs/{COMPAIRA_LOG_FILE}"

# Console configs
LINE_DIVIDER = "==========" * 5


