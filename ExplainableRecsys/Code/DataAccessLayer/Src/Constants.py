import time

# Movilens 100K path configs
MOVIELENS_100K_DATA_ROOT_PATH = "ThirdParty/recoxplainer_master/datasets/ml-100k"
MOVIELENS_100K_MODEL_OUTPUT_PATH = f"{MOVIELENS_100K_DATA_ROOT_PATH}/output"
MOVIELENS_RATING_PATH = f"{MOVIELENS_100K_DATA_ROOT_PATH}/u.data"
MOVIELENS_ITEM_PATH = f"{MOVIELENS_100K_DATA_ROOT_PATH}/u.item"
MOVIELENS_USER_PATH = f"{MOVIELENS_100K_DATA_ROOT_PATH}/u.user"
MOVIELENS_GENRE_PATH = f"{MOVIELENS_100K_DATA_ROOT_PATH}/u.genre"
MOVIELENS_100K_MODEL_FILENAME = f""

# Movilens 100k column configs
MOVIELENS_RATING_COLUMNS = ['userId', 'itemId', 'rating', 'timestamp']
MOVIELENS_USER_COLUMNS = ['userId', 'age', 'gender', 'occupation', 'zip code']
MOVIELENS_GENRE_COLUMNS = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
MOVIELENS_ITEM_COLUMNS = ['itemId', 'movie_title','release_date',
                          'video release date', 'IMDb URL'] + MOVIELENS_GENRE_COLUMNS
MOVIELENS_USER_ITEM_INTERACTION_COLUMNS = ['userId', 'itemId']
MOVIELENS_USER_ITEM_INTERACTION_WITH_RATING_COLUMNS = ['userId', 'itemId', 'rating']

MOVIELENS_USER_ID_COLUMN_KEY = "userId"
MOVIELENS_ITEM_ID_COLUMN_KEY = "itemId"
MOVIELENS_USER_ID_SEQUENTIAL_COLUMN_KEY = "user_id"
MOVIELENS_ITEM_ID_SEQUENTIAL_COLUMN_KEY = "item_Id"
MOVIELENS_RATING_COLUMN_KEY = "rating"
MOVIELENS_RATING_PREDICTION_COLUMN_KEY = "prediction"
MOVIELENS_RANK_COLUMN_KEY = "rank"
MOVIELENS_RECOMMENDATION_COLUMNS = [
    MOVIELENS_USER_ID_COLUMN_KEY,
    MOVIELENS_ITEM_ID_COLUMN_KEY,
    MOVIELENS_RANK_COLUMN_KEY]



# Movielens feature columns for different scenarios
MOVIELENS_FEATURE_COLUMNS_TYPE_0 = ['userId', 'itemId']
MOVIELENS_FEATURE_COLUMNS_TYPE_1 = ['userId', 'itemId', 'age', 'gender', 'occupation']
MOVIELENS_FEATURE_COLUMNS_TYPE_2 = ['userId', 'itemId', 'genre']
MOVIELENS_FEATURE_COLUMNS_TYPE_3 = ['userId', 'itemId'] + MOVIELENS_GENRE_COLUMNS
MOVIELENS_FEATURE_COLUMNS_TYPE_4 = ['userId', 'itemId', 'age', 'gender', 'occupation', 'genre']
MOVIELENS_FEATURE_COLUMNS_TYPE_5 = ['userId', 'itemId', 'age', 'gender', 'occupation'] + MOVIELENS_GENRE_COLUMNS

# Movielens miscellaneous file configs
MOVIELENS_RATING_FILE_DELIMITER = "\t"
MOVIELENS_USER_FILE_DELIMITER = "|"
MOVIELENS_USERS_FILE_ENCODING = "latin-1"
MOVIELENS_GENRE_USE_COLUMN_INDEX = 0

# Movielens pre-processing config
IS_CONSOLIDATE_GENRE_VALUES = False
MOVIELENS_TEST_DATA_SPLIT_FRACTION = 0.2




