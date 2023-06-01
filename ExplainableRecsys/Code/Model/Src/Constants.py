
# Cache paths configs
TEMP_FOLDER = "C:/Temp"
CACHE_RECOMMENDATIONS_PATH = f"{TEMP_FOLDER}/cache_recommendations.csv"
CACHE_TEST_PARTITION_DATA_PATH = f"{TEMP_FOLDER}/cache_test_data.csv"
MODEL_DATA_ROOT_FOLDER = "Model/Data"
MF_MODEL_CACHE_PATH = f"{MODEL_DATA_ROOT_FOLDER}/MF"
MF_PERFORMANCE_PLOT_PATH = f"{MODEL_DATA_ROOT_FOLDER}/performance-plots/MF_valid_performance.png"
MF_RECOMMENDATIONS_PATH = f"{MODEL_DATA_ROOT_FOLDER}/recommendations"
MF_TRAIN_RECOMMENDATIONS_FILENAME = "cache_train_recommendations.csv"

# Best recommendation, explanation models/results
BEST_RESULTS_FOLDER = f"{MODEL_DATA_ROOT_FOLDER}/best_results"

MF_BEST_TRAIN_RECOMMENDER_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_model.pl"
MF_BEST_RECOMMENDATIONS_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_recommendations.csv"

MF_BEST_AR_TRAIN_RECOMMENDER_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_ar_model.pl"
MF_BEST_AR_RECOMMENDATIONS_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_ar_recommendations.csv"
MF_BEST_AR_EXPLANATION_MODEL_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_ar_explanation_model.pl"
MF_BEST_AR_EXPLANATIONS_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_ar_recommendations.csv"

MF_BEST_KNN_TRAIN_RECOMMENDER_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_knn_model.pl"
MF_BEST_KNN_RECOMMENDATIONS_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_knn_recommendations.csv"
MF_BEST_KNN_EXPLANATION_MODEL_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_knn_explanation_model.pl"
MF_BEST_KNN_EXPLANATIONS_PATH = f"{BEST_RESULTS_FOLDER}/mf_best_knn_recommendations.csv"

FM_BEST_TRAIN_RECOMMENDER_PATH = f"{BEST_RESULTS_FOLDER}/fm_best_model.pl"
FM_BEST_RECOMMENDATIONS_PATH = f"{BEST_RESULTS_FOLDER}/fm_best_recommendations.csv"
FM_BEST_LIME_EXPLANATION_MODEL_PATH = f"{BEST_RESULTS_FOLDER}/fm_best_lime_explanation_model.pl"
FM_BEST_EXPLANATIONS_PATH = f"{BEST_RESULTS_FOLDER}/fm_best_explanations.csv"

ALS_BEST_TRAIN_RECOMMENDER_PATH = f"{BEST_RESULTS_FOLDER}/als_best_model.pl"
ALS_BEST_RECOMMENDATIONS_PATH = f"{BEST_RESULTS_FOLDER}/als_best_recommendations.csv"
ALS_BEST_EXPLANATION_MODEL_PATH = f"{BEST_RESULTS_FOLDER}/als_best_lime_explanation_model.pl"
ALS_BEST_EXPLANATIONS_PATH = f"{BEST_RESULTS_FOLDER}/als_best_explanations.csv"

EMF_BEST_TRAIN_RECOMMENDER_PATH = f"{BEST_RESULTS_FOLDER}/emf_best_model.pl"
EMF_BEST_RECOMMENDATIONS_PATH = f"{BEST_RESULTS_FOLDER}/emf_best_recommendations.csv"
EMF_BEST_EXPLANATION_MODEL_PATH = f"{BEST_RESULTS_FOLDER}/emf_best_lime_explanation_model.pl"
EMF_BEST_EXPLANATIONS_PATH = f"{BEST_RESULTS_FOLDER}/emf_best_explanations.csv"



# Factorization Machine configs
FM_MOVILENS_SAMPLE_DATA_PATH = "ThirdParty/recoxplainer_master/datasets/ml-fm/movielens_sample.txt"
FM_MOVIELENS_COLUMNS = ["movie_id", "user_id",
                           "gender", "age", "occupation", "zip"]
FM_MOVIELENS_RATING_COLUMN = ['rating']

# Factorization model parameters
FM_PARAMS_RANK_KEY = "rank"
FM_PARAMS_SEED_KEY = "random_seed"
FM_PARAM_N_ITER_KEY = "n_iter"
FM_PARAMS_N_KEPT_SAMPLES_KEY = "n_kept_samples"
FM_PARAMS_RANK_VALUE = 10
FM_PARAMS_SEED_VALUE = 100
FM_PARAM_N_ITER_VALUE = 200
FM_PARAMS_N_KEPT_SAMPLES_VALUE = 200
FM_PREDICT_PARTITION_SIZE = 10

# Local sample explanation configs
N_LOCAL_SAMPLES = 2000

# LIME regression prediction (R2 Score) threshold
R2_SCORE_THRESHOLD = 0.2