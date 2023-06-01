import json
import os
import numpy as np
import pandas as pd
from pprint import pprint
import json
import matplotlib.pyplot as plt

from Code.ThirdParty.lime_rs.experiment import utils
from Code.ThirdParty.lime_rs.src import data_utils
from Code.ThirdParty.lime_rs.src.lime_rs import LimeRSExplainer

logger = utils.get_logger("limers")


def extract_features(explanation_all_ids, feature_type, feature_map):
    filtered_dict = dict()
    if feature_type == "features":
        for tup in explanation_all_ids:
            if not (feature_map[tup[0]].startswith('user_id') or
                    feature_map[tup[0]].startswith('item_id')):
                filtered_dict[feature_map[tup[0]]] = round(tup[1], 3)

    elif feature_type == "item":
        top_features = 500
        for tup in explanation_all_ids:
            if feature_map[tup[0]].startswith('item_id') and len(filtered_dict) <= top_features:
                filtered_dict[feature_map[tup[0]]] = round(tup[1], 3)

    return filtered_dict


def generate_explanations(
        instances_to_explain,
        explainer,
        rec_model,
        feature_type='features',
        is_use_item_side_info=False
):
    result = list()

    for instance in instances_to_explain.itertuples(index=False):
        logger.info("explaining-> (user: {}, item: {})".format(instance.user_id, instance.item_id))

        exp = explainer.explain_instance(instance,
                                         rec_model,
                                         neighborhood_entity="item",
                                         labels=[0],
                                         num_samples=1000,
                                         num_features=10
                                         )

        # filter
        filtered_features = extract_features(exp.local_exp[0],
                                             feature_type=feature_type,
                                             feature_map=explainer.feature_map)
        #
        explanation_str = json.dumps(filtered_features)
        output_df = pd.DataFrame({'user_id': [instance.user_id], 'item_id': [instance.item_id],
                                  'explanations': [explanation_str],
                                  'local_prediction': [round(exp.local_pred[0], 3)]})

        result.append(output_df)

    return pd.concat(result), exp

def readExplainerResults():
    """

    """
    exp_setup = utils.setup()
    output_filename = "limers_explanations-{}".format(exp_setup.rec_name)
    explainer_results_path = f"{data_utils.DEFAULT_OUTPUT_FOLDER}/{output_filename}"
    explanations_df = pd.read_csv(explainer_results_path, delimiter="\t")
    feature_coefs = json.loads(explanations_df.explanations.iloc[0])
    return explanations_df

def plotExplainerResults(explanations_df, title="Movielens Lime Explainability"):
    lime_feature_coefs = json.loads(explanations_df.explanations.iloc[0])
    feature_names = list(lime_feature_coefs.keys())
    feature_names.reverse()
    coef_values = list(lime_feature_coefs.values())
    coef_values.reverse()
    pos = np.arange(len(coef_values)) + .5
    colors = ['green' if x > 0 else 'red' for x in coef_values]
    plt.barh(pos, coef_values, align='center', color=colors)
    plt.yticks(pos, feature_names, rotation=45)
    plt.title(title)
    plt.show()

def visualizeLimeExplainerResults():
    explanations_df = readExplainerResults()
    plotExplainerResults(explanations_df)

def main():
    #runLimeExplainer()
    visualizeLimeExplainerResults()


def runLimeExplainer():
    # setup
    exp_setup = utils.setup()
    # load data and rec model
    logger.info("Load data and recommender")
    rec_model = data_utils.load_dump(exp_setup.rec_name)
    # setup explainer
    feature_names = rec_model.one_hot_columns
    feature_map = {i: rec_model.one_hot_columns[i] for i in range(len(list(rec_model.one_hot_columns)))}
    explainer = LimeRSExplainer(rec_model.dataset.training_df,
                                feature_names=feature_names,
                                feature_map=feature_map,
                                mode='regression',
                                class_names=np.array(['rec']),
                                feature_selection='none')
    #
    instances_to_explain = pd.DataFrame([("1", "5")], columns=["user_id", "item_id"])
    explanations, exp = generate_explanations(
        instances_to_explain,
        explainer,
        rec_model,
        feature_type="features"
    )
    # save
    logger.info("Save LimeRS explanations")
    output_filename = "limers_explanations-{}".format(exp_setup.rec_name)
    explanations.to_csv(path_or_buf=os.path.join(data_utils.DEFAULT_OUTPUT_FOLDER, output_filename),
                        sep='\t', index=False, header=True)
    print("\n\nExplanations:")
    pprint(exp.as_list(label=0, n_samples_to_report=20))
    fig = exp.as_pyplot_figure(label=0, n_samples_to_report=20)
    output_filename = f"limers_explanations_chart-{exp_setup.rec_name}"
    explain_results_chart_path = os.path.join(data_utils.DEFAULT_OUTPUT_FOLDER, output_filename)
    fig.savefig(explain_results_chart_path)
    # exp.show_in_notebook(show_table=True, show_all=True)


if __name__ == '__main__':
    main()
