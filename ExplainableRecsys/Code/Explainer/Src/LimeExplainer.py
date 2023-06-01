import lime.explanation
import numpy as np
import pandas as pd
import sklearn
from lime import lime_base, explanation, lime_tabular
from sklearn.utils import check_random_state
from typing import List, Dict, Tuple, Optional, Any

import Code.DataAccessLayer.Src.Constants as c
from Code.Model.Src.FactorizationMachineModel import FMModel
from Code.DataAccessLayer.Src.MovieLensDataPreProcessing import Movielens100KPreprocessor as mv
from Code.DataAccessLayer.Src.MovielensDataset import MovielenDataset
from Code.Explainer.Src.LimeRSExplanations import LimeRSExplanations


class LimeRSExplainer(object):
    """
    LIME (Local Interpretable Model-agnostic Explanations) explainer model component

    This model is inspired by reference to work developed by Caio Nóbrega
    (github repo: https://github.com/caionobrega/explaining-recommendations)
    """
    def __init__(self,
                 dataset: MovielenDataset,
                 feature_names: np.ndarray,
                 feature_map: Dict[int, str],
                 mode: str = "classification",
                 kernel_width: int = 25,
                 verbose: bool = False,
                 class_names: np.ndarray = None,
                 feature_selection: str = 'auto',
                 random_state: int = None,
                 is_use_features_side_info: bool = True
                 ):
        """
        Constructor
        :param dataset: FM dataset
        :param feature_names: Feature names
        :param feature_map: Feature map
        :param mode: Problem type i.e., Classification or Regression
        :param kernel_width: Kernel width
        :param verbose: Verbosity flag
        :param class_names: Class names
        :param feature_selection: Feature selection
        :param random_state: Random state
        :param is_use_features_side_info: Flag to indicate the use of item/user feature side info
        """

        def kernel(distance) -> float:
            """
            Exponential kernel function
            :param: distance: Distance
            """
            return np.sqrt(np.exp(-(distance ** 2) / kernel_width ** 2))

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel, verbose,
                                       random_state=self.random_state)

        self.feature_names = feature_names
        self.feature_map = feature_map
        self.mode = mode
        self.class_names = class_names
        self.feature_selection = feature_selection

        self.categorical_features = list(range(len(self.feature_names)))
        self.training_df = dataset.train_users_items_df
        self.item_side_info_df = dataset.items_side_info_modified_df
        self.n_rows = self.training_df.shape[0]

        self.user_freq = self.training_df[c.MOVIELENS_USER_ID_COLUMN_KEY].value_counts(normalize=True)
        self.item_freq = self.training_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY].value_counts(normalize=True)

    @staticmethod
    def convertAndRound(values):
        """
        Converts to floats that are round to 2 decimal places
        """
        return [f'{np.round(v,2)}' for v in values]

    def explainInstance(self,
                        instance: pd.DataFrame,
                        rec_model: FMModel,
                        neighborhood_entity: str,
                        labels: List[int] = (1,),
                        num_features: int = 10,
                        num_samples: int = 50,
                        distance_metric: str = 'cosine',
                        model_regressor: Any = None
                        ) -> lime.explanation.Explanation:
        """
        Explains the response of the model to the input instance.
        Internally this method is invoking the 'explain_instance_with_data' method in the LimeBase class in the
        Lime library (i.e., further details in
        this github repo: https://github.com/marcotcr/lime/blob/master/lime/lime_base.py)

        :param instance: Instance of the model input
        :param rec_model: RecSys model (i.e., FM in this case)
        :param neighborhood_entity: Perturbed data, 2d array. first element is assumed to be the original data point
        :param labels: Corresponding perturbed labels. should have as many columns as the number of possible labels
        :param num_features: Maximum number of features in explanation
        :param num_samples: Number of samples
        :param distance_metric: Distance metric i.e., Cosine distance in this case
        :param model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()
        :return: Returns the model explanation
        """

        # get neighborhood
        neighborhood_df = self.generateNeighborhood(
            instance,
            neighborhood_entity,
            num_samples)

        # compute distance based mon interpretable format
        data, _ = mv.transformFMModelItemSideDatasetV2(
            user_item_interaction_df=neighborhood_df,
            item_side_df=self.item_side_info_df,
            ohe_feature_columns=rec_model.one_hot_columns,
            is_consolidate_genre_values=False
        )
        # data, _ = mv.calculateOneHotEncoding(neighborhood_df, one_hot_encoder=rec_model.one_hot_encoder)
        #data, _ = mv.convertToPyFmFormat(neighborhood_df, columns=rec_model.one_hot_columns)
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        X_neighborhood = data.copy()
        yss = np.array(rec_model.predict(X_neighborhood.toarray()))

        # for classification, the model needs to provide a list of tuples - classes along with prediction probabilities
        if self.mode == "classification":
            raise NotImplementedError("LIME-RS does not currently support classifier models.")
        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                            numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        lime_rs_domain_mapper = lime_tabular.TableDomainMapper(
            feature_names=rec_model.one_hot_columns,
            feature_values=data[0].reshape(1, -1),
            scaled_row=data[0].reshape(1, -1),
            categorical_features=[x for x in range(len(rec_model.one_hot_columns))]
        )
        ret_exp = explanation.Explanation(domain_mapper=lime_rs_domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            raise NotImplementedError("LIME-RS does not currently support classifier models.")
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data.toarray(),
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        return ret_exp

    def generateNeighborhood(
            self,
            instance: pd.DataFrame,
            entity: str,
            num_samples: int
    ) -> pd.DataFrame:
        """
        Generate the local neighborhood sample around an instance of X
        :param instance: Explained instance of the dataset
        :param entity: Item or user entity that is being perturbed randomly to create the local neighborhood samples
        :param num_samples: Number of generated samples
        :return: Generated samples
        """
        samples = list()
        samples.append({c.MOVIELENS_USER_ID_COLUMN_KEY: instance[c.MOVIELENS_USER_ID_COLUMN_KEY][0],
                        c.MOVIELENS_ITEM_ID_COLUMN_KEY: instance[c.MOVIELENS_ITEM_ID_COLUMN_KEY][0]})
        if entity == 'user':
            sample_users = np.random.choice(self.user_freq.index.tolist(), num_samples - 1, replace=False,
                                            p=self.user_freq.values.tolist())
            for u in sample_users:
                samples.append({c.MOVIELENS_USER_ID_COLUMN_KEY: u,
                                c.MOVIELENS_ITEM_ID_COLUMN_KEY: instance[c.MOVIELENS_ITEM_ID_COLUMN_KEY][0]})

        elif entity == 'item':
            sample_items = np.random.choice(self.item_freq.index.tolist(), num_samples - 1, replace=False,
                                            p=self.item_freq.values.tolist())
            for i in sample_items:
                samples.append({c.MOVIELENS_USER_ID_COLUMN_KEY: instance[c.MOVIELENS_USER_ID_COLUMN_KEY][0],
                                c.MOVIELENS_ITEM_ID_COLUMN_KEY: i})
        else:
            sample_rows = np.random.choice(range(self.n_rows), num_samples - 1, replace=False)
            for s in self.training_df.iloc[sample_rows].itertuples():
                samples.append({c.MOVIELENS_USER_ID_COLUMN_KEY: s[c.MOVIELENS_USER_ID_COLUMN_KEY][0],
                                c.MOVIELENS_ITEM_ID_COLUMN_KEY: s[c.MOVIELENS_ITEM_ID_COLUMN_KEY][0]})
        samples_df = pd.DataFrame(samples)
        user_item_columns = c.MOVIELENS_FEATURE_COLUMNS_TYPE_0
        samples_df = samples_df[user_item_columns]
        return samples_df

    @staticmethod
    def createSampleLocalInstances(sample_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Creates sample of local instances for Lime explainability
        :param sample_df: Sample dataset
        :return: List of local instances
        """
        user_ids = sample_df[c.MOVIELENS_USER_ID_COLUMN_KEY].tolist()
        item_ids = sample_df[c.MOVIELENS_ITEM_ID_COLUMN_KEY].tolist()
        local_instances = [pd.DataFrame({
            c.MOVIELENS_USER_ID_COLUMN_KEY: [user_ids[i]],
            c.MOVIELENS_ITEM_ID_COLUMN_KEY: [item_ids[i]]
        }) for i in range(len(user_ids))]
        return local_instances


