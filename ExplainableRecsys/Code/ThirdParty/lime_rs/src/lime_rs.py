import numpy as np
import pandas as pd
import sklearn
from lime import lime_base, explanation, lime_tabular
from sklearn.utils import check_random_state

from Code.ThirdParty.lime_rs.src.dataset import Dataset


class LimeRSExplanations(explanation.Explanation):
    """
    Derived explanation functionality from the Lime package
    Used to override the behaviour of the methods:
        - to_list()
        - to_pyplot_figure()
    """
    def __init__(self,
                 domain_mapper,
                 mode,
                 class_names,
                 random_state):
        """
        Constructor
        :param domain_mapper: Domain mapper
        :param mode: Problem mode i.e. 'regression' or 'classification'
        :param class_names: Class names
        :param random_state: Random state
        """
        super().__init__(domain_mapper, mode, class_names, random_state)

    def as_list(self, label=1, n_samples_to_report=None, **kwargs):
        """Returns the explanation as a list.
        Args:
            label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
                Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper
        Returns:
            list of tuples (representation, weight), where representation is
            given by domain_mapper. Weight is a float.
        """
        label_to_use = label
        ans = self.domain_mapper.map_exp_ids(self.local_exp[label_to_use])
        if n_samples_to_report:
            ans = [(x[0], float(x[1])) for x in ans][:n_samples_to_report]
        return ans

    def as_pyplot_figure(self, label=0, figsize=(4, 4), n_samples_to_report=None, **kwargs):
        """Returns the explanation as a pyplot figure.
        Will throw an error if you don't have matplotlib installed
        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
                   Will be ignored for regression explanations.
            figsize: desired size of pyplot in tuple format, defaults to (4,4).
            kwargs: keyword arguments, passed to domain_mapper
        Returns:
            pyplot figure (barchart).
        """
        import matplotlib.pyplot as plt
        exp = self.as_list(label=label, n_samples_to_report=n_samples_to_report, **kwargs)
        fig = plt.figure(figsize=figsize)
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        if self.mode == "classification":
            title = 'Local explanation for class %s' % self.class_names[label]
        else:
            title = 'Local explanation'
        plt.title(title)
        return fig





class LimeRSExplainer():

    def __init__(self,
                 training_df,
                 feature_names,
                 feature_map,
                 mode="classification",
                 kernel_width=25,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 random_state=None):

        # exponential kernel
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel, verbose,
                                       random_state=self.random_state)

        self.feature_names = list(feature_names)
        self.feature_map = feature_map
        self.mode = mode
        self.class_names = class_names
        self.feature_selection = feature_selection

        self.categorical_features = list(range(feature_names.shape[0]))

        self.n_rows = training_df.shape[0]
        self.training_df = training_df
        self.user_freq = training_df['user_id'].value_counts(normalize=True)
        self.item_freq = training_df['item_id'].value_counts(normalize=True)

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    def explain_instance(self,
                         instance,
                         rec_model,
                         neighborhood_entity,
                         labels=(1,),
                         num_features=10,
                         num_samples=50,
                         distance_metric='cosine',
                         model_regressor=None
                         ):

        # get neighborhood
        neighborhood_df = self.generate_neighborhood(
            instance,
            neighborhood_entity,
            num_samples,
            rec_model=rec_model
        )

        # compute distance based on interpretable format
        data, _ = Dataset.convert_to_pyfm_format(neighborhood_df, columns=rec_model.one_hot_columns)
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        # get predictions from original complex model
        yss = np.array(rec_model.predict(neighborhood_df))

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
        ret_exp = LimeRSExplanations(
            domain_mapper=lime_rs_domain_mapper,
            mode=self.mode,
            class_names=self.class_names,
            random_state=100
        )
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
                data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        return ret_exp

    def generate_neighborhood(
            self,
            instance,
            entity,
            num_samples,
            rec_model
    ):
        samples = list()
        samples.append({"user_id": str(instance.user_id), "item_id": str(instance.item_id)})
        if entity == 'user':
            sample_users = np.random.choice(self.user_freq.index.tolist(), num_samples - 1, replace=False,
                                            p=self.user_freq.values.tolist())
            for u in sample_users:
                samples.append({"user_id": str(u), "item_id": str(instance.item_id)})

        elif entity == 'item':
            sample_items = np.random.choice(self.item_freq.index.tolist(), num_samples - 1, replace=False,
                                            p=self.item_freq.values.tolist())
            for i in sample_items:
                samples.append({"user_id": str(instance.user_id), "item_id": str(i)})
        else:
            sample_rows = np.random.choice(range(self.n_rows), num_samples - 1, replace=False)
            for s in self.training_df.iloc[sample_rows].itertuples():
                samples.append({"user_id": str(s.user_id), "item_id": str(s.item_id)})

        samples_df = pd.DataFrame(samples)
        samples_df = samples_df[['user_id', 'item_id']]
        if rec_model.uses_features:
            samples_df = samples_df.merge(
                rec_model.dataset.item_features,
                how="left",
                on=["item_id"]
            )
        return samples_df
