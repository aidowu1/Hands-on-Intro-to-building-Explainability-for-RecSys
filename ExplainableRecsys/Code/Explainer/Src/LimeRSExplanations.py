import numpy as np
from lime import lime_base, explanation, lime_tabular

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

    def as_list(self, label=0, n_samples_to_report=None, **kwargs):
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