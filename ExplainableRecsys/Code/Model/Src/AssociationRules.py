from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from copy import deepcopy

from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.Recommender.Src.Constants import (ASSOCIATION_RULE_COLUMNS)

class AssociationRulesModel(object):
    """
    Association Rules recommendation model
    """
    def __init__(self,
                 min_support: float = .1,
                 max_len: int = 2,
                 metric: str = "lift",
                 min_threshold: float = .1,
                 min_confidence: float = .1,
                 min_lift: float = .1
                 ):
        """
        Constructor
        :param min_support: Minimum support
        :param max_len: Maximum length of the itemsets generated
        :param metric: Metric to evaluate if a rule is of interest
        :param min_threshold: Minimal threshold for the evaluation metric,
                              via the `metric` parameter, to decide whether a candidate rule is of interest.
        :param min_confidence: Minimum confidence value
        :param min_lift: Minimum lift value
        """
        self._min_support = min_support
        self._max_len = max_len
        self._metric = metric
        self._min_threshold = min_threshold
        self._min_confidence = min_confidence
        self._min_lift = min_lift
        self._dataset_metadata = None
        self._dataset = None
        self._catalogue = None
        self._rules_df = None

    def fit(self, dataset_metadata: DataReader, rating_threshold: int=3) -> bool:
        """
        Fit/train the model on the input dataset
        :param dataset_metadata: Dataset metadata
        :param rating_threshold: Rating threshold for binarization
        """
        self._dataset_metadata = dataset_metadata
        self._dataset = self._dataset_metadata.dataset
        self._dataset_metadata.makeConsecutiveIdsInDataset()
        self._dataset_metadata.binarize(binary_threshold=rating_threshold)
        self._catalogue = set(self._dataset['itemId'])
        self._computeAssociationRules()
        return True

    def _computeAssociationRules(self):
        """
        Compute association rules
        """
        item_sets = [
            [item for item in self._dataset[self._dataset.userId == user].itemId]
            for user in self._dataset.userId.unique()
        ]

        te = TransactionEncoder()
        te_ary = te.fit(item_sets).transform(item_sets)

        df = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(df,
                                    min_support=self._min_support,
                                    use_colnames=True,
                                    max_len=self._max_len)

        rules_df = association_rules(frequent_itemsets,
                                  metric=self._metric,
                                  min_threshold=self._min_threshold)
        rules_df = rules_df[(rules_df['confidence'] > self._min_confidence) &
                      (rules_df['lift'] > self._min_lift)]

        rules_df.consequents = [list(row.consequents)[0] for _, row in rules_df.iterrows()]
        rules_df.antecedents = [list(row.antecedents)[0] for _, row in rules_df.iterrows()]

        association_rules_columns = deepcopy(ASSOCIATION_RULE_COLUMNS) + [self._metric]
        self._rules_df = rules_df[association_rules_columns]

    @property
    def rules(self) -> pd.DataFrame:
        """
        Getter property for the computed association rules
        :return: Rules (as a table)
        """
        return self._rules_df

    @property
    def metric(self) -> str:
        """
        Getter property for the association rule metric
        :return: metric
        """
        return self._metric


