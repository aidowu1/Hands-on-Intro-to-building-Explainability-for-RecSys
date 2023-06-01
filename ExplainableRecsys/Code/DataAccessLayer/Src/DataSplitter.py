import sys
import random
import pandas as pd
import copy
from typing import Tuple

from Code.DataAccessLayer.Src.DataProvider import DataReader

class Splitter:
    """
    Super Splitting Class.
    args:
        data: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
    """

    def __init__(self):
        pass

    @staticmethod
    def splitLeaveLatestOut(data, n_latest: int = 1) -> Tuple[DataReader, pd.DataFrame]:
        """
        Leave N latest interactions out train/test split.
        Ref:
        Campos, Pedro G., Fernando Díez, and Iván Cantador. "Time-aware recommender systems: a comprehensive survey and
        analysis of existing evaluation protocols." User Modeling and User-Adapted Interaction 24.1-2 (2014): 67-119.
        :param data:
        :param n_latest: int, number of latest interactions to be in the the test set.
        :returns train as DataReader, test as data.frames
        """
        # group items by suer id and rank them by timestamp
        rank_latest = data._dataset_df.groupby(['userId'])['timestamp'] \
            .rank(method='first', ascending=False)

        # keep in test items that are ranked higher than n_latest
        test = data._dataset_df[rank_latest <= n_latest]
        # keep in train the rest
        train = copy.deepcopy(data)
        train._dataset_df = data._dataset_df[rank_latest > n_latest]

        return train, test

    @staticmethod
    def splitLeaveNOut(
            data, n: int = 1,
            frac: float = None,
            random_state_seed: int = 100
    ) -> Tuple[DataReader, pd.DataFrame]:
        """
        Leave N latest interactions out train/test split.
        Ref:
        Shani, Guy, and Asela Gunawardana. "Evaluating recommendation systems." Recommender systems handbook. Springer,
        Boston, MA, 2011. 257-297.
        :param data:
        :param n int, number of interactions to be in the test set.
        :param frac float, fraction.
        :returns dataframe train and test
        """
        min_nr_ratings_user = min(data._dataset_df['userId'].value_counts())

        if min_nr_ratings_user < n:
            sys.exit("split_leave_n_out: There are users with less ratings than n (required number of interactions "
                     "in the test set).")

        if frac is not None and frac > 1:
            sys.exit("f (i.e.) fraction should be smaller than 1.")


        # group items by user id and extraxt a random number of items per user
        grouped = data._dataset_df.groupby(['userId'])
        if frac is not None:
            test = grouped.apply(lambda x: x.sample(frac=frac, random_state=random_state_seed))
        else:
            test = grouped.apply(lambda x: x.sample(n=n, random_state=random_state_seed))

        test = test.reset_index(drop=True)
        train_pd = pd.merge(data._dataset_df, test, on=list(data._dataset_df.columns), how="outer", indicator=True)
        train_pd = train_pd[train_pd['_merge'] == 'left_only']
        train_pd = train_pd.drop(columns="_merge")

        train = copy.deepcopy(data)
        train._dataset_df = train_pd

        assert test.shape[0] + train_pd.shape[0] == data._dataset_df.shape[0]

        return train, test

    def relPlusN(self,
                   data,
                   negative_sample_size: int = 99,
                   splitting: str = "latest",
                   n: int = 1
                   ) -> Tuple[DataReader, pd.DataFrame]:
        """
        RelPlusN: We build the users test set by extracting one relevant random item ($HR_u$) from the entire set of
        rated items. Then  a set of random items with unknown relevance ($NR_u$), is extracted for each user $u$, where $u$
        had no previous interaction with these items. Finally, for each item $i$ in $HR_u$, the algorithm requests a ranking
        of the top-$N$ items from the set $ {i} cup NR_u$, on which the evaluation is performed. The evaluation metrics
        are averaged over all the items in $HR_u$ and later over all the users. In the following, all experiments have been
        conducted according to this protocol.
        Ref:
        - Paolo Cremonesi, Yehuda Koren, and Roberto Turrin. 2010.   Performance of Recommender Algorithms on Top-n
        Recommendation Tasks. InProceedings ofthe Fourth ACM Conference on Recommender Systems (RecSys ’10).
        - Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative
        Filtering. In Proceedings of the 26th InternationalConference on World Wide Web (WWW ’17).
        :param data
        :param negative_sample_size how many negative items to compute
        :param splitting either latest for leave n latest out, or n for leave n out
        :param n how many to leave out

        """

        if splitting == "latest":
            train, test = self.splitLeaveLatestOut(data, n)
        elif splitting == "n":
            train, test = self.splitLeaveNOut(data, n)
        else:
            sys.exit("splitting can be either \"latest\" or \"n\". ")

        neg_sample = self.sample_negative(data, negative_sample_size)

        return train, test.append(neg_sample)

    @staticmethod
    def sample_negative(data, negative_sample_size):
        """return all negative items """

        item_catalogue = set(data._dataset_df['itemId'])

        interact_status = data._dataset_df\
            .groupby('userId')['itemId']\
            .apply(set)\
            .reset_index()\
            .rename(columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items']\
            .apply(lambda x: item_catalogue - x)
        interact_status['negative_samples'] = interact_status['negative_items']\
            .apply(lambda x: random.sample(x, negative_sample_size))
        interact_status = interact_status[['userId', 'negative_samples']]

        userId = []
        itemId = []
        for row in interact_status.itertuples():
            for i in range(negative_sample_size):
                userId.append(int(row.userId))
                itemId.append(int(row.negative_samples[i]))

        return pd.DataFrame.from_dict({'userId': userId, 'itemId': itemId})

