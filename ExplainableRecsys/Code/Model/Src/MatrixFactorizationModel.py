import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Union, Any
from tqdm.auto import tqdm
import  numpy as np

from Code.ThirdParty.recoxplainer_master.recoxplainer.evaluator import Evaluator, ExplanationEvaluator
from Code.Model.Src.InteractionMatrixDataset import UserItemRatingDataset
from Code.DataAccessLayer.Src.DataProvider import DataReader
from Code.DataAccessLayer.Src.DataSplitter import Splitter
from Code.Utils.Src.Visualization import Visualizer
import Code.Model.Src.Constants as c



class MFModel(torch.nn.Module):
    """
    Matrix Factorization Model
    """
    def __init__(self,
                 learning_rate: float,
                 weight_decay: float,
                 latent_dim: int,
                 epochs: int,
                 batch_size: int,
                 split_fraction: float=0.0,
                 device_id=None,
                 is_provide_visualization: bool = True,
                 is_use_bias: bool = True,
                 is_train_and_validate = False,
                 performance_plot_save_path=c.MF_PERFORMANCE_PLOT_PATH
                 ):
        """
        Constructor
        :param learning_rate: Learning rate
        :param weight_decay: Weight decay
        :param latent_dim: Latent dimension (rank)
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param split_fraction: Data split fraction
        :param device_id: Device ID either "cpu" or "gpu"
        """
        super().__init__()
        self._weight_decay = weight_decay
        self._latent_dim = latent_dim
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._device_id = device_id
        self._split_fraction = split_fraction
        self._criterion = torch.nn.MSELoss()
        self._train_metadata = None
        self._test_df = None
        self.embedding_user = None
        self.embedding_item = None
        self.embedding_user_biases = None
        self.embedding_item_biases = None
        self._dataset_metadata = None
        self._train_dataset = None
        self._test_dataset = None
        self._is_provide_visualization = is_provide_visualization
        self._is_train_and_validate = is_train_and_validate
        self._is_use_bias = is_use_bias
        self._performance_plot_save_path = performance_plot_save_path


    def forward(self,
                user_indices: torch.LongTensor,
                item_indices: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward propagation logic
        :param user_indices: User indices
        :param item_indices: Item indices
        :return: Rating
        """
        user_embeddings = self.embedding_user(user_indices)
        item_embeddings = self.embedding_item(item_indices)
        if not self._is_use_bias:
            predictions = (user_embeddings * item_embeddings).sum(dim=1)
        else:
            predictions = self.embedding_item_biases(item_indices)
            predictions += self.embedding_user_biases(user_indices)
            predictions += torch.tensor([self.global_rating])
            predictions += (user_embeddings * item_embeddings).sum(dim=1, keepdims=True)
        return predictions.squeeze()

    def _preProcessData(self, dataset_metadata: DataReader) -> None:
        """
        Pre-processes the dataset prior to training.
        Pre-processing steps include:
            - Make the data consecutive
            - Split the data into train/validation partitions
        :param dataset_metadata: Metadata of problem dataset
        :return: Split partitions of the 'train' and 'test' metadata
        """
        dataset_metadata.makeConsecutiveIdsInDataset()
        sp = Splitter()
        self._train_metadata, self._test_df = sp.splitLeaveNOut(dataset_metadata, frac=self._split_fraction)

    def fit(self, dataset_metadata: DataReader) -> bool:
        """
        Computes the model fittness i.e. trains the model
        :param dataset_metadata: Metadata of the problem dataset
        :return: Status
        """
        self._dataset_metadata = dataset_metadata
        if self._is_train_and_validate:
            self._preProcessData(dataset_metadata)

        if self._is_use_bias:
            self.embedding_user_biases = torch.nn.Embedding(
                num_embeddings=self._dataset_metadata.num_user,
                embedding_dim=1
            )

            self.embedding_item_biases = torch.nn.Embedding(
                num_embeddings=self._dataset_metadata.num_item,
                embedding_dim=1
            )

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self._dataset_metadata.num_user,
            embedding_dim=self._latent_dim)

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self._dataset_metadata.num_item,
            embedding_dim=self._latent_dim)

        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=self._learning_rate,
                                         weight_decay=self._weight_decay)

        self._train_dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(self._dataset_metadata.dataset.userId.tolist()),
            item_tensor=torch.LongTensor(self._dataset_metadata.dataset.itemId.tolist()),
            target_tensor=torch.FloatTensor(self._dataset_metadata.dataset.rating.tolist()))

        if self._is_train_and_validate:
            self._test_dataset = UserItemRatingDataset(
                user_tensor=torch.LongTensor(self._test_df.userId.tolist()),
                item_tensor=torch.LongTensor(self._test_df.itemId.tolist()),
                target_tensor=torch.FloatTensor(self._test_df.rating.tolist()))

        if not self._is_train_and_validate:
            status = self._runTrainCycles()
        else:
            status = self._runTrainAndValidCycles()
        return status

    def _runTrainCycles(self) -> bool:
        """
        Executes the matrix factorization train cycles
        :param dataset: Movielens training data
        :return: Run status
        """
        loss_values = []
        with tqdm(
                enumerate(range(self._epochs)),
                total=self._epochs,
                desc="Training cycle updates"
        ) as progress:
            for idx, epoch in progress:
                train_data_loader = self.getTrainDataLoader(
                    self._train_dataset,
                    self._batch_size)

                # Train batch
                self.train()
                losses, nums = zip(
                    *[self._runLossBatch(user, item, rating, self.optimizer) for (user, item, rating) in train_data_loader])
                loss_value = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                loss_values.append(loss_value)

                message = {
                    'Epoch': idx + 1,
                    'Train cycles MSE loss (in progress)': f"{loss_value:.4f}"}
                progress.update(1)
                progress.set_postfix(message)
            if self._is_provide_visualization:
                Visualizer.plotModelPerformance(
                    metric=loss_values,
                    title="Model Performance (MSE Losses)",
                    plot_save_path=self._performance_plot_save_path
                )
        return True

    def _runTrainAndValidCycles(self) -> bool:
        """
        Executes the matrix factorization train-validation cycles
        :param dataset: Movielens training data
        :return: Run status
        """
        loss_values = []
        with tqdm(
                enumerate(range(self._epochs)),
                total=self._epochs,
                desc="Training-validation cycle updates"
        ) as progress:
            for idx, epoch in progress:
                train_data_loader, test_data_loader = self.getDataLoader(
                    self._train_dataset,
                    self._test_df,
                    self._batch_size)

                # Train batch
                self.train()
                for _, batch in enumerate(train_data_loader):
                    user, item, rating = batch[0], batch[1], batch[2]
                    self._runLossBatch(user, item, rating, self.optimizer)

                # Evaluate batch
                self.eval()
                with torch.no_grad():
                    losses, nums = zip(
                        *[self._runLossBatch(user, item, rating)
                          for (user, item, rating) in test_data_loader])

                loss_value = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                loss_values.append(loss_value)
                message = {
                    'Epoch': idx + 1,
                    'Train cycles MSE loss (in progress)': f"{loss_value:.4f}"}
                progress.update(1)
                progress.set_postfix(message)
            if self._is_provide_visualization:
                Visualizer.plotModelPerformance(
                    metric=loss_values,
                    title="Model Performance (MSE Losses)",
                    plot_save_path=self._performance_plot_save_path
                )
        return True

    def _runLossBatch(
            self,
            user,
            item,
            rating,
            optimizer=None) -> Tuple[Any, Any]:
        """
        Computation of the loss and back-propagation gradient
        :param user: Row batch
        :param item: Column batch
        :param rating: Rating batch
        :param optimizer: optimizer
        :return loss value and batch size
        """
        rating = rating.float()
        predicted_rating = self(user, item).squeeze()
        loss = self._criterion(predicted_rating, rating)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.item(), len(rating)


    def predict(self,
                user_id: Union[int, List[int]],
                item_id: Union[int, List[int]]
                ) -> List[float]:
        """
        Method used to invoke the prediction of the rating
        :param user_id: User ID
        :param item_id: Item ID
        :return: Predicted rating
        """
        if type(user_id) == 'int':
            user_id = [user_id]
        if type(item_id) == 'int':
            item_id = [item_id]
        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor(item_id)
        with torch.no_grad():
            pred = self.forward(user_id, item_id).cpu().tolist()
            return pred

    @property
    def train_metadata(self) -> DataReader:
        """
        Getter property for Train partition of the dataset metadata
        :return: Train partition of the dataset metadata
        """
        return self._train_metadata

    @property
    def test_df(self) -> pd.DataFrame:
        """
        Getter property for Test partition of the dataset metadata
        :return: Test partition of the dataset metadata
        """
        return self._test_df

    @property
    def dataset_metadata(self) -> DataReader:
        """
        Getter for the dataset metadata (after post-processing i.e. make data consecutive)
        :return: Dataset metadata
        """
        return self._dataset_metadata

    @staticmethod
    def getTrainDataLoader(train_ds, batch_size) -> Any:
        """
        Gets the training data loader
        :param train_ds: Train dataset
        :param batch_size: Batch size
        :return: Train data loader
        """
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    @staticmethod
    def getDataLoader(train_ds, test_ds, batch_size) -> Tuple[Any, Any]:
        """
        Gets the training and validation data loaders
        :param train_ds: Train dataset
        :param test_ds: Validation dataset
        :param batch_size: Batch size
        :return: Train and validation data loaders
        """
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=True),
        )
    @property
    def global_rating(self) -> np.ndarray:
        """
        Global bias i.e. mean of all non-zero ratings in the user-item interaction dataset
        """
        ratings = self._dataset_metadata.dataset.rating.values
        ratings = ratings[np.where(ratings != 0)]
        return np.mean(ratings)

