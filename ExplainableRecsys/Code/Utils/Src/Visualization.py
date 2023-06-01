import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Any, Dict, Tuple, Union, Optional

class Visualizer(object):
    """
    Visualization (plots/charts) component
    """

    @staticmethod
    def plotModelPerformance(
            metric: List[float],
            title: str = "Model Performance Plot",
            xlabel: str = "Epochs",
            ylabel: str = "Loss",
            plot_save_path: Optional[str] = None
    ) -> None:
        """
        Charts the model performance metrics
        param: metric: Performance metric e.g. Loss, Accuracy etc.
        param: title: Title of the chart
        param: xlabel: X-axis label
        param: ylabel: Y-axis label
        """
        plt.plot(metric)
        plt.title(title)
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        if plot_save_path is not None:
            plt.savefig(plot_save_path)
        plt.show()

    @staticmethod
    def plotReducedDimensionEmbeddings(
            embeddings: np.ndarray,
            labels: np.array = None,
            title: str = "UMAP projection of Movielens Item Embeddings"
    ):
        """
        Creates a scatter plot of the reduced dimensional embeddings
        :param embeddings: embeddings
        :param title: Plot title
        """
        if labels is not None:
            new_labels = [0 if x <= 0 else int(x) for x in labels]
            palette = sns.color_palette('deep', np.unique(new_labels).max() + 1)
            colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in new_labels]
            plt.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=colors)
        else:
            plt.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
            )
        plt.title(title, fontsize=24)
        plt.show()


