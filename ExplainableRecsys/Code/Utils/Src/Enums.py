import enum

class MetricTypes(enum.Enum):
  """
  Enums of metric types
  """
  mse = enum.auto()
  mape = enum.auto()
  mae = enum.auto()
  rmse = enum.auto()


class DimensionReductionTypes(enum.Enum):
  """
  Dimensionality reduction types
  """
  umap = enum.auto()
  pca = enum.auto()

class ExplanationType(enum.Enum):
  """
  Explanation model types
  """
  ar_explainer = enum.auto()
  knn_explainer = enum.auto()
  als_explainer = enum.auto()
  emf_explainer = enum.auto()
  fmlime_explainer = enum.auto()
  posthoc_noexplainer = enum.auto()