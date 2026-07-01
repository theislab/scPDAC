"""Tools: SCANVI atlas extension and hierarchical MLP classification."""

from ._classifier import MLP, HierarchicalClassifier, derive_malignant_mask, predict_labels
from ._mapping import embed_and_predict, extend_atlas
from ._data import scDataset, densify_rows, predict_indices

__all__ = [
    "extend_atlas",
    "embed_and_predict",
    "predict_labels",
    "HierarchicalClassifier",
    "MLP",
    "derive_malignant_mask",
    "scDataset",
    "densify_rows",
    "predict_indices",
]
