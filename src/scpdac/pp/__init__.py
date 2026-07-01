"""Preprocessing: gene-panel alignment and per-cell quantile binning."""

from ._binning import _bin_data, bin_data, slice_and_bin
from ._genes import HUMAN_MANUAL_GENES, MOUSE_MANUAL_GENES, align_to_genes, get_genes

__all__ = [
    "bin_data",
    "slice_and_bin",
    "_bin_data",
    "get_genes",
    "align_to_genes",
    "HUMAN_MANUAL_GENES",
    "MOUSE_MANUAL_GENES",
]
