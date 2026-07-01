from typing import TypeVar

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from .variables import HUMAN_MANUAL_GENES, MOUSE_MANUAL_GENES

MuData = TypeVar("MuData")
SpatialData = TypeVar("SpatialData")
ScverseDataStructures = AnnData | MuData | SpatialData


def _bin_data(adata, binning, key_to_process=None):
    """
    Bins numerical data into discrete categories based on quantiles.

    Parameters
    ----------
        adata (AnnData): The input data object.
        binning (int): Number of bins (must be an integer).
        key_to_process (str): Key in `adata.layers` to process.

    Raises
    ------
        ValueError: If `binning` is not an integer or data contains negative values.
    """
    if not isinstance(binning, int):
        raise ValueError(f"Binning must be an integer, but got {binning}.")

    layer_data = adata.layers[key_to_process] if key_to_process is not None else adata.X

    if issparse(layer_data):
        mat = layer_data.tocsr()
        n_obs, n_vars = mat.shape

        indptr = mat.indptr
        indices = mat.indices
        data = mat.data

        binned_data_values = []
        binned_data_indices = []
        binned_indptr = [0]
        bin_edges = []

        for i in range(n_obs):
            start = indptr[i]
            end = indptr[i + 1]

            if start == end:
                binned_indptr.append(binned_indptr[-1])
                bin_edges.append(np.array([0] * binning))
                continue

            row_vals = data[start:end]
            row_cols = indices[start:end]

            if row_vals.max() == 0:
                binned_indptr.append(binned_indptr[-1])
                bin_edges.append(np.array([0] * binning))
                continue

            bins = np.quantile(row_vals, np.linspace(0, 1, binning - 1))
            digits = np.digitize(row_vals, bins).astype(np.int64)

            binned_data_values.extend(digits.tolist())
            binned_data_indices.extend(row_cols.tolist())
            binned_indptr.append(binned_indptr[-1] + len(digits))
            bin_edges.append(np.concatenate([[0], bins]))

        binned_csr = csr_matrix(
            (
                np.array(binned_data_values, dtype=np.int64),
                np.array(binned_data_indices, dtype=np.int32),
                np.array(binned_indptr, dtype=np.int32),
            ),
            shape=(n_obs, n_vars),
        )

        adata.layers["binned_data"] = binned_csr
        adata.obsm["bin_edges"] = np.stack(bin_edges)
        return

    if layer_data.min() < 0:
        raise ValueError(f"Expecting non-negative data, but got min value {layer_data.min()}.")

    binned_rows = []
    bin_edges = []

    for row in layer_data:
        if row.max() == 0:
            binned_rows.append(np.zeros_like(row, dtype=np.int64))
            bin_edges.append(np.array([0] * binning))
            continue

        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]

        bins = np.quantile(non_zero_row, np.linspace(0, 1, binning - 1))
        non_zero_digits = np.digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits

        binned_rows.append(binned_row)
        bin_edges.append(np.concatenate([[0], bins]))

    adata.layers["binned_data"] = np.stack(binned_rows)
    adata.obsm["bin_edges"] = np.stack(bin_edges)


def slice_and_bin(adata: AnnData, species: str, binning: int, layer_key: str) -> AnnData:
    """
    Slice to species-specific manual genes and bin the specified layer.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object.
    species : str
        Species name ('mouse' or 'human').
    binning : int
        Number of bins for binning.
    layer_key : str
        Key in `adata.layers` to apply binning on.

    Returns
    -------
    AnnData
        Processed AnnData with sliced genes and binned data.

    Raises
    ------
    ValueError
        If the species is not recognized.
    """
    if species not in ["mouse", "human"]:
        raise ValueError(f"Unknown species: {species}")

    _adata = adata.copy()
    genes = MOUSE_MANUAL_GENES if species == "mouse" else HUMAN_MANUAL_GENES
    _adata = _adata[:, _adata.var_names.isin(genes)].copy()
    _bin_data(adata=_adata, binning=binning, key_to_process=layer_key)

    return _adata
