"""Per-cell quantile binning of expression values into discrete bins."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from . import _genes


def _bin_data(
    adata: AnnData,
    binning: int,
    key_to_process: str | None = None,
    result_binned_key: str = "binned_data",
) -> None:
    """Bin numerical expression data into discrete categories based on per-cell quantiles.

    The binned matrix is written to ``adata.layers[result_binned_key]`` and the
    per-cell bin edges to ``adata.obsm["bin_edges"]``. Operates in place and
    keeps sparse inputs sparse / dense inputs dense.

    Parameters
    ----------
    adata
        The input data object (modified in place).
    binning
        Number of bins (must be an integer).
    key_to_process
        Key in ``adata.layers`` to process; if ``None``, ``adata.X`` is used.
    result_binned_key
        Name of the output layer that will hold the binned matrix.

    Raises
    ------
    ValueError
        If ``binning`` is not an integer or (dense) data contains negative values.
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

        adata.layers[result_binned_key] = binned_csr
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

    adata.layers[result_binned_key] = np.stack(binned_rows)
    adata.obsm["bin_edges"] = np.stack(bin_edges)


def bin_data(adata: AnnData, n_bins: int = 50, layer: str | None = None) -> AnnData:
    """Discretize expression values into ``n_bins`` per-cell quantile bins.

    Thin, user-facing wrapper around the internal ``_bin_data`` routine. Safely
    handles dense arrays and sparse (``csr``/``csc``) matrices. The binned matrix is written
    to ``adata.layers["binned_data"]`` and bin edges to ``adata.obsm["bin_edges"]``.

    Parameters
    ----------
    adata
        The input object (modified in place).
    n_bins
        Number of bins. Defaults to 50, matching the packaged SCANVI models.
    layer
        Layer to bin; if ``None``, ``adata.X`` is used.

    Returns
    -------
    The same ``adata``, with the binned layer attached (returned for chaining).
    """
    _bin_data(adata, binning=n_bins, key_to_process=layer, result_binned_key="binned_data")
    return adata


def slice_and_bin(
    adata: AnnData,
    species: str,
    binning: int = 50,
    layer_key: str | None = None,
) -> AnnData:
    """Subset to species-specific manual genes and bin the chosen layer.

    Parameters
    ----------
    adata
        The input AnnData object (not modified in place).
    species
        Species name (``"mouse"`` or ``"human"``).
    binning
        Number of bins for binning. Defaults to 50.
    layer_key
        Key in ``adata.layers`` to bin; if ``None``, ``adata.X`` is used.

    Returns
    -------
    A new AnnData with sliced genes and a ``"binned_data"`` layer.

    Raises
    ------
    ValueError
        If the species is not recognized.
    """
    genes = sorted(_genes.get_genes(species))
    _adata = _genes.align_to_genes(adata, genes, missing="zeros", layer=layer_key)
    _bin_data(adata=_adata, binning=binning, key_to_process=layer_key, result_binned_key="binned_data")
    return _adata
