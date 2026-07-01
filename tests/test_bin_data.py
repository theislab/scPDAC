import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

import scpdac


def make_sparse_adata():
    data = np.array([0, 1, 2, 0, 3, 0, 4, 5, 0, 0, 6, 7], dtype=float)
    mat = csr_matrix(data.reshape(3, 4))
    ad = AnnData(X=csr_matrix(np.zeros((3, 4))))
    ad.layers["counts"] = mat
    return ad


def test_sparse_input_preserved_and_binned():
    ad = make_sparse_adata()
    original = ad.layers["counts"]
    assert issparse(original)

    scpdac.pp._bin_data(ad, binning=3, key_to_process="counts", result_binned_key="binned_data")

    assert "binned_data" in ad.layers
    binned = ad.layers["binned_data"]
    assert issparse(binned)
    assert binned.shape == original.shape

    orig_nz = set(zip(*original.nonzero(), strict=True))
    binned_nz = set(zip(*binned.nonzero(), strict=True))
    assert orig_nz == binned_nz
    assert np.issubdtype(binned.data.dtype, np.integer)
    assert binned.data.min() >= 0
    assert binned.data.max() <= 2
    assert "bin_edges" in ad.obsm
    be = ad.obsm["bin_edges"]
    assert be.shape == (ad.n_obs, 3)


def test_dense_input_keeps_dense_output():
    X = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]])
    ad = AnnData(X=X.copy())
    scpdac.pp._bin_data(ad, binning=2, key_to_process=None, result_binned_key="binned_dense")
    assert "binned_dense" in ad.layers
    out = ad.layers["binned_dense"]
    assert isinstance(out, np.ndarray)
    assert out.shape == X.shape
    assert np.array_equal(out[X == 0], np.zeros_like(out[X == 0]))


def test_non_integer_binning_raises():
    ad = make_sparse_adata()
    with pytest.raises(ValueError, match="Binning must be an integer"):
        scpdac.pp._bin_data(ad, binning=2.5, key_to_process="counts")


def test_dense_negative_values_raise():
    X = np.array([[0.0, -1.0], [2.0, 3.0]])
    ad = AnnData(X=X)
    with pytest.raises(ValueError, match="Expecting non-negative data"):
        scpdac.pp._bin_data(ad, binning=3)


def test_sparse_all_zero_rows_and_edges_shape():
    data = csr_matrix(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 5, 0, 0],
                [1, 2, 0, 0],
            ],
            dtype=float,
        )
    )
    ad = AnnData(X=csr_matrix((3, 4)))
    ad.layers["counts"] = data

    scpdac.pp._bin_data(ad, binning=4, key_to_process="counts", result_binned_key="binned")
    out = ad.layers["binned"]
    assert issparse(out)
    assert out.shape == data.shape

    assert out[0].nnz == 0
    be = ad.obsm["bin_edges"]
    assert be.shape == (ad.n_obs, 4)
    assert np.allclose(be[:, 0], 0)
    assert all(np.all(np.diff(row) >= 0) for row in be)


def test_key_to_process_does_not_modify_X():
    X = np.array([[0.0, 1.0], [2.0, 0.0]])
    ad = AnnData(X=X.copy())
    ad.layers["counts"] = csr_matrix(np.array([[0, 3], [4, 0]], dtype=float))

    scpdac.pp._bin_data(ad, binning=3, key_to_process="counts", result_binned_key="binned_counts")

    assert np.array_equal(ad.X.toarray() if issparse(ad.X) else ad.X, X)
    assert "binned_counts" in ad.layers
    assert issparse(ad.layers["binned_counts"])


def test_bin_values_respect_range_dense():
    X = np.array([[0.0, 0.1, 0.2, 0.3]])
    ad = AnnData(X=X)
    scpdac.pp._bin_data(ad, binning=5, key_to_process=None, result_binned_key="binned")
    out = ad.layers["binned"]
    assert out.dtype.kind in ("i", "u")
    assert out.min() >= 0
    assert out.max() <= 4


def test_bin_data_public_wrapper_defaults_to_binned_data():
    X = np.array([[0.0, 1.0, 2.0, 3.0]])
    ad = AnnData(X=X)
    out = scpdac.pp.bin_data(ad, n_bins=4)
    assert out is ad
    assert "binned_data" in ad.layers
    assert "bin_edges" in ad.obsm


def test_slice_and_bin_filters_genes_and_returns_new(monkeypatch):
    X = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [3.0, 0.0, 4.0, 5.0],
        ]
    )
    var_names = np.array(["G1", "G2", "G3", "G4"])
    ad = AnnData(X=X.copy())
    ad.var_names = var_names
    monkeypatch.setattr(scpdac.pp._genes, "MOUSE_MANUAL_GENES", ["G1", "G3"], raising=False)
    monkeypatch.setattr(scpdac.pp._genes, "HUMAN_MANUAL_GENES", ["HX"], raising=False)

    out = scpdac.pp.slice_and_bin(ad, species="mouse", binning=3)

    assert np.array_equal(ad.X, X)
    assert list(ad.var_names) == list(var_names)
    assert isinstance(out, AnnData)
    assert list(out.var_names) == ["G1", "G3"]
    assert "binned_data" in out.layers
    assert "bin_edges" in out.obsm
    assert out.obsm["bin_edges"].shape[0] == out.n_obs


def test_slice_and_bin_invalid_species_raises():
    ad = AnnData(X=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="Unknown species"):
        scpdac.pp.slice_and_bin(ad, species="doge", binning=3)
