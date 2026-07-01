import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

import scpdac
from scpdac.pp import align_to_genes, get_genes


def test_get_genes_dispatch():
    assert get_genes("human") is scpdac.pp.HUMAN_MANUAL_GENES
    assert get_genes("mouse") is scpdac.pp.MOUSE_MANUAL_GENES


def test_get_genes_invalid_species():
    with pytest.raises(ValueError, match="Unknown species"):
        get_genes("doge")


def _adata():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ad = AnnData(X=X)
    ad.var_names = ["A", "B", "C"]
    return ad


def test_align_to_genes_reorders_present_genes():
    out = align_to_genes(_adata(), ["C", "A"])
    assert list(out.var_names) == ["C", "A"]
    assert np.array_equal(out.X, np.array([[3.0, 1.0], [6.0, 4.0]]))


def test_align_to_genes_missing_error():
    with pytest.raises(ValueError, match="missing from input"):
        align_to_genes(_adata(), ["A", "Z"], missing="error")


def test_align_to_genes_missing_zeros_warns_and_imputes():
    with pytest.warns(UserWarning, match="imputing"):
        out = align_to_genes(_adata(), ["A", "Z", "C"], missing="zeros")
    assert list(out.var_names) == ["A", "Z", "C"]
    assert np.array_equal(out[:, "Z"].X.ravel(), np.zeros(2))
    assert np.array_equal(out[:, "A"].X.ravel(), np.array([1.0, 4.0]))


def test_align_to_genes_sparse_zero_impute():
    ad = AnnData(X=csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]])))
    ad.var_names = ["A", "B"]
    with pytest.warns(UserWarning):
        out = align_to_genes(ad, ["A", "Z"], missing="zeros")
    assert list(out.var_names) == ["A", "Z"]


def test_align_to_genes_invalid_missing_option():
    with pytest.raises(ValueError, match="Unknown 'missing' option"):
        align_to_genes(_adata(), ["A"], missing="oops")
