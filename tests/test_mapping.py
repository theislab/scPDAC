import numpy as np
import pytest
from anndata import AnnData

from scpdac.tl import embed_and_predict, extend_atlas


def _adata(n=4, with_batch=True):
    ad = AnnData(X=np.random.default_rng(0).random((n, 3)).astype(np.float32))
    ad.var_names = ["A", "B", "C"]
    ad.layers["counts"] = ad.X.copy()
    if with_batch:
        ad.obs["Sample_ID"] = "s1"
    return ad


def test_embed_and_predict_requires_batch_key():
    ad = _adata(with_batch=False)
    with pytest.raises(ValueError, match="Sample_ID"):
        embed_and_predict(ad, species="human")


def test_extend_atlas_requires_batch_key():
    query = _adata(with_batch=False)
    atlas = _adata()
    with pytest.raises(ValueError, match="Sample_ID"):
        extend_atlas(query, atlas, species="human")
