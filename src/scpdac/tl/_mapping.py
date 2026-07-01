"""Atlas extension and reference-based prediction via SCANVI / scArches."""

from __future__ import annotations

import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc
from anndata import AnnData

from scpdac.models import load_scanvi_model
from scpdac.pp import slice_and_bin

UNLABELED = "Unknown"
EMB_KEY = "X_scANVI_emb"


def embed_and_predict(
    adata: AnnData,
    species: str,
    *,
    layer: str = "counts",
    batch_key: str = "Sample_ID",
) -> AnnData:
    """Embed a query dataset with a packaged SCANVI model and predict cell types.

    This is the lightweight, no-surgery path: it assumes the query batches are
    compatible with the reference model registry. Use :func:`extend_atlas` when
    the query introduces new batches.

    Parameters
    ----------
    adata
        Query data. Must contain raw counts in ``adata.layers[layer]`` and a
        ``batch_key`` column in ``adata.obs``.
    species
        ``"human"`` or ``"mouse"`` — selects the packaged model and gene panel.
    layer
        Layer holding raw counts to bin before inference.
    batch_key
        Column in ``adata.obs`` identifying the batch covariate.

    Returns
    -------
    The input ``adata`` with the latent embedding in ``adata.obsm["X_scANVI_emb"]``
    and predicted labels in ``adata.obs["predicted_celltype"]``.

    Raises
    ------
    ValueError
        If ``batch_key`` is absent from ``adata.obs``.
    """
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Input AnnData must have {batch_key!r} in obs to use the model.")

    query = slice_and_bin(adata, species=species, binning=50, layer_key=layer)
    query.obs["Level_4"] = UNLABELED
    model = load_scanvi_model(species, adata=query)
    model.is_trained = True
    adata.obsm[EMB_KEY] = model.get_latent_representation(adata=query)
    adata.obs["predicted_celltype"] = model.predict(adata=query)
    return adata


def extend_atlas(
    query: AnnData,
    atlas: AnnData,
    species: str,
    *,
    layer: str = "counts",
    max_epochs: int = 10,
    batch_key: str = "Sample_ID",
    freeze_dropout: bool = True,
) -> AnnData:
    """Map a query dataset onto the reference atlas via scArches surgery.

    The pretrained SCANVI model is adapted to the query with
    :meth:`~scvi.model.SCANVI.load_query_data` (scArches architectural surgery),
    which tolerates batches not present in the reference registry. The query and
    reference are then embedded in a shared latent space and concatenated into an
    expanded reference object, **retaining all user-supplied query metadata**.

    Parameters
    ----------
    query
        Query data with raw counts in ``adata.layers[layer]`` and ``batch_key`` in obs.
        All of its ``obs`` columns are preserved in the returned object.
    atlas
        Reference atlas AnnData (raw counts in the same ``layer``).
    species
        ``"human"`` or ``"mouse"``.
    layer
        Layer holding raw counts to bin.
    max_epochs
        Epochs to fine-tune the surgery (query) model.
    batch_key
        Column in ``adata.obs`` identifying the batch covariate.
    freeze_dropout
        Passed to scArches surgery; freezes dropout in transferred layers.

    Returns
    -------
    An expanded reference :class:`~anndata.AnnData` containing both atlas and
    query cells in a shared ``obsm["X_scANVI_emb"]`` space, with a ``"source"``
    column ("atlas"/"query"), a joint UMAP, and predicted labels for the query in
    ``obs["predicted_celltype"]``.

    Raises
    ------
    ValueError
        If ``batch_key`` is absent from ``query.obs``.
    """
    if batch_key not in query.obs.columns:
        raise ValueError(f"Input AnnData must have {batch_key!r} in obs to use the model.")

    import scarches as sca

    atlas_binned = slice_and_bin(atlas, species=species, binning=50, layer_key=layer)
    query_binned = slice_and_bin(query, species=species, binning=50, layer_key=layer)
    query_binned.obs["Level_4"] = UNLABELED

    ref_model = load_scanvi_model(species, adata=query_binned, freeze_dropout=freeze_dropout)

    ref_model._unlabeled_indices = np.arange(query_binned.n_obs)
    ref_model._labeled_indices = []
    ref_model.train(max_epochs=max_epochs)

    atlas_manager = ref_model.adata_manager.transfer_fields(atlas_binned, extend_categories=True)
    query_manager = ref_model.adata_manager.transfer_fields(query_binned, extend_categories=True)

    ref_model._register_manager_for_instance(atlas_manager)
    atlas_binned = atlas_manager.adata

    atlas_binned.obsm[EMB_KEY] = ref_model.get_latent_representation(adata=atlas_binned)
    atlas_binned.obs["predicted_celltype"] = ref_model.predict(adata=atlas_binned)

    ref_model._register_manager_for_instance(query_manager)
    query_binned = query_manager.adata

    query_binned.obsm[EMB_KEY] = ref_model.get_latent_representation(adata=query_binned)
    query_binned.obs["predicted_celltype"] = ref_model.predict(adata=query_binned)

    for col in atlas.obs.columns:
        if col not in query.obs.columns:
            if isinstance(atlas.obs[col].dtype, pd.CategoricalDtype):
                query.obs[col] = pd.Series(index=query.obs.index, dtype=atlas.obs[col].dtype)
            else:
                query.obs[col] = np.nan

    expanded = ad.concat(
        [atlas, query],
        join="inner",
        label="source",
        keys=["atlas", "query"],
        merge="first",
    )
    expanded.obsm[EMB_KEY] = np.concatenate([atlas_binned.obsm[EMB_KEY], query_binned.obsm[EMB_KEY]], axis=0)
    expanded.obs["predicted_celltype"] = np.concatenate([atlas_binned.obs["predicted_celltype"], query_binned.obs["predicted_celltype"]], axis=0)

    return expanded
