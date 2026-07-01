"""Basic plotting helpers for inspecting predictions and mappings."""

from __future__ import annotations

from anndata import AnnData


def plot_label_distribution(adata: AnnData, key: str = "predicted_celltype", *, ax=None):
    """Plot a horizontal bar chart of the value counts of an ``obs`` column.

    Handy for inspecting the output of :func:`scpdac.tl.predict_labels`.

    Parameters
    ----------
    adata
        AnnData whose ``obs[key]`` will be summarised.
    key
        The ``obs`` column to count (e.g. ``"predicted_celltype"`` or
        ``"predicted_malignant"``).
    ax
        Optional Matplotlib axis to draw on; created if not provided.

    Returns
    -------
    The Matplotlib axis containing the plot.

    Raises
    ------
    KeyError
        If ``key`` is not present in ``adata.obs``.
    """
    import matplotlib.pyplot as plt

    if key not in adata.obs.columns:
        raise KeyError(f"{key!r} not found in adata.obs.")

    counts = adata.obs[key].value_counts().sort_values()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(2, 0.3 * len(counts))))
    counts.plot.barh(ax=ax)
    ax.set_xlabel("number of cells")
    ax.set_title(f"Distribution of {key}")
    return ax


def basic_plot(adata: AnnData) -> int:
    """Placeholder plotting function kept for template/back-compat purposes.

    Parameters
    ----------
    adata
        The AnnData object to plot.

    Returns
    -------
    Always ``0``.
    """
    print("Use scpdac.pl.plot_label_distribution or scanpy plotting utilities.")
    return 0
