"""Hierarchical MLP classifier for Level-4 cell-type prediction.

A 3-model, 2-step hierarchy operating on log-normalised expression
(``adata.layers["log1p_norm"]`` by default, **not** binned data):

1. **Root** — binary ``Malignant`` vs ``Non-Malignant``.
2. **Malignant sub-classifier** — Level-4 labels for malignant cells.
3. **Non-malignant sub-classifier** — Level-4 labels for non-malignant cells.

Cells are routed through step 2 based on the step-1 prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import issparse
from torch import nn

from scpdac.pp import align_to_genes

MALIGNANT = "Malignant"
NON_MALIGNANT = "Non-Malignant"


class MLP(nn.Module):
    """A simple feed-forward classifier with BatchNorm + dropout hidden blocks.

    Parameters
    ----------
    n_genes
        Number of input features (genes).
    n_classes
        Number of output classes.
    hidden
        Sizes of the hidden layers.
    dropout
        Dropout probability applied after each hidden block.
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_genes
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for a ``(n_cells, n_genes)`` input tensor."""
        return self.net(x)


def is_malignant_label(label: str) -> bool:
    """Return ``True`` if a Level-4 label denotes a malignant population.

    Malignant Level-4 cell types in the atlas always carry ``"Malignant"`` in
    their name (e.g. ``"Malignant Basal"``), so membership is a simple substring
    test. The explicit ``"Non-Malignant"`` label (emitted by the root classifier)
    is excluded, since it also contains the substring ``"Malignant"``.
    """
    s = str(label)
    return "Malignant" in s and "Non-Malignant" not in s


def derive_malignant_mask(level4: pd.Series | np.ndarray | list) -> np.ndarray:
    """Derive a boolean malignant mask from Level-4 labels.

    Parameters
    ----------
    level4
        Per-cell Level-4 label values.

    Returns
    -------
    Boolean array, ``True`` where the label is malignant.
    """
    return np.array([is_malignant_label(v) for v in np.asarray(level4)])


def _load_mlp(ckpt: dict, device: str = "cpu") -> MLP:
    """Instantiate and load an :class:`MLP` from a checkpoint dict."""
    model = MLP(
        n_genes=len(ckpt["genes"]),
        n_classes=len(ckpt["classes"]),
        hidden=tuple(ckpt.get("hidden", (512, 256))),
        dropout=float(ckpt.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def _features(adata: AnnData, genes: list[str], layer: str, device: str = "cpu") -> torch.Tensor:
    """Align ``adata`` to ``genes`` and return a dense float32 feature tensor.

    Zero-imputes any missing genes (with a warning) so the feature order always
    matches the model's training panel. The input ``adata`` is not modified.
    """
    aligned = align_to_genes(adata, genes, missing="zeros", layer=layer)
    mat = aligned.layers[layer] if layer in aligned.layers else aligned.X
    if issparse(mat):
        mat = mat.toarray()
    return torch.as_tensor(np.asarray(mat, dtype=np.float32), device=device)


class HierarchicalClassifier:
    """Wraps the three MLPs of the hierarchical classifier for one species.

    Construct via :meth:`from_species` to load the packaged weights, or pass
    three checkpoint dicts directly (useful for testing).

    Parameters
    ----------
    root
        Checkpoint dict for the root (malignant vs non-malignant) model.
    malignant
        Checkpoint dict for the malignant Level-4 sub-classifier.
    non_malignant
        Checkpoint dict for the non-malignant Level-4 sub-classifier.
    device
        Torch device to run inference on.
    """

    def __init__(self, root: dict, malignant: dict, non_malignant: dict, device: str = "cpu"):
        self.device = device
        self._ckpts = {"root": root, "malignant": malignant, "non_malignant": non_malignant}
        self.root = _load_mlp(root, device)
        self.malignant = _load_mlp(malignant, device)
        self.non_malignant = _load_mlp(non_malignant, device)

    @classmethod
    def from_species(cls, species: str, device: str = "cpu") -> HierarchicalClassifier:
        """Load the packaged classifier checkpoints for ``species``."""
        from scpdac.models import load_classifier_checkpoints

        ckpts = load_classifier_checkpoints(species, map_location=device)
        return cls(ckpts["root"], ckpts["malignant"], ckpts["non_malignant"], device=device)

    def _predict_one(self, model: MLP, ckpt: dict, adata: AnnData, layer: str) -> np.ndarray:
        feats = _features(adata, ckpt["genes"], layer, self.device)
        with torch.no_grad():
            idx = model(feats).argmax(dim=1).cpu().numpy()
        return np.asarray(ckpt["classes"])[idx]

    def predict(self, adata: AnnData, layer: str = "log1p_norm") -> tuple[np.ndarray, np.ndarray]:
        """Run the 2-step hierarchy.

        Returns
        -------
        A tuple ``(malignant_labels, celltype_labels)`` of length ``adata.n_obs``.
        """
        root_labels = self._predict_one(self.root, self._ckpts["root"], adata, layer)
        mal_mask = derive_malignant_mask(root_labels)

        celltype = np.empty(adata.n_obs, dtype=object)
        if mal_mask.any():
            celltype[mal_mask] = self._predict_one(self.malignant, self._ckpts["malignant"], adata[mal_mask], layer)
        if (~mal_mask).any():
            celltype[~mal_mask] = self._predict_one(
                self.non_malignant, self._ckpts["non_malignant"], adata[~mal_mask], layer
            )
        malignant_labels = np.where(mal_mask, MALIGNANT, NON_MALIGNANT)
        return malignant_labels, celltype.astype(str)


def predict_labels(
    adata: AnnData,
    species: str,
    *,
    layer: str = "log1p_norm",
    device: str = "cpu",
) -> AnnData:
    """Predict hierarchical Level-4 cell-type labels for a query dataset.

    Routes cells through the root classifier then the matching Level-4
    sub-classifier, appending predictions to ``adata.obs`` in place.

    Parameters
    ----------
    adata
        Query data. Log-normalised expression is read from ``adata.layers[layer]``
        (falling back to ``adata.X`` if the layer is absent). The original object
        is preserved; a gene-aligned copy is used internally for inference.
    species
        ``"human"`` or ``"mouse"``.
    layer
        Layer holding the log-normalised counts (``"log1p_norm"`` by default).
    device
        Torch device for inference.

    Returns
    -------
    The input ``adata`` with ``obs["predicted_malignant"]`` (``Malignant`` /
    ``Non-Malignant``) and ``obs["predicted_celltype"]`` (Level-4 labels).
    """
    clf = HierarchicalClassifier.from_species(species, device=device)
    malignant_labels, celltype = clf.predict(adata, layer=layer)
    adata.obs["predicted_malignant"] = pd.Categorical(malignant_labels)
    adata.obs["predicted_celltype"] = pd.Categorical(celltype)
    return adata
