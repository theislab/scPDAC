import numpy as np
import torch
from anndata import AnnData

import scpdac
from scpdac.tl import MLP, HierarchicalClassifier, derive_malignant_mask, predict_labels
from scpdac.tl._classifier import is_malignant_label

GENES = ["A", "B", "C"]


def test_mlp_forward_shape():
    model = MLP(n_genes=10, n_classes=3, hidden=(8,)).eval()
    out = model(torch.randn(4, 10))
    assert out.shape == (4, 3)


def test_is_malignant_label():
    assert is_malignant_label("Malignant_Ductal")
    assert is_malignant_label("Malignant")
    assert not is_malignant_label("Non-Malignant Ductal")
    assert not is_malignant_label("T cell")


def test_derive_malignant_mask():
    labels = ["Malignant_A", "Non-Malignant_B", "T cell", "Malignant"]
    assert list(derive_malignant_mask(labels)) == [True, False, False, True]


def _make_ckpt(genes, classes, hidden=(8,)):
    model = MLP(n_genes=len(genes), n_classes=len(classes), hidden=hidden)
    return {
        "state_dict": model.state_dict(),
        "genes": list(genes),
        "classes": list(classes),
        "hidden": list(hidden),
        "input_layer": "log1p_norm",
    }


def _make_classifier():
    root = _make_ckpt(GENES, ["Malignant", "Non-Malignant"])
    malignant = _make_ckpt(GENES, ["Malignant_Ductal", "Malignant_Acinar"])
    non_malignant = _make_ckpt(GENES, ["T cell", "B cell", "Macrophage"])
    return HierarchicalClassifier(root, malignant, non_malignant), {
        "Malignant_Ductal",
        "Malignant_Acinar",
    }, {"T cell", "B cell", "Macrophage"}


def _make_adata(n=12):
    rng = np.random.default_rng(0)
    ad = AnnData(X=rng.random((n, len(GENES))).astype(np.float32))
    ad.var_names = GENES
    ad.layers["log1p_norm"] = ad.X.copy()
    return ad


def test_hierarchical_routing_consistency():
    clf, mal_classes, nonmal_classes = _make_classifier()
    adata = _make_adata()
    malignant_labels, celltype = clf.predict(adata, layer="log1p_norm")

    assert len(malignant_labels) == adata.n_obs
    assert len(celltype) == adata.n_obs
    assert set(malignant_labels) <= {"Malignant", "Non-Malignant"}

    for mal, ct in zip(malignant_labels, celltype, strict=True):
        if mal == "Malignant":
            assert ct in mal_classes
        else:
            assert ct in nonmal_classes


def test_predict_labels_appends_obs(monkeypatch):
    clf, _, _ = _make_classifier()
    monkeypatch.setattr(
        scpdac.tl._classifier.HierarchicalClassifier,
        "from_species",
        classmethod(lambda cls, species, device="cpu": clf),
    )
    adata = _make_adata()
    out = predict_labels(adata, species="human")
    assert out is adata
    assert "predicted_malignant" in adata.obs
    assert "predicted_celltype" in adata.obs
    assert adata.obs["predicted_celltype"].notna().all()


def test_predict_labels_zero_imputes_missing_genes(monkeypatch):
    clf, _, _ = _make_classifier()
    monkeypatch.setattr(
        scpdac.tl._classifier.HierarchicalClassifier,
        "from_species",
        classmethod(lambda cls, species, device="cpu": clf),
    )
    rng = np.random.default_rng(1)
    adata = AnnData(X=rng.random((5, 2)).astype(np.float32))
    adata.var_names = ["A", "B"]
    adata.layers["log1p_norm"] = adata.X.copy()
    out = predict_labels(adata, species="human")
    assert out.n_obs == 5
