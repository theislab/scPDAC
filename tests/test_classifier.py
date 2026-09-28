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


def test_predict_default_returns_labels_only():
    clf, _, _ = _make_classifier()
    assert len(clf.predict(_make_adata(), layer="log1p_norm")) == 2


def test_predict_return_uncertainties():
    clf, _, _ = _make_classifier()
    adata = _make_adata()
    labels_only = clf.predict(adata, layer="log1p_norm")
    malignant_labels, celltype, mal_unc, ct_unc = clf.predict(adata, layer="log1p_norm", return_uncertainties=True)

    np.testing.assert_array_equal(malignant_labels, labels_only[0])
    np.testing.assert_array_equal(celltype, labels_only[1])
    for unc in (mal_unc, ct_unc):
        assert unc.shape == (adata.n_obs,)
        assert np.isfinite(unc).all()
        assert ((unc >= 0) & (unc <= 1 + 1e-6)).all()


def test_softmax_entropy_bounds():
    from scpdac.tl._classifier import _softmax_entropy

    ent = _softmax_entropy(torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]))
    assert torch.allclose(ent, torch.tensor([1.0, 0.0]), atol=1e-5)


def test_predict_labels_uncertainty_columns(monkeypatch):
    clf, _, _ = _make_classifier()
    monkeypatch.setattr(
        scpdac.tl._classifier.HierarchicalClassifier,
        "from_species",
        classmethod(lambda cls, species, device="cpu": clf),
    )
    cols = ["predicted_malignant_uncertainty_score", "predicted_celltype_uncertainty_score"]

    adata = predict_labels(_make_adata(), species="human")
    assert not set(cols) & set(adata.obs.columns)

    adata = predict_labels(_make_adata(), species="human", return_uncertainties=True)
    for col in cols:
        assert adata.obs[col].dtype.kind == "f"
        assert adata.obs[col].notna().all()


def _linear_ckpt(genes, classes, weight, bias):
    """Checkpoint for a hidden-layer-free MLP with hand-set ``logits = x @ weight.T + bias``."""
    model = MLP(n_genes=len(genes), n_classes=len(classes), hidden=())
    with torch.no_grad():
        model.net[0].weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
        model.net[0].bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    return {"state_dict": model.state_dict(), "genes": list(genes), "classes": list(classes), "hidden": []}


def _constant_ckpt(genes, classes, logits):
    """Checkpoint whose logits are ``logits`` for every cell, regardless of input."""
    return _linear_ckpt(genes, classes, np.zeros((len(classes), len(genes))), logits)


def _normalised_entropy(logits):
    p = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=-1).numpy()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


MAL_CLASSES = ["Malignant_Ductal", "Malignant_Acinar"]
NONMAL_CLASSES = ["T cell", "B cell", "Macrophage"]


def test_softmax_entropy_matches_scipy():
    from scipy.stats import entropy

    from scpdac.tl._classifier import _softmax_entropy

    logits = torch.randn(20, 5, generator=torch.Generator().manual_seed(0))
    expected = entropy(torch.softmax(logits, dim=1).numpy(), axis=1) / np.log(5)
    np.testing.assert_allclose(_softmax_entropy(logits).numpy(), expected, rtol=1e-5)


def test_softmax_entropy_single_class_is_zero():
    from scpdac.tl._classifier import _softmax_entropy

    assert torch.equal(_softmax_entropy(torch.randn(4, 1)), torch.zeros(4))


def test_uncertainty_confident_root_uniform_subclassifier():
    # Root is ~certain every cell is malignant; the malignant sub-model is a coin flip.
    root = _constant_ckpt(GENES, ["Malignant", "Non-Malignant"], [20.0, -20.0])
    malignant = _constant_ckpt(GENES, MAL_CLASSES, [0.0, 0.0])
    non_malignant = _constant_ckpt(GENES, NONMAL_CLASSES, [0.0, 0.0, 0.0])
    clf = HierarchicalClassifier(root, malignant, non_malignant)

    malignant_labels, _, mal_unc, ct_unc = clf.predict(_make_adata(), return_uncertainties=True)

    assert (malignant_labels == "Malignant").all()
    np.testing.assert_allclose(mal_unc, 0.0, atol=1e-6)
    # Every cell took the malignant branch, so none is left at the NaN fill value.
    np.testing.assert_allclose(ct_unc, 1.0, atol=1e-6)


def test_uncertainty_non_malignant_branch_value():
    root = _constant_ckpt(GENES, ["Malignant", "Non-Malignant"], [-1.0, 1.0])
    malignant = _constant_ckpt(GENES, MAL_CLASSES, [0.0, 0.0])
    non_malignant = _constant_ckpt(GENES, NONMAL_CLASSES, [2.0, 1.0, 0.0])
    clf = HierarchicalClassifier(root, malignant, non_malignant)

    malignant_labels, celltype, mal_unc, ct_unc = clf.predict(_make_adata(), return_uncertainties=True)

    assert (malignant_labels == "Non-Malignant").all()
    assert (celltype == "T cell").all()
    np.testing.assert_allclose(mal_unc, _normalised_entropy([-1.0, 1.0]), rtol=1e-5)
    np.testing.assert_allclose(ct_unc, _normalised_entropy([2.0, 1.0, 0.0]), rtol=1e-5)


def _mixed_routing_classifier():
    """Root routes on gene A (A=1 -> Malignant, A=0 -> Non-Malignant); sub-models have distinct entropies."""
    root = _linear_ckpt(GENES, ["Malignant", "Non-Malignant"], [[3.0, 0, 0], [-3.0, 0, 0]], [-1.5, 1.5])
    malignant = _constant_ckpt(GENES, MAL_CLASSES, [3.0, 0.0])
    non_malignant = _constant_ckpt(GENES, NONMAL_CLASSES, [0.5, 0.0, 0.0])
    return HierarchicalClassifier(root, malignant, non_malignant)


def _mixed_adata():
    adata = _make_adata(n=10)
    adata.X[:, 0] = [1, 0] * 5
    adata.layers["log1p_norm"] = adata.X.copy()
    return adata


def test_celltype_uncertainty_comes_from_routed_model():
    clf = _mixed_routing_classifier()
    adata = _mixed_adata()

    malignant_labels, _, mal_unc, ct_unc = clf.predict(adata, return_uncertainties=True)
    mal_mask = malignant_labels == "Malignant"

    np.testing.assert_array_equal(mal_mask, [True, False] * 5)
    np.testing.assert_allclose(mal_unc, _normalised_entropy([1.5, -1.5]), rtol=1e-5)
    np.testing.assert_allclose(ct_unc[mal_mask], _normalised_entropy([3.0, 0.0]), rtol=1e-5)
    np.testing.assert_allclose(ct_unc[~mal_mask], _normalised_entropy([0.5, 0.0, 0.0]), rtol=1e-5)


def test_predict_labels_uncertainty_columns_match_predict(monkeypatch):
    clf = _mixed_routing_classifier()
    monkeypatch.setattr(
        scpdac.tl._classifier.HierarchicalClassifier,
        "from_species",
        classmethod(lambda cls, species, device="cpu": clf),
    )
    adata = _mixed_adata()
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    _, _, mal_unc, ct_unc = clf.predict(adata, return_uncertainties=True)

    predict_labels(adata, species="human", return_uncertainties=True)

    np.testing.assert_allclose(adata.obs["predicted_malignant_uncertainty_score"].to_numpy(), mal_unc)
    np.testing.assert_allclose(adata.obs["predicted_celltype_uncertainty_score"].to_numpy(), ct_unc)
