#!/usr/bin/env python
r"""Train and evaluate the 3-model hierarchical MLP classifier on a PDAC atlas.

Models (all trained on log-normalised expression over the species manual-gene
panel — **not** binned data):

1. root           — Malignant vs Non-Malignant
2. malignant      — Level-3 labels among malignant cells
3. nonmalignant   — Level-3 labels among non-malignant cells

A stratified held-out **test** split is carved out up front; the three models
never see it during training. After training, the script evaluates every model
on that held-out test set and writes accuracy bar plots + a metrics table:

* ``accuracy_root.png``                 — Malignant vs Non-Malignant
* ``accuracy_malignant_level3.png``     — Level-3 within malignant cells
* ``accuracy_nonmalignant_level3.png``  — Level-3 within non-malignant cells
* ``accuracy_combined_level3.png``      — full hierarchy, end-to-end Level-3

Each model is saved as a ``.pt`` checkpoint holding ``state_dict``, the gene
order, the class list, and the architecture, so inference can realign features.

Example
-------
    python scripts/train_classifier.py \\
        --species human \\
        --atlas /home/daniele/atlases/Human_Atlas_Harmonised.zarr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn

from scpdac.tl import MLP, derive_malignant_mask
from scpdac.tl._data import make_loader, predict_indices

# Reuse the exact preprocessing the inference path applies, so train and serve
# see identical feature panels and ordering.
from scpdac.pp import align_to_genes, get_genes

DEFAULT_OUT = "src/scpdac/models/classifier"
DEFAULT_EVAL = "scripts/eval_outputs"
MALIGNANT, NON_MALIGNANT = "Malignant", "Non-Malignant"


def read_atlas(path: str) -> ad.AnnData:
    """Read an atlas from a ``.zarr`` or ``.h5ad`` file."""
    p = Path(path)
    print(f"Reading atlas from {path} ...")
    if p.suffix == ".zarr" or p.is_dir():
        return ad.read_zarr(path)
    return ad.read_h5ad(path)


def ensure_log1p_norm(adata: ad.AnnData, layer: str, counts_layer: str) -> None:
    """Ensure ``adata.layers[layer]`` holds log1p-normalised counts."""
    if layer in adata.layers:
        return
    src = counts_layer if counts_layer in adata.layers else None
    tmp = ad.AnnData(X=adata.layers[src].copy() if src else adata.X.copy())
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    adata.layers[layer] = tmp.X


def stratified_split(y: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, test_idx)`` stratified on ``y`` where possible.

    Falls back to a plain random split if any class is too small to stratify.
    """
    idx = np.arange(len(y))
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=stratify)
    return train_idx, test_idx


def train_mlp(
    x,
    y: np.ndarray,
    classes: list[str],
    *,
    hidden: tuple[int, ...],
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    device: str,
    num_workers: int = 0,
) -> tuple[dict, MLP]:
    """Train an :class:`~scpdac.tl.MLP` and return ``(checkpoint, best_model)``.

    ``x`` may be sparse: rows are densified one mini-batch at a time via
    :class:`~scpdac.tl._data.scDataset`, so the full dense matrix is
    never materialised. The model is selected on an internal validation split
    (carved from the training data only) and returned in ``eval`` mode.
    """
    enc = LabelEncoder().fit(classes)
    y_idx = enc.transform(y)

    n = len(y_idx)
    perm = np.random.RandomState(0).permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_val, y_val = x[val_idx], y_idx[val_idx]
    loader = make_loader(
        x[train_idx], y_idx[train_idx], batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    n_batches = len(loader)

    model = MLP(n_genes=x.shape[1], n_classes=len(classes), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for b, (xb, yb) in enumerate(loader):
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            if b % 500 == 0:
                print(f"  epoch {epoch + 1}/{epochs}  batch {b}/{n_batches}  loss={loss.item():.3f}", flush=True)
        pred = predict_indices(model, x_val, device=device, batch_size=max(batch_size, 1024))
        acc = float((pred == y_val).mean())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch + 1}/{epochs}  val_acc={acc:.3f}")

    model.load_state_dict(best_state)
    model.to(device).eval()
    ckpt = {
        "state_dict": best_state,
        "classes": list(enc.classes_),
        "hidden": list(hidden),
        "best_val_acc": best_acc,
    }
    return ckpt, model


def mlp_predict(model: MLP, x, classes: list[str], device: str) -> np.ndarray:
    """Run ``model`` on ``x`` (densifying in batches) and map indices to labels."""
    idx = predict_indices(model, x, device=device)
    return np.asarray(classes, dtype=object)[idx] if len(idx) else np.empty(0, dtype=object)


def plot_accuracy_bars(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> dict[str, float]:
    """Save a horizontal per-class accuracy (recall) bar plot and return summary metrics."""
    import matplotlib.pyplot as plt

    classes = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    rec = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    order = np.argsort(rec)
    classes_sorted = [classes[i] for i in order]
    rec_sorted = rec[order]

    overall = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.42 * len(classes) + 1.2)))
    colors = plt.cm.viridis(np.clip(rec_sorted, 0, 1))
    ax.barh(range(len(classes_sorted)), rec_sorted, color=colors, edgecolor="white", height=0.78)
    ax.set_yticks(range(len(classes_sorted)))
    ax.set_yticklabels(classes_sorted)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-class accuracy (recall)")
    ax.set_title(f"{title}\noverall acc = {overall:.3f}   macro-F1 = {macro_f1:.3f}", fontsize=11)
    for i, v in enumerate(rec_sorted):
        ax.text(min(v + 0.012, 0.98), i, f"{v:.2f}", va="center", ha="left", fontsize=8, color="#222")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}  (acc={overall:.3f}, macro-F1={macro_f1:.3f})")

    return {
        "accuracy": overall,
        "macro_f1": macro_f1,
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "n_cells": int(len(y_true)),
    }


def main() -> None:
    """Parse CLI args, train the three classifiers, and evaluate on a held-out split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=["human", "mouse"], required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--out-dir", default=None, help=f"Checkpoint dir. Default: {DEFAULT_OUT}/<species>")
    parser.add_argument("--eval-dir", default=None, help=f"Plots/metrics dir. Default: {DEFAULT_EVAL}/<species>")
    parser.add_argument("--log1p-layer", default="log_norm")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--labels-key", default="Level_3")
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (within the train split).")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Held-out test fraction for evaluation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for on-the-fly densifying.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or f"{DEFAULT_OUT}/{args.species}")
    eval_dir = Path(args.eval_dir or f"{DEFAULT_EVAL}/{args.species}")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    hidden = tuple(args.hidden)

    atlas = read_atlas(args.atlas)
    ensure_log1p_norm(atlas, args.log1p_layer, args.counts_layer)

    genes = get_genes(args.species)
    atlas = align_to_genes(atlas, genes, missing="zeros", layer=args.log1p_layer)
    print(atlas)
    print(
        f"Training on {len(atlas)} cells, {atlas.shape[1]} genes, {len(set(atlas.obs[args.labels_key]))} Level-3 classes"
    )
    # Keep the feature matrix in its native (typically sparse) form; rows are
    # densified per mini-batch downstream to avoid a full dense atlas in RAM.
    x = atlas.layers[args.log1p_layer]

    level3 = atlas.obs[args.labels_key].astype(str).to_numpy()
    mal_mask = derive_malignant_mask(level3)
    root_y = np.where(mal_mask, MALIGNANT, NON_MALIGNANT)

    # Hold out a stratified test set shared across all three models.
    train_idx, test_idx = stratified_split(level3, args.test_frac, args.seed)
    print(f"{len(train_idx)} train / {len(test_idx)} held-out test cells")

    def save(name: str, ckpt: dict) -> None:
        ckpt["genes"] = list(genes)
        ckpt["input_layer"] = args.log1p_layer
        torch.save(ckpt, out_dir / f"{name}.pt")
        print(f"Saved {name}.pt ({len(ckpt['classes'])} classes, val_acc={ckpt['best_val_acc']:.3f})")

    train_kwargs = {
        "hidden": hidden,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_frac": args.val_frac,
        "device": args.device,
        "num_workers": args.num_workers,
    }

    # --- Train the three models on the training split only -------------------
    mal_train = mal_mask[train_idx]

    print("[1/3] root (Malignant vs Non-Malignant)")
    root_ckpt, root_model = train_mlp(x[train_idx], root_y[train_idx], [MALIGNANT, NON_MALIGNANT], **train_kwargs)
    save("root", root_ckpt)

    print("[2/3] malignant sub-classifier")
    mal_classes = sorted(set(level3[train_idx][mal_train]))
    mal_ckpt, mal_model = train_mlp(x[train_idx][mal_train], level3[train_idx][mal_train], mal_classes, **train_kwargs)
    save("malignant", mal_ckpt)

    print("[3/3] non-malignant sub-classifier")
    nonmal_classes = sorted(set(level3[train_idx][~mal_train]))
    nonmal_ckpt, nonmal_model = train_mlp(
        x[train_idx][~mal_train], level3[train_idx][~mal_train], nonmal_classes, **train_kwargs
    )
    save("nonmalignant", nonmal_ckpt)

    # --- Evaluate every model on the held-out test set -----------------------
    print(f"\nEvaluating on {len(test_idx)} held-out cells; writing plots to {eval_dir}")
    x_test = x[test_idx]
    level3_test = level3[test_idx]
    root_y_test = root_y[test_idx]
    mal_test = mal_mask[test_idx]

    metrics: dict[str, dict] = {}

    # 1. Root: Malignant vs Non-Malignant.
    root_pred = mlp_predict(root_model, x_test, root_ckpt["classes"], args.device)
    metrics["root_malignant"] = plot_accuracy_bars(
        root_y_test, root_pred, "Root — Malignant vs Non-Malignant", eval_dir / "accuracy_root.png"
    )

    # 2. Malignant Level-3 (on truly malignant test cells).
    metrics["malignant_level3"] = plot_accuracy_bars(
        level3_test[mal_test],
        mlp_predict(mal_model, x_test[mal_test], mal_ckpt["classes"], args.device),
        "Malignant Level-3 sub-classifier",
        eval_dir / "accuracy_malignant_level3.png",
    )

    # 3. Non-malignant Level-3 (on truly non-malignant test cells).
    metrics["nonmalignant_level3"] = plot_accuracy_bars(
        level3_test[~mal_test],
        mlp_predict(nonmal_model, x_test[~mal_test], nonmal_ckpt["classes"], args.device),
        "Non-malignant Level-3 sub-classifier",
        eval_dir / "accuracy_nonmalignant_level3.png",
    )

    # 4. Combined: full hierarchy end-to-end (root routes cells to a sub-model).
    pred_route_mal = root_pred == MALIGNANT
    combined = np.empty(len(test_idx), dtype=object)
    if pred_route_mal.any():
        combined[pred_route_mal] = mlp_predict(mal_model, x_test[pred_route_mal], mal_ckpt["classes"], args.device)
    if (~pred_route_mal).any():
        combined[~pred_route_mal] = mlp_predict(
            nonmal_model, x_test[~pred_route_mal], nonmal_ckpt["classes"], args.device
        )
    metrics["combined_level3"] = plot_accuracy_bars(
        level3_test, combined.astype(str), "Combined hierarchy — Level-3", eval_dir / "accuracy_combined_level3.png"
    )

    metrics_df = pd.DataFrame(metrics).T.rename_axis("task").reset_index()
    metrics_df.to_csv(eval_dir / "metrics.csv", index=False)
    print("\n" + metrics_df.to_string(index=False))
    print(f"\nDone. Checkpoints -> {out_dir}   |   evaluation -> {eval_dir}")


if __name__ == "__main__":
    main()
