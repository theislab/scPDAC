#!/usr/bin/env python
r"""Train and evaluate the 3-model hierarchical MLP classifier on a PDAC atlas.

Models (all trained on log-normalised expression over the species manual-gene
panel — **not** binned data):

1. root           — Malignant vs Non-Malignant
2. malignant      — Level-3 labels among malignant cells
3. nonmalignant   — Level-3 labels among non-malignant cells

The **test** set is not a random split: the two datasets (``adata.obs['Dataset']``)
with the fewest cells are held out entirely, so evaluation measures transfer to
studies the models never saw. The three models never see those cells during
training. After training, the script evaluates every model on that held-out set
and writes per-class F1 bar plots + a metrics table:

* ``f1_root.png``                 — Malignant vs Non-Malignant
* ``f1_malignant_level3.png``     — Level-3 within malignant cells
* ``f1_nonmalignant_level3.png``  — Level-3 within non-malignant cells
* ``f1_combined_level3.png``      — full hierarchy, end-to-end Level-3

The datasets chosen as the hold-out are recorded in ``test_split.csv`` (one row
per dataset, with its cell count and train/test role) and in every checkpoint.
Every held-out call is also dumped to ``predictions.csv``, from which
``scripts/plot_performance.py`` rebuilds the figures without retraining.

Each model is saved as a ``.pt`` checkpoint holding ``state_dict``, the gene
order, the class list, the architecture, and the held-out dataset names, so
inference can realign features.

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
from sklearn.preprocessing import LabelEncoder
from torch import nn

# Plot code lives next door so the figures can be rebuilt from predictions.csv
# without retraining; this script and `plot_performance.py` share it.
from plot_performance import TASKS, plot_f1_bars

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


def dataset_holdout_split(
    datasets: np.ndarray, n_test: int, explicit: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series]:
    """Hold out whole datasets, returning ``(train_idx, test_idx, test_names, counts)``.

    Unless ``explicit`` names the datasets to hold out, the ``n_test`` datasets
    with the **fewest cells** become the test set. Ties are broken by name so the
    choice is reproducible. Splitting on ``Dataset`` rather than at random means
    no study contributes cells to both sides, so the reported scores describe
    transfer to an unseen study rather than in-distribution performance.
    """
    counts = pd.Series(datasets).value_counts()
    if explicit:
        missing = sorted(set(explicit) - set(counts.index))
        if missing:
            raise ValueError(f"--test-datasets not found in the atlas: {missing}")
        test_names = list(explicit)
    else:
        if n_test >= len(counts):
            raise ValueError(f"--n-test-datasets={n_test} but the atlas only has {len(counts)} datasets")
        # Sort by (count, name) so equal-sized datasets order deterministically.
        test_names = list(counts.sort_index().sort_values(kind="stable").index[:n_test])

    is_test = np.isin(datasets, test_names)
    if not is_test.any():
        raise ValueError(f"held-out datasets {test_names} select no cells")
    return np.flatnonzero(~is_test), np.flatnonzero(is_test), test_names, counts


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
    seed: int = 0,
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
    perm = np.random.RandomState(seed).permutation(n)
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


def main() -> None:
    """Parse CLI args, train the three classifiers, and evaluate on the held-out datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=["human", "mouse"], required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--out-dir", default=None, help=f"Checkpoint dir. Default: {DEFAULT_OUT}/<species>")
    parser.add_argument("--eval-dir", default=None, help=f"Plots/metrics dir. Default: {DEFAULT_EVAL}/<species>")
    parser.add_argument("--log1p-layer", default="log_norm")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--labels-key", default="Level_3")
    parser.add_argument("--dataset-key", default="Dataset", help="obs column holding the study of origin.")
    parser.add_argument("--n-test-datasets", type=int, default=4, help="Hold out the N smallest datasets as test.")
    parser.add_argument(
        "--test-datasets", nargs="+", default=None, help="Hold out these datasets instead of the N smallest."
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (within the train split).")
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

    # Hold out whole datasets (the smallest ones) shared across all three models.
    datasets = atlas.obs[args.dataset_key].astype(str).to_numpy()
 
    train_idx, test_idx, test_datasets, dataset_counts = dataset_holdout_split(
        datasets, args.n_test_datasets, args.test_datasets
    )
    print(f"\nHeld-out datasets: {', '.join(test_datasets)}")
    print(f"{len(train_idx)} train / {len(test_idx)} held-out test cells")

    missing_in_test = sorted(set(level3[train_idx]) - set(level3[test_idx]))
    if missing_in_test:
        print(f"Classes absent from the hold-out (not scored): {', '.join(missing_in_test)}")

    # Record the split so the docs can state exactly which studies were held out.
    split_df = pd.DataFrame(
        {
            "dataset": dataset_counts.index.astype(str),
            "n_cells": dataset_counts.to_numpy(),
            "split": np.where(np.isin(dataset_counts.index.astype(str), test_datasets), "test", "train"),
        }
    ).sort_values(["split", "n_cells"], ascending=[True, False])
    split_df.to_csv(eval_dir / "test_split.csv", index=False)
    print(f"  wrote {eval_dir / 'test_split.csv'}")

    def save(name: str, ckpt: dict) -> None:
        ckpt["genes"] = list(genes)
        ckpt["input_layer"] = args.log1p_layer
        ckpt["test_datasets"] = list(test_datasets)
        torch.save(ckpt, out_dir / f"{name}.pt")
        print(f"Saved {name}.pt ({len(ckpt['classes'])} classes, val_acc={ckpt['best_val_acc']:.3f})")

    train_kwargs = {
        "hidden": hidden,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_frac": args.val_frac,
        "device": args.device,
        "seed": args.seed,
        "num_workers": args.num_workers,
    }

    # --- Train the three models on the training datasets only ----------------
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

    # --- Evaluate every model on the held-out datasets -----------------------
    print(f"\nEvaluating on {len(test_idx)} held-out cells; writing plots to {eval_dir}")
    x_test = x[test_idx]
    level3_test = level3[test_idx]
    root_y_test = root_y[test_idx]
    mal_test = mal_mask[test_idx]

    # 1. Root: Malignant vs Non-Malignant.
    root_pred = mlp_predict(root_model, x_test, root_ckpt["classes"], args.device)

    # 2/3. Level-3 sub-classifiers, each on its *true* cells.
    mal_pred = mlp_predict(mal_model, x_test[mal_test], mal_ckpt["classes"], args.device)
    nonmal_pred = mlp_predict(nonmal_model, x_test[~mal_test], nonmal_ckpt["classes"], args.device)

    # 4. Combined: full hierarchy end-to-end (root routes cells to a sub-model).
    pred_route_mal = root_pred == MALIGNANT
    combined = np.empty(len(test_idx), dtype=object)
    if pred_route_mal.any():
        combined[pred_route_mal] = mlp_predict(mal_model, x_test[pred_route_mal], mal_ckpt["classes"], args.device)
    if (~pred_route_mal).any():
        combined[~pred_route_mal] = mlp_predict(
            nonmal_model, x_test[~pred_route_mal], nonmal_ckpt["classes"], args.device
        )

    # Which held-out cells each task scores, and what it predicted for them.
    all_rows = np.arange(len(test_idx))
    scored = {
        "root_malignant": (all_rows, root_y_test, root_pred),
        "malignant_level3": (np.flatnonzero(mal_test), level3_test[mal_test], mal_pred),
        "nonmalignant_level3": (np.flatnonzero(~mal_test), level3_test[~mal_test], nonmal_pred),
        "combined_level3": (all_rows, level3_test, combined.astype(str)),
    }

    cells_test = atlas.obs_names.to_numpy()[test_idx]
    datasets_test = datasets[test_idx]
    metrics: dict[str, dict] = {}
    frames: list[pd.DataFrame] = []
    for task, (rows, y_true, y_pred) in scored.items():
        title, filename = TASKS[task]
        if len(rows) == 0:
            print(f"  skipping {task}: no held-out cells")
            continue
        metrics[task] = plot_f1_bars(y_true, y_pred, title, eval_dir / filename)
        frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "cell": cells_test[rows],
                    "dataset": datasets_test[rows],
                    "y_true": np.asarray(y_true, dtype=str),
                    "y_pred": np.asarray(y_pred, dtype=str),
                }
            )
        )

    # Persist the raw calls so the figures can be rebuilt (and remetriced) later
    # with scripts/plot_performance.py, no retraining required.
    pd.concat(frames, ignore_index=True).to_csv(eval_dir / "predictions.csv", index=False)
    print(f"  wrote {eval_dir / 'predictions.csv'}")

    metrics_df = pd.DataFrame(metrics).T.rename_axis("task").reset_index()
    metrics_df.insert(1, "test_datasets", "|".join(test_datasets))
    metrics_df.to_csv(eval_dir / "metrics.csv", index=False)
    print("\n" + metrics_df.to_string(index=False))
    print(f"\nDone. Checkpoints -> {out_dir}   |   evaluation -> {eval_dir}")


if __name__ == "__main__":
    main()
