#!/usr/bin/env python
r"""Rebuild the docs performance figures from stored hold-out predictions.

``train_classifier.py`` writes one ``predictions.csv`` per species holding the
true and predicted label of every held-out cell, for each stage of the hierarchy:

===============  =========================================================
``task``         one of the keys of :data:`TASKS`
``cell``         obs name of the held-out cell
``dataset``      the held-out study the cell came from
``y_true``       atlas label
``y_pred``       model call
===============  =========================================================

Everything the plots show — per-class F1, class supports, weighted-F1 — is
recomputed from those columns, so restyling a figure never costs a retrain.

The evaluation directories are hardcoded in :data:`EVAL_DIRS`; plots are written
back next to the ``predictions.csv`` they came from.

Example
-------
    python scripts/plot_performance.py              # both species
    python scripts/plot_performance.py --species mouse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where each species' ``predictions.csv`` lives and where its plots are written.
EVAL_DIRS = {
    "human": REPO_ROOT / "scripts" / "eval_outputs" / "human",
    "mouse": REPO_ROOT / "scripts" / "eval_outputs" / "mouse",
}

PREDICTIONS_CSV = "predictions.csv"

#: ``task`` -> (plot title, output filename). Shared with ``train_classifier.py``
#: so a figure looks identical whether it is written during training or replotted.
TASKS: dict[str, tuple[str, str]] = {
    "root_malignant": ("Root — Malignant vs Non-Malignant", "f1_root.png"),
    "malignant_level3": ("Malignant Level-3 sub-classifier", "f1_malignant_level3.png"),
    "nonmalignant_level3": ("Non-malignant Level-3 sub-classifier", "f1_nonmalignant_level3.png"),
    "combined_level3": ("Combined hierarchy — Level-3", "f1_combined_level3.png"),
}


def plot_f1_bars(y_true, y_pred, title: str, path: Path) -> dict[str, float]:
    """Save a horizontal per-class F1 bar plot and return summary metrics.

    Only classes **present in the hold-out** are scored: a class with no cells in
    the held-out datasets has no F1 to report, and including it would drag
    macro-F1 down for a population the test set cannot measure. Predictions that
    fall on such a class still count as errors against the true label.

    The bars are per-class F1; the title reports the support-weighted F1, i.e.
    the hierarchy's performance on the hold-out's actual cell mixture.
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    classes = sorted(set(y_true))
    f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    support = pd.Series(y_true).value_counts()

    order = np.argsort(f1)
    classes_sorted = [classes[i] for i in order]
    f1_sorted = f1[order]
    labels = [f"{c}  (n={support[c]:,})" for c in classes_sorted]

    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)

    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.42 * len(classes) + 1.2)))
    colors = plt.cm.viridis(np.clip(f1_sorted, 0, 1))
    ax.barh(range(len(classes_sorted)), f1_sorted, color=colors, edgecolor="white", height=0.78)
    ax.set_yticks(range(len(classes_sorted)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-class F1")
    ax.set_title(f"{title}\nweighted-F1 = {weighted_f1:.3f}", fontsize=11)
    for i, v in enumerate(f1_sorted):
        ax.text(min(v + 0.012, 0.98), i, f"{v:.2f}", va="center", ha="left", fontsize=8, color="#222")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}  (weighted-F1={weighted_f1:.3f}, macro-F1={macro_f1:.3f})")

    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "macro_precision": precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "n_cells": int(len(y_true)),
        "n_classes": len(classes),
    }


def plot_from_predictions(predictions: pd.DataFrame, eval_dir: Path) -> pd.DataFrame:
    """Write every :data:`TASKS` figure for one species and return its metrics."""
    metrics: dict[str, dict] = {}
    for task, (title, filename) in TASKS.items():
        sub = predictions[predictions["task"] == task]
        if sub.empty:
            print(f"  skipping {task}: no rows in {PREDICTIONS_CSV}")
            continue
        metrics[task] = plot_f1_bars(sub["y_true"], sub["y_pred"], title, eval_dir / filename)
    return pd.DataFrame(metrics).T.rename_axis("task").reset_index()


def replot_species(species: str, eval_dir: Path | None = None) -> pd.DataFrame:
    """Rebuild one species' figures from its stored predictions."""
    eval_dir = Path(eval_dir) if eval_dir else EVAL_DIRS[species]
    path = eval_dir / PREDICTIONS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/train_classifier.py --species {species} once to write it."
        )
    predictions = pd.read_csv(path)
    held_out = sorted(predictions["dataset"].astype(str).unique()) if "dataset" in predictions else []
    print(f"\n{species}: {path} ({len(predictions):,} rows, held-out datasets: {', '.join(held_out) or 'unknown'})")
    return plot_from_predictions(predictions, eval_dir)


def main() -> None:
    """Replot one or both species from their stored hold-out predictions."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species", choices=[*EVAL_DIRS, "both"], default="both")
    parser.add_argument("--eval-dir", default=None, help="Override the hardcoded dir (single species only).")
    args = parser.parse_args()

    species_list = list(EVAL_DIRS) if args.species == "both" else [args.species]
    if args.eval_dir and len(species_list) > 1:
        parser.error("--eval-dir requires a single --species")

    for species in species_list:
        metrics_df = replot_species(species, args.eval_dir)
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
