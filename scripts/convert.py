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


def main():
    # atlas = ad.read_zarr("/data/Daniele/atlases/intermediate/Human_Atlas_Harmonised_genes_filtered.zarr")
    # atlas.layers['counts'] = atlas.X.copy()
    # atlas.obs["Sample_ID"] = atlas.obs["Dataset"].astype(str) + atlas.obs["Sample_ID"].astype(str)
    # atlas.write_zarr('/data/Daniele/atlases/final_versions/Human.zarr')
    atlas = ad.read_zarr("/data/Daniele/atlases/final_versions/Human.zarr")
    print(atlas.layers["counts"][:10000, :10000].max())


if __name__ == "__main__":
    main()
