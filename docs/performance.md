# Classifier performance & limitations

Every number on this page is computed on a **held-out set of whole datasets**:
Four studies from each atlas are withheld from training entirely, so no
study contributes cells to training. The models are then scored on
donors, protocols, and batches they have never seen, which makes these numbers a
**cross-study transfer** estimate rather than an in-distribution ceiling.

Read them as a **prior, not a guarantee**. Use the
per-class F1 values to know which labels the model resolves confidently and which
it does not, then check the predictions against *your* data — canonical markers,
expected compositions, known biology — before treating any call as fact.

```{note}
Both the **human** and the **mouse** atlas are trained and scored at **Level-3**
resolution.
```

## The held-out datasets

The split is made by `scripts/train_classifier.py`, which ranks
`adata.obs['Dataset']` by cell count and holds out four studies. The
full ranking, with each dataset's role, is written to
`scripts/eval_outputs/<species>/test_split.csv`, and the held-out names are also
stored inside every `.pt` checkpoint under `test_datasets`.

| Species | Held-out datasets | Test cells |
| --- | --- | ---: |
| Human | `Kemp_2020`, `Schlesinger_2020`, `Lin_2020`, `Elyada_2019` | 40,999 |
| Mouse | `Chen_2021`, `Erdem_2024`, `Han_2023`, `Rupert_2025` | 77,848 |

## Headline metrics

Each row is one stage of the hierarchy. `weighted-F1` averages the per-class F1
scores weighted by each class's support in the hold-out, so it describes the
hierarchy as applied to a realistic cell mixture and is dominated by the abundant
populations. It says nothing about how the rare classes fare individually — for
that, read the [per-class bars](#per-class-f1) below, where every class counts the
same regardless of size.

### Human (Level-3)

| Stage | Cells | Weighted-F1 |
| --- | ---: | ---: |
| Root — Malignant vs Non-Malignant | 40,999 | 0.981 |
| Malignant Level-3 sub-classifier | 15,329 | 0.953 |
| Non-malignant Level-3 sub-classifier | 25,670 | 0.929 |
| **Combined hierarchy (end-to-end)** | 40,999 | **0.921** |

### Mouse (Level-3)

| Stage | Cells | Weighted-F1 |
| --- | ---: | ---: |
| Root — Malignant vs Non-Malignant | 77,848 | 0.982 |
| Malignant Level-3 sub-classifier | 21,570 | 0.935 |
| Non-malignant Level-3 sub-classifier | 56,278 | 0.866 |
| **Combined hierarchy (end-to-end)** | 77,848 | **0.876** |

## Per-class F1

The `combined` plots show the **end-to-end** behaviour of
`scpdac.tl.predict_labels`: each held-out cell is routed by the root model, then
labelled by the matching sub-classifier. The per-branch plots score each
sub-classifier in isolation on its *true* cells, so the difference between a branch
plot and the combined plot is the price of root-level routing errors. The cell
count next to each class name is its support in the hold-out.

### Human (Level-3)

```{figure} _static/performance/human/f1_combined_level3.png
:alt: Human combined Level-3 per-class F1
:width: 100%

End-to-end per-class F1 for the full human hierarchy.
```

### Mouse (Level-3)

```{figure} _static/performance/mouse/f1_combined_level3.png
:alt: Mouse combined Level-3 per-class F1
:width: 100%

End-to-end per-class F1 for the full mouse hierarchy.
```

## Limitations

### Routing errors are unrecoverable

Cells are dispatched by the root call, so a *Malignant* / *Non-Malignant* mistake
sends a cell to a sub-classifier that cannot emit its true label. The root call is
near ceiling (weighted-F1 0.981 human, 0.982 mouse), so the effect is small — but
it is a hard ceiling on end-to-end performance.

### Zero-imputed genes and uncalibrated outputs

At inference the query is realigned to the training gene panel and **missing genes
are zero-imputed** — a query lacking many panel genes degrades without
raising an error.
We encourage to validate predictions against canonical markers (see the end of the
[classifier tutorial](notebooks/classifier)) before drawing biological
conclusions.

## Reproducing these figures

The plots, `metrics.csv`, `test_split.csv`, and `predictions.csv` are written by
the training script, which carves out the dataset-level hold-out and evaluates
every model on it:

```bash
python scripts/train_classifier.py \
    --species human \
    --atlas /path/to/Human_Atlas_Harmonised.zarr
```

Outputs land in `scripts/eval_outputs/<species>/`. Pass `--n-test-datasets N` to
hold out a different number of studies, or `--test-datasets A B` to
name them explicitly.

`predictions.csv` keeps the true and predicted label of every held-out cell, so
the figures can be restyled or recomputed **without retraining**:
