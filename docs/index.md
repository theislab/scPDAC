# scPDAC

[![Tests](https://img.shields.io/github/actions/workflow/status/theislab/scPDAC/test.yaml?branch=main)](https://github.com/theislab/scPDAC/actions/workflows/test.yaml)
[![Documentation](https://img.shields.io/readthedocs/scPDAC)](https://scPDAC.readthedocs.io)

**scPDAC** maps and annotates single-cell RNA-seq data against pancreatic ductal
adenocarcinoma (PDAC) reference atlases. It ships pretrained models for **human**
and **mouse** and turns a raw-count `AnnData` into an annotated, atlas-integrated
one in a few lines — aligning your genes to each model's panel automatically and
writing predictions straight back into `.obs`.

## Two workflows

**🧬 Atlas mapping** — project a query into the reference SCANVI latent space and
share an embedding with the atlas.

- `scpdac.tl.extend_atlas` — scArches surgery; tolerates new batches and returns an
  expanded atlas with your query metadata intact.
- `scpdac.tl.embed_and_predict` — fast, no-surgery embedding + label transfer.

  → **[Atlas mapping tutorial](notebooks/mapping)**

**🏷️ Hierarchical annotation** — label cells with a 3-model hierarchical MLP
classifier.

- `scpdac.tl.predict_labels` — splits *Malignant* vs *Non-Malignant*, then assigns
  fine-grained cell types with a dedicated sub-classifier per branch.
- Runs on log-normalised expression and realigns to the training gene panel.

  → **[Hierarchical classifier tutorial](notebooks/classifier)**

## Installation

You need Python 3.11 or newer. If you don't have Python, we recommend [uv][].

Install the latest development version:

```bash
pip install scpdac
```

## Where to go next

- 🧬 **[Atlas mapping tutorial](notebooks/mapping)** — map a query onto a reference
  PDAC atlas end-to-end.
- 🏷️ **[Hierarchical classifier tutorial](notebooks/classifier)** — annotate an
  unlabelled dataset in a single call.
- 📊 **[Performance & limitations](performance)** — benchmarks on held-out studies
  and, just as importantly, where **not** to trust the models.
- 📚 **[API reference](api)** — every public function and class in `scpdac`.

## Citation

If you use **scPDAC** in your research, please cite {cite:t}`Lucarelli2026`:

> Lucarelli D, Parikh S, Jiménez S, *et al.* Cross-species single-cell atlases
> chart progression, therapy-driven remodelling and immune evasion in pancreatic
> cancer. *bioRxiv* (2026).
> doi:[10.64898/2026.03.19.712924](https://doi.org/10.64898/2026.03.19.712924)

## Getting help

For questions, help requests, or to report a bug, please open an issue on the
[issue tracker](https://github.com/theislab/scPDAC/issues).

[uv]: https://github.com/astral-sh/uv

```{toctree}
:hidden: true
:maxdepth: 2

api.md
tutorials.md
```

```{toctree}
:hidden: true
:maxdepth: 1

performance.md
changelog.md
contributing.md
references.md
```
