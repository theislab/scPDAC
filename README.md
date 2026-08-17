# scPDAC

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/scPDAC/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scPDAC

**scPDAC** maps and annotates single-cell RNA-seq data against pancreatic ductal
adenocarcinoma (PDAC) reference atlases. It ships pretrained models for **human**
and **mouse** and exposes two complementary workflows:

- 🧬 **Atlas mapping** — project a query dataset into a reference SCANVI latent
  space, either with scArches surgery (`scpdac.tl.extend_atlas`, tolerates new
  batches and returns an expanded atlas) or a fast embed-only label transfer
  (`scpdac.tl.embed_and_predict`) — all while retaining your original metadata.
- 🏷️ **Hierarchical annotation** — label cells with a 3-model hierarchical MLP
  classifier (`scpdac.tl.predict_labels`) that first splits *Malignant* vs
  *Non-Malignant* and then assigns fine-grained cell types with a dedicated
  sub-classifier for each branch.

Both paths align your genes to the model's panel automatically and write the
predictions straight back into your `AnnData`.

## Getting started

Please refer to the [documentation][]. The [atlas mapping][] and
[hierarchical classifier][] tutorials walk through both workflows end-to-end,
the [performance][] page reports held-out benchmarks and known limitations, and
the [API documentation][] lists every public function.

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].


1) Install the latest release of `scPDAC` from [PyPI][]:

```bash
pip install scPDAC
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, or to report a bug, please open an issue.

## Citation

If you use scPDAC in your research, please cite:

> Lucarelli D, Parikh S, Jiménez S, *et al.* Cross-species single-cell atlases
> chart progression, therapy-driven remodelling and immune evasion in pancreatic
> cancer. *bioRxiv* (2026). doi:[10.64898/2026.03.19.712924](https://doi.org/10.64898/2026.03.19.712924)

```bibtex
@article{Lucarelli2026,
  author    = {Lucarelli, Daniele and Parikh, Shrey and Jim{\'e}nez, Sara and others},
  title     = {Cross-species single-cell atlases chart progression, therapy-driven remodelling and immune evasion in pancreatic cancer},
  journal   = {bioRxiv},
  year      = {2026},
  doi       = {10.64898/2026.03.19.712924},
  publisher = {Cold Spring Harbor Laboratory},
  url       = {https://www.biorxiv.org/content/10.1101/2026.03.19.712924},
}
```

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/theislab/scPDAC/issues
[tests]: https://github.com/theislab/scPDAC/actions/workflows/test.yaml
[documentation]: https://scPDAC.readthedocs.io
[changelog]: https://scPDAC.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scPDAC.readthedocs.io/en/latest/api.html
[atlas mapping]: https://scPDAC.readthedocs.io/en/latest/notebooks/mapping.html
[hierarchical classifier]: https://scPDAC.readthedocs.io/en/latest/notebooks/classifier.html
[performance]: https://scPDAC.readthedocs.io/en/latest/performance.html
[pypi]: https://pypi.org/project/scPDAC
