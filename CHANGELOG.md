# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

## [0.1.0] - 2026-07-01

### Added

- Atlas mapping via scArches surgery (`tl.extend_atlas`) and fast embed-only
  label transfer (`tl.embed_and_predict`).
- Hierarchical MLP classifier (`tl.predict_labels`, `HierarchicalClassifier`)
  splitting malignant vs non-malignant before fine-grained typing.
- Gene-panel alignment and quantile binning (`pp.get_genes`, `pp.align_to_genes`,
  `pp.bin_data`, `pp.slice_and_bin`) with manual human/mouse panels.
- Pretrained human and mouse SCANVI models and classifier checkpoints, with
  `models.load_scanvi_model` and `models.load_classifier_checkpoints`.
- Label-distribution plot (`pl.plot_label_distribution`).
- Training, conversion and SLURM scripts, plus tutorial notebooks and a
  performance benchmark page.

### Changed

- Load reference SCANVI models with scArches `load_query_data`.
- Made internal modules private (`_basic`, `_binning`, `_genes`).

### Removed

- Placeholder `basic` tool and example notebook.

[unreleased]: https://github.com/MDLDan/scPDAC/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MDLDan/scPDAC/releases/tag/v0.1.0
