# Tutorials

Two end-to-end walkthroughs that run the `scpdac` workflows on the same public
PDAC dataset ([Chen et al. 2025](https://www.cell.com/cancer-cell/fulltext/S1535-6108(25)00393-9)):

- 🧬 **[Atlas mapping](notebooks/mapping)** — map a query onto a reference PDAC
  atlas with scArches surgery (`extend_atlas`) or a fast embed-only label transfer
  (`embed_and_predict`), then visualise both in a shared UMAP.
- 🏷️ **[Hierarchical classifier](notebooks/classifier)** — annotate an unlabelled
  dataset with fine-grained cell-type labels via `predict_labels`, then sanity-check
  the calls on a UMAP and with canonical markers.

```{toctree}
:maxdepth: 1

notebooks/mapping
notebooks/classifier
```
