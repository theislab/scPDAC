#!/usr/bin/env python
r"""Pretrain a SCANVI reference model on a (human or mouse) PDAC atlas.

The atlas is sliced to the species manual-gene panel and 50-bin quantised; an
SCVI model is trained on the binned layer and converted to SCANVI using the
``Level_4`` labels. The resulting SCANVI model is saved with ``SCANVI.save``.

Example
-------
    python scripts/pretrain_scanvi.py \\
        --species human \\
        --atlas /home/daniele/atlases/Human_Atlas_Harmonised.zarr \\
        --out src/scpdac/models/scanvi/human_scanvi
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import scvi

from scpdac.pp import slice_and_bin


def read_atlas(path: str) -> ad.AnnData:
    """Read an atlas from a ``.zarr`` or ``.h5ad`` file."""
    p = Path(path)
    if p.suffix == ".zarr" or p.is_dir():
        return ad.read_zarr(path)
    return ad.read_h5ad(path)


def main() -> None:
    """Parse CLI args and train/save the SCANVI model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=["human", "mouse"], required=True)
    parser.add_argument("--atlas", required=True, help="Path to the atlas (.zarr or .h5ad).")
    parser.add_argument("--out", required=True, help="Output directory for the SCANVI model.")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--labels-key", default="Level_4")
    parser.add_argument("--batch-key", default="Sample_ID")
    parser.add_argument("--unlabeled-category", default="Unknown")
    parser.add_argument("--max-epochs-scvi", type=int, default=100)
    parser.add_argument("--max-epochs-scanvi", type=int, default=10)
    parser.add_argument("--n-bins", type=int, default=50)
    args = parser.parse_args()
    atlas = read_atlas(args.atlas)
    atlas = slice_and_bin(atlas, species=args.species, binning=args.n_bins, layer_key=args.counts_layer)

    scvi.model.SCVI.setup_anndata(atlas, layer="binned_data", batch_key=args.batch_key)
    vae = scvi.model.SCVI(atlas)
    vae.train(max_epochs=args.max_epochs_scvi)

    scanvi = scvi.model.SCANVI.from_scvi_model(
        vae,
        unlabeled_category=args.unlabeled_category,
        labels_key=args.labels_key,
    )
    scanvi.train(max_epochs=args.max_epochs_scanvi)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    scanvi.save(args.out, overwrite=True)
    print(f"Saved SCANVI model for {args.species} to {args.out}")


if __name__ == "__main__":
    main()
