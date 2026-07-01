#!/bin/bash
#SBATCH --job-name=scpdac_scanvi
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail

ATLAS_DIR="${ATLAS_DIR:-/data/Daniele/atlases/final_versions/}"

export MAMBA_ROOT_PREFIX="/environments/daniele/micromamba"
nvidia-smi || true

micromamba run -n atlas_review python scripts/train_scanvi.py \
    --species human \
    --atlas "${ATLAS_DIR}/Human.zarr" \
    --out src/scpdac/models/scanvi/human_scanvi \
    --batch-key "Sample_ID"

#micromamba run -n atlas_review python scripts/train_scanvi.py \
#    --species mouse \
#    --atlas "${ATLAS_DIR}/Mouse.zarr" \
#    --out src/scpdac/models/scanvi/mouse_scanvi \
#    --batch-key "Sample_ID"