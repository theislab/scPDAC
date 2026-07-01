#!/bin/bash
#SBATCH --job-name=scpdac_clf
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=124G
#SBATCH --time=24:00:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail

#ATLAS_DIR="${ATLAS_DIR:-/data/Daniele/atlases/}"
ATLAS_DIR="${ATLAS_DIR:-/home/daniele/atlases/}"

export MAMBA_ROOT_PREFIX="/environments/daniele/micromamba"
# Unbuffered stdout/stderr so progress shows up in the slurm .out file live
# (otherwise Python block-buffers when output is a file, not a terminal).
export PYTHONUNBUFFERED=1

nvidia-smi || true

micromamba run -n atlas_review python -u scripts/train_classifier.py \
    --species human \
    --atlas "${ATLAS_DIR}/Human_Atlas_Harmonised.zarr" \
    --device cuda \
    --hidden 512 256 128 64

micromamba run -n atlas_review python -u scripts/train_classifier.py \
    --species mouse \
    --atlas "${ATLAS_DIR}/Mouse_Atlas_Harmonised.zarr" \
    --device cuda \
    --hidden 512 256 128 64


#Human_Atlas_Harmonised_30k.h5ad" 