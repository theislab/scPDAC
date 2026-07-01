#!/bin/bash
#SBATCH --job-name=check_atlas
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail

export MAMBA_ROOT_PREFIX="/environments/daniele/micromamba"
nvidia-smi || true

micromamba run -n atlas_review python scripts/convert.py

