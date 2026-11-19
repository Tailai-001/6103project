#!/bin/bash
# SLURM template for C's formal runs. Set the partition, then submit from the repo root
# (mkdir -p logs first, SLURM needs it for --output). Activate your env or pass PY=... .
# train.py has no resume, so --time must cover a whole run; one arm per job is safest:
#   C_ARMS=e3  sbatch scripts/train_member_c.slurm
#   C_ARMS=e4  sbatch scripts/train_member_c.slurm
#   C_ARMS=e4r sbatch scripts/train_member_c.slurm
# --dependency=singleton keeps jobs with this name one-at-a-time (clean walltimes, no
# concurrent registry writes). Running all three in one job needs the sum of their times.
#SBATCH --job-name=AI6103_C
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/c_%j.out
#SBATCH --error=logs/c_%j.err
#SBATCH --dependency=singleton
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this job from the repo root}"
export PY=${PY:-python}
nvidia-smi
bash scripts/run_member_c.sh
