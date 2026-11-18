#!/bin/bash
#SBATCH --job-name=AI6103_E0
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e0_%j.out
#SBATCH --error=logs/e0_%j.err

set -e

cd /home/msai/li0034qi/6103/6103project

echo "===== AI6103 E0 Training ====="
date
hostname

nvidia-smi

echo "===== Environment ====="

CONDA_ENV=/home/msai/li0034qi/.conda/envs/ai6103

"$CONDA_ENV/bin/python" --version

echo "===== Start Training ====="

/home/msai/li0034qi/.conda/envs/ai6103/bin/python \
    -u train.py \
    --config configs/main.yaml \
    --policy random \
    --epochs 2 \
    --max-batches 2 \
    --run-name e1_smoke_v1
echo "===== Finished ====="
date
