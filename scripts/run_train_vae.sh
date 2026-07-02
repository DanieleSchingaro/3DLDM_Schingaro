#!/bin/bash
#SBATCH --job-name=train_vae
#SBATCH --output=logs/train_vae_%j.log
#SBATCH --error=logs/train_vae_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=500:00:00

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro

# stampa info ambiente
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "GPU disponibili: $(nvidia-smi --list-gpus | wc -l)"

MASTER_PORT=$((29000 + RANDOM % 2000))
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m src.training.train_vae

echo "End: $(date)"