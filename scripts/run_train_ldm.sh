#!/bin/bash

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
mkdir -p logs

# stampa info ambiente
echo "Start: $(date)"
echo "GPU disponibili: $(nvidia-smi --list-gpus | wc -l)"

MASTER_PORT=$((29000 + RANDOM % 2000))
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m src.training.train_ldm \
    2>&1 | tee logs/train_ldm_v5_$(date +%Y%m%d_%H%M).log

echo "End: $(date)"