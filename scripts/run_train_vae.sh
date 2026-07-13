#!/bin/bash
# NB v4: il VAE v4 (warmup lineare) e' risultato peggiore del v2. La pipeline v4
# adotta il VAE v2 gia' addestrato, quindi questo script NON si rilancia in v4;
# resta come documentazione dell'esperimento.

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
mkdir -p logs

# stampa info ambiente
echo "Start: $(date)"
echo "GPU disponibili: $(nvidia-smi --list-gpus | wc -l)"

MASTER_PORT=$((29000 + RANDOM % 2000))
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m src.training.train_vae \
    2>&1 | tee logs/train_vae_v5_$(date +%Y%m%d_%H%M).log

echo "End: $(date)"