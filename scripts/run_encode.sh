#!/bin/bash
# v4: encode con il VAE v2 (--checkpoint). Output in embeddings_v4.
# Lo stesso VAE v2 deve decodificare in sample/eval (config trained_autoencoder_path).

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
mkdir -p logs

# stampa info ambiente
echo "Start: $(date)"
echo "GPU disponibili: $(nvidia-smi --list-gpus | wc -l)"

MASTER_PORT=$((29000 + RANDOM % 2000))
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m src.data.encode_dataset \
    --checkpoint outputs/models_v2/autoencoder_best.pt \
    --embedding_dir data/processed/embeddings_v5 \
    2>&1 | tee logs/encode_v5_$(date +%Y%m%d_%H%M).log

echo "End: $(date)"