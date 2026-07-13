#!/bin/bash

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
mkdir -p logs

# ============================================================
# PARAMETRI v4 (autoguidance) - MODIFICA QUI PRIMA DI LANCIARE
# ------------------------------------------------------------
# GOOD: checkpoint FID-best ottenuto da checkpoint_selection.
# BAD : checkpoint 'cattivo' per l'autoguidance = epoca precoce
#       dello STESSO run (tipicamente ~30% dell'epoca del good).
# W   : scala dell'autoguidance (sweep tipico 1.5 - 3.0).
# Per generare la BASELINE senza autoguidance: aggiungere --no_autoguidance
# alla riga torchrun e lasciare BAD vuoto.
# ============================================================
LDM_CKPT="./outputs/models_v5/ldm_unet_epoch900.pt"       # <-- GOOD (FID-best): metti il tuo
LDM_CKPT_BAD="./outputs/models_v5/ldm_unet_epoch300.pt"   # <-- BAD (~30% del good): metti il tuo
GUIDANCE_SCALE=2.0

# numero di campioni da generare (default 100, sovrascrivibile da CLI)
N_SAMPLES=${1:-100}

# stampa info ambiente
echo "Start: $(date)"
echo "GPU disponibili: $(nvidia-smi --list-gpus | wc -l)"
echo "Campioni da generare: $N_SAMPLES"
echo "LDM good: $LDM_CKPT"
echo "LDM bad : $LDM_CKPT_BAD"
echo "Guidance scale: $GUIDANCE_SCALE"

MASTER_PORT=$((29000 + RANDOM % 2000))
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m src.inference.sample \
    --n_samples $N_SAMPLES \
    --ldm_ckpt "$LDM_CKPT" \
    --ldm_ckpt_bad "$LDM_CKPT_BAD" \
    --guidance_scale $GUIDANCE_SCALE \
    2>&1 | tee logs/sample_v5_$(date +%Y%m%d_%H%M).log

echo "End: $(date)"