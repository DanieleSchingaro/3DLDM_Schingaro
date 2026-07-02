#!/bin/bash
#SBATCH --job-name=sample_ldm
#SBATCH --output=logs/sample_ldm_%j.log
#SBATCH --error=logs/sample_ldm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro

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
LDM_CKPT="./outputs/models_v4/ldm_unet_epoch800.pt"       # <-- GOOD (FID-best): metti il tuo
LDM_CKPT_BAD="./outputs/models_v4/ldm_unet_epoch200.pt"   # <-- BAD (~30% del good): metti il tuo
GUIDANCE_SCALE=2.0

# numero di campioni da generare (default 100, sovrascrivibile da CLI)
N_SAMPLES=${1:-100}

# stampa info ambiente
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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
    --guidance_scale $GUIDANCE_SCALE

echo "End: $(date)"