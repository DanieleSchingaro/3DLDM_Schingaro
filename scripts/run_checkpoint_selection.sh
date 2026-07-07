#!/bin/bash

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
mkdir -p logs

# ============================================================
# PARAMETRI (modificabili)
# ------------------------------------------------------------
# N_SAMPLES     : campioni generati per checkpoint (default 100).
# REFINE_TOP    : quanti top checkpoint rivalutare con autoguidance (fase 2).
# GUIDANCE_SCALE: scala w dell'autoguidance nella fase 2 (uguale a run_sample.sh).
# NB: checkpoint_selection.py gestisce il multi-GPU internamente con mp.spawn,
#     quindi NON si usa torchrun: si lancia con python3 e lo script distribuisce
#     i checkpoint su tutte le GPU disponibili da solo.
# ============================================================
N_SAMPLES=${1:-100}
REFINE_TOP=3
GUIDANCE_SCALE=2.0

# stampa info ambiente
echo "Start: $(date)"
echo "GPU disponibili: $(nvidia-smi --list-gpus | wc -l)"
echo "Campioni per checkpoint: $N_SAMPLES"
echo "Refine top: $REFINE_TOP | Guidance scale: $GUIDANCE_SCALE"

python3 -m src.evaluation.checkpoint_selection \
    --models_dir outputs/models_v4 \
    --work_dir outputs/checkpoint_selection_v4 \
    --n_samples $N_SAMPLES \
    --refine_top $REFINE_TOP \
    --guidance_scale $GUIDANCE_SCALE \
    2>&1 | tee logs/cksel_v4_$(date +%Y%m%d_%H%M).log

echo "End: $(date)"