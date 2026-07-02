#!/bin/bash
#SBATCH --job-name=cksel_ldm
#SBATCH --output=logs/cksel_ldm_%j.log
#SBATCH --error=logs/cksel_ldm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=35:00:00

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro

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
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "GPU disponibili: $(nvidia-smi --list-gpus | wc -l)"
echo "Campioni per checkpoint: $N_SAMPLES"
echo "Refine top: $REFINE_TOP | Guidance scale: $GUIDANCE_SCALE"

python3 -m src.evaluation.checkpoint_selection \
    --n_samples $N_SAMPLES \
    --refine_top $REFINE_TOP \
    --guidance_scale $GUIDANCE_SCALE

echo "End: $(date)"