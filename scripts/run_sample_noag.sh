#!/bin/bash
# BASELINE SENZA AUTOGUIDANCE.
#
# Scopo: test causale sulle traslazioni quantizzate (+-32 voxel) osservate nei
# campioni generati con autoguidance (w=2.0). Stesso checkpoint 'good', stesso
# VAE v2 per il decode, stessi seed -> l'UNICA variabile e' la guida.
#
# I seed sono deterministici e dipendono solo dall'indice del campione
# (set_determinism(base_seed + idx)), quindi hc_synth_0011 qui parte dallo STESSO
# rumore iniziale di hc_synth_0011 generato con guida: il confronto e' appaiato.
# In particolare 0011 (l'unico campione veramente degenere) e' l'indice 10,
# quindi bastano >=11 campioni per includerlo; 50 danno un tasso affidabile.
#
# Output in cartelle SEPARATE: non sovrascrive nulla della generazione guidata.

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
mkdir -p logs

# ============================================================
# PARAMETRI
# ------------------------------------------------------------
# GOOD: lo STESSO checkpoint usato per la generazione guidata (FID-best).
# Nessun BAD: l'autoguidance e' disattivata con --no_autoguidance.
# ============================================================
LDM_CKPT="./outputs/models_v4/ldm_unet_epoch900.pt"

# numero di campioni (default 50; usare 100 per il confronto appaiato completo)
N_SAMPLES=${1:-50}

# stampa info ambiente
echo "Start: $(date)"
echo "GPU disponibili: $(nvidia-smi --list-gpus | wc -l)"
echo "Campioni da generare: $N_SAMPLES"
echo "LDM good: $LDM_CKPT"
echo "Autoguidance: DISATTIVATA (baseline)"

MASTER_PORT=$((29000 + RANDOM % 2000))
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m src.inference.sample \
    --n_samples $N_SAMPLES \
    --ldm_ckpt "$LDM_CKPT" \
    --no_autoguidance \
    --out_dir data/synthetic_uniform_noag \
    --png_dir outputs/generated/synthetic_v4_noag \
    2>&1 | tee logs/sample_v4_noag_$(date +%Y%m%d_%H%M).log

echo "End: $(date)"
echo ""
echo "Prossimo passo: quantificare le traslazioni sulla baseline e confrontare."
echo "  python3 -m tests.check_degenerate_samples \\"
echo "      --synth_dir data/synthetic_v4_noag \\"
echo "      --real_source test \\"
echo "      --out_json outputs/metrics/degenerate_noag.json"