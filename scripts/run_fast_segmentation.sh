#!/bin/bash
# Segmentazione FSL FAST (3 tessuti: CSF/GM/WM) di tutti i volumi del dataset,
# via container Singularity. Output: le maschere _pveseg a risoluzione nativa
# 181x217x181 in data/masks_fsl_prePad_181/. Il padding a 256^3 e' un passo
# separato (pad_masks_to256.py), per isolare lo stadio lento (FAST) da quello
# fragile (padding).
#
# Caratteristiche:
#   - parallelo su N_JOBS processi (FAST e' single-thread)
#   - RESUME-SAFE: salta i volumi che hanno gia' la maschera _pveseg
#   - pulizia al volo: tiene solo _pveseg.nii.gz, cancella gli altri output FAST
#   - la lista dei volumi arriva da src.data.list_volumes (niente parsing in bash)
#
# Uso:
#   bash scripts/run_fast_segmentation.sh
#   bash scripts/run_fast_segmentation.sh 14      # override N_JOBS
#
# Lanciare in tmux: ~4 min/volume, ~5-6h totali con 12 job.

set -u

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
mkdir -p logs

REPO="/mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro"
SIF="$HOME/containers/fsl.sif"
OUT_DIR="data/masks_fsl_prePad_181"
N_JOBS="${1:-12}"

mkdir -p "$OUT_DIR"

[ -f "$SIF" ] || { echo "container non trovato: $SIF"; exit 1; }

LOG="logs/fast_seg_$(date +%Y%m%d_%H%M).log"
echo "Start: $(date)" | tee "$LOG"
echo "N_JOBS: $N_JOBS | out: $OUT_DIR" | tee -a "$LOG"

# ---- 1) lista dei volumi (path relativi alla repo) da list_volumes ----
VOL_LIST="logs/_vol_list.txt"
python3 -m src.data.list_volumes --splits_path data/splits/dataset.json > "$VOL_LIST"
TOTAL=$(wc -l < "$VOL_LIST")
echo "Volumi totali da processare: $TOTAL" | tee -a "$LOG"

# ---- 2) funzione di segmentazione di UN volume (esportata per xargs) ----
segment_one() {
    local rel="$1"
    local base
    base=$(basename "$rel" .nii.gz)
    local out_prefix="$OUT_DIR/$base"

    if [ -f "${out_prefix}_pveseg.nii.gz" ]; then
        echo "SKIP (gia' fatto): $base"
        return 0
    fi

    singularity exec -B "$REPO":/mnt "$SIF" \
        fast -t 1 -n 3 -o "/mnt/${out_prefix}" "/mnt/${rel}" \
        > /dev/null 2>&1

    if [ -f "${out_prefix}_pveseg.nii.gz" ]; then
        rm -f "${out_prefix}_seg.nii.gz" \
              "${out_prefix}_mixeltype.nii.gz" \
              "${out_prefix}_pve_0.nii.gz" \
              "${out_prefix}_pve_1.nii.gz" \
              "${out_prefix}_pve_2.nii.gz" 2>/dev/null
        echo "OK: $base"
    else
        echo "FALLITO: $base"
    fi
}
export -f segment_one
export REPO SIF OUT_DIR

# ---- 3) esecuzione parallela ----
cat "$VOL_LIST" \
    | xargs -P "$N_JOBS" -I {} bash -c 'segment_one "$@"' _ {} \
    | tee -a "$LOG"

# ---- 4) resoconto ----
DONE=$(ls "$OUT_DIR"/*_pveseg.nii.gz 2>/dev/null | wc -l)
echo "" | tee -a "$LOG"
echo "Maschere prodotte: $DONE / $TOTAL" | tee -a "$LOG"
echo "End: $(date)" | tee -a "$LOG"

if [ "$DONE" -lt "$TOTAL" ]; then
    echo "ATTENZIONE: mancano $((TOTAL - DONE)) maschere. Rilancia lo script (resume automatico)." | tee -a "$LOG"
fi