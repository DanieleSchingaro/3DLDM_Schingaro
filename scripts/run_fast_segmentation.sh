#!/bin/bash
# Segmentazione FSL FAST (3 tessuti: CSF/GM/WM) di tutti i volumi del dataset,
# via container Singularity. Output: le maschere _pveseg a risoluzione nativa
# 181x217x181 in data/masks_fsl_prePad_181/. Il padding a 256^3 e' un passo
# separato (pad_masks_to256.py).
#
# NB: FAST scrive l'output in /tmp DENTRO il container (path assoluto isolato),
# poi si copia il solo _pveseg sull'host. Questo evita che FAST ricostruisca il
# path assoluto dell'host (/mnt/data/home-ubuntu/...) come cartelle relative.
#
# Caratteristiche:
#   - parallelo su N_JOBS processi (FAST e' single-thread)
#   - RESUME-SAFE: salta i volumi che hanno gia' la maschera _pveseg
#   - solo _pveseg tenuto sull'host; gli altri output FAST restano nel /tmp effimero
#   - lista volumi da src.data.list_volumes
#
# Uso:
#   bash scripts/run_fast_segmentation.sh
#   bash scripts/run_fast_segmentation.sh 14
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

# ---- 1) lista dei volumi (path relativi alla repo) ----
VOL_LIST="logs/_vol_list.txt"
python3 -m src.data.list_volumes --splits_path data/splits/dataset.json > "$VOL_LIST"
TOTAL=$(wc -l < "$VOL_LIST")
echo "Volumi totali da processare: $TOTAL" | tee -a "$LOG"

# ---- 2) segmentazione di UN volume ----
segment_one() {
    local rel="$1"
    local base
    base=$(basename "$rel" .nii.gz)
    local final="$OUT_DIR/${base}_pveseg.nii.gz"

    # RESUME
    if [ -f "$final" ]; then
        echo "SKIP (gia' fatto): $base"
        return 0
    fi

    # FAST scrive in /tmp DENTRO il container: path assoluto isolato, nessuna
    # gerarchia host da ricostruire. La repo e' montata su /mnt (per leggere
    # l'input) e /tmp e' privato del container.
    # -o /tmp/<base>  ->  /tmp/<base>_pveseg.nii.gz  ecc.
    # Poi copiamo il solo _pveseg sull'host montando OUT_DIR su /out.
    singularity exec \
        -B "$REPO":/mnt \
        -B "$REPO/$OUT_DIR":/out \
        "$SIF" \
        bash -c "fast -t 1 -n 3 -o /tmp/${base} /mnt/${rel} > /dev/null 2>&1 && cp /tmp/${base}_pveseg.nii.gz /out/${base}_pveseg.nii.gz"

    if [ -f "$final" ]; then
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
    echo "ATTENZIONE: mancano $((TOTAL - DONE)) maschere. Rilancia (resume automatico)." | tee -a "$LOG"
fi