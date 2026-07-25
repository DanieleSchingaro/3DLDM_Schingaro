#!/bin/bash
# Segmentazione FSL FAST (3 tessuti: CSF/GM/WM) di tutti i volumi del dataset,
# via container Singularity. Output: maschere _pveseg a 181x217x181 in
# data/masks_fsl_prePad_181/. Il padding a 256^3 e' un passo separato.
#
# NB path: si usa "$PWD" (risolto a runtime) invece di hardcodare il percorso
# della repo, per evitare ambiguita' tra /home/ubuntu/... e /mnt/data/home-ubuntu/...
# FAST scrive in /tmp DENTRO il container (isolato); si copia il solo _pveseg
# sull'host tramite il mount /out.
#
#   - parallelo su N_JOBS processi (FAST e' single-thread)
#   - RESUME-SAFE
#   - lista volumi da src.data.list_volumes
#
# Uso: bash scripts/run_fast_segmentation.sh [N_JOBS]
# In tmux: ~4 min/volume, ~5-6h con 12 job.

set -u

# vai nella repo (dir dello script/../) e attiva il venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"
source "$REPO/.venv/bin/activate"
mkdir -p logs

SIF="$HOME/containers/fsl.sif"
OUT_DIR="data/masks_fsl_prePad_181"
N_JOBS="${1:-12}"

mkdir -p "$OUT_DIR"
[ -f "$SIF" ] || { echo "container non trovato: $SIF"; exit 1; }

echo "REPO (runtime): $REPO"
LOG="logs/fast_seg_$(date +%Y%m%d_%H%M).log"
echo "Start: $(date)" | tee "$LOG"
echo "N_JOBS: $N_JOBS | out: $OUT_DIR" | tee -a "$LOG"

VOL_LIST="logs/_vol_list.txt"
python3 -m src.data.list_volumes --splits_path data/splits/dataset.json > "$VOL_LIST"
TOTAL=$(wc -l < "$VOL_LIST")
echo "Volumi totali: $TOTAL" | tee -a "$LOG"

segment_one() {
    local rel="$1"
    local base
    base=$(basename "$rel" .nii.gz)
    local final="$OUT_DIR/${base}_pveseg.nii.gz"

    if [ -f "$final" ]; then
        echo "SKIP: $base"
        return 0
    fi

    # $REPO montato su /mnt (input); $REPO/$OUT_DIR su /out (output).
    # FAST scrive in /tmp del container, poi copia il solo _pveseg su /out.
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

cat "$VOL_LIST" \
    | xargs -P "$N_JOBS" -I {} bash -c 'segment_one "$@"' _ {} \
    | tee -a "$LOG"

DONE=$(ls "$OUT_DIR"/*_pveseg.nii.gz 2>/dev/null | wc -l)
echo "" | tee -a "$LOG"
echo "Maschere prodotte: $DONE / $TOTAL" | tee -a "$LOG"
echo "End: $(date)" | tee -a "$LOG"
[ "$DONE" -lt "$TOTAL" ] && echo "Mancano $((TOTAL-DONE)). Rilancia (resume)." | tee -a "$LOG"