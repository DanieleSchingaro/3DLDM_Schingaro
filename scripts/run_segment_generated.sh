#!/bin/bash
# Ri-segmenta con FAST i volumi generati dalla ControlNet, per il calcolo del DSC.
# Per ogni <base>_synth.nii.gz produce <base>_synth_pveseg.nii.gz nella stessa cartella.
#
# I volumi decodificati (in [0,1], troppo lisci) non sono segmentabili da FAST cosi'
# come sono. Prima di FAST vengono preparati da src/evaluation/prepare_for_fast.py
# (range clinico + rumore leggero, seed deterministico). La preparazione e' SOLO per
# la segmentazione: i volumi salvati in [0,1] restano intatti per le metriche FID.
#
# Uso:
#   bash scripts/run_segment_generated.sh <gen_dir>
# Esempio:
#   bash scripts/run_segment_generated.sh data/controlnet_gen_val/epoch100

set -e

GENDIR="$1"
if [ -z "$GENDIR" ]; then
    echo "Uso: bash scripts/run_segment_generated.sh <gen_dir>"
    exit 1
fi

REPO=/mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
SIF=~/containers/fsl.sif
NPROC=12
PREP_DIR="$GENDIR/_fast_prep"

source "$REPO/.venv/bin/activate"
cd "$REPO"
echo "Start: $(date)"
echo "Segmento i generati in: $GENDIR"
mkdir -p "$PREP_DIR"

mapfile -t VOLS < <(ls "$GENDIR"/*_synth.nii.gz 2>/dev/null)
echo "Volumi da segmentare: ${#VOLS[@]}"

seg_one() {
    local vol="$1"
    local base
    base=$(basename "$vol" _synth.nii.gz)
    local prep="$PREP_DIR/${base}_prep.nii.gz"
    local out_rel="${GENDIR#$REPO/}/${base}_synth"

    if [ -f "$REPO/${out_rel}_pveseg.nii.gz" ]; then
        return 0
    fi

    # preparazione per FAST (modulo python dedicato)
    python3 -m src.evaluation.prepare_for_fast --in "$vol" --out "$prep"

    # FAST sul volume preparato
    singularity exec -B "$REPO":/mnt "$SIF" \
        fast -t 1 -n 3 -o "/mnt/${out_rel}" "/mnt/${PREP_DIR#$REPO/}/${base}_prep.nii.gz" > /dev/null 2>&1

    # tieni solo _pveseg; rimuovi output FAST superflui e temporaneo
    rm -f "$REPO/${out_rel}_seg.nii.gz" "$REPO/${out_rel}_mixeltype.nii.gz" \
          "$REPO/${out_rel}_pve_0.nii.gz" "$REPO/${out_rel}_pve_1.nii.gz" "$REPO/${out_rel}_pve_2.nii.gz" \
          "$prep" 2>/dev/null
}
export -f seg_one
export REPO SIF GENDIR PREP_DIR

printf "%s\n" "${VOLS[@]}" | xargs -P "$NPROC" -I {} bash -c 'seg_one "$@"' _ {}

rmdir "$PREP_DIR" 2>/dev/null || true
NDONE=$(ls "$GENDIR"/*_synth_pveseg.nii.gz 2>/dev/null | wc -l)
echo "Segmentazioni prodotte: $NDONE"
echo "End: $(date)"