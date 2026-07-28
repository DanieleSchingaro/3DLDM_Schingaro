#!/bin/bash
# Ri-segmenta con FAST i volumi generati dalla ControlNet, per il DSC.
# Per ogni <base>_synth.nii.gz produce <base>_synth_pveseg.nii.gz nella stessa cartella.
#
# I volumi decodificati (in [0,1], troppo lisci) non sono segmentabili da FAST cosi'
# come sono: vengono preparati da src/evaluation/prepare_for_fast.py (range clinico +
# rumore leggero, seed deterministico) solo per la segmentazione. I volumi in [0,1]
# restano intatti per le metriche FID.
#
# Uso (lanciare dalla RADICE della repo):
#   bash scripts/segment_generated.sh data/controlnet_gen_val/epoch100
# GENDIR puo' essere relativo alla repo o assoluto: viene normalizzato.

set -e

GENDIR_ARG="$1"
if [ -z "$GENDIR_ARG" ]; then
    echo "Uso: bash scripts/segment_generated.sh <gen_dir>"
    exit 1
fi

# REPO = path canonico della repo, ricavato dalla posizione dello script (robusto:
# niente path hardcodato). Lo script sta in <repo>/scripts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SIF=~/containers/fsl.sif
NPROC=12

source "$REPO/.venv/bin/activate"
cd "$REPO"

# GENDIR normalizzato ad ASSOLUTO e canonico, poi reso RELATIVO alla repo per i path
# interni al container (che monta la repo su /mnt).
GENDIR_ABS="$(cd "$GENDIR_ARG" && pwd)"
GENDIR_REL="${GENDIR_ABS#$REPO/}"
if [ "$GENDIR_REL" = "$GENDIR_ABS" ]; then
    echo "ERRORE: $GENDIR_ABS non e' dentro la repo $REPO"
    exit 1
fi

PREP_DIR="$GENDIR_ABS/_fast_prep"
mkdir -p "$PREP_DIR"

echo "Start: $(date)"
echo "Repo:   $REPO"
echo "GenDir: $GENDIR_REL (rel. alla repo)"

mapfile -t VOLS < <(ls "$GENDIR_ABS"/*_synth.nii.gz 2>/dev/null)
echo "Volumi da segmentare: ${#VOLS[@]}"

seg_one() {
    local vol="$1"
    local base
    base=$(basename "$vol" _synth.nii.gz)
    local prep="$PREP_DIR/${base}_prep.nii.gz"
    local out_rel="${GENDIR_REL}/${base}_synth"        # path relativo alla repo (-> /mnt/<rel> nel container)
    local prep_rel="${GENDIR_REL}/_fast_prep/${base}_prep.nii.gz"

    if [ -f "$REPO/${out_rel}_pveseg.nii.gz" ]; then
        return 0
    fi

    python3 -m src.evaluation.prepare_for_fast --in "$vol" --out "$prep"

    singularity exec -B "$REPO":/mnt "$SIF" \
        fast -t 1 -n 3 -o "/mnt/${out_rel}" "/mnt/${prep_rel}" > /dev/null 2>&1

    rm -f "$REPO/${out_rel}_seg.nii.gz" "$REPO/${out_rel}_mixeltype.nii.gz" \
          "$REPO/${out_rel}_pve_0.nii.gz" "$REPO/${out_rel}_pve_1.nii.gz" "$REPO/${out_rel}_pve_2.nii.gz" \
          "$prep" 2>/dev/null
}
export -f seg_one
export REPO SIF GENDIR_REL GENDIR_ABS PREP_DIR

printf "%s\n" "${VOLS[@]}" | xargs -P "$NPROC" -I {} bash -c 'seg_one "$@"' _ {}

rmdir "$PREP_DIR" 2>/dev/null || true
NDONE=$(ls "$GENDIR_ABS"/*_synth_pveseg.nii.gz 2>/dev/null | wc -l)
echo "Segmentazioni prodotte: $NDONE"
echo "End: $(date)"