#!/bin/bash
# Ri-segmenta con FAST i volumi generati dalla ControlNet, per il calcolo del DSC.
# Per ogni <base>_synth.nii.gz nella cartella di input produce <base>_synth_pveseg.nii.gz.
# I generati sono gia' 256^3 -> nessun padding, FAST diretto.
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
NPROC=12   # FAST e' single-thread: parallelizza sui core

cd "$REPO"
echo "Start: $(date)"
echo "Segmento i generati in: $GENDIR"

# lista dei synth da segmentare (esclude eventuali _condmask e _pveseg gia' fatti)
mapfile -t VOLS < <(ls "$GENDIR"/*_synth.nii.gz 2>/dev/null)
echo "Volumi da segmentare: ${#VOLS[@]}"

# funzione di segmentazione di un singolo volume (path relativo alla repo -> /mnt nel container)
seg_one() {
    local vol="$1"
    local rel="${vol#$REPO/}"                    # path relativo alla repo
    local out="${rel%_synth.nii.gz}_synth"       # prefisso output (FAST aggiunge _pveseg)
    # salta se gia' fatto
    if [ -f "$REPO/${out}_pveseg.nii.gz" ]; then
        return 0
    fi
    singularity exec -B "$REPO":/mnt "$SIF" \
        fast -t 1 -n 3 -o "/mnt/${out}" "/mnt/${rel}" > /dev/null 2>&1
    # tieni solo il _pveseg, cancella gli altri output di FAST (risparmio spazio)
    rm -f "$REPO/${out}_seg.nii.gz" "$REPO/${out}_mixeltype.nii.gz" \
          "$REPO/${out}_pve_0.nii.gz" "$REPO/${out}_pve_1.nii.gz" "$REPO/${out}_pve_2.nii.gz" 2>/dev/null
}
export -f seg_one
export REPO SIF

# parallelizza sui core
printf "%s\n" "${VOLS[@]}" | xargs -P "$NPROC" -I {} bash -c 'seg_one "$@"' _ {}

NDONE=$(ls "$GENDIR"/*_synth_pveseg.nii.gz 2>/dev/null | wc -l)
echo "Segmentazioni prodotte: $NDONE"
echo "End: $(date)"