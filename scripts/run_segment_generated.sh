#!/bin/bash
# Ri-segmenta con FAST i volumi generati dalla ControlNet, per il calcolo del DSC.
# Per ogni <base>_synth.nii.gz produce <base>_synth_pveseg.nii.gz nella stessa cartella.
#
# I volumi generati NON vengono piu' clippati a 1.0 nell'inferenza (il clip creava un
# muro di voxel saturi che rompeva la stima EM di FAST). Nel loro range nativo FAST li
# segmenta DIRETTAMENTE, senza preparazione ne' rumore artificiale.
#
# FALLBACK: se un volume dovesse comunque collassare (meno di 4 classi), si ritenta con
# la preparazione di src/evaluation/prepare_for_fast.py (range clinico + rumore leggero)
# a sigma crescente. E' un'eccezione, non la norma.
#
# Uso (dalla radice della repo):
#   bash scripts/run_segment_generated.sh data/controlnet_gen_val/epoch100

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SIF=~/containers/fsl.sif
NPROC=12
FALLBACK_SIGMAS="10 20 30"

GENDIR_ARG="$1"
if [ -z "$GENDIR_ARG" ]; then
    echo "Uso: bash scripts/run_segment_generated.sh <gen_dir>"
    exit 1
fi

source "$REPO/.venv/bin/activate"
cd "$REPO"

GENDIR_ABS="$(cd "$GENDIR_ARG" && pwd)"
GENDIR_REL="${GENDIR_ABS#$REPO/}"
if [ "$GENDIR_REL" = "$GENDIR_ABS" ]; then
    echo "ERRORE: $GENDIR_ABS non e' dentro la repo $REPO"; exit 1
fi

PREP_DIR="$GENDIR_ABS/_fast_prep"
STATUS_DIR="$GENDIR_ABS/_fast_status"
mkdir -p "$PREP_DIR" "$STATUS_DIR"

echo "Start: $(date)"
echo "GenDir: $GENDIR_REL"

mapfile -t VOLS < <(ls "$GENDIR_ABS"/*_synth.nii.gz 2>/dev/null)
TOT=${#VOLS[@]}
echo "Volumi da segmentare: $TOT"

seg_one() {
    local vol="$1"
    local base
    base=$(basename "$vol" _synth.nii.gz)
    local out_rel="${GENDIR_REL}/${base}_synth"

    # gia' fatto e valido? salta
    if [ -f "$REPO/${out_rel}_pveseg.nii.gz" ] && \
       python3 -m src.evaluation.check_mask_classes --mask "$REPO/${out_rel}_pveseg.nii.gz" --quiet; then
        echo "skip" > "$STATUS_DIR/${base}.status"
        return 0
    fi

    # --- tentativo PRINCIPALE: FAST diretto sul volume nativo (nessuna preparazione) ---
    singularity exec -B "$REPO":/mnt "$SIF" \
        fast -t 1 -n 3 -o "/mnt/${out_rel}" "/mnt/${GENDIR_REL}/${base}_synth.nii.gz" > /dev/null 2>&1 || true
    rm -f "$REPO/${out_rel}_seg.nii.gz" "$REPO/${out_rel}_mixeltype.nii.gz" \
          "$REPO/${out_rel}_pve_"*.nii.gz 2>/dev/null
    if [ -f "$REPO/${out_rel}_pveseg.nii.gz" ] && \
       python3 -m src.evaluation.check_mask_classes --mask "$REPO/${out_rel}_pveseg.nii.gz" --quiet; then
        echo "ok diretto" > "$STATUS_DIR/${base}.status"
        return 0
    fi

    # --- FALLBACK: preparazione con rumore crescente ---
    local prep="$PREP_DIR/${base}_prep.nii.gz"
    local prep_rel="${GENDIR_REL}/_fast_prep/${base}_prep.nii.gz"
    for sigma in $FALLBACK_SIGMAS; do
        python3 -m src.evaluation.prepare_for_fast --in "$vol" --out "$prep" --noise_std "$sigma"
        singularity exec -B "$REPO":/mnt "$SIF" \
            fast -t 1 -n 3 -o "/mnt/${out_rel}" "/mnt/${prep_rel}" > /dev/null 2>&1 || true
        rm -f "$REPO/${out_rel}_seg.nii.gz" "$REPO/${out_rel}_mixeltype.nii.gz" \
              "$REPO/${out_rel}_pve_"*.nii.gz 2>/dev/null
        if [ -f "$REPO/${out_rel}_pveseg.nii.gz" ] && \
           python3 -m src.evaluation.check_mask_classes --mask "$REPO/${out_rel}_pveseg.nii.gz" --quiet; then
            echo "ok fallback sigma=$sigma" > "$STATUS_DIR/${base}.status"
            rm -f "$prep"
            return 0
        fi
    done

    echo "FAILED" > "$STATUS_DIR/${base}.status"
    rm -f "$prep"
    return 0
}
export -f seg_one
export REPO SIF GENDIR_REL GENDIR_ABS PREP_DIR STATUS_DIR FALLBACK_SIGMAS

printf "%s\n" "${VOLS[@]}" | xargs -P "$NPROC" -I {} bash -c 'seg_one "$@"' _ {} &
XPID=$!
while kill -0 $XPID 2>/dev/null; do
    DONE=$(ls "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
    echo "  progresso: $DONE/$TOT   ($(date +%H:%M:%S))"
    sleep 60
done
wait $XPID || true

DIRETTO=$(grep -l "ok diretto" "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
FALLBACK=$(grep -l "ok fallback" "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
SKIP=$(grep -l "skip" "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
FAIL=$(grep -l "FAILED" "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
NPVE=$(ls "$GENDIR_ABS"/*_synth_pveseg.nii.gz 2>/dev/null | wc -l)

echo ""
echo "=== RIEPILOGO ==="
echo "  maschere prodotte  : $NPVE/$TOT"
echo "  FAST diretto (ok)  : $DIRETTO"
echo "  serviti da fallback: $FALLBACK"
echo "  gia' presenti      : $SKIP"
echo "  FALLITI            : $FAIL"
if [ "$FALLBACK" -gt 0 ]; then
    echo "  volumi che hanno richiesto il fallback:"
    grep -l "ok fallback" "$STATUS_DIR"/*.status 2>/dev/null | while read f; do
        echo "    - $(basename "$f" .status): $(cat "$f")"
    done
fi
if [ "$FAIL" -gt 0 ]; then
    echo "  volumi falliti:"
    grep -l "FAILED" "$STATUS_DIR"/*.status 2>/dev/null | while read f; do echo "    - $(basename "$f" .status)"; done
fi

rmdir "$PREP_DIR" 2>/dev/null || true
echo "End: $(date)"