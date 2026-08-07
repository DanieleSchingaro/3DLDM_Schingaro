#!/bin/bash
# Ri-segmenta con FAST i volumi generati dalla ControlNet, per il calcolo del DSC.
# Per ogni <base>_synth.nii.gz produce <base>_synth_pveseg.nii.gz nella stessa cartella.
#
# I volumi decodificati (in [0,1], troppo lisci) non sono segmentabili da FAST cosi'
# come sono: la stima EM diverge (variance nan) e la maschera collassa a 2 classi.
# Ogni volume viene quindi preparato da src/evaluation/prepare_for_fast.py (range
# clinico + rumore leggero, seed deterministico) prima di FAST. La preparazione e'
# SOLO per la segmentazione: i volumi in [0,1] restano intatti per le metriche FID.
#
# ROBUSTEZZA: alcuni volumi resistono al rumore di default (sigma=10) e collassano
# comunque. Lo script verifica che la maschera abbia 4 classi e, in caso contrario,
# RITENTA automaticamente con sigma crescente (10 -> 20 -> 30). A fine esecuzione
# riporta quanti volumi sono riusciti al primo colpo, quanti con retry, quali falliti.
#
# Uso (dalla radice della repo):
#   bash scripts/run_segment_generated.sh data/controlnet_gen_val/epoch100

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SIF=~/containers/fsl.sif
NPROC=12
SIGMAS="10 20 30"        # rumore: default, poi retry crescenti

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
echo "Repo:   $REPO"
echo "GenDir: $GENDIR_REL"

mapfile -t VOLS < <(ls "$GENDIR_ABS"/*_synth.nii.gz 2>/dev/null)
TOT=${#VOLS[@]}
echo "Volumi da segmentare: $TOT"

seg_one() {
    local vol="$1"
    local base
    base=$(basename "$vol" _synth.nii.gz)
    local out_rel="${GENDIR_REL}/${base}_synth"
    local prep="$PREP_DIR/${base}_prep.nii.gz"
    local prep_rel="${GENDIR_REL}/_fast_prep/${base}_prep.nii.gz"

    # gia' fatto e valido? salta
    if [ -f "$REPO/${out_rel}_pveseg.nii.gz" ]; then
        if python3 -m src.evaluation.check_mask_classes --mask "$REPO/${out_rel}_pveseg.nii.gz" --quiet; then
            echo "skip" > "$STATUS_DIR/${base}.status"
            return 0
        fi
    fi

    local attempt=0
    for sigma in $SIGMAS; do
        attempt=$((attempt+1))
        python3 -m src.evaluation.prepare_for_fast --in "$vol" --out "$prep" --noise_std "$sigma"
        singularity exec -B "$REPO":/mnt "$SIF" \
            fast -t 1 -n 3 -o "/mnt/${out_rel}" "/mnt/${prep_rel}" > /dev/null 2>&1 || true
        # pulizia output extra di FAST
        rm -f "$REPO/${out_rel}_seg.nii.gz" "$REPO/${out_rel}_mixeltype.nii.gz" \
              "$REPO/${out_rel}_pve_"*.nii.gz 2>/dev/null
        # la maschera ha 4 classi?
        if [ -f "$REPO/${out_rel}_pveseg.nii.gz" ] && \
           python3 -m src.evaluation.check_mask_classes --mask "$REPO/${out_rel}_pveseg.nii.gz" --quiet; then
            echo "ok sigma=$sigma attempt=$attempt" > "$STATUS_DIR/${base}.status"
            rm -f "$prep"
            return 0
        fi
    done

    # nessun sigma ha funzionato
    echo "FAILED" > "$STATUS_DIR/${base}.status"
    rm -f "$prep"
    return 0
}
export -f seg_one
export REPO SIF GENDIR_REL GENDIR_ABS PREP_DIR STATUS_DIR SIGMAS

printf "%s\n" "${VOLS[@]}" | xargs -P "$NPROC" -I {} bash -c 'seg_one "$@"' _ {} &
XPID=$!

# progresso ogni 60s finche' xargs lavora
while kill -0 $XPID 2>/dev/null; do
    DONE=$(ls "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
    echo "  progresso: $DONE/$TOT   ($(date +%H:%M:%S))"
    sleep 60
done
wait $XPID || true

# riepilogo
OK1=$(grep -l "sigma=10" "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
RETRY=$(grep -lE "sigma=(20|30)" "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
SKIP=$(grep -l "skip" "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
FAIL=$(grep -l "FAILED" "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
NPVE=$(ls "$GENDIR_ABS"/*_synth_pveseg.nii.gz 2>/dev/null | wc -l)

echo ""
echo "=== RIEPILOGO ==="
echo "  maschere prodotte : $NPVE/$TOT"
echo "  ok al primo colpo : $OK1"
echo "  ok dopo retry     : $RETRY"
echo "  gia' presenti     : $SKIP"
echo "  FALLITI           : $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "  volumi falliti:"
    grep -l "FAILED" "$STATUS_DIR"/*.status 2>/dev/null | while read f; do
        echo "    - $(basename "$f" .status)"
    done
fi
if [ "$RETRY" -gt 0 ]; then
    echo "  volumi che hanno richiesto rumore maggiore:"
    grep -lE "sigma=(20|30)" "$STATUS_DIR"/*.status 2>/dev/null | while read f; do
        echo "    - $(basename "$f" .status): $(cat "$f")"
    done
fi

rmdir "$PREP_DIR" 2>/dev/null || true
echo "End: $(date)"