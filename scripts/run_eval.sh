#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO/.venv/bin/activate"
cd "$REPO"
mkdir -p logs

# sorgente reali: "test" (102, default) oppure "all" (1007)
REAL_SOURCE=${1:-test}
# argomenti opzionali per la ControlNet:
#   $2 = synth_dir, $3 = tag, $4 = "cn" per attivare pattern+normalizzazione
SYNTH_DIR=${2:-}
TAG=${3:-v5}
MODE=${4:-ldm}

echo "Start: $(date)"
echo "Reali di riferimento: $REAL_SOURCE"

# la valutazione usa 1 sola GPU (niente torchrun): FID/MMD/MS-SSIM non sono in DDP
EXTRA=""
if [ "$MODE" = "cn" ]; then
    EXTRA='--synth_pattern *_synth.nii.gz --normalize'
fi

python3 -m src.evaluation.eval --real_source "$REAL_SOURCE" \
    ${SYNTH_DIR:+--synth_dir "$SYNTH_DIR"} \
    --tag "$TAG" $EXTRA \
    2>&1 | tee logs/eval_${REAL_SOURCE}_${TAG}_$(date +%Y%m%d_%H%M).log

echo "End: $(date)"