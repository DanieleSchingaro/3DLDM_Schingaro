#!/bin/bash
# Inferenza ControlNet condizionata su maschere.
# Uso:
#   bash scripts/run_sample_controlnet.sh <controlnet_ckpt> <json_data_list> <out_dir>
# Esempi:
#   # checkpoint selection su validation (un lancio per checkpoint candidato):
#   bash scripts/run_sample_controlnet.sh outputs/controlnet_v6/controlnet_epoch60.pt  data/splits/controlnet_infer_val.json data/controlnet_gen_val/epoch60
#   bash scripts/run_sample_controlnet.sh outputs/controlnet_v6/controlnet_epoch80.pt  data/splits/controlnet_infer_val.json data/controlnet_gen_val/epoch80
#   bash scripts/run_sample_controlnet.sh outputs/controlnet_v6/controlnet_epoch100.pt data/splits/controlnet_infer_val.json data/controlnet_gen_val/epoch100
#   # valutazione finale sul test col checkpoint scelto:
#   bash scripts/run_sample_controlnet.sh outputs/controlnet_v6/controlnet_epoch80.pt  data/splits/controlnet_infer_test.json data/controlnet_gen_test/epoch80

set -e

CKPT="$1"
JSON="$2"
OUTDIR="$3"
CS="${4:-1.0}"      # conditioning scale (default 1.0 = comportamento originale)
STEPS="${5:-30}"    # numero di step di inferenza

if [ -z "$CKPT" ] || [ -z "$JSON" ] || [ -z "$OUTDIR" ]; then
    echo "Uso: bash scripts/run_sample_controlnet.sh <controlnet_ckpt> <json_data_list> <out_dir>"
    exit 1
fi

# path della repo ricavato dalla posizione dello script (robusto: niente hardcoding)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO/.venv/bin/activate"
cd "$REPO"
mkdir -p logs

echo "Start: $(date)"
echo "ControlNet: $CKPT"
echo "Maschere:   $JSON"
echo "Output:     $OUTDIR"
echo "cond_scale: $CS | steps: $STEPS"

MASTER_PORT=$((29000 + RANDOM % 2000))
LOGNAME=$(basename "$OUTDIR")
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m src.inference.sample_controlnet \
    --controlnet_ckpt "$CKPT" \
    --ldm_ckpt outputs/models_v5/ldm_unet_epoch800.pt \
    --json_data_list "$JSON" \
    --out_dir "$OUTDIR" \
    --num_inference_steps "$STEPS" \
    --cond_scale "$CS" \
    2>&1 | tee logs/infer_controlnet_${LOGNAME}_$(date +%Y%m%d_%H%M).log

echo "End: $(date)"