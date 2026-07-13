#!/bin/bash

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
mkdir -p logs

# sorgente reali: "test" (102, default) oppure "all" (1007)
REAL_SOURCE=${1:-test}

echo "Start: $(date)"
echo "Reali di riferimento: $REAL_SOURCE"

# la valutazione usa 1 sola GPU (niente torchrun): FID/MMD/MS-SSIM non sono in DDP
python3 -m src.evaluation.eval --real_source $REAL_SOURCE \
    2>&1 | tee logs/eval_${REAL_SOURCE}_v5_$(date +%Y%m%d_%H%M).log

echo "End: $(date)"