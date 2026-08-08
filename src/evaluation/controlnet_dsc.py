#src/evaluation/controlnet_dsc.py
"""
Calcolo del DSC per la generazione condizionata (ControlNet).

Per ogni volume generato si confronta:
    - la maschera CONDIZIONE : <base>_condmask.nii.gz     (quella data in input)
    - la maschera del GENERATO: <base>_synth_pveseg.nii.gz (ri-segmentata con FAST)
Un DSC alto significa che il volume generato rispetta la maschera richiesta.

Output: un JSON in outputs/metrics/ (coerente con le metriche dell'LDM) contenente
mean/generalized DSC, numero di coppie e il DSC per-volume (per vedere la distribuzione,
non solo la media). Utile sia per la checkpoint selection (su validation) sia per la
valutazione finale (su test).

Lo stesso script misura anche il TETTO della pipeline: confrontando la maschera-condizione
con la ri-segmentazione del REALE DECODIFICATO (--gen_suffix _realdec_pveseg.nii.gz) si
ottiene il DSC massimo ottenibile, che nessun modello generativo puo' superare.

Uso:
    #DSC dei generati
    python3 src/evaluation/controlnet_dsc.py --gen_dir data/controlnet_gen_val/epoch100 --tag val_epoch100
    #tetto della pipeline
    python3 src/evaluation/controlnet_dsc.py --gen_dir data/tetto --tag tetto \
        --gen_suffix _realdec_pveseg.nii.gz
"""

import os
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, AsDiscreted
from monai.metrics import DiceMetric, GeneralizedDiceScore

def evaluate(gen_dir, num_classes=4, batch_size=1, num_workers=1, logger=None,
             gen_suffix="_synth_pveseg.nii.gz"):
    log=logger or logging.getLogger(__name__)

    cond_files=sorted(Path(gen_dir).glob("*_condmask.nii.gz"))
    if not cond_files:
        raise ValueError(f"Nessun _condmask in {gen_dir}")

    data_dicts=[]
    names=[]
    missing=0
    for cond in cond_files:
        base=cond.name.replace("_condmask.nii.gz", "")
        gen=Path(gen_dir)/f"{base}{gen_suffix}"
        if gen.exists():
            data_dicts.append({"cond":str(cond), "gen":str(gen)})
            names.append(base)
        else:
            missing+=1
            log.warning(f"manca la ri-segmentazione per {base}")
    log.info(f"coppie valide: {len(data_dicts)} | mancanti: {missing}")
    if not data_dicts:
        raise ValueError("nessuna coppia cond/gen valida")

    keys=["cond", "gen"]
    transforms=Compose([
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        EnsureTyped(keys=keys),
        AsDiscreted(keys=keys, to_onehot=num_classes),
    ])
    loader=DataLoader(Dataset(data=data_dicts, transform=transforms),
                      batch_size=batch_size, num_workers=num_workers)

    #metriche aggregate + per-volume (reduction="mean_batch" dà il dice per campione)
    mean_dice=DiceMetric(include_background=False, reduction="mean")
    gen_dice=GeneralizedDiceScore(include_background=False, weight_type="square")
    per_volume=DiceMetric(include_background=False, reduction="mean")  #per il dettaglio per-volume

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"valuto {len(data_dicts)} coppie su {device}...")

    per_vol_scores=[]
    with torch.no_grad():
        for i, batch in enumerate(loader):
            y_true=batch["cond"].to(device)   #maschera richiesta
            y_pred=batch["gen"].to(device)    #maschera del generato
            mean_dice(y_pred=y_pred, y=y_true)
            gen_dice(y_pred=y_pred, y=y_true)
            #dice del singolo volume
            per_volume.reset()
            per_volume(y_pred=y_pred, y=y_true)
            per_vol_scores.append(float(per_volume.aggregate().item()))

    final_mean=mean_dice.aggregate().item()
    final_gen=gen_dice.aggregate().item()
    mean_dice.reset(); gen_dice.reset()

    per_vol=np.array(per_vol_scores)
    result={
        "n_pairs":len(data_dicts),
        "n_missing":missing,
        "mean_dsc":final_mean,
        "generalized_dsc":final_gen,
        "per_volume_mean":float(per_vol.mean()),
        "per_volume_std":float(per_vol.std()),
        "per_volume_min":float(per_vol.min()),
        "per_volume_max":float(per_vol.max()),
        "per_volume":{names[i]:per_vol_scores[i] for i in range(len(names))},
    }

    log.info("-"*40)
    log.info(f"Mean DSC:        {final_mean:.4f}")
    log.info(f"Generalized DSC: {final_gen:.4f}")
    log.info(f"per-volume: mean={per_vol.mean():.4f} std={per_vol.std():.4f} "
             f"min={per_vol.min():.4f} max={per_vol.max():.4f}")
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="cartella coi generati + ri-segmentazioni + condmask")
    ap.add_argument("--tag", required=True, help="etichetta per i file di output, es. val_epoch100 o test_epoch80")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--gen_suffix", default="_synth_pveseg.nii.gz",
                    help="suffisso della maschera da confrontare con _condmask. "
                         "Default: _synth_pveseg.nii.gz (generati). "
                         "Per misurare il TETTO della pipeline usare _realdec_pveseg.nii.gz "
                         "(reale decodificato: DSC massimo ottenibile).")
    ap.add_argument("--metrics_dir", default="outputs/metrics", help="cartella dei JSON delle metriche")
    args=ap.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)
    log_path=os.path.join(args.metrics_dir, f"dsc_{args.tag}.txt")
    json_path=os.path.join(args.metrics_dir, f"dsc_{args.tag}.json")

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()])
    logger=logging.getLogger(__name__)

    logger.info(f"=== DSC [{args.tag}] su {args.gen_dir} ===")
    result=evaluate(args.gen_dir, num_classes=args.num_classes, logger=logger,
                    gen_suffix=args.gen_suffix)

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"JSON salvato: {json_path}")

if __name__=="__main__":
    main()