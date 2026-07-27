#src/evaluation/controlnet_dsc.py
"""
Calcolo del DSC per la generazione condizionata (ControlNet).

Per ogni volume generato si confronta:
    - la maschera del GENERATO   : <base>_synth_pveseg.nii.gz  (ri-segmentato con FAST)
    - la maschera CONDIZIONE      : <base>_condmask.nii.gz      (quella data in input)
Un DSC alto significa che il volume generato rispetta la maschera richiesta.


Uso:
    python3 src/evaluation/controlnet_dsc.py --gen_dir data/controlnet_gen_val/epoch100
"""

import os
import argparse
import logging
from pathlib import Path
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, AsDiscreted
from monai.metrics import DiceMetric, GeneralizedDiceScore

def evaluate(gen_dir, num_classes=4, batch_size=1, num_workers=1, logger=None):
    log=logger or logging.getLogger(__name__)

    #accoppia <base>_condmask (richiesta) con <base>_synth_pveseg (del generato)
    cond_files=sorted(Path(gen_dir).glob("*_condmask.nii.gz"))
    if not cond_files:
        raise ValueError(f"Nessun _condmask in {gen_dir}")

    data_dicts=[]
    missing=0
    for cond in cond_files:
        base=cond.name.replace("_condmask.nii.gz", "")
        gen=Path(gen_dir)/f"{base}_synth_pveseg.nii.gz"
        if gen.exists():
            data_dicts.append({"cond":str(cond), "gen":str(gen)})
        else:
            missing+=1
            log.warning(f"manca la ri-segmentazione per {base} (atteso {gen.name})")
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

    mean_dice=DiceMetric(include_background=False, reduction="mean")
    gen_dice=GeneralizedDiceScore(include_background=False, weight_type="square")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"valuto {len(data_dicts)} coppie su {device}...")
    with torch.no_grad():
        for batch in loader:
            y_true=batch["cond"].to(device)   #maschera richiesta (ground truth della condizione)
            y_pred=batch["gen"].to(device)    #maschera del generato
            mean_dice(y_pred=y_pred, y=y_true)
            gen_dice(y_pred=y_pred, y=y_true)

    m=mean_dice.aggregate().item()
    g=gen_dice.aggregate().item()
    mean_dice.reset(); gen_dice.reset()

    log.info("-"*40)
    log.info(f"Mean DSC:        {m:.4f}")
    log.info(f"Generalized DSC: {g:.4f}")
    return m, g

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="cartella coi generati + ri-segmentazioni + condmask")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--log_file", default=None, help="se dato, salva il log su file")
    args=ap.parse_args()

    handlers=[logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
    logger=logging.getLogger(__name__)

    logger.info(f"=== DSC su {args.gen_dir} ===")
    evaluate(args.gen_dir, num_classes=args.num_classes, logger=logger)

if __name__=="__main__":
    main()