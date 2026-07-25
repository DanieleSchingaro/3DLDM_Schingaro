#tests/check_mask_alignment.py

"""
Verifica di ALLINEAMENTO tra le maschere di segmentazione (paddate a 256^3) e i
volumi immagine da cui derivano gli embedding.

Questo e' il controllo critico della fase ControlNet: la maschera e' la
condizione del generatore, quindi deve corrispondere voxel-per-voxel al latente
dell'immagine. Se maschera e immagine sono geometricamente sfasate (per un
disallineamento di orientamento/affine tra FAST e il preprocessing), la
ControlNet impara un condizionamento sbagliato SENZA che le loss lo segnalino.

Metrica: IoU tra la maschera binaria di tessuto della segmentazione (voxel > 0)
e quella dell'immagine preprocessata (intensita' > soglia). Entrambe passano per
la stessa geometria (Orientationd RAS -> ResizeWithPadOrCrop 256^3), quindi un
IoU alto (~0.9+) conferma l'allineamento; un IoU basso indica sfasamento.

Puo' lavorare in due modi:
  - sulle maschere GIA' paddate in out_dir (default): confronto diretto;
  - con --from_181: padda al volo dalla cartella 181 (per verificare PRIMA di
    aver eseguito il padding di tutto il set).

Lancio:
    python3 -m tests.check_mask_alignment                 # campione da postPad_256
    python3 -m tests.check_mask_alignment --n 20          # 20 volumi
    python3 -m tests.check_mask_alignment --from_181 --n 5  # dry-run pre-padding
"""

import os
import glob
import json
import argparse
import numpy as np
import nibabel as nib

from src.data.transforms import get_encoding_transforms
from src.data.pad_masks_to256 import get_mask_geometry_transforms, pad_one


IOU_WARN = 0.85   # sotto questa soglia la coppia e' segnalata come sospetta


def tissue_iou(mask_arr:np.ndarray, img_arr:np.ndarray, img_thr:float)->float:
    """IoU tra tessuto della maschera (>0) e tessuto dell'immagine (>img_thr)."""
    m=mask_arr > 0
    im=img_arr > img_thr
    inter=np.logical_and(m, im).sum()
    union=np.logical_or(m, im).sum()
    return float(inter / union) if union > 0 else 0.0


def find_volume(mask_base:str, splits)->str:
    """Ritrova il path del volume originale dal basename della maschera."""
    for split in ("training", "validation", "test"):
        for item in splits.get(split, []):
            vpath=item["image"] if isinstance(item, dict) else item
            vbase=os.path.basename(vpath)
            for ext in (".nii.gz", ".nifti", ".nii"):
                if vbase.endswith(ext):
                    vbase=vbase[: -len(ext)]
                    break
            if vbase == mask_base:
                return vpath
    return ""


def main():
    parser=argparse.ArgumentParser(description="Verifica allineamento maschera vs immagine")
    parser.add_argument("--masks_256_dir", type=str, default="data/masks_fsl_postPad_256")
    parser.add_argument("--masks_181_dir", type=str, default="data/masks_fsl_prePad_181")
    parser.add_argument("--splits_path", type=str, default="data/splits/dataset.json")
    parser.add_argument("--img_thr", type=float, default=0.01,
                        help="soglia intensita' per il tessuto nell'immagine preprocessata [0,1]")
    parser.add_argument("--n", type=int, default=10, help="quanti volumi campionare")
    parser.add_argument("--from_181", action="store_true",
                        help="padda al volo dalle maschere 181 invece di leggere le 256 gia' scritte")
    args=parser.parse_args()

    with open(args.splits_path) as f:
        splits=json.load(f)
    img_tf=get_encoding_transforms()
    mask_tf=get_mask_geometry_transforms() if args.from_181 else None

    src_dir=args.masks_181_dir if args.from_181 else args.masks_256_dir
    masks=sorted(glob.glob(f"{src_dir}/*_pveseg.nii.gz"))
    if not masks:
        print(f"Nessuna maschera in {src_dir}")
        return

    #campionamento uniforme lungo il set
    step=max(1, len(masks) // args.n)
    sample=masks[::step][: args.n]
    print(f"Verifica allineamento su {len(sample)} volumi (fonte: {src_dir})")
    print(f"{'volume':44s}{'IoU':>8}")
    print("-" * 52)

    ious=[]
    suspects=[]
    for mfile in sample:
        base=os.path.basename(mfile).replace("_pveseg.nii.gz", "")

        if args.from_181:
            mask_arr=pad_one(mfile, mask_tf)
        else:
            mask_arr=np.asarray(nib.load(mfile).dataobj)

        vpath=find_volume(base, splits)
        if not vpath:
            print(f"{base[:44]:44s}  volume non trovato")
            continue

        vol=img_tf({"image": vpath})["image"]
        if hasattr(vol, "as_tensor"):
            vol=vol.as_tensor()
        vol=vol.squeeze(0).numpy()

        iou=tissue_iou(mask_arr, vol, args.img_thr)
        ious.append(iou)
        flag="  <-- SOSPETTO" if iou < IOU_WARN else ""
        if iou < IOU_WARN:
            suspects.append((base, iou))
        print(f"{base[:44]:44s}{iou:>8.4f}{flag}")

    if ious:
        print("-" * 52)
        print(f"IoU medio: {np.mean(ious):.4f} | min: {np.min(ious):.4f} | max: {np.max(ious):.4f}")
        if suspects:
            print(f"\n{len(suspects)} coppie sotto soglia ({IOU_WARN}): possibile disallineamento.")
            print("NON procedere col training finche' non e' chiarito.")
        else:
            print(f"\nTutte le coppie sopra soglia ({IOU_WARN}): maschere e immagini allineate.")


if __name__=="__main__":
    main()