#src/data/pad_masks_to256.py

"""
Porta le maschere di segmentazione FSL FAST dalla risoluzione nativa
(181x217x181) a 256^3, applicando la STESSA geometria del preprocessing con cui
sono stati prodotti gli embedding (Orientationd RAS -> ResizeWithPadOrCropd 256^3).
Cosi' la maschera e il latente dell'immagine corrispondente vivono nello stesso
spazio voxel-per-voxel, condizione necessaria perche' la ControlNet impari un
condizionamento corretto.

Differenza cruciale rispetto all'immagine: la maschera contiene ETICHETTE intere
(0=bg, 1=CSF, 2=GM, 3=WM). Il resize da 181 a 256 e' solo PADDING (nessun
ridimensionamento), quindi le etichette restano intere; il padding e' a costante
0 (background), coerente con l'immagine.

La verifica che maschera e immagine risultino allineate e' un passo separato:
tests/check_mask_alignment.py (da lanciare PRIMA di fidarsi dell'intero set).

Lancio:
    python3 -m src.data.pad_masks_to256
    python3 -m src.data.pad_masks_to256 --masks_dir ... --out_dir ...
"""

import os
import glob
import logging
import argparse
import numpy as np
import nibabel as nib
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ResizeWithPadOrCropd,
    EnsureTyped,
)


def setup_logging()->logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("pad_masks")


def get_mask_geometry_transforms()->Compose:
    """
    Solo la parte GEOMETRICA del preprocessing dell'immagine, applicata alla
    maschera (etichette intere, dtype uint8). Nessuna normalizzazione di
    intensita': le classi non sono intensita'.
    """
    return Compose([
        LoadImaged(keys=["mask"]),
        EnsureChannelFirstd(keys=["mask"]),
        Orientationd(keys=["mask"], axcodes="RAS"),
        ResizeWithPadOrCropd(
            keys=["mask"],
            spatial_size=(256, 256, 256),
            mode="constant",
            constant_values=0,
        ),
        EnsureTyped(keys=["mask"], dtype=torch.uint8),
    ])


def pad_one(mask_path:str, tf:Compose)->np.ndarray:
    """Applica la geometria e ritorna la maschera 256^3 come array uint8 [X,Y,Z]."""
    out=tf({"mask": mask_path})["mask"]
    if hasattr(out, "as_tensor"):
        out=out.as_tensor()
    return out.squeeze(0).numpy().astype(np.uint8)


def main():
    parser=argparse.ArgumentParser(description="Padding maschere FSL a 256^3 (geometria dell'encoding)")
    parser.add_argument("--masks_dir", type=str, default="data/masks_fsl_prePad_181")
    parser.add_argument("--out_dir", type=str, default="data/masks_fsl_postPad_256")
    args=parser.parse_args()

    logger=setup_logging()
    os.makedirs(args.out_dir, exist_ok=True)

    mask_tf=get_mask_geometry_transforms()
    masks=sorted(glob.glob(f"{args.masks_dir}/*_pveseg.nii.gz"))
    logger.info(f"Maschere da processare: {len(masks)}")

    written=0
    for i, mfile in enumerate(masks):
        base=os.path.basename(mfile).replace("_pveseg.nii.gz", "")
        out_path=os.path.join(args.out_dir, f"{base}_pveseg.nii.gz")
        if os.path.exists(out_path):
            continue
        padded=pad_one(mfile, mask_tf)
        #affine identita': spazio voxel canonico 256^3 post-resize
        nib.save(nib.Nifti1Image(padded, affine=np.eye(4)), out_path)
        written+=1
        if (i + 1) % 100 == 0:
            logger.info(f"  ...{i + 1}/{len(masks)}")

    logger.info(f"Maschere paddate scritte: {written} (gia' presenti saltate)")
    logger.info("Prossimo passo: python3 -m tests.check_mask_alignment  (verifica allineamento)")


if __name__=="__main__":
    main()