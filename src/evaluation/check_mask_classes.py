#src/evaluation/check_mask_classes.py
"""
Verifica che una maschera di segmentazione abbia tutte le classi attese.

Serve alla pipeline di ri-segmentazione (run_segment_generated.sh): dopo FAST si
controlla che la maschera abbia 4 classi (0=bg, 1=CSF, 2=GM, 3=WM). Se ne ha meno,
FAST e' collassato (variance nan) e il volume va ritentato con piu' rumore.

Con --quiet non stampa nulla e usa solo l'exit code (0=ok, 1=classi mancanti), cosi'
e' usabile direttamente in un if di bash.

Uso:
    python3 -m src.evaluation.check_mask_classes --mask maschera.nii.gz
    python3 -m src.evaluation.check_mask_classes --mask maschera.nii.gz --quiet
    python3 -m src.evaluation.check_mask_classes --dir data/controlnet_gen_val/epoch100
"""

import sys
import glob
import argparse
import numpy as np
import nibabel as nib

def mask_classes(path):
    return np.unique(np.asarray(nib.load(path).dataobj)).astype(int).tolist()

def check_one(path, expected, quiet=False):
    cls=mask_classes(path)
    ok=(cls==expected)
    if not quiet:
        status="OK" if ok else "ANOMALA"
        print(f"{status}: {path.split('/')[-1]} -> classi {cls}")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mask", help="singola maschera da verificare")
    ap.add_argument("--dir", help="cartella: verifica tutti i *_pveseg.nii.gz")
    ap.add_argument("--expected", default="0,1,2,3", help="classi attese, es. 0,1,2,3")
    ap.add_argument("--quiet", action="store_true", help="nessun output, solo exit code")
    args=ap.parse_args()

    expected=[int(x) for x in args.expected.split(",")]

    if args.mask:
        ok=check_one(args.mask, expected, quiet=args.quiet)
        sys.exit(0 if ok else 1)

    if args.dir:
        files=sorted(glob.glob(f"{args.dir}/*_pveseg.nii.gz"))
        bad=[]
        for f in files:
            if not check_one(f, expected, quiet=True):
                bad.append((f.split("/")[-1], mask_classes(f)))
        print(f"{args.dir}: {len(files)} maschere | corrette: {len(files)-len(bad)}/{len(files)}")
        for name, cls in bad:
            print(f"  ANOMALA {name}: classi {cls}")
        sys.exit(0 if not bad else 1)

    ap.error("specificare --mask o --dir")

if __name__=="__main__":
    main()