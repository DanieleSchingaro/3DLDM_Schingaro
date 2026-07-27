#src/data/create_controlnet_inference_json.py
"""
Genera i JSON di INFERENZA per la ControlNet.

A differenza del JSON di training (che accoppia embedding + maschera per addestrare),
l'inferenza parte SOLO dalla maschera: la ControlNet genera da rumore condizionando
sulla maschera, senza mai vedere il volume/embedding reale. Quindi il campo "image"
NON serve (non si usa il latente reale: sarebbe barare).

Produce DUE liste, per rispettare l'hold-out:
    - validation (i 100 del validation split): per la CHECKPOINT SELECTION via DSC
      -> si sceglie il checkpoint ControlNet migliore su questi
    - test (i 102 del test split): per la VALUTAZIONE FINALE col checkpoint scelto
      -> il test non viene mai usato per selezionare, resta hold-out puro

Ogni item:
    "label"  : maschera 256^3 (_pveseg.nii.gz) = condizione della generazione
    "dim"    : [256,256,256] (dimensione del volume di output; costante nel progetto)
    "spacing": [1.0,1.0,1.0] (obbligatorio nel dataloader MAISI)
NON si scrive "image" (nessun embedding reale in inferenza) ne' "modality".

Uso:
    python3 src/data/create_controlnet_inference_json.py \
        --split data/splits/dataset.json \
        --mask_dir data/masks_fsl_postPad_256 \
        --out_val data/splits/controlnet_infer_val.json \
        --out_test data/splits/controlnet_infer_test.json
"""

import os
import re
import json
import glob
import argparse


def basename_from_raw(raw_path):
    return re.sub(r"\.nii\.gz$", "", os.path.basename(raw_path))


def find_one(pattern, what, base):
    hits=glob.glob(pattern, recursive=True)
    if len(hits)==0:
        raise FileNotFoundError(f"{what} non trovato per '{base}' (pattern: {pattern})")
    if len(hits)>1:
        raise RuntimeError(f"{what} ambiguo per '{base}': {len(hits)} match")
    return hits[0]


def build_items(split_items, mask_dir, repo_root):
    out=[]
    for it in split_items:
        raw=it["image"] if isinstance(it, dict) else it
        base=basename_from_raw(raw)
        msk=find_one(os.path.join(mask_dir, "**", f"{base}_pveseg.nii.gz"), "maschera", base)
        out.append({
            "label": os.path.relpath(msk, repo_root),
            "dim": [256, 256, 256],
            "spacing": [1.0, 1.0, 1.0],
        })
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/dataset.json")
    ap.add_argument("--mask_dir", default="data/masks_fsl_postPad_256")
    ap.add_argument("--out_val", default="data/splits/controlnet_infer_val.json")
    ap.add_argument("--out_test", default="data/splits/controlnet_infer_test.json")
    ap.add_argument("--repo_root", default=".")
    args=ap.parse_args()

    repo_root=os.path.abspath(args.repo_root)
    split=json.load(open(args.split))

    val_items=build_items(split["validation"], args.mask_dir, repo_root)
    test_items=build_items(split["test"], args.mask_dir, repo_root)

    # il dataloader MAISI legge la lista sotto la chiave "training": si riusa quella
    # convenzione anche qui (una sola lista letta per intero, nessuno split per fold
    # in inferenza -> tutti gli item vengono processati).
    for path, items, name in [
        (args.out_val, val_items, "validation"),
        (args.out_test, test_items, "test"),
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"training": items}, f, indent=2)
        print(f"{name}: {len(items)} maschere -> {path}")

    print("\nEsempio record:")
    print(json.dumps(val_items[0], indent=2))
    print("\nNOTA: la checkpoint selection va fatta su controlnet_infer_val.json.")
    print("      Il test (controlnet_infer_test.json) SOLO per la valutazione finale.")


if __name__=="__main__":
    main()