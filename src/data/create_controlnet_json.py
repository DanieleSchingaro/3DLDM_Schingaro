#src/data/create_controlnet_json.py

"""
Genera il JSON di training per la ControlNet, accoppiando per ogni volume:
  - image : embedding latente v4  (.npz, prodotto dal VAE v2 -> base del curriculum)
  - label : maschera FSL-FAST 256^3 (_pveseg.nii.gz, valori 0/1/2/3)

Split (OPZIONE A, mantiene lo split originale 805/100/102):
  - training   (805) -> fold 1   (usato in training dal dataloader MAISI)
  - validation (100) -> fold 0   (usato in validazione: fold==0 -> val)
  - test       (102) -> ESCLUSO dal JSON. Resta l'hold-out per la valutazione
                        finale con DSC/FID; non deve mai entrare nel training.

Campi per item:
  image, label, fold, spacing.
  NON si scrive "modality": con num_class_embeds=null (la UNet di Daniele)
  il dataloader passa modality_mapping=None, e una lambda su "modality" con
  mapping None solleverebbe TypeError. Assente la chiave, il problema non si pone.
  "spacing" e' invece OBBLIGATORIO (la lambda su spacing non ha allow_missing_keys).

Uso:
  python3 src/data/create_controlnet_json.py \
      --split data/splits/dataset.json \
      --emb_dir data/processed/embeddings_v4 \
      --mask_dir data/masks_fsl_postPad_256 \
      --out data/splits/controlnet_dataset.json
"""

import os
import re
import json
import glob
import argparse


def basename_from_raw(raw_path: str) -> str:
    """Nome base del volume: basename del raw senza estensione .nii.gz."""
    return re.sub(r"\.nii\.gz$", "", os.path.basename(raw_path))


def find_one(pattern: str, what: str, base: str):
    """Trova esattamente un file per il pattern; errore chiaro se 0 o >1."""
    hits=glob.glob(pattern, recursive=True)
    if len(hits)==0:
        raise FileNotFoundError(f"{what} non trovato per '{base}' (pattern: {pattern})")
    if len(hits)>1:
        raise RuntimeError(f"{what} ambiguo per '{base}': {len(hits)} match:\n  " +
                           "\n  ".join(hits))
    return hits[0]


def build_items(split_items, emb_dir, mask_dir, fold, repo_root):
    """Per ogni item dello split, costruisce il record accoppiato."""
    out=[]
    for it in split_items:
        raw=it["image"] if isinstance(it, dict) else it
        base=basename_from_raw(raw)

        emb=find_one(os.path.join(emb_dir, "**", f"{base}_emb.npz"), "embedding", base)
        msk=find_one(os.path.join(mask_dir, "**", f"{base}_pveseg.nii.gz"), "maschera", base)

        # path relativi alla repo (data_base_dir="." nel dataloader)
        emb_rel=os.path.relpath(emb, repo_root)
        msk_rel=os.path.relpath(msk, repo_root)

        out.append({
            "image":emb_rel,          # latente .npz (letto dal dataloader custom)
            "label":msk_rel,          # maschera 256^3 (condizione)
            "fold":fold,              # 1 = training, 0 = validation
            "spacing":[1.0, 1.0, 1.0] # obbligatorio nel dataloader MAISI
        })
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/dataset.json")
    ap.add_argument("--emb_dir", default="data/processed/embeddings_v4")
    ap.add_argument("--mask_dir", default="data/masks_fsl_postPad_256")
    ap.add_argument("--out", default="data/splits/controlnet_dataset.json")
    ap.add_argument("--repo_root", default=".",
                    help="radice per i path relativi nel JSON (default: cwd)")
    args=ap.parse_args()

    repo_root=os.path.abspath(args.repo_root)
    with open(args.split) as f:
        split=json.load(f)

    n_train=len(split.get("training", []))
    n_val=len(split.get("validation", []))
    n_test=len(split.get("test", []))
    print(f"Split originale: training={n_train}, validation={n_val}, test={n_test}")
    print("Opzione A: training->fold 1, validation->fold 0, test ESCLUSO.\n")

    # training -> fold 1 ; validation -> fold 0
    train_items=build_items(split["training"], args.emb_dir, args.mask_dir, 1, repo_root)
    val_items=build_items(split["validation"], args.emb_dir, args.mask_dir, 0, repo_root)

    all_items=train_items + val_items
    data={"training": all_items}

    # --- controlli di sanita' prima di scrivere ---
    n_fold1=sum(1 for d in all_items if d["fold"] == 1)
    n_fold0=sum(1 for d in all_items if d["fold"] == 0)
    assert n_fold1==n_train, f"fold1={n_fold1} != training={n_train}"
    assert n_fold0==n_val, f"fold0={n_fold0} != validation={n_val}"
    # nessun test finito dentro per errore
    assert len(all_items)==n_train + n_val, "conteggio totale inatteso"
    # tutti i file esistono (find_one avrebbe gia' sollevato, ma doppio check)
    for d in all_items:
        assert os.path.exists(os.path.join(repo_root, d["image"])), d["image"]
        assert os.path.exists(os.path.join(repo_root, d["label"])), d["label"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Scritti {len(all_items)} item ({n_fold1} training / {n_fold0} validation).")
    print(f"Test ({n_test}) escluso, resta hold-out per la valutazione finale.")
    print(f"JSON: {args.out}")
    print("\nEsempio record:")
    print(json.dumps(all_items[0], indent=2))


if __name__=="__main__":
    main()