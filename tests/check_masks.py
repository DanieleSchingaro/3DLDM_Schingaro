#tests/check_masks.py

"""
Controllo di sanita' delle maschere di segmentazione FSL FAST (_pveseg).

Per ogni maschera verifica:
  - shape attesa (default 181x217x181, la risoluzione nativa pre-padding)
  - classi presenti: attese [0, 1, 2, 3] (background, CSF, GM, WM)
  - dimensione file non anomala (una maschera troncata pesa pochissimo)
  - frazione di tessuto per classe: intercetta segmentazioni "formalmente valide"
    (4 classi presenti) ma degeneri (una classe quasi assente)

Stampa una progress ogni PROGRESS_EVERY file, cosi' si vede l'avanzamento su
dataset grandi, e un riepilogo finale con l'elenco dei file sospetti.

Lancio:
    python3 -m tests.check_masks
    python3 -m tests.check_masks --masks_dir data/masks_fsl_prePad_181
    python3 -m tests.check_masks --expected_shape 181 217 181
"""

import glob
import argparse
import numpy as np
import nibabel as nib


PROGRESS_EVERY=100


def check_masks(
    masks_dir:str,
    expected_shape:tuple,
    expected_classes:list,
    min_frac:float,
):
    """
    Scorre le maschere _pveseg in masks_dir e raccoglie le anomalie. Ritorna
    un dizionario con le liste dei file problematici per ciascun tipo di check.
    """
    files=sorted(glob.glob(f"{masks_dir}/*_pveseg.nii.gz"))
    n=len(files)
    print(f"Maschere trovate: {n} in {masks_dir}")
    if n==0:
        return None

    bad_shape=[]       # shape diversa dall'attesa
    bad_classes=[]     # insieme di classi diverso da [0,1,2,3]
    small_file=[]      # file sospettosamente piccolo (troncato)
    degenerate=[]      # una classe di tessuto quasi assente

    #soglia dimensione file: la mediana / 5 e' una soglia robusta per "troppo piccolo"
    sizes=[__import__("os").path.getsize(f) for f in files]
    size_thr=np.median(sizes) / 5.0

    for i, f in enumerate(files):
        name=f.split("/")[-1]

        if sizes[i]<size_thr:
            small_file.append((name, sizes[i]))

        d=np.asarray(nib.load(f).dataobj)

        if d.shape!=expected_shape:
            bad_shape.append((name, d.shape))

        classes=np.unique(d).astype(int).tolist()
        if classes!=expected_classes:
            bad_classes.append((name, classes))

        # frazione di ciascuna classe di tessuto (1,2,3) sul tessuto totale
        total_tissue=(d>0).sum()
        if total_tissue>0:
            for c in (1, 2, 3):
                frac=(d==c).sum()/total_tissue
                if frac<min_frac:
                    degenerate.append((name, c, round(float(frac), 4)))
                    break

        if (i+1)%PROGRESS_EVERY==0:
            print(f"  ...{i + 1}/{n} controllate")

    return dict(
        n=n,
        bad_shape=bad_shape,
        bad_classes=bad_classes,
        small_file=small_file,
        degenerate=degenerate,
    )


def main():
    parser=argparse.ArgumentParser(description="Controllo di sanita' delle maschere FSL FAST")
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="data/masks_fsl_prePad_181",
        help="Cartella con le maschere _pveseg",
    )
    parser.add_argument(
        "--expected_shape",
        type=int,
        nargs=3,
        default=[181, 217, 181],
        help="Shape attesa delle maschere (pre-padding)",
    )
    parser.add_argument(
        "--min_frac",
        type=float,
        default=0.02,
        help="Frazione minima di una classe sul tessuto totale sotto cui la maschera e' 'degenere'",
    )
    args=parser.parse_args()

    res=check_masks(
        masks_dir=args.masks_dir,
        expected_shape=tuple(args.expected_shape),
        expected_classes=[0, 1, 2, 3],
        min_frac=args.min_frac,
    )
    if res is None:
        print("Nessuna maschera trovata.")
        return

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"Maschere totali      : {res['n']}")
    print(f"Shape anomala        : {len(res['bad_shape'])}")
    print(f"Classi anomale       : {len(res['bad_classes'])}")
    print(f"File troppo piccolo  : {len(res['small_file'])}")
    print(f"Classe degenere      : {len(res['degenerate'])}")

    if res["bad_shape"]:
        print("\n-- shape anomala --")
        for name, sh in res["bad_shape"][:10]:
            print(f"  {name}: {sh}")

    if res["bad_classes"]:
        print("\n-- classi anomale (atteso [0,1,2,3]) --")
        for name, cl in res["bad_classes"][:10]:
            print(f"  {name}: {cl}")

    if res["small_file"]:
        print("\n-- file sospettosamente piccoli --")
        for name, sz in res["small_file"][:10]:
            print(f"  {name}: {sz} byte")

    if res["degenerate"]:
        print("\n-- classe di tessuto quasi assente --")
        for name, c, frac in res["degenerate"][:10]:
            tissue = {1: "CSF", 2: "GM", 3: "WM"}[c]
            print(f"  {name}: {tissue} = {frac} del tessuto")

    all_ok=not (res["bad_shape"] or res["bad_classes"] or res["small_file"] or res["degenerate"])
    print("\n" + ("Tutte le maschere sono sane." if all_ok else "Alcune maschere richiedono un controllo manuale (vedi sopra)."))


if __name__=="__main__":
    main()