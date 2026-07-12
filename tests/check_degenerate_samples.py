#tests/check_degenerate_samples.py
"""
QUANTIFICA I CAMPIONI DEGENERI nelle MRI sintetiche.

Motivazione: l'ispezione visiva ha mostrato che alcuni campioni generati con
autoguidance (es. hc_synth_0011) hanno il cervello minuscolo e/o traslato nel
volume 256^3, mentre i reali sono allineati e centrati. Il FID non lo cattura
bene perche' in eval.py le slice quasi vuote vengono scartate (drop_empty=True),
quindi un volume in gran parte nero contribuisce con poche slice.
La MS-SSIM, a sua volta, confonde "diversita' anatomica" con "disallineamento
geometrico": due cervelli identici ma traslati danno MS-SSIM bassa.

Questo script misura, per ogni volume, tre proprieta' GEOMETRICHE (non di texture):
  - frazione di voxel di tessuto (volume cerebrale relativo)
  - centroide del tessuto, normalizzato in [0,1] sui tre assi (posizione)
  - estensione della bounding box del tessuto, normalizzata (scala)

Poi costruisce una distribuzione di RIFERIMENTO dai volumi REALI e segnala come
degenere ogni sintetica che esce da quella distribuzione (z-score robusto,
basato su mediana e MAD, insensibile agli outlier stessi).

Uso:
    # sintetiche con autoguidance vs reali del test set
    python3 -m tests.check_degenerate_samples \
        --synth_dir data/synthetic_v4 \
        --splits data/splits/dataset.json --real_source test

    # confronto fra due generazioni (es. con vs senza guida)
    python3 -m tests.check_degenerate_samples --synth_dir data/synthetic_v4_w15 ...

Nota: usa gli stessi transform di eval.py per i reali, cosi' reali e sintetiche
vivono nello stesso spazio (256^3, [0,1]) e le misure sono confrontabili.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import nibabel as nib


def _tissue_mask(vol: np.ndarray, thr: float) -> np.ndarray:
    """Maschera binaria del tessuto: voxel sopra soglia di intensita'."""
    return vol > thr


def geometry_stats(vol: np.ndarray, thr: float):
    """
    Ritorna le statistiche geometriche di un volume gia' in [0,1], shape [X,Y,Z]:
      frac      -> frazione di voxel di tessuto (proxy del volume cerebrale)
      cx,cy,cz  -> centroide del tessuto normalizzato in [0,1] per asse
      ex,ey,ez  -> estensione della bounding box normalizzata in [0,1] per asse
    Se il volume e' vuoto, ritorna NaN (verra' segnalato come degenere).
    """
    mask = _tissue_mask(vol, thr)
    n = int(mask.sum())
    if n == 0:
        return dict(frac=0.0, cx=np.nan, cy=np.nan, cz=np.nan,
                    ex=np.nan, ey=np.nan, ez=np.nan, n_vox=0)

    frac = n / mask.size
    idx = np.argwhere(mask)                       # [n,3]
    shape = np.array(vol.shape, dtype=float)

    centroid = idx.mean(axis=0) / shape           # in [0,1]
    lo = idx.min(axis=0)
    hi = idx.max(axis=0)
    extent = (hi - lo + 1) / shape                # in [0,1]

    return dict(frac=frac,
                cx=centroid[0], cy=centroid[1], cz=centroid[2],
                ex=extent[0], ey=extent[1], ez=extent[2],
                n_vox=n)


def robust_z(values, med, mad):
    """
    z-score robusto: (x - mediana) / (1.4826 * MAD).
    Il fattore 1.4826 rende la MAD uno stimatore consistente della deviazione
    standard per dati gaussiani. Robusto perche' mediana e MAD non vengono
    trascinate dagli outlier (a differenza di media e std), che e' esattamente
    cio' che serve quando gli outlier sono l'oggetto della ricerca.
    """
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = 1e-9
    return (values - med) / scale


def load_real(item, tf_cache={}):
    """Carica una reale applicando get_encoding_transforms() (come eval.py)."""
    from src.data.transforms import get_encoding_transforms
    if "tf" not in tf_cache:
        tf_cache["tf"] = get_encoding_transforms()
    path = item["image"] if isinstance(item, dict) else item
    out = tf_cache["tf"]({"image": path})
    img = out["image"]
    if hasattr(img, "as_tensor"):
        img = img.as_tensor()
    return img.squeeze(0).float().numpy()


def load_synth(path):
    return nib.load(path).get_fdata().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Quantifica i campioni sintetici degeneri")
    ap.add_argument("--synth_dir", type=str, default="data/synthetic_v4")
    ap.add_argument("--splits", type=str, default="data/splits/dataset.json")
    ap.add_argument("--real_source", type=str, default="test", choices=["test", "all"])
    ap.add_argument("--thr", type=float, default=0.05,
                    help="soglia di intensita' per considerare un voxel 'tessuto' (volumi in [0,1])")
    ap.add_argument("--z_thr", type=float, default=3.5,
                    help="|z| robusto oltre il quale un campione e' segnalato come degenere")
    ap.add_argument("--n_real", type=int, default=0,
                    help="quanti reali usare per il riferimento (0 = tutti quelli dello split)")
    ap.add_argument("--out_json", type=str, default="outputs/metrics/degenerate_report.json")
    args = ap.parse_args()

    # ---------- reali: distribuzione di riferimento ----------
    with open(args.splits) as f:
        splits = json.load(f)
    if args.real_source == "test":
        real_items = splits["test"]
    else:
        real_items = splits["training"] + splits["validation"] + splits["test"]
    if args.n_real > 0:
        real_items = real_items[: args.n_real]

    print(f"Riferimento reale: {len(real_items)} volumi ({args.real_source})")
    real_stats = []
    for i, it in enumerate(real_items):
        vol = load_real(it)
        real_stats.append(geometry_stats(vol, args.thr))
        if (i + 1) % 20 == 0:
            print(f"  reali processati: {i+1}/{len(real_items)}")

    keys = ["frac", "cx", "cy", "cz", "ex", "ey", "ez"]
    R = {k: np.array([s[k] for s in real_stats], dtype=float) for k in keys}
    ref = {}
    for k in keys:
        v = R[k][np.isfinite(R[k])]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        ref[k] = (med, mad)

    print("\n=== DISTRIBUZIONE DI RIFERIMENTO (reali) ===")
    print(f"{'metrica':<8}{'mediana':>12}{'MAD':>12}")
    for k in keys:
        print(f"{k:<8}{ref[k][0]:>12.4f}{ref[k][1]:>12.4f}")

    # ---------- sintetiche ----------
    synth_files = sorted(glob.glob(os.path.join(args.synth_dir, "hc_synth_*.nii.gz")))
    if not synth_files:
        print(f"Nessun volume in {args.synth_dir}")
        return
    print(f"\nSintetiche: {len(synth_files)} volumi da {args.synth_dir}")

    results = []
    for i, p in enumerate(synth_files):
        vol = load_synth(p)
        s = geometry_stats(vol, args.thr)

        # z robusto su ogni proprieta'; il campione e' degenere se ALMENO una sfora
        zs = {}
        for k in keys:
            val = s[k]
            if not np.isfinite(val):
                zs[k] = float("inf")
            else:
                zs[k] = float(abs(robust_z(np.array([val]), *ref[k])[0]))
        worst_key = max(zs, key=lambda k: zs[k])
        worst_z = zs[worst_key]
        degenerate = (s["n_vox"] == 0) or (worst_z > args.z_thr)

        results.append(dict(
            file=os.path.basename(p),
            **{k: (None if not np.isfinite(s[k]) else round(float(s[k]), 5)) for k in keys},
            n_vox=s["n_vox"],
            worst_metric=worst_key,
            worst_z=(None if not np.isfinite(worst_z) else round(worst_z, 2)),
            degenerate=bool(degenerate),
        ))
        if (i + 1) % 20 == 0:
            print(f"  sintetiche processate: {i+1}/{len(synth_files)}")

    # ---------- report ----------
    bad = [r for r in results if r["degenerate"]]
    print("\n" + "=" * 68)
    print(f"CAMPIONI DEGENERI: {len(bad)} / {len(results)}  "
          f"({100.0*len(bad)/len(results):.1f}%)   [soglia |z| > {args.z_thr}]")
    print("=" * 68)
    if bad:
        print(f"{'file':<24}{'metrica':<10}{'|z|':>8}{'frac':>10}{'cz':>8}")
        for r in sorted(bad, key=lambda r: -(r["worst_z"] or 1e9)):
            wz = "inf" if r["worst_z"] is None else f"{r['worst_z']:.2f}"
            fr = "-" if r["frac"] is None else f"{r['frac']:.4f}"
            cz = "-" if r["cz"] is None else f"{r['cz']:.3f}"
            print(f"{r['file']:<24}{r['worst_metric']:<10}{wz:>8}{fr:>10}{cz:>8}")

    # riepilogo delle proprieta' geometriche: sintetiche vs reali
    print("\n=== GEOMETRIA: sintetiche vs reali (mediana) ===")
    print(f"{'metrica':<8}{'reali':>12}{'sintetiche':>14}{'scarto':>12}")
    for k in keys:
        sv = np.array([r[k] for r in results if r[k] is not None], dtype=float)
        if sv.size == 0:
            continue
        s_med = float(np.median(sv))
        r_med = ref[k][0]
        print(f"{k:<8}{r_med:>12.4f}{s_med:>14.4f}{s_med - r_med:>+12.4f}")

    print("\nLegenda: frac = frazione di voxel di tessuto (volume); "
          "cx/cy/cz = centroide normalizzato (posizione); "
          "ex/ey/ez = estensione bounding box (scala).")

    out = dict(
        synth_dir=args.synth_dir,
        real_source=args.real_source,
        n_synth=len(results),
        n_degenerate=len(bad),
        rate=len(bad) / len(results),
        thr=args.thr,
        z_thr=args.z_thr,
        reference_median_mad={k: dict(median=ref[k][0], mad=ref[k][1]) for k in keys},
        samples=results,
    )
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nReport salvato in {args.out_json}")


if __name__ == "__main__":
    main()