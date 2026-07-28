#src/evaluation/prepare_for_fast.py
"""
Prepara un volume generato (decodificato dal VAE, in [0,1]) per la segmentazione FAST.

I volumi decodificati sono troppo "lisci" e in range compresso: FAST (EM + campo di
Markov) non separa i 3 tessuti e collassa a 2 classi (variance nan). Si porta il
volume a range clinico (x1000) e si aggiunge un rumore gaussiano leggero, con seed
DETERMINISTICO derivato dal nome del file (riproducibile), per rompere l'omogeneita'.

Questo NON altera i volumi salvati: opera su una copia temporanea usata solo per FAST.
I volumi in [0,1] restano intatti per le metriche di realismo (FID/MMD/MS-SSIM).

Uso (singolo file):
    python3 src/evaluation/prepare_for_fast.py --in vol_synth.nii.gz --out vol_prep.nii.gz
"""

import argparse
import hashlib
import numpy as np
import nibabel as nib

def prepare_volume(in_path, out_path, scale=1000.0, noise_std=10.0, brain_thr=0.01):
    im=nib.load(in_path)
    d=np.asarray(im.dataobj).astype(np.float32)
    brain=d>brain_thr

    d2=d.copy()
    d2[brain]=d[brain]*scale

    #seed deterministico dal nome file -> riproducibile
    key=in_path.split("/")[-1]
    seed=int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    rng=np.random.default_rng(seed)
    d2[brain]+=rng.normal(0.0, noise_std, brain.sum()).astype(np.float32)
    d2=np.clip(d2, 0.0, None)

    nib.save(nib.Nifti1Image(d2.astype(np.float32), im.affine), out_path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--scale", type=float, default=1000.0)
    ap.add_argument("--noise_std", type=float, default=10.0)
    args=ap.parse_args()
    prepare_volume(args.in_path, args.out_path, args.scale, args.noise_std)

if __name__=="__main__":
    main()