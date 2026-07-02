#src/evaluation/eval.py
"""
Valutazione delle MRI cerebrali HC sintetiche generate dall'LDM.

Confronta le SINTETICHE (data/synthetic_v4) con le REALI di riferimento:
  - FID 2.5D   (reali vs sintetiche, distribuzionale)
  - MMD        (reali vs sintetiche, distribuzionale)
  - MS-SSIM    diversita' intra-set su sintetiche E reali (mode collapse).

NB v4: le sintetiche valutate sono quelle prodotte da sample.py CON autoguidance,
quindi il FID/MMD/MS-SSIM qui riflettono gia' la configurazione finale (guidata).
 
GESTIONE MEMORIA: i volumi NON vengono tenuti tutti in RAM. Si usa VolumeStream
(da metrics.py) che carica un volume alla volta da disco. Le reali vengono
preprocessate al volo con get_encoding_transforms() (256^3, [0,1]) -> stesso
spazio delle sintetiche. Cosi' si puo' valutare anche --real_source all (1007
volumi) senza OOM, perche' in memoria c'e' sempre al piu' 1-2 volumi alla volta.
 
Sorgente reali (--real_source):
  "test" : 102 volumi hold-out (mai visti dal VAE/LDM). Pulito ma rumoroso.
  "all"  : 1007 volumi (805 visti in training). FID a varianza piu' bassa.
 
Esempi:
    python3 -m src.evaluation.eval --real_source test
    python3 -m src.evaluation.eval --real_source all --mmd_max_pairs 100
 
Risultati in outputs/metrics/eval_<real_source>_v4.json
"""
import os 
import json 
import glob
import argparse
import numpy as np 
import nibabel as nib 
import torch 
from src.data.transforms import get_encoding_transforms
from src.evaluation.metrics import(
    VolumeStream,
    compute_fid_2p5d,
    compute_msssim_diversity,
    compute_mmd,
)

#loader per VolumeStream
def _load_synth_volume(path:str)->torch.Tensor:
    """Carica una sintetica .nii.gz"""
    data=nib.load(path).get_fdata().astype(np.float32)
    return torch.from_numpy(data)

#transform delle immagini reali. Riusato dal loader
_REAL_TF=None

def _load_real_volume(item)->torch.Tensor:
    """Carica una reale dal path e la preprocessa con get_encoding_transforms()"""
    global _REAL_TF
    if _REAL_TF is None:
        _REAL_TF=get_encoding_transforms()
    path=item["image"] if isinstance(item, dict) else item
    out=_REAL_TF({"image": path})
    img=out["image"]
    if hasattr(img, "as_tensor"):
        img=img.as_tensor()
    return img.squeeze(0).float() #--> [256,256,256]

def build_synth_stream(synth_dir:str):
    files=sorted(glob.glob(os.path.join(synth_dir, "hc_synth_*.nii.gz")))
    return VolumeStream(files, _load_synth_volume), files 

def build_real_stream(splits_path:str, real_source:str):
    with open(splits_path) as f:
        splits=json.load(f)
    if real_source=="test":
        items=splits["test"]
    elif real_source=="all":
        items=splits["training"]+splits["validation"]+splits["test"]
    else:
        raise ValueError(f"real_source sconosciuto: {real_source}")
    return VolumeStream(items, _load_real_volume)

def main():
    parser = argparse.ArgumentParser(description="Valutazione MRI HC sintetiche (FID/MMD/MS-SSIM)")
    parser.add_argument("--synth_dir", type=str, default="data/synthetic_v4")
    parser.add_argument("--splits", type=str, default="data/splits/dataset.json")
    parser.add_argument("--real_source", type=str, default="test", choices=["test", "all"])
    parser.add_argument("--out_dir", type=str, default="outputs/metrics")
    parser.add_argument("--n_pairs", type=int, default=200,
                        help="coppie casuali per la diversita' MS-SSIM")
    parser.add_argument("--mmd_max_pairs", type=int, default=100,
                        help="max coppie per l'MMD (limita il costo con dataset grandi)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="slice per batch nell'estrazione feature FID")
    parser.add_argument("--no_drop_empty", action="store_true",
                        help="NON scartare le slice quasi vuote (default: le scarta)")
    args = parser.parse_args()

    device="cuda" if torch.cuda.is_available() else "cpu"
    drop_empty=not args.no_drop_empty
    print(f"Device: {device}")
    print(f"Reali di riferimento: {args.real_source}")

    synth_stream, synth_files=build_synth_stream(args.synth_dir)
    real_stream=build_real_stream(args.splits, args.real_source)
    print(f"Sintetiche: {len(synth_stream)} volumi da {args.synth_dir}")
    print(f"Reali: {len(real_stream)} volumi ({args.real_source}, preprocessate al volo)")

    if len(synth_stream)==0 or len(real_stream)==0:
        print("Errore: nessun volume trovato. Controlla i percorsi")
        return
    
    results={
        "real_source":args.real_source,
        "n_synthetic":len(synth_stream),
        "n_real":len(real_stream),
    }

    #FID 2.5D
    print("\n===FID 2.5D (reali vs sintetiche)===")
    fid_res=compute_fid_2p5d(
        real_stream, synth_stream, device=device,
        drop_empty=drop_empty, batch_size=args.batch_size, verbose=True,
    )
    results.update(fid_res)

    #MMD
    print("\n===MMD (reali vs sintetiche)===")
    results["mmd"]=compute_mmd(
        real_stream, synth_stream, device=device,
        max_pairs=args.mmd_max_pairs, verbose=True,
    )
    
    #MS-SSIM diversità (sintetiche e reali)
    print("\n===MS-SSIM===")
    print("Sintetiche:")
    results["msssim_synth"]=compute_msssim_diversity(
        synth_stream, device=device, n_pairs=args.n_pairs, verbose=True)
    print("Reali:")
    results["msssim_real"]=compute_msssim_diversity(
        real_stream, device=device, n_pairs=args.n_pairs, verbose=True)
    
    #riepilogo
    print("\n"+"="*55)
    print(f"RIEPILOGO VALUTAZIONE (reali: {args.real_source})")
    print("="*55)
    print(f"FID XY/YZ/ZX: {results['fid_xy']:.3f}/{results['fid_yz']:.3f}/{results['fid_zx']:.3f}")
    print(f"FID medio: {results['fid_avg']:.3f}")
    print(f"MMD: {results['mmd']:.6f}")
    print(f"MS-SSIM sint: {results['msssim_synth']:.4f}   (diversita' sintetiche)")
    print(f"MS-SSIM reali: {results['msssim_real']:.4f}   (diversita' reali, riferimento)")
    print("="*55)
    print("Nota: MS-SSIM sint vicino a MS-SSIM reali = buona varieta'.")
    print("MS-SSIM sint molto piu' alto = possibile mode collapse.")

    #salvataggio
    os.makedirs(args.out_dir, exist_ok=True)
    out_path=os.path.join(args.out_dir, f"eval_{args.real_source}_v4.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRisultati salvati in {out_path}")

if __name__=="__main__":
    main()