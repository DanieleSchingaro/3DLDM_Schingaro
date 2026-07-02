#src/evaluation/plot_fid_curve.py
"""
Grafico della curva FID-vs-epoca a partire da fid_by_checkpoint.json
Mostra l'evoluzione della qualita' di generazione (FID 2.5D) durante il training
dell'LDM: la media sui 3 piani piu' le tre curve per piano (XY/YZ/ZX). Evidenzia
il checkpoint migliore (FID medio minimo) e la zona di saturazione.

Uso: 
    python3 -m src.evaluation.plot_fid_curve
    python3 -m src.evaluation.plot_fid_curve --json outputs/checkpoint_selection/fid_by_checkpoint_v4.json
"""

import os 
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

def plot_fid_curve(json_path, save_path=None, show_planes=True, title=None):
    """
    Disegna la curva FID-vs-epoca dal .json dei checkpoint.
    Args:
        json_path: percorso a fid_by_checkpoint.json
        save_path: se dato, salva la figura qui (l'estensione decide il formato);
                   salva anche un .pdf affianco se save_path termina in .png.
        show_planes: se True disegna anche le 3 curve per piano (XY/YZ/ZX).
        title: titolo personalizzato (altrimenti uno di default).
 
    Returns:
        (fig, best_epoch, best_fid) - la Figure matplotlib e il checkpoint migliore.
    """
    with open(json_path) as f:
        data=json.load(f)
    
    #ordina per epoca
    items=sorted(data.values(), key=lambda d: d["epoch"])
    epochs=[d["epoch"] for d in items]
    fid_avg=[d["fid_avg"] for d in items]
    fid_xy=[d["fid_xy"] for d in items]
    fid_yz=[d["fid_yz"] for d in items]
    fid_zx=[d["fid_zx"] for d in items]

    #checkpoint migliore (FID medio minimo)
    best_i=min(range(len(fid_avg)), key=lambda i: fid_avg[i])
    best_epoch=epochs[best_i]
    best_fid=fid_avg[best_i]

    fig,ax=plt.subplots(figsize=(9,5.5))

    #curve per piano
    if show_planes:
        ax.plot(epochs, fid_xy, "--", color="#7aa6c2", linewidth=1.3, marker="o",
                markersize=4, label="FID XY", alpha=0.8)
        ax.plot(epochs, fid_yz, "--", color="#c27a9e", linewidth=1.3, marker="s",
                markersize=4, label="FID YZ", alpha=0.8)
        ax.plot(epochs, fid_zx, "--", color="#7ac2a0", linewidth=1.3, marker="^",
                markersize=4, label="FID ZX", alpha=0.8)
    
    #curva media
    ax.plot(epochs, fid_avg, "-", color="#2b3a67", linewidth=2.6, marker="D",
            markersize=6, label="FID medio (3 piani)", zorder=5)
    
    #evidenzia il punto migliore
    ax.scatter([best_epoch], [best_fid], s=180, facecolors="none",
               edgecolors="#d62728", linewidths=2.2, zorder=6)
    ax.annotate(f"min: epoch {best_epoch}\nFID {best_fid:.2f}",
                xy=(best_epoch, best_fid),
                xytext=(best_epoch-180, best_fid+18),
                fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3))
    
    ax.set_xlabel("Epoca di training (LDM)", fontsize=11)
    ax.set_ylabel("FID 2.5D (più basso=meglio)", fontsize=11)
    ax.set_title(title or "Evoluzione del FID durante il training dell'LDM",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xticks(epochs)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    if save_path:
        out_dir=os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Salvato: {save_path}")
        #salva anche in PDF
        if save_path.lower().endswith(".png"):
            pdf_path=save_path[:-4]+".pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"Salvato: {pdf_path}")
    
    return fig, best_epoch, best_fid

def main():
    ap=argparse.ArgumentParser(description="Curva FID-vs-epoca")
    ap.add_argument("--json", type=str, default="outputs/checkpoint_selection/fid_by_checkpoint_v4.json")
    ap.add_argument("--out", type=str, default="outputs/metrics/fid_vs_epoch_v4.png")
    ap.add_argument("--no_planes", action="store_true", help="mostra solo la media")
    args=ap.parse_args()
 
    fig, best_epoch, best_fid=plot_fid_curve(
        args.json, save_path=args.out, show_planes=not args.no_planes)
    print(f"\nCheckpoint migliore: epoch {best_epoch} (FID {best_fid:.3f})")
    print("Nota: con 100 campioni differenze <5 tra checkpoint vicini sono")
    print("entro il rumore statistico del FID.")

if __name__=="__main__":
    main()