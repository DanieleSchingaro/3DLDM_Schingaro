#test/test_vae_reconstruction.py

"""
Verifica visiva e quantitativa delle ricostruzioni del VAE trainato.
Carica autoencoder_best.pt e ricostruisce alcuni volumi del validation set,
e salva confronti input/output in formato PNG + metriche SSIM, L1 e PSNR.
Usa sliding_window_inference, come MAISI, per gestire il volume di 256^3.
Esegui con: python3 -m tests.test_vae_reconstruction
"""

import os
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torch.amp import autocast
from monai.inferers import sliding_window_inference
from monai.metrics import SSIMMetric, PSNRMetric
from src.data.encode_dataset import load_autoencoder, setup_logging
from src.data.transforms import get_encoding_transforms
from src.data.dataset import load_splits

#CONFIG
CHECKPOINT_PATH="outputs/models_v3/autoencoder_best.pt"
SPLITS_PATH="data/splits/dataset.json"
OUTPUT_DIR="outputs/generated/reconstructions_v3"
N_VOLUMES=4
ROI_SIZE=(128,128,128) #finestra sliding_window
SW_BATCH_SIZE=1
OVERLAP=0.25

def reconstruct_volume(autoencoder, image, device):
    """
    Ricostruisce un volume intero con sliding window inference.
    AutoencoderKLMaisi.forward(x) restituisce (reconstruction, z_mu, z_sigma):
    prendiamo solo la ricostruzione (indice 0).
    """
    def _infer(x):
        with autocast("cuda", enabled=(device.type=="cuda")):
            out=autoencoder(x)
            return out[0] if isinstance(out, (tuple, list)) else out 
    
    with torch.inference_mode():
        recon=sliding_window_inference(
            inputs=image,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=_infer,
            overlap=OVERLAP,
            mode="gaussian",
            sw_device=device,
            device=device,
        )
    
    return recon 

def save_comparison(original, reconstruction, idx, ssim_val, l1_val, psnr_val, out_dir):
    """
    Salva un PNG con 3 viste (assiale, coronale, sagittale)
    di originale vs ricostruzione affiancati.
    Original, reconstruction: numpy [X,Y,Z] con valori in [0,1].
    """
    os.makedirs(out_dir, exist_ok=True)
    #slice centrale per ogni piano anatomico
    x,y,z=original.shape
    views={
        "Assiale": (original[:,:,z//2], reconstruction[:,:,z//2]),
        "Coronale": (original[:,y//2,:], reconstruction[:,y//2,:]),
        "Sagittale": (original[x//2,:,:], reconstruction[x//2,:,:]),
    }

    fig, axes=plt.subplots(2,3,figsize=(12,8))
    fig.suptitle(
        f"Volume {idx} | SSIM={ssim_val:.4f} | L1={l1_val:.4f} | PSNR={psnr_val:.2f} dB", fontsize=14,
    )

    for col, (name,(orig_slice, recon_slice)) in enumerate(views.items()):
        axes[0, col].imshow(np.rot90(orig_slice), cmap="gray")
        axes[0, col].set_title(f"Input - {name}")
        axes[0, col].axis("off")

        axes[1, col].imshow(np.rot90(recon_slice), cmap="gray")
        axes[1, col].set_title(f"Ricostruzione - {name}")
        axes[1, col].axis("off")
    
    plt.tight_layout()
    out_path=os.path.join(out_dir, f"recon_vol{idx}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

def main():
    logger=setup_logging()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    #caricamento VAE
    logger.info(f"Caricamento VAE da {CHECKPOINT_PATH}")
    autoencoder=load_autoencoder(CHECKPOINT_PATH, device)

    #transforms
    transforms=get_encoding_transforms()

    #N volumi dal validation set per misurare la generalizzazione
    _, val_files, _=load_splits(SPLITS_PATH)
    val_files=val_files[:N_VOLUMES]
    logger.info(f"Ricostruisco {len(val_files)} volumi dal validation set")

    #metriche in range dati [0,1]
    ssim_metric=SSIMMetric(spatial_dims=3, data_range=1.0)
    psnr_metric=PSNRMetric(max_val=1.0)
    l1_loss=torch.nn.L1Loss()

    all_ssim, all_l1, all_psnr=[],[],[]

    for idx, item in enumerate(val_files):
        image_path=item["image"]
        logger.info(f"[{idx+1}/{len(val_files)}] {image_path}")

        #preprocessing identico all'encoding --> [1,1,256,256,256]
        data=transforms({"image": image_path})
        image=data["image"].unsqueeze(0).to(device)

        #ricostruzione con sliding_window
        recon=reconstruct_volume(autoencoder, image, device)
        #clamp ricostruzione 
        recon=torch.clamp(recon, 0.0, 1.0)
        #metriche
        ssim_val=ssim_metric(recon.float(), image.float()).mean().item()
        psnr_val=psnr_metric(recon.float(), image.float()).mean().item()
        l1_val=l1_loss(recon.float(), image.float()).mean().item()

        all_ssim.append(ssim_val)
        all_psnr.append(psnr_val)
        all_l1.append(l1_val)

        #conversione numpy [X,Y,Z] per plot
        orig_np=image.squeeze().cpu().float().numpy()
        recon_np=recon.squeeze().cpu().float().numpy()

        out_path=save_comparison(orig_np, recon_np, idx+1, ssim_val, l1_val, psnr_val, OUTPUT_DIR)
        logger.info(f"SSIM={ssim_val:.4f} | L1={l1_val:.4f} | PSNR={psnr_val:.2f} dB | {out_path}")
    
    #riepilogo finale
    logger.info(f"\n{'='*55}")
    logger.info(f"RIEPILOGO RICOSTRUZIONE VAE ({len(val_files)} volumi val set)")
    logger.info(f"SSIM medio: {np.mean(all_ssim):.4f}  (+/- {np.std(all_ssim):.4f})")
    logger.info(f"PSNR medio: {np.mean(all_psnr):.2f} dB (+/- {np.std(all_psnr):.2f})")
    logger.info(f"L1 medio:   {np.mean(all_l1):.4f}  (+/- {np.std(all_l1):.4f})")
    logger.info(f"{'='*55}")
    logger.info(f"PNG di confronto salvati in {OUTPUT_DIR}")
    logger.info("Interpretazione: SSIM>0.90 eccellente, 0.80-0.90 buono, <0.80 da indagare")

if __name__=="__main__":
    main()