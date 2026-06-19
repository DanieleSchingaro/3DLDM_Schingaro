#src/evaluation/checkpoint_selection.py
"""
SELEZIONE DEL CHECKPOINT LDM MIGLIORE VIA FID
Per i modelli di diffusione la val_loss NON predice la qualita' di generazione:
il checkpoint con val_loss minima non e' necessariamente quello che genera le
immagini migliori. Questo script confronta piu' checkpoint LDM in modo oggettivo:
 
  per ogni checkpoint ldm_unet_epoch{N}.pt:
    1. genera N_SAMPLES immagini sintetiche (default 100) in una cartella dedicata
    2. calcola il FID 2.5D di quelle immagini vs il test set reale
  alla fine: classifica i checkpoint per FID e salva un JSON con la curva
  FID-vs-epoca (utile anche come figura per la tesi).
 
E' SELF-CONTAINED: non modifica sample.py ne' eval.py. Riusa solo le funzioni di
metrics.py (FID) e replica la logica minima di generazione di sample.py.
 
Lancio (1 GPU, niente DDP - generazione sequenziale per semplicita' e controllo):
    python3 -m src.evaluation.checkpoint_selection --n_samples 100
"""
import os
import json
import glob
import argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
import numpy as np 
import nibabel as nib 
import torch 
from torch.amp import autocast
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism
from tqdm import tqdm
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from src.data.transforms import get_encoding_transforms
from src.evaluation.metrics import VolumeStream, compute_fid_2p5d

 
# Caricamento modelli (replica minima di sample.py, senza DDP)
def load_autoencoder(config_net, ckpt_path, device):
    ae=config_net["autoencoder_def"]
    net=AutoencoderKlMaisi(
        spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
        num_channels=ae.get("num_channels", [128, 256, 512]),
        num_res_blocks=ae.get("num_res_blocks", [2, 2, 2]),
        norm_num_groups=ae.get("norm_num_groups", 32),
        norm_eps=ae.get("norm_eps", 1e-6),
        attention_levels=ae.get("attention_levels", [False, False, False]),
        with_encoder_nonlocal_attn=ae.get("with_encoder_nonlocal_attn", False),
        with_decoder_nonlocal_attn=ae.get("with_decoder_nonlocal_attn", False),
        use_checkpointing=False,
        use_convtranspose=ae.get("use_convtranspose", False),
        norm_float16=ae.get("norm_float16", True),
        num_splits=ae.get("num_splits", 4),
        dim_split=ae.get("dim_split", 1),
    ).to(device)

    ckpt=torch.load(ckpt_path, map_location=device)
    state=ckpt["autoencoder_state_dict"] if "autoencoder_state_dict" in ckpt else ckpt
    state={k.replace("module.", "", 1): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    for p in net.parameters():
        p.requires_grad=False
    return net
 
 
def load_unet(config_net, ckpt_path, device):
    nc=config_net["diffusion_unet_def"]
    net=DiffusionModelUNetMaisi(
        spatial_dims=3, in_channels=4, out_channels=4,
        num_channels=nc.get("num_channels", [64, 128, 256, 512]),
        attention_levels=nc.get("attention_levels", [False, False, True, True]),
        num_head_channels=nc.get("num_head_channels", [0, 0, 32, 32]),
        num_res_blocks=nc.get("num_res_blocks", 2),
        use_flash_attention=nc.get("use_flash_attention", True),
        resblock_updown=nc.get("resblock_updown", True),
        include_fc=nc.get("include_fc", True),
        with_conditioning=False, num_class_embeds=None,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        include_spacing_input=False,
    ).to(device)

    ckpt=torch.load(ckpt_path, map_location=device, weights_only=False)
    state={k.replace("module.", "", 1): v for k, v in ckpt["unet_state_dict"].items()}
    net.load_state_dict(state, strict=True)
    sf=ckpt["scale_factor"]
    if isinstance(sf, torch.Tensor):
        sf=sf.to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad=False
    return net, sf
 
 
class ReconModel(torch.nn.Module):
    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder=autoencoder
        self.scale_factor=scale_factor
 
    def forward(self, z):
        return self.autoencoder.decode_stage_2_outputs(z/self.scale_factor)
 
 
@torch.inference_mode()
def generate_one(unet, recon_model, scheduler, latent_shape, steps, device, inferer):
    noise=torch.randn((1, *latent_shape), device=device)
    image=noise
    scheduler.set_timesteps(num_inference_steps=steps,
                            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])))
    all_t=scheduler.timesteps
    all_next=torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))
    with autocast("cuda", enabled=True):
        for t, nt in zip(all_t, all_next):
            out=unet(x=image, timesteps=torch.Tensor((t,)).to(device))
            image, _=scheduler.step(out, t, image, nt)
        synth=inferer(network=recon_model, inputs=image) if inferer is not None else recon_model(image)
    data=synth.squeeze().cpu().float().numpy()
    return np.clip(data, 0.0, 1.0)
 
 
# Caricamento reali (test set)
_REAL_TF=None
 
def _load_real(item):
    global _REAL_TF
    if _REAL_TF is None:
        _REAL_TF=get_encoding_transforms()
    path=item["image"] if isinstance(item, dict) else item
    out=_REAL_TF({"image": path})
    img=out["image"]
    if hasattr(img, "as_tensor"):
        img=img.as_tensor()
    return img.squeeze(0).float()
 
 
def _load_synth(path):
    return torch.from_numpy(nib.load(path).get_fdata().astype(np.float32))
 
# Main
def main():
    ap=argparse.ArgumentParser(description="Selezione checkpoint LDM via FID")
    ap.add_argument("--config", type=str, default="configs/config_diff_model.json")
    ap.add_argument("--network", type=str, default="configs/config_network.json")
    ap.add_argument("--splits", type=str, default="data/splits/dataset.json")
    ap.add_argument("--models_dir", type=str, default="outputs/models_v2")
    ap.add_argument("--work_dir", type=str, default="outputs/checkpoint_selection_v2",
                    help="dove salvare le immagini temporanee e il JSON dei risultati")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--epochs", type=str, default="100,200,300,400,500,600,700,800,900,1000",
                    help="lista epoche dei checkpoint da testare, separate da virgola")
    args=ap.parse_args()
 
    device="cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.work_dir, exist_ok=True)
    results_path=os.path.join(args.work_dir, "fid_by_checkpoint_v2.json")
 
    # carica risultati gia' presenti (ripresa incrementale)
    results={}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results=json.load(f)
        print(f"Risultati esistenti caricati: {list(results.keys())}")
 
    with open(args.config) as f:
        config=json.load(f)
    with open(args.network) as f:
        config_net=json.load(f)
 
    infer_cfg=config["diffusion_unet_inference"]
    output_size=tuple(infer_cfg.get("dim", [256, 256, 256]))
    spacing=tuple(infer_cfg.get("spacing", [1.0, 1.0, 1.0]))
    steps=infer_cfg.get("num_inference_steps", 30)
    base_seed=infer_cfg.get("random_seed", 42)
    latent_channels=config_net.get("latent_channels", 4)
    latent_shape=(latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)
 
    paths=config["paths"]
    ae_ckpt=paths.get("trained_autoencoder_path", "./outputs/models_v2/autoencoder_best.pt")
 
    # il VAE e' lo stesso per tutti i checkpoint LDM: caricalo UNA volta
    print(f"Carico VAE da {ae_ckpt}")
    autoencoder=load_autoencoder(config_net, ae_ckpt, device)
 
    sched_cfg=config_net["noise_scheduler"]
    scheduler=RFlowScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        use_discrete_timesteps=sched_cfg.get("use_discrete_timesteps", False),
        use_timestep_transform=sched_cfg.get("use_timestep_transform", True),
        scale=sched_cfg.get("scale", 1.4),
        sample_method=sched_cfg.get("sample_method", "uniform"),
    )
    inferer=SlidingWindowInferer(roi_size=[64, 64, 64], sw_batch_size=1, progress=False,
                                   mode="gaussian", overlap=0.4, sw_device=device, device=device)
 
    # reali del test set (stream pigro, riusato per ogni FID)
    with open(args.splits) as f:
        test_items=json.load(f)["test"]
    real_stream=VolumeStream(test_items, _load_real)
    print(f"Test set reale: {len(real_stream)} volumi")
 
    epochs=[int(e) for e in args.epochs.split(",")]
 
    for ep in epochs:
        key=f"epoch{ep}"
        if key in results:
            print(f"[{key}] gia' valutato (FID={results[key]['fid_avg']:.3f}), salto.")
            continue
 
        ckpt_path=os.path.join(args.models_dir, f"ldm_unet_epoch{ep}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[{key}] checkpoint non trovato ({ckpt_path}), salto.")
            continue
 
        print(f"\n{'='*55}\n[{key}] genero {args.n_samples} campioni\n{'='*55}")
        unet, scale_factor=load_unet(config_net, ckpt_path, device)
        recon=ReconModel(autoencoder, scale_factor).to(device)
 
        synth_dir=os.path.join(args.work_dir, f"synth_{key}")
        os.makedirs(synth_dir, exist_ok=True)
 
        for idx in tqdm(range(args.n_samples), desc=key):
            set_determinism(seed=base_seed + idx)
            data=generate_one(unet, recon, scheduler, latent_shape, steps, device, inferer)
            affine=np.eye(4)
            for i in range(3):
                affine[i, i]=spacing[i]
            nib.save(nib.Nifti1Image(data.astype(np.float32), affine),
                     os.path.join(synth_dir, f"hc_synth_{idx+1:04d}.nii.gz"))
 
        # libera la UNet prima del FID
        del unet, recon
        if device != "cpu":
            torch.cuda.empty_cache()
 
        # FID di questi campioni vs test set
        print(f"[{key}] calcolo FID...")
        synth_files=sorted(glob.glob(os.path.join(synth_dir, "hc_synth_*.nii.gz")))
        synth_stream=VolumeStream(synth_files, _load_synth)
        fid_res=compute_fid_2p5d(real_stream, synth_stream, device=device,
                                   drop_empty=True, batch_size=32, verbose=True)
        results[key]={"epoch": ep, **fid_res}
 
        # salva in modo incrementale (se interrotto, non perdi il lavoro)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{key}] FID medio = {fid_res['fid_avg']:.3f}  (salvato in {results_path})")
 
    # classifica finale
    print(f"\n{'='*55}\nCLASSIFICA CHECKPOINT PER FID (piu' basso = meglio)\n{'='*55}")
    ranked=sorted(results.items(), key=lambda kv: kv[1]["fid_avg"])
    for rank, (key, r) in enumerate(ranked, 1):
        print(f"{rank}. {key:>10}  FID medio = {r['fid_avg']:.3f}  "
              f"(XY {r['fid_xy']:.1f} / YZ {r['fid_yz']:.1f} / ZX {r['fid_zx']:.1f})")
    if ranked:
        best_key, best_r=ranked[0]
        print(f"\nMIGLIORE: {best_key} con FID {best_r['fid_avg']:.3f}")
        print("NB: con 100 campioni il FID ha ancora varianza; differenze piccole")
        print("(<5) tra checkpoint vicini potrebbero non essere significative.")
 
 
if __name__=="__main__":
    main()