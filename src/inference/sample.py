#src/inference/sample.py
"""
Generazione di MRI cerebrali T1 skull-stripped sintetiche (HC) con l'LDM trainato.
Basato su diff_model_infer.py di NV-Generate-CTMR (MAISI), adattato al caso
Incondizionato: nessuna modality/region/spacing, solo rumore->denoising->decode.

MODIFICHE v4:
    - latent_mean PER-CANALE: il training v5 normalizza (z-mean)*scale, quindi in
      inferenza si de-normalizza z/scale + mean (ReconModel). latent_mean e
      scale_factor sono tensori [1,C,1,1,1] letti dal checkpoint.
    - AUTOGUIDANCE (sostituto EMA per modelli incondizionati): si guida il modello
      buono con una versione peggiore di se stesso (un checkpoint precoce dello
      STESSO run). velocity = v_bad + w*(v_good - v_bad).
      Attiva di default (obbligatoria); disattivabile con --no_autoguidance per
      generare la baseline di confronto.
    - scheduler: aggiunto loc, default scale 1.0 (coerente con train_ldm.py v4).

Pipeline:
    1. rumore gaussiano [1,4,64,64,64]
    2. denoising RFlow (num_inference_steps step, default 30), con autoguidance
    3. decode del latente con il VAE (de-normalizzando con scale_factor + mean)
    4. salvataggio come .nii.gz in --out_dir
    5. (solo rank 0) PNG di anteprima in --png_dir
"""

import os
import json
import glob
import argparse
from datetime import datetime
#riduce la frammentazione della memoria CUDA: va settato prima di importare torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
import numpy as np 
import nibabel as nib
import torch
import torch.distributed as dist
from torch.amp import autocast
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism 
from tqdm import tqdm 
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi 
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi 

#ReconModel -> decodifica il latente in immagine
class ReconModel(torch.nn.Module):
    """
    Wrapper che decodifica un latente in immagine de-normalizzando (v5).
    Il latente generato dall'LDM e' nello spazio normalizzato+centrato
    (z_train = (z - latent_mean) * scale_factor), quindi va de-normalizzato con
    l'operazione inversa: z = z_gen / scale_factor + latent_mean, prima del decoder.
    scale_factor e latent_mean sono tensori [1,C,1,1,1] (per-canale).
    """
    def __init__(self, autoencoder, scale_factor, latent_mean):
        super().__init__()
        self.autoencoder=autoencoder
        self.scale_factor=scale_factor
        self.latent_mean=latent_mean
    
    def forward(self, z):
        z=z/self.scale_factor + self.latent_mean
        recon=self.autoencoder.decode_stage_2_outputs(z)
        return recon


#DDP setup
def setup_ddp_optional():
    """
    Inizializza DDP se lanciato con torchrun, altrimenti singola GPU.
    """
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank=int(os.environ["LOCAL_RANK"])
        world_size=dist.get_world_size()
    else:
        local_rank=0
        world_size=1
    torch.cuda.set_device(local_rank)
    device=torch.device(f"cuda:{local_rank}")
    return local_rank, world_size, device

#Caricamento modelli
def load_autoencoder(config_net:dict, checkpoint_path:str, device:torch.device):
    """
    Carica il VAE trainato in eval, frozen.
    use_checkpointing=False in inferenza.
    """
    ae_cfg=config_net["autoencoder_def"]
    autoencoder=AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=ae_cfg.get("num_channels", [128, 256, 512]),
        num_res_blocks=ae_cfg.get("num_res_blocks", [2, 2, 2]),
        norm_num_groups=ae_cfg.get("norm_num_groups", 32),
        norm_eps=ae_cfg.get("norm_eps", 1e-6),
        attention_levels=ae_cfg.get("attention_levels", [False, False, False]),
        with_encoder_nonlocal_attn=ae_cfg.get("with_encoder_nonlocal_attn", False),
        with_decoder_nonlocal_attn=ae_cfg.get("with_decoder_nonlocal_attn", False),
        use_checkpointing=False,
        use_convtranspose=ae_cfg.get("use_convtranspose", False),
        norm_float16=ae_cfg.get("norm_float16", True),
        num_splits=ae_cfg.get("num_splits", 4),
        dim_split=ae_cfg.get("dim_split", 1),
    ).to(device)

    ckpt=torch.load(checkpoint_path, map_location=device)
    state=ckpt["autoencoder_state_dict"] if "autoencoder_state_dict" in ckpt else ckpt
    state={k.replace("module.", "", 1): v for k, v in state.items()}
    autoencoder.load_state_dict(state)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad=False
    
    return autoencoder

def build_unet(config_net:dict, device:torch.device):
    """Costruisce la UNet di diffusione (architettura, senza pesi)."""
    net_cfg=config_net["diffusion_unet_def"]
    unet=DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=net_cfg.get("num_channels", [64, 128, 256, 512]),
        attention_levels=net_cfg.get("attention_levels", [False, False, True, True]),
        num_head_channels=net_cfg.get("num_head_channels", [0, 0, 32, 32]),
        num_res_blocks=net_cfg.get("num_res_blocks", 2),
        use_flash_attention=net_cfg.get("use_flash_attention", True),
        resblock_updown=net_cfg.get("resblock_updown", True),
        include_fc=net_cfg.get("include_fc", True),
        with_conditioning=False,
        num_class_embeds=None,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        include_spacing_input=False,
    ).to(device)
    return unet

def load_unet(config_net:dict, checkpoint_path:str, device:torch.device):
    """
    Carica la UNet di diffusione trainata + scale_factor + latent_mean +
    num_train_timesteps dal checkpoint dell'LDM (v5).
    """
    unet=build_unet(config_net, device)
    ckpt=torch.load(checkpoint_path, map_location=device, weights_only=False)
    unet_state={k.replace("module.", "", 1): v for k, v in ckpt["unet_state_dict"].items()}
    unet.load_state_dict(unet_state, strict=True)

    scale_factor=ckpt["scale_factor"]
    if isinstance(scale_factor, torch.Tensor):
        scale_factor=scale_factor.to(device)
    #v4: latent_mean per-canale. Retrocompatibile: se assente (checkpoint pre-v4),
    #si usa 0 -> de-normalizzazione = solo /scale, come le versioni precedenti.
    latent_mean=ckpt.get("latent_mean", 0.0)
    if isinstance(latent_mean, torch.Tensor):
        latent_mean=latent_mean.to(device)

    num_train_timesteps=ckpt.get("num_train_timesteps", 1000)

    unet.eval()
    for p in unet.parameters():
        p.requires_grad=False
    return unet, scale_factor, latent_mean, num_train_timesteps

def load_unet_weights_only(config_net:dict, checkpoint_path:str, device:torch.device):
    """
    Carica SOLO i pesi di una UNet (per il modello 'bad' dell'autoguidance).
    Non rilegge scale_factor/latent_mean: quelli del run sono presi dal 'good'
    (i due checkpoint appartengono allo STESSO run, quindi condividono la stessa
    normalizzazione).
    """
    unet=build_unet(config_net, device)
    ckpt=torch.load(checkpoint_path, map_location=device, weights_only=False)
    unet_state={k.replace("module.", "", 1): v for k, v in ckpt["unet_state_dict"].items()}
    unet.load_state_dict(unet_state, strict=True)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad=False
    return unet

#Generazione del singolo volume
@torch.inference_mode()
def generate_one(
    unet, unet_bad, guidance_scale,
    recon_model, noise_scheduler,
    latent_shape, num_inference_steps, device, inferer,
):
    """
    Genera un singolo volume sintetico con autoguidance:
    rumore -> denoising RFlow guidato -> decode -> numpy [X,Y,Z] in [0,1].

    Autoguidance (se unet_bad is not None):
        v = v_bad + w * (v_good - v_bad)
    Con unet_bad=None si genera con la sola v_good (baseline, --no_autoguidance).
    """
    noise=torch.randn((1, *latent_shape), device=device)
    image=noise 

    #imposta i timestep RFlow
    noise_scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
    )

    all_timesteps=noise_scheduler.timesteps 
    all_next=torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))

    with autocast("cuda", enabled=True):
        for t, next_t in zip(all_timesteps, all_next):
            t_in=torch.Tensor((t,)).to(device)
            #velocity del modello buono
            v_good=unet(x=image, timesteps=t_in)
            if unet_bad is not None:
                #velocity del modello "cattivo" (checkpoint precoce stesso run)
                v_bad=unet_bad(x=image, timesteps=t_in)
                #estrapolazione autoguidance
                model_output=v_bad + guidance_scale*(v_good - v_bad)
            else:
                model_output=v_good
            #step RFlow
            image,_=noise_scheduler.step(model_output, t, image, next_t)
        #decode del latente -> immagine (de-normalizzazione dentro ReconModel)
        synthetic=inferer(network=recon_model, inputs=image) if inferer is not None else recon_model(image)
    
    data=synthetic.squeeze().cpu().float().numpy()
    #clamp di sicurezza
    data=np.clip(data, 0.0, 1.0)
    return data

#Salvataggio volume
def save_nifti(data: np.ndarray, spacing:tuple, output_path:str):
    """
    Salva il volume come .nii.gz con affine diagonale dato lo spacing.
    """
    affine=np.eye(4)
    for i in range(3):
        affine[i, i]=spacing[i]
    img=nib.Nifti1Image(data.astype(np.float32), affine=affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(img, output_path)


#Salvataggio PNG di anteprima (solo rank 0, a fine generazione)
def save_previews(volumes_dir:str, png_dir:str):
    """
    Crea un PNG per OGNI volume sintetico, 3 viste ortogonali della slice centrale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files=sorted(glob.glob(os.path.join(volumes_dir, "hc_synth_*.nii.gz")))
    if not files:
        print(f"Nessun volume trovato in {volumes_dir}, salto le anteprime.")
        return
    os.makedirs(png_dir, exist_ok=True)

    for f in files:
        vol=nib.load(f).get_fdata()
        x, y, z=[s//2 for s in vol.shape]
        fig, ax=plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(np.rot90(vol[x, :, :]), cmap="gray"); ax[0].set_title("Sagittale"); ax[0].axis("off")
        ax[1].imshow(np.rot90(vol[:, y, :]), cmap="gray"); ax[1].set_title("Coronale"); ax[1].axis("off")
        ax[2].imshow(np.rot90(vol[:, :, z]), cmap="gray"); ax[2].set_title("Assiale"); ax[2].axis("off")
        name=os.path.basename(f).replace(".nii.gz", "")
        plt.suptitle(name, fontsize=12)
        plt.tight_layout()
        png_path=os.path.join(png_dir, f"{name}.png")
        plt.savefig(png_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"Generati {len(files)} PNG (uno per volume) in {png_dir}")


def main():
    parser=argparse.ArgumentParser(description="Generazione MRI HC sintetiche con LDM (v5, autoguidance)")
    parser.add_argument("--config", type=str, default="configs/config_diff_model.json")
    parser.add_argument("--network", type=str, default="configs/config_network.json")
    parser.add_argument("--n_samples", type=int, default=100, help="numero totale di campioni da generare")
    parser.add_argument("--out_dir", type=str, default="data/synthetic_v5",
                        help="cartella dei volumi .nii.gz sintetici")
    parser.add_argument("--png_dir", type=str, default="outputs/generated/synthetic_v5",
                        help="cartella dei PNG di anteprima")
    parser.add_argument("--no_png", action="store_true", help="se presente, NON genera i PNG di anteprima")
    #checkpoint LDM: il 'good' e' il FID-best da checkpoint_selection (passato a mano)
    parser.add_argument("--ldm_ckpt", type=str, default=None,
                        help="path del checkpoint LDM 'good' (FID-best). Se assente usa paths.model_dir/model_filename del config.")
    #autoguidance
    parser.add_argument("--ldm_ckpt_bad", type=str, default=None,
                        help="path del checkpoint LDM 'bad' (epoca precoce dello STESSO run) per l'autoguidance.")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="scala w dell'autoguidance: v_bad + w*(v_good - v_bad). Tipico 1.5-3.")
    parser.add_argument("--no_autoguidance", action="store_true",
                        help="disattiva l'autoguidance (genera la baseline con la sola v_good).")
    args=parser.parse_args()
 
    with open(args.config) as f:
        config=json.load(f)
    with open(args.network) as f:
        config_net=json.load(f)
 
    local_rank, world_size, device=setup_ddp_optional()
    is_main=local_rank==0
 
    # parametri di inferenza
    infer_cfg=config["diffusion_unet_inference"]
    output_size=tuple(infer_cfg.get("dim", [256, 256, 256]))
    spacing=tuple(infer_cfg.get("spacing", [1.0, 1.0, 1.0]))
    num_inference_steps=infer_cfg.get("num_inference_steps", 30)
    base_seed=infer_cfg.get("random_seed", 42)
 
    paths=config["paths"]
    ae_ckpt=paths.get("trained_autoencoder_path", "./outputs/models_v2/autoencoder_best.pt")
    #good checkpoint: da --ldm_ckpt se dato, altrimenti dal config
    if args.ldm_ckpt is not None:
        ldm_ckpt=args.ldm_ckpt
    else:
        ldm_ckpt=os.path.join(paths.get("model_dir", "./outputs/models_v5"),
                              paths.get("model_filename", "ldm_unet_best.pt"))

    #AUTOGUIDANCE: attiva di default (obbligatoria) salvo --no_autoguidance.
    #Se attiva, il checkpoint 'bad' e' OBBLIGATORIO: senza, il job si ferma
    #subito con errore chiaro (niente fallback silenzioso, niente spreco GPU).
    use_autoguidance=not args.no_autoguidance
    if use_autoguidance and args.ldm_ckpt_bad is None:
        if is_main:
            print("ERRORE: autoguidance attiva ma --ldm_ckpt_bad non fornito.\n"
                  "        Passa il checkpoint 'bad' (epoca precoce dello stesso run),\n"
                  "        oppure genera la baseline con --no_autoguidance.")
        if dist.is_initialized():
            dist.barrier(); dist.destroy_process_group()
        return
    if use_autoguidance and not os.path.exists(args.ldm_ckpt_bad):
        if is_main:
            print(f"ERRORE: checkpoint 'bad' non trovato: {args.ldm_ckpt_bad}")
        if dist.is_initialized():
            dist.barrier(); dist.destroy_process_group()
        return

    latent_channels=config_net.get("latent_channels", 4)
 
    # carica modelli
    if is_main:
        print(f"Carico VAE da {ae_ckpt}")
        print(f"Carico LDM (good) da {ldm_ckpt}")
        if use_autoguidance:
            print(f"Autoguidance ATTIVA | bad: {args.ldm_ckpt_bad} | w={args.guidance_scale}")
        else:
            print("Autoguidance DISATTIVA (baseline, sola v_good)")
    autoencoder=load_autoencoder(config_net, ae_ckpt, device)
    unet, scale_factor, latent_mean, num_train_timesteps=load_unet(config_net, ldm_ckpt, device)

    #modello 'bad' per autoguidance (pesi soltanto; normalizzazione dal good)
    unet_bad=None
    if use_autoguidance:
        unet_bad=load_unet_weights_only(config_net, args.ldm_ckpt_bad, device)

    recon_model=ReconModel(
        autoencoder=autoencoder, scale_factor=scale_factor, latent_mean=latent_mean,
    ).to(device)
 
    if is_main:
        sf=scale_factor.flatten().tolist() if isinstance(scale_factor, torch.Tensor) else scale_factor
        print(f"scale_factor(per-canale)={sf}, num_train_timesteps={num_train_timesteps}")
 
    # scheduler RFlow (stessi parametri del training v4: logit-normal, loc, scale)
    sched_cfg=config_net["noise_scheduler"]
    noise_scheduler=RFlowScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        use_discrete_timesteps=sched_cfg.get("use_discrete_timesteps", False),
        use_timestep_transform=sched_cfg.get("use_timestep_transform", True),
        loc=sched_cfg.get("loc", 0.0),
        scale=sched_cfg.get("scale", 1.0),
        sample_method=sched_cfg.get("sample_method", "uniform"),
    )
 
    # latent shape: output_size / 4 (compressione VAE)
    latent_shape=(
        latent_channels,
        output_size[0]//4,
        output_size[1]//4,
        output_size[2]//4,
    )
 
    # sliding window inferer per il decode
    inferer=SlidingWindowInferer(
        roi_size=[64, 64, 64],
        sw_batch_size=1,
        progress=False,
        mode="gaussian",
        overlap=0.4,
        sw_device=device,
        device=device,
    )
 
    # suddivisione dei campioni tra i rank
    all_indices=list(range(args.n_samples))
    my_indices=all_indices[local_rank::world_size]
 
    if is_main:
        print(f"Genero {args.n_samples} campioni totali su {world_size} GPU")
        print(f"Volumi -> {args.out_dir}")
 
    progress=tqdm(my_indices, desc=f"rank{local_rank}", disable=not is_main)
    for idx in progress:
        set_determinism(seed=base_seed+idx)
        data=generate_one(
            unet, unet_bad, args.guidance_scale,
            recon_model, noise_scheduler,
            latent_shape, num_inference_steps, device, inferer,
        )
        out_path=os.path.join(args.out_dir, f"hc_synth_{idx+1:04d}.nii.gz")
        save_nifti(data, spacing, out_path)
 
    if dist.is_initialized():
        dist.barrier()
 
    if is_main and not args.no_png:
        save_previews(args.out_dir, args.png_dir)
 
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
 
    if is_main:
        print(f"Generazione completata. Volumi in {args.out_dir}, anteprime in {args.png_dir}")
 
 
if __name__=="__main__":
    main()