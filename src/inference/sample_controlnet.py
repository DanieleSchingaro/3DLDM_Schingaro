#src/inference/sample_controlnet.py
"""
Generazione CONDIZIONATA di MRI cerebrali T1 skull-stripped tramite ControlNet.

Data una maschera di segmentazione (FSL-FAST, 3 tessuti), la ControlNet guida l'LDM
curriculum (congelato) a generare un volume che rispetti quella maschera. Per ogni
maschera in --json_data_list si genera un volume; si salva sia il volume sintetico
sia la maschera-condizione (serve poi al calcolo del DSC: si ri-segmenta il volume
generato e si confronta con questa).

Struttura: e' il sample.py dell'LDM, con due differenze:
    - a ogni step la ControlNet riceve il latente rumoroso + la maschera binarizzata e
      produce residui, iniettati nella UNet (down_block_additional_residuals /
      mid_block_additional_residual). Schema identico a compute_output di
      train_controlnet.py (gia' validato).
    - l'input non e' un contatore di campioni, ma la lista di maschere del JSON.

Normalizzazione: la UNet e' congelata e tarata su (z-latent_mean)*scale_factor; il
latente generato va de-normalizzato nel decode (ReconModel), con scale_factor e
latent_mean per-canale letti dal checkpoint dell'LDM (NON dalla controlnet).

Lancio (DDP, 4 GPU) dalla radice della repo:
    torchrun --nproc_per_node=4 -m src.inference.infer_controlnet \
        --controlnet_ckpt outputs/controlnet_v6/controlnet_epoch100.pt \
        --json_data_list data/splits/controlnet_infer_val.json \
        --out_dir data/controlnet_gen_val/epoch100
"""

import os
import json
import argparse
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
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi

from src.data.binarize import binarize_labels

#ReconModel: decodifica il latente de-normalizzando (come sample.py dell'LDM)
class ReconModel(torch.nn.Module):
    def __init__(self, autoencoder, scale_factor, latent_mean):
        super().__init__()
        self.autoencoder=autoencoder
        self.scale_factor=scale_factor
        self.latent_mean=latent_mean

    def forward(self, z):
        z=z/self.scale_factor + self.latent_mean
        return self.autoencoder.decode_stage_2_outputs(z)

def setup_ddp_optional():
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

def load_autoencoder(config_net, checkpoint_path, device):
    ae_cfg=config_net["autoencoder_def"]
    autoencoder=AutoencoderKlMaisi(
        spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
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

def build_unet(config_net, device):
    net_cfg=config_net["diffusion_unet_def"]
    return DiffusionModelUNetMaisi(
        spatial_dims=3, in_channels=4, out_channels=4,
        num_channels=net_cfg.get("num_channels", [64, 128, 256, 512]),
        attention_levels=net_cfg.get("attention_levels", [False, False, True, True]),
        num_head_channels=net_cfg.get("num_head_channels", [0, 0, 32, 32]),
        num_res_blocks=net_cfg.get("num_res_blocks", 2),
        use_flash_attention=net_cfg.get("use_flash_attention", True),
        resblock_updown=net_cfg.get("resblock_updown", True),
        include_fc=net_cfg.get("include_fc", True),
        with_conditioning=False, num_class_embeds=None,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        include_spacing_input=False,
    ).to(device)

def load_unet(config_net, checkpoint_path, device):
    """UNet dell'LDM curriculum congelata + scale_factor/latent_mean per-canale."""
    unet=build_unet(config_net, device)
    ckpt=torch.load(checkpoint_path, map_location=device, weights_only=False)
    state={k.replace("module.", "", 1): v for k, v in ckpt["unet_state_dict"].items()}
    unet.load_state_dict(state, strict=True)
    scale_factor=ckpt["scale_factor"]
    if isinstance(scale_factor, torch.Tensor):
        scale_factor=scale_factor.to(device)
    latent_mean=ckpt.get("latent_mean", 0.0)
    if isinstance(latent_mean, torch.Tensor):
        latent_mean=latent_mean.to(device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad=False
    return unet, scale_factor, latent_mean

def build_controlnet(config_net, checkpoint_path, device):
    """ControlNet allineata alla UNet + pesi addestrati (controlnet_v6)."""
    cn_cfg=config_net["controlnet_def"]
    controlnet=ControlNetMaisi(
        spatial_dims=3, in_channels=4,
        num_channels=cn_cfg.get("num_channels", [64, 128, 256, 512]),
        attention_levels=cn_cfg.get("attention_levels", [False, False, True, True]),
        num_head_channels=cn_cfg.get("num_head_channels", [0, 0, 32, 32]),
        num_res_blocks=cn_cfg.get("num_res_blocks", 2),
        resblock_updown=cn_cfg.get("resblock_updown", True),
        include_fc=cn_cfg.get("include_fc", True),
        use_flash_attention=cn_cfg.get("use_flash_attention", True),
        conditioning_embedding_in_channels=cn_cfg.get("conditioning_embedding_in_channels", 8),
        conditioning_embedding_num_channels=cn_cfg.get("conditioning_embedding_num_channels", [8, 32, 64]),
        num_class_embeds=None,
    ).to(device)
    ckpt=torch.load(checkpoint_path, map_location=device, weights_only=False)
    #il checkpoint puo' contenere l'EMA: se presente e richiesto, si usa quello
    state=ckpt["controlnet_state_dict"]
    state={k.replace("module.", "", 1): v for k, v in state.items()}
    controlnet.load_state_dict(state, strict=True)
    controlnet.eval()
    for p in controlnet.parameters():
        p.requires_grad=False
    return controlnet

#Generazione di UN volume condizionato su UNA maschera
@torch.inference_mode()
def generate_one(mask, controlnet, unet, recon_model, noise_scheduler,
                 latent_shape, num_inference_steps, device, inferer, cond_scale=1.0,
                 cond_scale_end=None, unet_bad=None, guidance_scale=2.0):
    """
    mask: [1,1,X,Y,Z] intero (0/1/2/3). Genera un volume che la rispetta.
    A ogni step: controlnet(noisy, t, cond)->residui; unet(noisy, t, +residui)->velocity.
    """
    #condizione: maschera binarizzata bit-plane [1,8,X,Y,Z] (COSTANTE per tutti gli step)
    controlnet_cond=binarize_labels(mask.to(torch.long)).float()

    noise=torch.randn((1, *latent_shape), device=device)
    image=noise

    noise_scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
    )
    all_timesteps=noise_scheduler.timesteps
    all_next=torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))

    with autocast("cuda", enabled=True):
        n_steps=len(all_timesteps)
        for i, (t, next_t) in enumerate(zip(all_timesteps, all_next)):
            #scale dello step corrente (costante, oppure lineare start->end)
            if cond_scale_end is None:
                cs=cond_scale
            else:
                frac=i/max(n_steps-1, 1)
                cs=cond_scale+(cond_scale_end-cond_scale)*frac
            t_in=torch.Tensor((t,)).to(device)
            down_res, mid_res=controlnet(x=image, timesteps=t_in, controlnet_cond=controlnet_cond)
            #conditioning scale: >1 rafforza l'aderenza alla maschera, <1 la allenta
            if cs!=1.0:
                down_res=[r*cs for r in down_res]
                mid_res=mid_res*cs
            v_good=unet(
                x=image, timesteps=t_in,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            )
            if unet_bad is not None:
                #stessi residui della ControlNet anche sulla UNet 'bad'
                v_bad=unet_bad(
                    x=image, timesteps=t_in,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                )
                model_output=v_bad + guidance_scale*(v_good - v_bad)
            else:
                model_output=v_good
            image,_=noise_scheduler.step(model_output, t, image, next_t)
        synthetic=inferer(network=recon_model, inputs=image) if inferer is not None else recon_model(image)

    data=synthetic.squeeze().cpu().float().numpy()
    data=np.clip(data, 0.0, None)   #niente tetto: il clip a 1.0 creava un muro di voxel saturi che rompe FAST
    return data

def save_nifti(data, spacing, output_path):
    affine=np.eye(4)
    for i in range(3):
        affine[i, i]=spacing[i]
    img=nib.Nifti1Image(data.astype(np.float32), affine=affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(img, output_path)

def load_mask(path):
    """Carica la maschera come tensore [1,1,X,Y,Z] intero (nessuna interpolazione)."""
    arr=np.asarray(nib.load(path).dataobj).astype(np.int64)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def main():
    parser=argparse.ArgumentParser(description="Inferenza ControlNet condizionata su maschera")
    parser.add_argument("--config", type=str, default="configs/config_diff_model.json")
    parser.add_argument("--network", type=str, default="configs/config_network.json")
    parser.add_argument("--controlnet_ckpt", type=str, required=True,
                        help="checkpoint della ControlNet (es. outputs/controlnet_v6/controlnet_epoch100.pt)")
    parser.add_argument("--ldm_ckpt", type=str, default="outputs/models_v5/ldm_unet_epoch800.pt",
                        help="checkpoint dell'LDM curriculum (UNet congelata + scale/mean)")
    parser.add_argument("--json_data_list", type=str, required=True,
                        help="lista maschere (controlnet_infer_val.json o _test.json)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="cartella dei volumi generati + maschere-condizione")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cond_scale", type=float, default=1.0,
                        help="fattore sui residui della ControlNet (>1 rafforza il condizionamento)")
    parser.add_argument("--ldm_ckpt_bad", type=str, default=None,
                        help="checkpoint LDM 'bad' (epoca precoce dello stesso run) per l'autoguidance")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="peso w dell'autoguidance: v = v_bad + w*(v_good - v_bad)")
    parser.add_argument("--cond_scale_end", type=float, default=None,
                        help="se dato, il fattore scende linearmente da --cond_scale (primo step) a questo valore (ultimo step)")
    parser.add_argument("--base_seed", type=int, default=42)
    args=parser.parse_args()

    config=json.load(open(args.config))
    config_net=json.load(open(args.network))
    local_rank, world_size, device=setup_ddp_optional()
    is_main=local_rank==0

    paths=config["paths"]
    ae_ckpt=paths.get("trained_autoencoder_path", "./outputs/models_v2/autoencoder_best.pt")

    if is_main:
        print(f"VAE: {ae_ckpt}")
        print(f"LDM (congelato): {args.ldm_ckpt}")
        print(f"ControlNet: {args.controlnet_ckpt}")
        print(f"Maschere: {args.json_data_list}")
        sched="costante" if args.cond_scale_end is None else f"-> {args.cond_scale_end} (lineare)"
        print(f"cond_scale={args.cond_scale} {sched} | steps={args.num_inference_steps}")

    autoencoder=load_autoencoder(config_net, ae_ckpt, device)
    unet, scale_factor, latent_mean=load_unet(config_net, args.ldm_ckpt, device)
    #UNet 'bad' per l'autoguidance (opzionale)
    unet_bad=None
    if args.ldm_ckpt_bad is not None:
        unet_bad,_,_=load_unet(config_net, args.ldm_ckpt_bad, device)
        if is_main:
            print(f"autoguidance: bad={args.ldm_ckpt_bad}, w={args.guidance_scale}")

    controlnet=build_controlnet(config_net, args.controlnet_ckpt, device)
    recon_model=ReconModel(autoencoder, scale_factor, latent_mean).to(device)

    if is_main:
        sf=scale_factor.flatten().tolist() if isinstance(scale_factor, torch.Tensor) else scale_factor
        print(f"scale_factor(per-canale)={sf}")

    #scheduler RFlow (uniform in inferenza, coerente con il training controlnet)
    sched_cfg=config_net["noise_scheduler"]
    noise_scheduler=RFlowScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        use_discrete_timesteps=sched_cfg.get("use_discrete_timesteps", False),
        use_timestep_transform=sched_cfg.get("use_timestep_transform", True),
        loc=sched_cfg.get("loc", 0.0),
        scale=sched_cfg.get("scale", 1.0),
        sample_method="uniform",
    )

    #lista maschere
    items=json.load(open(args.json_data_list))["training"]
    output_size=tuple(items[0].get("dim", [256, 256, 256]))
    latent_channels=config_net.get("latent_channels", 4)
    latent_shape=(latent_channels, output_size[0]//4, output_size[1]//4, output_size[2]//4)

    inferer=SlidingWindowInferer(
        roi_size=[64, 64, 64], sw_batch_size=1, progress=False,
        mode="gaussian", overlap=0.4, sw_device=device, device=device,
    )

    #suddivisione delle maschere tra i rank
    my_items=items[local_rank::world_size]
    if is_main:
        print(f"Genero {len(items)} volumi condizionati su {world_size} GPU -> {args.out_dir}")

    for i, item in enumerate(tqdm(my_items, desc=f"rank{local_rank}", disable=not is_main)):
        #seed deterministico per riproducibilita' (indice globale)
        global_idx=local_rank + i*world_size
        set_determinism(seed=args.base_seed+global_idx)

        mask_path=item["label"]
        spacing=tuple(item.get("spacing", [1.0, 1.0, 1.0]))
        base=os.path.basename(mask_path).replace("_pveseg.nii.gz", "")

        mask=load_mask(mask_path).to(device)
        data=generate_one(mask, controlnet, unet, recon_model, noise_scheduler,
                          latent_shape, args.num_inference_steps, device, inferer,
                          cond_scale=args.cond_scale, cond_scale_end=args.cond_scale_end,
                          unet_bad=unet_bad, guidance_scale=args.guidance_scale)

        #salva il volume generato E la maschera-condizione (per il DSC)
        save_nifti(data, spacing, os.path.join(args.out_dir, f"{base}_synth.nii.gz"))
        mask_np=mask.squeeze().cpu().numpy().astype(np.float32)
        save_nifti(mask_np, spacing, os.path.join(args.out_dir, f"{base}_condmask.nii.gz"))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print(f"Inferenza completata. Output in {args.out_dir}")

if __name__=="__main__":
    main()