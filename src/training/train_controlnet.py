#src/training/train_controlnet.py
"""
Training della ControlNet sopra l'LDM curriculum (v5) CONGELATO.

Idea (Zhang et al.): si congela il modello di diffusione gia' addestrato e si addestra
una copia dell'encoder (la ControlNet) che riceve una CONDIZIONE — qui la maschera di
segmentazione FSL-FAST a 3 tessuti — e inietta i suoi residui nella UNet congelata.
La UNet non cambia: impara solo la ControlNet a guidare la generazione verso la maschera.

Adatta lo schema di NV-Generate-CTMR (MAISI) all'LDM di questo progetto. Differenze:
    1. UNet CONGELATA dal checkpoint del curriculum (models_v5), istanziata con la stessa
       firma di train_ldm.setup_unet.
    2. Normalizzazione latenti PER-CANALE + CENTERING: (z - latent_mean) * scale_factor,
       con latent_mean/scale_factor [1,4,1,1,1] CARICATI DAL CHECKPOINT (non ricalcolati).
    3. Niente spacing_tensor/class_labels/region index (la UNet non li usa).
    4. Timestep sampling=uniform (l'LDM e' congelato: la ControlNet impara solo a
       condizionare, quindi il curriculum non serve).
    5. Weighted loss disattivata (le 3 classi contano uniformemente).
    6. ControlNet inizializzata copiando i pesi dell'encoder della UNet (copy_model_state).

Loss e target IDENTICI a train_ldm.py: target=images-noise, loss=L1.
Tracciamento MLflow (esperimento dedicato "ControlNet_training") + grafico finale.

Lancio (DDP, 4 GPU) dalla radice della repo:
    torchrun --nproc_per_node=4 -m src.training.train_controlnet
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from monai.utils import set_determinism
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi
from monai.networks.schedulers.rectified_flow import RFlowScheduler
from monai.networks.utils import copy_model_state

from src.data.controlnet_dataset import setup_controlnet_dataloaders
from src.data.binarize import binarize_labels

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank=int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    return False, 0, 0, 1

def is_main():
    return (not dist.is_initialized()) or dist.get_rank()==0

def setup_unet(net_cfg, device):
    """UNet identica a train_ldm.setup_unet: stessa firma -> i pesi combaciano."""
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

def setup_controlnet(cn_cfg, device):
    """
    ControlNet allineata alla UNet. num_class_embeds DEVE essere None (nel
    config_network.json va messo null, altrimenti si aspetta class_labels).
    """
    assert cn_cfg.get("num_class_embeds", None) is None, \
        "controlnet_def.num_class_embeds deve essere null (allineato alla UNet)"
    controlnet=ControlNetMaisi(
        spatial_dims=3,
        in_channels=4,
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
    return controlnet

def setup_noise_scheduler(sched_cfg):
    """Identico a train_ldm.setup_noise_scheduler."""
    return RFlowScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        use_discrete_timesteps=sched_cfg.get("use_discrete_timesteps", False),
        use_timestep_transform=sched_cfg.get("use_timestep_transform", True),
        loc=sched_cfg.get("loc", 0.0),
        scale=sched_cfg.get("scale", 1.0),
        sample_method=sched_cfg.get("sample_method", "uniform"),
    )

def compute_output(images, labels, noise, timesteps, noise_scheduler, controlnet, unet):
    """
    Forward completo (versione ridotta di compute_model_output di MAISI, senza
    spacing/modality/region):
        controlnet_cond=binarize(mask) -> [B,8,X,Y,Z]
        noisy=add_noise(images, noise, t)
        controlnet(noisy, t, cond) -> residui
        unet(noisy, t, +residui) -> velocity
    """
    controlnet_cond=binarize_labels(labels.to(torch.long)).float()
    noisy_latent=noise_scheduler.add_noise(
        original_samples=images, noise=noise, timesteps=timesteps,
    )
    down_res, mid_res=controlnet(x=noisy_latent, timesteps=timesteps, controlnet_cond=controlnet_cond)
    model_output=unet(
        x=noisy_latent, timesteps=timesteps,
        down_block_additional_residuals=down_res,
        mid_block_additional_residual=mid_res,
    )
    return model_output

class EMA:
    """Media esponenziale dei pesi della ControlNet (opzionale, da config)."""
    def __init__(self, model, decay=0.999):
        self.decay=decay
        self.shadow={k:v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
            else:
                self.shadow[k].copy_(v)

def save_checkpoint(path, epoch, controlnet, optimizer, loss, ema=None):
    cn=controlnet.module if isinstance(controlnet, DistributedDataParallel) else controlnet
    payload={
        "epoch":epoch,
        "loss":loss,
        "controlnet_state_dict":cn.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
    }
    if ema is not None:
        payload["controlnet_ema"]=ema.shadow
    torch.save(payload, path)

def plot_loss_curve(train_hist, val_hist, save_path):
    """Grafico finale: train_loss per epoca + val_loss ai punti di validazione."""
    fig, ax=plt.subplots(figsize=(10, 6))
    tr_e=[e for e, _ in train_hist]
    tr_v=[v for _, v in train_hist]
    ax.plot(tr_e, tr_v, label="train_loss", color="#1f4e79")
    if val_hist:
        vl_e=[e for e, _ in val_hist]
        vl_v=[v for _, v in val_hist]
        ax.plot(vl_e, vl_v, label="val_loss", color="#c0504d", marker="o", linestyle="--")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss L1")
    ax.set_title("ControlNet - curva di training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def run_epoch(
    epoch, controlnet, unet, loader, optimizer, lr_scheduler,
    noise_scheduler, latent_mean, scale_factor, device, scaler,
    loss_pt, train=True, amp=True, ema=None,
):
    """Singola epoca. target=images-noise, loss L1 (come train_ldm.py)."""
    controlnet.train(train)
    loss_acc=torch.zeros(2, dtype=torch.float, device=device)

    tag="train" if train else "val"
    progress_bar=tqdm(
        loader,
        desc=f"Epoch {epoch+1} [{tag}|{noise_scheduler.sample_method}]",
        ncols=100,
        disable=(not is_main()),
    )

    for batch in progress_bar:
        #latente grezzo -> normalizzato per-canale e centrato (dal checkpoint)
        images=batch["image"].to(device)
        images=(images-latent_mean)*scale_factor
        labels=batch["label"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train), autocast("cuda", enabled=amp):
            noise=torch.randn_like(images)

            #timesteps campionati dallo scheduler (uniform per la controlnet)
            timesteps=noise_scheduler.sample_timesteps(images)

            model_output=compute_output(
                images, labels, noise, timesteps, noise_scheduler, controlnet, unet,
            )

            #target rectified flow (velocity), identico a train_ldm.py
            model_gt=images-noise
            loss=loss_pt(model_output.float(), model_gt.float())

        if train:
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if ema is not None:
                ema.update(controlnet.module if isinstance(controlnet, DistributedDataParallel) else controlnet)

        loss_acc[0]+=loss.detach()
        loss_acc[1]+=1
        if is_main():
            progress_bar.set_postfix(loss=float(loss.detach()))

    if dist.is_initialized():
        dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
    if train and lr_scheduler is not None:
        lr_scheduler.step()
    return (loss_acc[0]/loss_acc[1]).item()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--net_config", default="configs/config_network.json")
    parser.add_argument("--train_config", default="configs/config_controlnet.json")
    args=parser.parse_args()

    ddp, local_rank, rank, world=setup_ddp()
    device=torch.device(f"cuda:{local_rank}")
    set_determinism(seed=42+rank)

    net_cfg=json.load(open(args.net_config))
    tcfg=json.load(open(args.train_config))["controlnet_train"]

    if is_main():
        print(f"Device: {device}, world_size: {world}")
        print(f"Base LDM: {tcfg['trained_diffusion_ckpt']}")
        print(f"Output: {tcfg['controlnet_out_dir']}")
        mlflow.set_experiment("ControlNet_training")

    #UNet congelata + pesi dal curriculum
    unet=setup_unet(net_cfg["diffusion_unet_def"], device)
    ckpt=torch.load(tcfg["trained_diffusion_ckpt"], map_location="cpu", weights_only=False)
    unet.load_state_dict(ckpt["unet_state_dict"])
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    #latent_mean/scale_factor per-canale dal checkpoint (NON ricalcolati)
    latent_mean=ckpt["latent_mean"].to(device)
    scale_factor=ckpt["scale_factor"].to(device)
    if is_main():
        print(f"latent_mean per-canale: {latent_mean.flatten().tolist()}")
        print(f"scale_factor per-canale: {scale_factor.flatten().tolist()}")

    #ControlNet: copia dei pesi encoder della UNet
    controlnet=setup_controlnet(net_cfg["controlnet_def"], device)
    copy_model_state(controlnet, unet.state_dict())
    if ddp:
        controlnet=DistributedDataParallel(controlnet, device_ids=[local_rank], find_unused_parameters=False)

    #scheduler: uniform per il training controlnet
    sched_cfg=dict(net_cfg["noise_scheduler"])
    sched_cfg["sample_method"]=tcfg.get("sample_method", "uniform")
    noise_scheduler=setup_noise_scheduler(sched_cfg)
    if is_main():
        print(f"Timestep sampling: {noise_scheduler.sample_method}")

    #dataloaders
    train_loader, val_loader=setup_controlnet_dataloaders(
        config={
            "batch_size":tcfg.get("batch_size", 1),
            "num_workers":tcfg.get("num_workers", 2),
            "divisible_k":tcfg.get("divisible_k", 8),
        },
        json_path=tcfg["json_data_list"],
        repo_root=tcfg.get("data_base_dir", "."),
        fold=tcfg.get("fold", 0),
    )

    #optimizer / scheduler_lr / amp / ema
    lr=tcfg.get("lr", 1e-5)
    n_epochs=tcfg.get("n_epochs", 100)
    optimizer=torch.optim.AdamW(controlnet.parameters(), lr=lr)
    lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    amp=tcfg.get("amp", True)
    scaler=GradScaler("cuda", enabled=amp)
    loss_pt=torch.nn.L1Loss()
    ema=EMA(controlnet.module if ddp else controlnet, decay=tcfg.get("ema_decay", 0.999)) \
        if tcfg.get("use_ema", False) else None

    out_dir=tcfg["controlnet_out_dir"]
    if is_main():
        os.makedirs(out_dir, exist_ok=True)

    save_interval=tcfg.get("save_interval", 20)
    val_interval=tcfg.get("val_interval", 10)
    best_val=float("inf")
    train_hist=[]   #(epoch, train_loss)
    val_hist=[]     #(epoch, val_loss)

    if is_main():
        mlflow.log_params({
            "base_ldm":tcfg["trained_diffusion_ckpt"],
            "lr":lr,
            "n_epochs":n_epochs,
            "batch_size":tcfg.get("batch_size", 1),
            "sample_method":noise_scheduler.sample_method,
            "use_ema":bool(ema is not None),
            "weighted_loss":tcfg.get("weighted_loss", 1.0),
        })

    for epoch in range(n_epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss=run_epoch(
            epoch, controlnet, unet, train_loader, optimizer, lr_scheduler,
            noise_scheduler, latent_mean, scale_factor, device, scaler,
            loss_pt, train=True, amp=amp, ema=ema,
        )

        do_val=((epoch+1)%val_interval==0) or (epoch==n_epochs-1)
        val_loss=None
        if do_val:
            with torch.no_grad():
                val_loss=run_epoch(
                    epoch, controlnet, unet, val_loader, optimizer, None,
                    noise_scheduler, latent_mean, scale_factor, device, scaler,
                    loss_pt, train=False, amp=amp, ema=None,
                )

        if is_main():
            current_lr=optimizer.param_groups[0]["lr"]
            msg=f"[epoch {epoch+1}/{n_epochs}] train_loss={train_loss:.5f} | lr={current_lr:.2e}"
            if val_loss is not None:
                msg+=f" | val_loss={val_loss:.5f}"
            print(msg, flush=True)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)
            train_hist.append((epoch+1, train_loss))

            if val_loss is not None:
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                val_hist.append((epoch+1, val_loss))
                if val_loss<best_val:
                    best_val=val_loss
                    save_checkpoint(os.path.join(out_dir, "controlnet_best.pt"), epoch, controlnet, optimizer, val_loss, ema)
                    print(f"  -> nuovo best (val_loss={val_loss:.5f})")

            #salva sempre l'ultimo
            save_checkpoint(os.path.join(out_dir, "controlnet_last.pt"), epoch, controlnet, optimizer, train_loss, ema)
            #checkpoint periodico
            if (epoch+1)%save_interval==0:
                save_checkpoint(os.path.join(out_dir, f"controlnet_epoch{epoch+1}.pt"), epoch, controlnet, optimizer, train_loss, ema)

    if is_main():
        #grafico finale della loss
        plot_path=os.path.join(out_dir, "controlnet_loss_curve.png")
        plot_loss_curve(train_hist, val_hist, plot_path)
        mlflow.log_artifact(plot_path)
        print(f"Grafico salvato: {plot_path}")
        print("Training ControlNet completato.")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__=="__main__":
    main()