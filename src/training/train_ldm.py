#src/training/train_ldm.py
"""
Training dell'LDM sui latenti generati dal VAE.
Basato su diff_model_train.py di NV-Generate-CTMR (MAISI)

Obiettivo: generare MRI cerebrali T1 skull-stripped sintetiche di soggetti sani (HC).
Il modello è INCONDIZIONATO:
    -niente body region (top/bottom region index)
    -niente modality embedding
    -niente spacing input
Genera HC dal rumore puro.

Scheduler: RFlowScheduler (rectified flow), target=images-noise

MODIFICHE v4:
    - scale_factor PER-CANALE + CENTERING: invece di uno scalare globale stimato
      su un solo batch, si calcolano media e std PER-CANALE su TUTTO il train set.
      Il latente viene normalizzato come (z - latent_mean) * scale_factor, con
      broadcast su tensori [1,C,1,1,1]. Migliora il condizionamento dei 4 canali
      per l'LDM (canali con std diverse non piu' sbilanciati) e centra il
      bersaglio, avvicinandolo al supporto del rumore N(0,I).
    - latent_mean SALVATO nel checkpoint: serve al sampling per de-normalizzare
      (z/scale_factor + latent_mean) prima del decode VAE.
    - WARMUP dell'optimizer LDM: rampa lineare iniziale del lr (lr_warmup_epochs)
      combinata col decay polinomiale, prima assente.
    - timestep logit-normal: NON qui, si attiva da config_network.json
      (sample_method="logit-normal"); lo scheduler lo legge da li'.

MODIFICHE v5 (CURRICULUM a due fasi sul timestep sampling):
    - Il campionamento dei timestep evolve durante il training:
        fase 1 (epoca < curriculum_switch_epoch): UNIFORM
            -> consolida la struttura GLOBALE (posizione/scala del cervello),
               che si stabilisce ai timestep ad alto rumore (i "bordi" della
               traiettoria), poco campionati dal logit-normal.
        fase 2 (epoca >= curriculum_switch_epoch): LOGIT-NORMAL (loc=0, scale=1)
            -> affina la TEXTURE (mid-range della traiettoria), recuperando la
               nitidezza persa con l'uniform puro.
    - Motivazione: nella v4 il logit-normal statico dava FID migliore ma ~20% di
      campioni con traslazione globale (la struttura globale, appresa male, si
      quantizzava a 1 voxel del bottleneck UNet). L'uniform statico azzerava le
      traslazioni ma perdeva nitidezza. Il curriculum mira ad avere entrambe.
    - Implementazione: si commuta l'attributo mutabile noise_scheduler.sample_method
      all'inizio di ogni epoca (nessuna ricostruzione dello scheduler). Lo switch
      e' NETTO. curriculum_switch_epoch e' letto dal config (default 500).
    - Riferimento: curriculum sui timestep per flow matching (2026), che mostra
      come una distribuzione non-stazionaria (struttura->dettaglio) superi sia
      uniform sia logit-normal statici.
"""

import os
import json
import argparse
import torch
import torch.distributed as dist 
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from monai.utils import set_determinism, first
from monai.networks.schedulers import RFlowScheduler
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi 
from tqdm import tqdm
import mlflow
from src.data.ldm_dataset import setup_ldm_dataloaders

#DDP setup
def setup_ddp()->tuple[int, torch.device]:
    """
    Inizializza DDP con torchrun (LOCAL_RANK, RANK, WORLD_SIZE già settati)
    """
    dist.init_process_group(backend="nccl")
    local_rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device=torch.device(f"cuda:{local_rank}")
    return local_rank, device

#Modello UNet di diffusione
def setup_unet(net_cfg:dict, device:torch.device, local_rank:int)->torch.nn.Module:
    """
    Inizializza la Unet di diffusione incondizionata.
    Tutti i conditioning sono DISATTIVATI per la generazione HC pura:
        - include_top/bottom_region_index_input=False  (niente body region)
        - include_spacing_input=False                  (niente spacing)
        - num_class_embeds=None                        (niente modality)
        - with_conditioning=False                      (niente cross-attention)
    Con questi flag il forward si chiama con solo (x, timesteps).
 
    Parametri letti da config_network["diffusion_unet_def"]. Le chiavi "_target_"
    e i riferimenti "@..." del config bundle vengono ignorati: passiamo i valori
    espliciti.
    """
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
        # conditioning disattivati (generazione HC incondizionata)
        with_conditioning=False,
        num_class_embeds=None,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        include_spacing_input=False,
    ).to(device)

    if dist.is_available() and dist.is_initialized():
        unet=DistributedDataParallel(
            unet, device_ids=[local_rank], find_unused_parameters=False,
        )
    return unet

def setup_noise_scheduler(sched_cfg: dict)->RFlowScheduler:
    """
    Crea il RFlowScheduler con i parametri del config (MAISI-style):
    num_train_timesteps, use_discrete_timesteps, use_timesteps_transform, scale, sample_method.
    NB: sample_method="logit-normal" (impostato in config_network.json) attiva il
    campionamento logit-normale dei timestep; qui non serve altro.
    """
    return RFlowScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        use_discrete_timesteps=sched_cfg.get("use_discrete_timesteps", False),
        use_timestep_transform=sched_cfg.get("use_timestep_transform", True),
        loc=sched_cfg.get("loc", 0.0),
        scale=sched_cfg.get("scale", 1.0),
        sample_method=sched_cfg.get("sample_method", "uniform"),
    )

#scale_factor + latent_mean PER-CANALE (v4)
def calculate_latent_stats(train_loader, device:torch.device):
    """
    Calcola media e std PER-CANALE su TUTTO il train set (non su un solo batch).
    Ritorna (latent_mean, scale_factor), entrambi tensori [1,C,1,1,1] per broadcast.

    Metodo numericamente robusto e DDP-esatto:
      - si accumulano le SOMME per-canale: sum(z), sum(z^2), e il conteggio N di
        elementi per canale (B*X*Y*Z sommato su tutti i batch);
      - su DDP si riducono le SOMME e il conteggio con ReduceOp.SUM (NON si mediano
        le medie per-rank: sarebbe approssimato);
      - da sum e sum2 si ricavano mean e var globali per-canale.
    scale_factor = 1/std per-canale (clamp per sicurezza numerica).
    """
    sum_c=None      # somma per-canale       [C]
    sumsq_c=None    # somma dei quadrati     [C]
    count=0.0       # numero di elementi per canale (scalare, uguale per ogni canale)

    for batch in train_loader:
        z=batch["latent"].to(device)                 # [B,C,X,Y,Z]
        c=z.shape[1]
        # riduci su batch + dimensioni spaziali, tieni il canale
        s=z.sum(dim=(0,2,3,4))                        # [C]
        s2=(z*z).sum(dim=(0,2,3,4))                   # [C]
        n=z.shape[0]*z.shape[2]*z.shape[3]*z.shape[4] # elementi per canale in questo batch

        sum_c=s if sum_c is None else sum_c+s
        sumsq_c=s2 if sumsq_c is None else sumsq_c+s2
        count+=float(n)

    # riduzione DDP ESATTA: somma dei totali + conteggio, poi calcolo
    if dist.is_initialized():
        dist.barrier()
        count_t=torch.tensor([count], device=device)
        dist.all_reduce(sum_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(sumsq_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        count=count_t.item()

    mean_c=sum_c/count                                # [C]
    var_c=sumsq_c/count-mean_c*mean_c                 # [C]
    std_c=torch.sqrt(var_c.clamp_min(1e-8))           # [C]
    scale_c=1.0/std_c.clamp_min(1e-8)                 # [C]

    # reshape a [1,C,1,1,1] per broadcast sul latente [B,C,X,Y,Z]
    latent_mean=mean_c.view(1, -1, 1, 1, 1)
    scale_factor=scale_c.view(1, -1, 1, 1, 1)
    return latent_mean, scale_factor

def train_one_epoch(
    epoch,unet,train_loader,optimizer,lr_scheduler,
    loss_pt, scaler, latent_mean, scale_factor, noise_scheduler,
    device, local_rank, amp=True,
):
    """
    Singola epoca di training rectified flow.
    target=images-noise (velocity lungo il path lineare)
    Normalizzazione v4 PER-CANALE + CENTERING: (z - latent_mean) * scale_factor.
    """
    unet.train()
    loss_acc=torch.zeros(2, dtype=torch.float, device=device)

    #barra di avanzamento solo su rank 0. Mostra anche il regime timestep attivo
    #(curriculum v5): utile per verificare nel log che lo switch avvenga a
    #curriculum_switch_epoch.
    progress_bar=tqdm(
        train_loader,
        desc=f"Epoch {epoch+1} [{noise_scheduler.sample_method}]",
        ncols=100,
        disable=(local_rank!=0),
    )

    for train_data in progress_bar:
        #latente grezzo -> normalizzato per-canale e centrato (v4)
        images=train_data["latent"].to(device)
        images=(images-latent_mean)*scale_factor

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            noise=torch.randn_like(images)

            #RFlow: timesteps campionati dallo scheduler
            #(uniform o logit-normal a seconda di config_network.json)
            timesteps=noise_scheduler.sample_timesteps(images)

            #aggiunge rumore lungo il path lineare
            noisy_latent=noise_scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=timesteps,
            )

            #Unet incondizionata: solo x e timesteps
            model_output=unet(x=noisy_latent, timesteps=timesteps)

            #target rectified flow (velocity)
            model_gt=images-noise

            loss=loss_pt(model_output.float(), model_gt.float())
        
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        
        loss_acc[0]+=loss.item()
        loss_acc[1]+=1.0

        #aggiornamento della barra con la loss media corrente
        if local_rank==0:
            progress_bar.set_postfix({"loss": f"{(loss_acc[0]/loss_acc[1]).item():.5f}"})
    
    if dist.is_initialized():
        dist.all_reduce(loss_acc, op=torch.distributed.ReduceOp.SUM)
    
    return (loss_acc[0]/loss_acc[1]).item()

#Validazione
@torch.no_grad() #disattiva il calcolo dei gradienti
def validate(unet, val_loader, loss_pt, latent_mean, scale_factor, noise_scheduler, device, amp=True):
    """
    Loss di validazione (stesso obiettivo di rectified flow, senza backward)
    NB: per i modelli di diffusione la val_loss e' un indicatore debole della
    qualita' di generazione. La valutazione vera (FID, MMD, MS-SSIM) si fa sui
    campioni generati in eval.py. Qui serve solo a monitorare l'overfitting.
    Usa la stessa normalizzazione v4 per-canale + centering del training.
    """
    unet.eval()
    loss_acc=torch.zeros(2, dtype=torch.float, device=device)

    for val_data in val_loader:
        images=val_data["latent"].to(device)
        images=(images-latent_mean)*scale_factor

        with autocast("cuda", enabled=amp):
            noise=torch.randn_like(images)
            timesteps=noise_scheduler.sample_timesteps(images)
            noisy_latent=noise_scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=timesteps,
            )
            model_output=unet(x=noisy_latent, timesteps=timesteps)
            model_gt=images-noise
            loss=loss_pt(model_output.float(), model_gt.float())

        loss_acc[0]+=loss.item()
        loss_acc[1]+=1.0
    
    if dist.is_initialized():
        dist.all_reduce(loss_acc, op=torch.distributed.ReduceOp.SUM)
    
    return (loss_acc[0]/loss_acc[1]).item()

#Checkpoint
def save_checkpoint(epoch, unet, loss, latent_mean, scale_factor, num_train_timesteps, save_path):
    """
    Salva il checkpoint. latent_mean, scale_factor e num_train_timesteps inclusi:
    servono tutti al sampling per ricostruire lo scheduler e de-normalizzare
    (z/scale_factor + latent_mean) prima del decode VAE.
    """
    unet_state=unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    torch.save(
        {
            "epoch":epoch+1,
            "loss":loss,
            "num_train_timesteps":num_train_timesteps,
            "latent_mean":latent_mean,          # v4: media per-canale [1,C,1,1,1]
            "scale_factor":scale_factor,        # v4: scale per-canale [1,C,1,1,1]
            "unet_state_dict":unet_state,
        },
        save_path,
    )

def main():
    parser=argparse.ArgumentParser(description="Training LDM su latenti del VAE")
    parser.add_argument("--config", type=str, default="configs/config_diff_model.json")
    parser.add_argument("--network", type=str, default="configs/config_network.json")
    args=parser.parse_args()

    #caricamento delle config
    with open(args.config) as f:
        config=json.load(f) #diffusion_unet_train, paths ...
    with open(args.network) as f:
        config_net=json.load(f) #diffusion_unet_def, noise_scheduler ...
    
    #DDP
    local_rank, device=setup_ddp()
    is_main=local_rank==0
    set_determinism(seed=42)

    if is_main:
        print(f"Device: {device}, GPU: {torch.cuda.device_count()}")
        mlflow.set_experiment("LDM_training")
    
    #parametri di training (config_diff_model["diffusion_unet_train"])
    train_cfg=config["diffusion_unet_train"]
    n_epochs=train_cfg.get("n_epochs", 1000)
    lr=train_cfg.get("lr", 1e-5)
    batch_size=train_cfg.get("batch_size", 1)
    num_workers=train_cfg.get("num_workers", 4)
    val_interval=train_cfg.get("val_interval", 50)     # default 50 se assente
    save_interval=train_cfg.get("save_interval", 100)  # checkpoint periodici
    amp=train_cfg.get("amp", True)
    lr_warmup_epochs=train_cfg.get("lr_warmup_epochs", 50)  # v4: warmup optimizer LDM
    # v5: epoca di switch del curriculum sul timestep sampling.
    #   epoca <  switch -> uniform      (consolida la struttura globale)
    #   epoca >= switch -> logit-normal (affina la texture)
    # Se assente (0 o mancante) il curriculum e' DISATTIVATO e vale il sample_method
    # del config_network (comportamento v4).
    curriculum_switch_epoch=train_cfg.get("curriculum_switch_epoch", 0)

     #path (config_diff_model["paths"])
    paths=config["paths"]
    save_dir=paths.get("model_dir", "./outputs/models")
    splits_path=paths["splits_path"]   # embeddings_dataset.json
    os.makedirs(save_dir, exist_ok=True)
 
    #scheduler params (config_network["noise_scheduler"])
    sched_cfg=config_net["noise_scheduler"]
    num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000)

    #dataloaders
    loader_cfg={
        "batch_size":batch_size,
        "num_workers":num_workers,
        "divisible_k":8,
    }
    train_loader, val_loader, _=setup_ldm_dataloaders(loader_cfg, splits_path)

    #Unet (config_network["diffusion_unet_def"])
    unet=setup_unet(config_net["diffusion_unet_def"], device, local_rank)

    #v4: media + scale PER-CANALE su TUTTO il train set
    latent_mean, scale_factor=calculate_latent_stats(train_loader, device)
    if is_main:
        print(f"latent_mean per-canale: {latent_mean.flatten().tolist()}")
        print(f"scale_factor per-canale: {scale_factor.flatten().tolist()}")
    
    #scheduler RFlow
    noise_scheduler=setup_noise_scheduler(sched_cfg)

    # v5: metodo di fase 2 del curriculum. In fase 1 si forza "uniform"; in fase 2
    # si ripristina il metodo del config (tipicamente "logit-normal"). Cosi' il
    # config_network resta la fonte di verita' per loc/scale del logit-normal.
    phase2_method=sched_cfg.get("sample_method", "uniform")
    if is_main and curriculum_switch_epoch>0:
        print(f"[curriculum v5] fase1 (epoca<{curriculum_switch_epoch}): uniform | "
              f"fase2 (epoca>={curriculum_switch_epoch}): {phase2_method}")

    #optimizer (MAISI: Adam)
    optimizer=torch.optim.Adam(params=unet.parameters(), lr=lr)
    total_steps=n_epochs*len(train_loader)
    warmup_steps=lr_warmup_epochs*len(train_loader)

    #v4: WARMUP lineare + decay polinomiale (power 2.0).
    #Prima: solo PolynomialLR dal primo step (nessun warmup).
    #Ora: rampa lineare da ~0 a lr nei primi warmup_steps, poi decay poly fino a 0.
    def lr_lambda(step):
        if step<warmup_steps:
            return float(step)/float(max(1, warmup_steps))
        #decay polinomiale sui restanti step (coerente con MAISI, power 2.0)
        progress=float(step-warmup_steps)/float(max(1, total_steps-warmup_steps))
        progress=min(1.0, progress)
        return (1.0-progress)**2.0
    lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    loss_pt=torch.nn.L1Loss()
    scaler=GradScaler("cuda", enabled=amp)

    best_val_loss=float("inf")

    #loop di training
    for epoch in range(n_epochs):
        # v5 CURRICULUM: commuta il regime di campionamento dei timestep in base
        # all'epoca corrente. Lo scheduler legge sample_method (e loc/scale) a ogni
        # chiamata di sample_timesteps(), quindi basta settare l'attributo mutabile
        # prima di iniziare l'epoca (switch a granularita' di epoca, netto).
        if curriculum_switch_epoch>0:
            desired="uniform" if epoch<curriculum_switch_epoch else phase2_method
            if noise_scheduler.sample_method!=desired:
                noise_scheduler.sample_method=desired
                if is_main:
                    print(f"[curriculum v5] epoca {epoch+1}: timestep sampling -> {desired}")

        train_loss=train_one_epoch(
            epoch, unet, train_loader, optimizer, lr_scheduler,
            loss_pt, scaler, latent_mean, scale_factor, noise_scheduler,
            device, local_rank, amp=amp,
        )

        if is_main:
            current_lr=optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{n_epochs} | train_loss: {train_loss:.5f} | lr: {current_lr:.2e}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)
            #salva sempre l'ultimo
            save_checkpoint(
                epoch, unet, train_loss, latent_mean, scale_factor, num_train_timesteps,
                os.path.join(save_dir, "ldm_unet_last.pt"),
            )

            #checkpoint periodico (per scegliere poi il best con FID, non con val_loss)
            if (epoch+1)%save_interval==0:
                save_checkpoint(
                    epoch, unet, train_loss, latent_mean, scale_factor, num_train_timesteps,
                    os.path.join(save_dir, f"ldm_unet_epoch{epoch+1}.pt"),
                )
        
        # validazione periodica (monitoraggio overfitting)
        if (epoch+1) % val_interval==0:
            val_loss=validate(
                unet, val_loader, loss_pt, latent_mean, scale_factor, noise_scheduler, device, amp=amp,
            )
            if is_main:
                print(f"  -> val_loss: {val_loss:.5f}")
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                if val_loss<best_val_loss:
                    best_val_loss=val_loss
                    save_checkpoint(
                        epoch, unet, val_loss, latent_mean, scale_factor, num_train_timesteps,
                        os.path.join(save_dir, "ldm_unet_best.pt"),
                    )
                    print(f"  -> nuovo best (val_loss={val_loss:.5f}), salvato ldm_unet_best.pt")
 
    if dist.is_initialized():
        dist.destroy_process_group()
 
 
if __name__=="__main__":
    main()