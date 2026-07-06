#src/data/encode_dataset.py

"""
Encoding del dataset HC sulla base del VAE trainato.
Converte le MRI raw in embedding latenti per il training dell'LDM.
Basato su diff_model_create_training_data.py di NV-Generate-CTMR.

VERSIONE MULTI-GPU (DDP opzionale):
    - se lanciato con torchrun, i volumi vengono suddivisi tra i rank
      (ogni GPU encoda la propria fetta, in parallelo)
    - se lanciato con python semplice, gira su singola GPU come prima
    L'encoding e' "embarrassingly parallel": nessuna comunicazione tra i rank
    durante il lavoro, ogni rank scrive i propri .npz in modo indipendente.
    L'unica collettiva e' una barrier finale + all_reduce dei contatori per il
    report aggregato.

Lancio multi-GPU (4 GPU):
    torchrun --nproc_per_node=4 -m src.data.encode_dataset
Lancio singola GPU (come prima):
    python3 -m src.data.encode_dataset
"""

import os
import json
import logging
import argparse
from typing import Optional, Tuple
import numpy as np 
import torch
import torch.distributed as dist
from torch.amp import autocast
from monai.transforms import Compose
from src.data.transforms import get_encoding_transforms
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

#DDP setup (opzionale)
def setup_ddp_optional():
    """
    Inizializza DDP se lanciato con torchrun (LOCAL_RANK presente),
    altrimenti gira su singola GPU. Stesso pattern di sample.py.
    Ritorna (local_rank, world_size, device).
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

def setup_logging(is_main:bool=True)->logging.Logger:
    """
    Configura il logger per l'encoding.
    In multi-GPU solo il rank 0 (is_main=True) logga a livello INFO; gli altri
    rank vengono alzati a ERROR per non sovrapporre 4 stream di log.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger=logging.getLogger("encode_dataset")
    if not is_main:
        logger.setLevel(logging.ERROR)
    return logger

def load_autoencoder(checkpoint_path:str, device:torch.device)->AutoencoderKlMaisi:
    """
    Carica il VAE dal checkpoint salvato durante il training.
    Il modello vien messo poi in modalità eval e frozen -
    durante l'encoding non aggiorniamo i pesi.
    Ogni rank carica indipendentemente lo stesso checkpoint sulla propria GPU.
    Il VAE e' deterministico in eval (usiamo z_mu, nessun sampling), quindi i
    latenti prodotti da rank diversi sono identici a quelli di un singolo rank.
    """
    autoencoder=AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=(128,256,512),
        num_res_blocks=(2,2,2),
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False,False,False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        num_splits=4,
        dim_split=1,
    )

    #caricamento checkpoint
    checkpoint=torch.load(checkpoint_path, map_location=device)

    #gestion del checkpoint con DataParallel
    if "autoencoder_state_dict" in checkpoint:
        state_dict=checkpoint["autoencoder_state_dict"]
    else:
        state_dict=checkpoint
    
    # rimuove il prefisso "module." aggiunto da DDP durante il training
    state_dict={k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
    autoencoder.load_state_dict(state_dict)
    autoencoder=autoencoder.to(device)

    #modalità eval e frozen - nessun gradiente durante l'encoding
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad=False
    
    return autoencoder

def encode_volume(
    image_path:str,
    autoencoder:AutoencoderKlMaisi,
    transforms:Compose,
    device:torch.device,
    logger:logging.Logger,
)->Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Encoding su un singolo volume MRI nello spazio latente.
    Il latente viene salvato in formato [C,X,Y,Z] -> standard pytorch.
    """
    try:
        #carica e processa il volume
        data=transforms({"image": image_path})
        image=data["image"]
        #affine per salvare il NIfTI
        affine=image.meta.get("original_affine", None)
        if affine is None:
            affine=image.meta.get("affine", None)
        if affine is not None:
            affine=np.array(affine)
        #aggiunta dimensione del batch
        pt_image=image.unsqueeze(0).to(device)

        with torch.inference_mode():
            #encode nello spazio latente
            z_mu, _=autoencoder.encode(pt_image)
            z=z_mu

        logger.info(
            f"Latent shape: {z.shape}, "
            f"min: {z.min().item():.3f}, "
            f"max: {z.max().item():.3f}"
        )

        #conversione in numpy: [1,C,X,Y,Z] -> [C,X,Y,Z]
        z_np=z.squeeze(0).cpu().float().numpy()

        return z_np, affine

    except Exception as e:
        logger.error(f"Errore encoding {image_path}: {e}")
        return None, None

def encode_dataset(
    splits_path:str,
    autoencoder:AutoencoderKlMaisi,
    transforms:Compose,
    device:torch.device,
    embedding_base_dir:str,
    logger:logging.Logger,
    local_rank:int=0,
    world_size:int=1,
)->None:
    """
    Encoda l'intero dataset (train, val e test) e salva gli embedding.
    In multi-GPU ogni rank processa la propria fetta dei file
    (all_files[local_rank::world_size]): nessuna sovrapposizione, nessun
    conflitto di scrittura (ogni rank scrive .npz diversi).
    Struttura output:
    data/processed/embeddings_v4/
        hc_adni_brain_mask/
            file1_emb.npz
            ...
        hc_nifd_brain_mask/
            ...
    
    Skippa i file già processati per permettere di riprendere 
    l'encoding se interrotto.
    """
    #carica gli split
    with open(splits_path, "r") as f:
        splits=json.load(f)
    
    #processa tutti gli split
    all_files=(
        splits["training"]+
        splits["validation"]+
        splits["test"]
    )

    #SUDDIVISIONE MULTI-GPU: ogni rank prende una fetta disgiunta dei file.
    #Con 4 rank -> ~250 volumi ciascuno invece di 1007 in fila.
    my_files=all_files[local_rank::world_size]

    logger.info(f"File totali: {len(all_files)} | questo rank ({local_rank}): {len(my_files)}")

    #contatori per report finale (per-rank; aggregati alla fine con all_reduce)
    processed=0
    skipped=0
    errors=0

    for i, item in enumerate(my_files):
        image_path=item["image"]

        #costruisce il path dell'embedding
        #es: data/raw/hc_adni.../file.nii.gz
        # -> data/processed/embeddings_v4/hc_adni.../file_emb.npz
        rel_path=image_path.replace("data/raw/", "")
        emb_filename=rel_path.replace(".nii.gz", "_emb.npz")
        emb_path=os.path.join(embedding_base_dir, emb_filename)

        #skippa se già processato
        if os.path.exists(emb_path):
            logger.info(f"[rank{local_rank}][{i+1}/{len(my_files)}] Già presente, salto: {emb_path}")
            skipped+=1
            continue
        
        logger.info(f"[rank{local_rank}][{i+1}/{len(my_files)}] Encoding: {image_path}")

        #encoda il volume
        z_np, affine=encode_volume(
            image_path=image_path,
            autoencoder=autoencoder,
            transforms=transforms,
            device=device,
            logger=logger,
        )

        if z_np is None:
            errors+=1
            continue
        
        #crea la cartella di output
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)

        #salva come .npz
        np.savez(emb_path, z=z_np, affine=affine)

        logger.info(f"Salvato: {emb_path} | shape: {z_np.shape}")
        processed+=1
    
    #tutti i rank finiscono di scrivere prima del report / prima che
    #embeddings_dataset.py vada a cercare i file
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    #aggregazione dei contatori tra i rank (somma globale)
    if world_size>1:
        counts=torch.tensor([processed, skipped, errors], device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        processed, skipped, errors=[int(x) for x in counts.tolist()]

    #report finale (solo rank 0 logga a livello INFO)
    logger.info(f"\n{'='*50}")
    logger.info(f"Encoding completato!")
    logger.info(f"Processati: {processed}")
    logger.info(f"Saltati (già esistenti): {skipped}")
    logger.info(f"Errori: {errors}")
    logger.info(f"{'='*50}")

def main():
    """
    Script principale per l'encoding del dataset.
    Carica il vae trainato e encoda le MRI in embedding latenti.
    Multi-GPU via torchrun (DDP opzionale).
    """
    parser=argparse.ArgumentParser(description="Encoding dataset con VAE trainato (multi-GPU)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/models_v2/autoencoder_best.pt",
        help="Path al checkpoint del VAE",
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        default="data/splits/dataset.json",
        help="Path al file degli split",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/processed/embeddings_v4",
        help="Cartella dove salvare gli embedding",
    )
    args=parser.parse_args()

    #DDP opzionale (torchrun -> multi-GPU, altrimenti singola GPU)
    local_rank, world_size, device=setup_ddp_optional()
    is_main=local_rank==0

    logger=setup_logging(is_main=is_main)
    logger.info(f"Device: {device} | world_size: {world_size}")

    #carica il VAE (ogni rank sulla propria GPU)
    logger.info(f"Carica VAE da {args.checkpoint}")
    autoencoder=load_autoencoder(args.checkpoint, device)
    logger.info("VAE caricato e impostato su frozen")

    #transforms
    transforms=get_encoding_transforms()

    #encoding (split dei file gestito dentro encode_dataset)
    encode_dataset(
        splits_path=args.splits_path,
        autoencoder=autoencoder,
        transforms=transforms,
        device=device,
        embedding_base_dir=args.embedding_dir,
        logger=logger,
        local_rank=local_rank,
        world_size=world_size,
    )

    #chiusura pulita del process group
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__=="__main__":
    main()