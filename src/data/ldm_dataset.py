#src/data/ldm_dataset.py
"""
Dataset e DataLoader per gli embedding latenti (.npz) prodotti dal VAE.
Usato per il training dell'LDM.
Ogni .npz contiene:
    -"z": il latente [C,X,Y,Z] (C=4)
    -"affine": l'affine del volume originale per la decodifica
DataLoader restituisce il latente grezzo (non normalizzato).
La normalizzazione con scale_factor avverrà in train_ldm.py
"""

import os
import json
from typing import Optional
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader 
import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler

def pad_to_divisible(latent:torch.Tensor, k:int=8)->torch.Tensor:
    """
    Padda le dimensioni spaziali del latente al multiplo di k più vicino (verso l'alto).
    Padding è Condizionale: se la shape è già divisibile, per k non viene fatto nulla.
    L'UNet di diffusione richiede dimensioni divisibili per 2^(num_downsample)
    Con num_channels=[64,128,256,512] ->3 downsampling ->divisibile per 8.

    Args:
        latent: tensore[C,X,Y,Z]
        k: divisiore richiesto (default 8)
    Returns:
        tensore[C,X',Y',Z'] con X',Y',Z' divisibili per k
    """
    c,x,y,z=latent.shape

    def _next_mult(v:int)->int:
        return ((v+k-1)//k)*k
    
    target=(_next_mult(x), _next_mult(y), _next_mult(z))

    #se è già divisibile, nessun padding
    if (x,y,z)==target:
        return latent
    
    pad_x=target[0]-x 
    pad_y=target[1]-y 
    pad_z=target[2]-z 

    pad=(0,pad_z,0,pad_y,0,pad_x)
    latent=torch.nn.functional.pad(latent,pad,mode="constant",value=0.0)

    return latent

class LatentDataset(Dataset):
    """
    Carica gli embedding latenti .npz per il training dell'LDM
    Restituisce un dict con:
        -"latent": tensore [C,X',Y',Z']
        -"source": stringa con il dataset di provenienza
    """
    def __init__(self, data_list:list[dict], k:int=8):
        """
        Args:
            data_list: lista di dict {"image": path_npz, "source": nome_dataset}
            k: divisore per il padding spaziale 
        """
        self.data_list=data_list
        self.k=k
    
    def __len__(self)->int:
        return len(self.data_list)
    
    def __getitem__(self,idx:int)->dict:
        item=self.data_list[idx]
        npz_path=item["image"]

        #carica il latente
        data=np.load(npz_path)
        z=data["z"]

        latent=torch.from_numpy(z).float()

        #padding condizionale a multiplo di k
        latent=pad_to_divisible(latent, k=self.k)

        return{
            "latent":latent,
            "source":item.get("source", "unknown"),
        }

def load_embeddings_splits(splits_path:str)->tuple[list,list,list]:
    """
    Carica gli split degli embedding dal JSON creato da embeddings_dataset.py
    """
    with open(splits_path, "r") as f:
        splits=json.load(f)
    return splits["training"], splits["validation"], splits["test"]

def create_ldm_dataloader(
    data_list:list[dict],
    batch_size:int=1,
    is_train:bool=True,
    num_workers:int=2,
    k:int=8,
    sampler=None,
)->DataLoader:
    """
    Crea un DataLoader per i latenti
    """
    dataset=LatentDataset(data_list, k=k)

    #shuffle solo in training e solo se non usiamo un sampler DDP
    shuffle_flag=is_train and sampler is None

    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=is_train,
    )
    return dataloader

def setup_ldm_dataloaders(
    config:dict,
    splits_path:str="data/splits/embeddings_dataset_v4.json",
)->tuple:
    """
    Crea i tre DataLoader (train,val,test) per il training dell'LDM.
    """
    train_files, val_files, test_files=load_embeddings_splits(splits_path)
    print(f"Embedding: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    batch_size=config.get("batch_size",1)
    num_workers=config.get("num_workers",2)
    k=config.get("divisible_k",8)

    is_distributed=dist.is_available() and dist.is_initialized()

    rank=dist.get_rank() if is_distributed else 0
    world_size=dist.get_world_size() if is_distributed else 1

    train_sampler=DistributedSampler(
        train_files, num_replicas=world_size, rank=rank, shuffle=True, seed=42,
    ) if is_distributed else None

    val_sampler=DistributedSampler(
        val_files, num_replicas=world_size, rank=rank, shuffle=False,
    ) if is_distributed else None

    test_sampler=DistributedSampler(
        test_files, num_replicas=world_size, rank=rank, shuffle=False,
    ) if is_distributed else None

    train_loader=create_ldm_dataloader(
        train_files, batch_size=batch_size, is_train=True,
        num_workers=num_workers, k=k, sampler=train_sampler,
    )
    val_loader=create_ldm_dataloader(
        val_files, batch_size=batch_size, is_train=False,
        num_workers=num_workers, k=k, sampler=val_sampler,
    )
    test_loader=create_ldm_dataloader(
        test_files, batch_size=batch_size, is_train=False,
        num_workers=num_workers, k=k, sampler=test_sampler,
    )

    return train_loader, val_loader, test_loader