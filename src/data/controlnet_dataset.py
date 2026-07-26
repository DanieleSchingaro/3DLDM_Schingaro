#src/data/controlnet_dataset.py
"""
Dataset e DataLoader per il training della ControlNet.

Estende la logica di ldm_dataset.py accoppiando, per ogni item:
    - "image": embedding latente .npz (chiave "z", [C,X,Y,Z], C=4) prodotto dal VAE v2
    - "label": maschera di segmentazione FSL-FAST 256^3 (_pveseg.nii.gz, valori 0/1/2/3)

Punti tecnici (allineati a train_controlnet.py di NV-Generate-CTMR e a train_ldm.py):

  1. LATENTE GREZZO. Come in LatentDataset, il dataset restituisce il latente NON
     normalizzato. La normalizzazione per-canale (images - latent_mean) * scale_factor
     avviene nel training loop, usando latent_mean/scale_factor CARICATI DAL CHECKPOINT
     del curriculum (non ricalcolati: la UNet congelata e' tarata su quei valori).

  2. MASCHERA INTERA, INTERPOLAZIONE NEAREST. La maschera e' a valori interi {0,1,2,3}.
     Va caricata SENZA interpolazione bilinear (che creerebbe classi frazionarie non
     valide). Qui la maschera e' gia' a 256^3 e allineata al latente, quindi non serve
     resampling: si carica, si porta a RAS + channel-first, dtype long. La
     binarizzazione bit-plane (binarize_labels) NON si fa qui: avviene nel training loop.

  3. PADDING. pad_to_divisible sul latente e' identico a ldm_dataset.py (no-op per 64^3,
     ma tenuto per coerenza). La maschera NON viene paddata dal dataset: e' gia' 256^3.

Il dizionario restituito usa le chiavi "image" e "label" (non "latent"), perche' e'
cio' che train_controlnet.py si aspetta (batch["image"], batch["label"]).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import nibabel as nib


def pad_to_divisible(latent: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Padda le dimensioni spaziali del latente al multiplo di k piu' vicino (verso l'alto).
    Identica a ldm_dataset.pad_to_divisible: con num_channels=[64,128,256,512] la UNet
    fa 3 downsampling -> serve divisibilita' per 8. Con latenti 64^3 e' un no-op, ma
    la si tiene per robustezza.
    """
    c, x, y, z=latent.shape

    def _next_mult(v: int)->int:
        return ((v+k-1)//k)*k

    target=(_next_mult(x), _next_mult(y), _next_mult(z))
    if (x, y, z)==target:
        return latent

    pad=(0,target[2]-z, 0, target[1]-y, 0, target[0]-x)
    return torch.nn.functional.pad(latent, pad, mode="constant", value=0.0)


def _load_mask(path: str)->torch.Tensor:
    """
    Carica la maschera _pveseg.nii.gz come tensore intero [1,X,Y,Z], senza
    interpolazione. La maschera FSL e' gia' in RAS 256^3 (verificato), quindi qui
    non si riorienta ne' si ricampiona: si legge e si aggiunge il canale.

    Nota: si usa nibabel invece dei transform MONAI per semplicita' e per garantire
    che i valori di classe {0,1,2,3} restino INTERI (nessun rischio di interpolazione).
    Se in futuro le maschere non fossero gia' in RAS, andrebbe reintrodotto Orientationd
    con mode="nearest".
    """
    arr=np.asarray(nib.load(path).dataobj)          # [X,Y,Z], int
    t=torch.from_numpy(arr.astype(np.int64))        # long
    return t.unsqueeze(0)                             # [1,X,Y,Z]


class ControlNetLatentMaskDataset(Dataset):
    """
    Restituisce, per ogni item:
        "image": latente grezzo [C,X',Y',Z'] (float)
        "label": maschera intera [1,X,Y,Z] (long, valori 0/1/2/3)
        "source": provenienza (se presente nel JSON), altrimenti "unknown"
    """

    def __init__(self, data_list: list[dict], repo_root: str=".", k: int=8):
        """
        Args:
            data_list: lista di dict {"image": path_npz, "label": path_mask, ...}
                       (i path sono relativi a repo_root, come nel controlnet_dataset.json)
            repo_root: radice a cui i path relativi sono agganciati
            k: divisore per il padding spaziale del latente
        """
        self.data_list=data_list
        self.repo_root=repo_root
        self.k=k

    def __len__(self)->int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        item=self.data_list[idx]

        npz_path=os.path.join(self.repo_root, item["image"])
        mask_path=os.path.join(self.repo_root, item["label"])

        # latente grezzo
        z=np.load(npz_path)["z"]
        latent=torch.from_numpy(z).float()
        latent=pad_to_divisible(latent, k=self.k)

        # maschera intera (nessuna interpolazione)
        label=_load_mask(mask_path)

        return{
            "image": latent,     # [C,X',Y',Z']  (grezzo: normalizzato nel training loop)
            "label": label,      # [1,X,Y,Z]     (intera: binarizzata nel training loop)
            "source": item.get("source", "unknown"),
        }


def load_controlnet_split(json_path: str, fold: int=0):
    """
    Legge il controlnet_dataset.json e lo divide in training/validation secondo il
    campo 'fold', REPLICANDO la logica di add_data_dir2path di MAISI:
        item con fold == <fold arg>  -> validation
        item con fold != <fold arg>  -> training
    Con il JSON generato (training->fold 1, validation->fold 0) e fold=0:
        fold 0 -> validation (100), fold 1 -> training (805).
    """
    with open(json_path) as f:
        items=json.load(f)["training"]

    train_files, val_files=[],[]
    for d in items:
        (val_files if d["fold"]==fold else train_files).append(d)
    return train_files, val_files


def create_controlnet_dataloader(
    data_list: list[dict],
    repo_root: str=".",
    batch_size: int=1,
    is_train: bool=True,
    num_workers: int=2,
    k: int=8,
    sampler=None,
)->DataLoader:
    dataset=ControlNetLatentMaskDataset(data_list, repo_root=repo_root, k=k)
    shuffle_flag=is_train and sampler is None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=is_train,
    )


def setup_controlnet_dataloaders(
    config: dict,
    json_path: str="data/splits/controlnet_dataset.json",
    repo_root: str=".",
    fold: int=0,
) -> tuple:
    """
    Crea i due DataLoader (train, val) per il training della ControlNet, con supporto
    DDP identico a setup_ldm_dataloaders.
    """
    train_files, val_files=load_controlnet_split(json_path, fold=fold)
    print(f"ControlNet: train={len(train_files)} (fold!={fold}), "
          f"val={len(val_files)} (fold=={fold})")

    batch_size=config.get("batch_size", 1)
    num_workers=config.get("num_workers", 2)
    k=config.get("divisible_k", 8)

    is_distributed=dist.is_available() and dist.is_initialized()
    rank=dist.get_rank() if is_distributed else 0
    world_size=dist.get_world_size() if is_distributed else 1

    train_sampler=DistributedSampler(
        train_files, num_replicas=world_size, rank=rank, shuffle=True, seed=42,
    ) if is_distributed else None
    val_sampler=DistributedSampler(
        val_files, num_replicas=world_size, rank=rank, shuffle=False,
    ) if is_distributed else None

    train_loader=create_controlnet_dataloader(
        train_files, repo_root=repo_root, batch_size=batch_size, is_train=True,
        num_workers=num_workers, k=k, sampler=train_sampler,
    )
    val_loader=create_controlnet_dataloader(
        val_files, repo_root=repo_root, batch_size=batch_size, is_train=False,
        num_workers=num_workers, k=k, sampler=val_sampler,
    )
    return train_loader, val_loader