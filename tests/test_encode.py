"""
Test rapido dell'encoding con il VAE trainato.
Verifica che l'encoding funzioni su 2 immagini prima di lanciarlo su tutto il dataset.
Esegui con: python -m tests.test_encode
"""

import torch
import numpy as np
from src.data.encode_dataset import load_autoencoder, get_encoding_transforms, encode_volume, setup_logging
from src.data.dataset import get_file_list, DATASET_DIRS

def test_encode():
    logger=setup_logging()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    #caricamento VAE
    checkpoint_path="outputs/models_v5/autoencoder_best.pt"
    autoencoder=load_autoencoder(checkpoint_path, device)
    transforms=get_encoding_transforms()

    #pescaggio di due file dal dataset
    files=get_file_list(DATASET_DIRS)[:2]

    for item in files:
        image_path=item["image"]
        logger.info(f"Encoding: {image_path}")

        z_np, affine=encode_volume(
            image_path=image_path,
            autoencoder=autoencoder,
            transforms=transforms,
            device=device,
            logger=logger,
        )

        assert z_np is not None, "Encoding fallito"
        assert len(z_np.shape)==4, f"Shape attesa [C,X,Y,Z], ottenuta {z_np.shape}"
        assert z_np.shape[0]==4, f"Canali latenti attesi 4, ottenuti {z_np.shape[0]}"
        assert not np.isnan(z_np).any(), "NaN nei latenti"

        logger.info(f"Shape: {z_np.shape}")
        logger.info(f"Min: {z_np.min():.4f}, Max: {z_np.max():.4f}")
        logger.info(f"Affine: {affine is not None}")
    
    logger.info("Test encoding superato")

if __name__=="__main__":
    test_encode()