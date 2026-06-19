#tests/test_sample.py
"""
Smoke test di src/inference/sample.py
Scopo: verificare che la PIPELINE MECCANICA di generazione giri senza errori,
NON la qualita' delle immagini. Usa modelli con PESI CASUALI (non addestrati),
quindi le immagini prodotte saranno spazzatura: qui controlliamo solo che
shape, API e flusso siano corretti.
 
Cosa verifica:
  1. ReconModel decodifica un latente [1,4,64,64,64] -> volume [1,1,256,256,256]
     (cioe' il fattore di upsampling 4x del VAE e' corretto)
  2. generate_one completa il loop di denoising RFlow senza crash e restituisce
     un numpy della shape attesa
  3. save_nifti scrive un .nii.gz valido e rileggibile

NOTE OPERATIVE:
  - Va eseguito su GPU (i modelli 3D Maisi sono pesanti). Lanciare quando una
    GPU e' libera (es. prima del training LDM, dopo che il VAE ha finito).
  - Usa pochi step di denoising (3 invece di 30) e nessun DDP, per essere rapido.
  - NON carica checkpoint: i pesi sono casuali, e' voluto.
 
Esegui con: python3 -m tests.test_sample
"""
import os
import tempfile
import shutil
import numpy as np 
import torch 
import nibabel as nib 
from monai.networks.schedulers import RFlowScheduler
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from src.inference.sample import ReconModel, generate_one, save_nifti

def build_small_models(device):
    """
    Istanzia VAE e Unet con pesi casuali ma con architettura coerente.
    num_channels piccoli per rendere il test leggero e veloce.
    """
    autoencoder=AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=(128, 256, 512),
        num_res_blocks=(2, 2, 2),
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        num_splits=4,
        dim_split=1,
    ).to(device).eval()
 
    unet=DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True],
        num_head_channels=[0, 0, 32, 32],
        num_res_blocks=2,
        use_flash_attention=False,
        resblock_updown=True,
        include_fc=True,
        with_conditioning=False,
        num_class_embeds=None,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        include_spacing_input=False,
    ).to(device).eval()
 
    return autoencoder, unet

def test_recon_model_decode(device):
    """
    Verifica 1: ReconModel decodifica [1,4,64,64,64]->[1,1,256,256,256]
    Il VAE comprime 4x.
    """
    autoencoder, _=build_small_models(device)
    scale_factor=torch.tensor(1.0, device=device)
    recon_model=ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    z=torch.randn(1,4,64,64,64, device=device)
    with torch.no_grad():
        out=recon_model(z)
    
    assert out.shape==(1,1,256,256,256), f"Shape decode errata: {out.shape}"
    print(f"OK ReconModel: latente {tuple(z.shape)} -> volume {tuple(out.shape)}")

def test_generate_one(device):
    """
    Verifica 2: generate_one completa il denoising RFlow e
    restituisce un numpy [256,256,256] con valori in [0,1]
    """
    autoencoder,unet=build_small_models(device)
    scale_factor=torch.tensor(1.0, device=device)
    recon_model=ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    noise_scheduler=RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,
        use_timestep_transform=True,
        scale=1.4,
        sample_method="uniform",
    )

    data=generate_one(
        unet=unet,
        recon_model=recon_model,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        latent_shape=(4, 64, 64, 64),
        num_inference_steps=3,
        device=device,
        inferer=None,
    )

    assert isinstance(data, np.ndarray), f"Output non numpy: {type(data)}"
    assert data.shape==(256,256,256), f"Shape volume errata: {data.shape}"
    assert data.min()>=0.0 and data.max()<=1.0, f"Valori fuori [0,1]: [{data.min()}, {data.max()}]"
    print(f"Ok generate_one: volume {data.shape}, range [{data.min():.3f}, {data.max():.3f}]")
    return data

def test_save_nifti(data):
    """
    Verifica 3: save_nifti scrive un .nii.gz rileggibile con shape corretta.
    """
    tmp_dir=tempfile.mkdtemp()
    out_path=os.path.join(tmp_dir, "test_synth.nii.gz")
    try:
        save_nifti(data, spacing=(1.0,1.0,1.0), output_path=out_path)
        assert os.path.exists(out_path), "File .nii.gz non creato"

        #rilettura per conferma
        img=nib.load(out_path)
        assert img.shape==(256,256,256), f"Shape nifti riletta errata: {img.shape}"
        #affine diagonale con spacing 1mm
        affine=img.affine
        assert np.allclose(np.diag(affine)[:3], [1.0,1.0,1.0]), f"Affine spacing errato: {np.diag(affine)}"
        print(f"OK save_nifti: file scritto e riletto, shape {img.shape}, spacing 1mm")
    finally:
        shutil.rmtree(tmp_dir)

def main():
    print("="*55)
    print("SMOKE TEST sample.py (pesi causali)")
    print("="*55)

    if not torch.cuda.is_available():
        print("Attenzione: nessuna GPU disponibile. I modelli 3D Maisi su CPU")
        print("sono molto lenti e pesanti in RAM. Esecuzione sconsigliata su CPU.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    test_recon_model_decode(device)
    data=test_generate_one(device)
    test_save_nifti(data)

    print("="*55)
    print("Tutti gli smoke test superati: la pipeline di sample.py e' meccanicamente corretta.")
    print("NB: la QUALITA' delle immagini dipende dall'addestramento dell'LDM,")
    print("e va valutata separatamente (eval.py con FID/MMD/MS-SSIM).")
    print("="*55)
 
 
if __name__=="__main__":
    main()