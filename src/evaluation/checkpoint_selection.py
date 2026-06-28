"""
SELEZIONE DEL CHECKPOINT LDM MIGLIORE VIA FID  (versione MULTI-GPU)

Per i modelli di diffusione la val_loss NON predice la qualita' di generazione:
il checkpoint con val_loss minima non e' necessariamente quello che genera le
immagini migliori. Questo script confronta piu' checkpoint LDM in modo oggettivo:

  per ogni checkpoint ldm_unet_epoch{N}.pt:
    1. genera N_SAMPLES immagini sintetiche (default 100) in una cartella dedicata
    2. calcola il FID 2.5D di quelle immagini vs il test set reale
  alla fine: classifica i checkpoint per FID, salva un UNICO JSON con la curva
  FID-vs-epoca e genera un grafico (utile come figura per la tesi).

MULTI-GPU: i checkpoint vengono distribuiti su tutte le GPU disponibili (una
sola istanza dello script, un solo comando). Ogni GPU lavora sui propri
checkpoint in modo indipendente. Tutti i processi scrivono lo STESSO file
fid_by_checkpoint_v2.json: per evitare corruzione/race condition la scrittura e'
serializzata con un lock condiviso (read-modify-write atomico).

Lancio (usa TUTTE le GPU automaticamente, un comando solo):
    python3 -m src.evaluation.checkpoint_selection --n_samples 100

Per limitare il numero di GPU (es. se la RAM e' poca):
    python3 -m src.evaluation.checkpoint_selection --n_samples 100 --n_gpus 2
"""
import os
import json
import glob
import argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
import numpy as np
import nibabel as nib
import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
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


def _read_results(results_path):
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _already_done(results_path, key, lock):
    """Controlla (sotto lock) se un checkpoint e' gia' nel JSON unico."""
    with lock:
        data=_read_results(results_path)
        if key in data:
            return data[key]
    return None


def _save_result(results_path, key, result, lock):
    """
    Scrittura atomica sul JSON unico: sotto lock rilegge il file corrente,
    aggiunge la chiave e riscrive. Cosi' processi su GPU diverse non si
    sovrascrivono a vicenda.
    """
    with lock:
        data=_read_results(results_path)
        data[key]=result
        tmp=results_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, results_path)  # rename atomico


def evaluate_one_checkpoint(ep, args, config, config_net, autoencoder, scheduler,
                            inferer, real_stream, latent_shape, steps, spacing,
                            base_seed, device, results_path, lock):
    """
    Genera i campioni di UN checkpoint, ne calcola il FID e lo scrive nel JSON
    unico (in modo atomico tramite lock). Ritorna il dict del risultato.
    """
    key=f"epoch{ep}"

    # ripresa incrementale: gia' nel JSON unico?
    done=_already_done(results_path, key, lock)
    if done is not None:
        print(f"[{device}] [{key}] gia' valutato (FID={done['fid_avg']:.3f}), salto.")
        return done

    ckpt_path=os.path.join(args.models_dir, f"ldm_unet_epoch{ep}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[{device}] [{key}] checkpoint non trovato ({ckpt_path}), salto.")
        return None

    print(f"\n{'='*55}\n[{device}] [{key}] genero {args.n_samples} campioni\n{'='*55}")
    unet, scale_factor=load_unet(config_net, ckpt_path, device)
    recon=ReconModel(autoencoder, scale_factor).to(device)

    synth_dir=os.path.join(args.work_dir, f"synth_{key}")
    os.makedirs(synth_dir, exist_ok=True)

    for idx in tqdm(range(args.n_samples), desc=f"{device} {key}"):
        set_determinism(seed=base_seed + idx)
        data=generate_one(unet, recon, scheduler, latent_shape, steps, device, inferer)
        affine=np.eye(4)
        for i in range(3):
            affine[i, i]=spacing[i]
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine),
                 os.path.join(synth_dir, f"hc_synth_{idx+1:04d}.nii.gz"))

    del unet, recon
    torch.cuda.empty_cache()

    print(f"[{device}] [{key}] calcolo FID...")
    synth_files=sorted(glob.glob(os.path.join(synth_dir, "hc_synth_*.nii.gz")))
    synth_stream=VolumeStream(synth_files, _load_synth)
    fid_res=compute_fid_2p5d(real_stream, synth_stream, device=device,
                             drop_empty=True, batch_size=32, verbose=True)
    result={"epoch": ep, **fid_res}

    _save_result(results_path, key, result, lock)
    print(f"[{device}] [{key}] FID medio = {fid_res['fid_avg']:.3f}  (salvato in {results_path})")
    return result


def worker(rank, n_gpus, args, all_epochs, lock):
    """
    Processo per UNA GPU. Valuta il sottoinsieme di epoche assegnato a questo rank
    (round-robin). Il VAE, lo scheduler e lo stream delle reali si caricano una
    volta e si riusano per tutti i checkpoint di competenza.
    """
    torch.cuda.set_device(rank)
    device=f"cuda:{rank}"

    my_epochs=all_epochs[rank::n_gpus]
    if not my_epochs:
        print(f"[{device}] nessun checkpoint assegnato, esco.")
        return
    print(f"[{device}] checkpoint assegnati: {my_epochs}")

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
    ae_ckpt=paths.get("trained_autoencoder_path", "./outputs/models_v3/autoencoder_best.pt")

    print(f"[{device}] carico VAE da {ae_ckpt}")
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

    with open(args.splits) as f:
        test_items=json.load(f)["test"]
    real_stream=VolumeStream(test_items, _load_real)
    print(f"[{device}] test set reale: {len(real_stream)} volumi")

    results_path=os.path.join(args.work_dir, "fid_by_checkpoint_v3.json")

    for ep in my_epochs:
        evaluate_one_checkpoint(
            ep, args, config, config_net, autoencoder, scheduler, inferer,
            real_stream, latent_shape, steps, spacing, base_seed, device,
            results_path, lock,
        )


def rank_and_plot(args, all_epochs):
    """
    Legge il JSON unico finale, stampa la classifica e genera il grafico
    FID-vs-epoca. Eseguito dal processo principale dopo che i worker hanno finito.
    """
    results_path=os.path.join(args.work_dir, "fid_by_checkpoint_v3.json")
    results=_read_results(results_path)
    if not results:
        print("Nessun risultato trovato.")
        return

    print(f"\n{'='*55}\nCLASSIFICA CHECKPOINT PER FID (piu' basso = meglio)\n{'='*55}")
    ranked=sorted(results.items(), key=lambda kv: kv[1]["fid_avg"])
    for rank, (key, r) in enumerate(ranked, 1):
        print(f"{rank}. {key:>10}  FID medio = {r['fid_avg']:.3f}  "
              f"(XY {r['fid_xy']:.1f} / YZ {r['fid_yz']:.1f} / ZX {r['fid_zx']:.1f})")
    best_key, best_r=ranked[0]
    print(f"\nMIGLIORE: {best_key} con FID {best_r['fid_avg']:.3f}")
    print("NB: con 100 campioni il FID ha ancora varianza; differenze piccole")
    print("(<5) tra checkpoint vicini potrebbero non essere significative.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps_sorted=sorted(results.values(), key=lambda r: r["epoch"])
        xs=[r["epoch"] for r in eps_sorted]
        ys=[r["fid_avg"] for r in eps_sorted]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, marker="o", color="C0")
        plt.xlabel("Epoca checkpoint LDM")
        plt.ylabel("FID medio (3 piani)")
        plt.title("Selezione checkpoint LDM via FID")
        plt.grid(True, alpha=0.3)
        plt.scatter([best_r["epoch"]], [best_r["fid_avg"]], color="C3", zorder=5,
                    label=f"best: ep{best_r['epoch']} (FID {best_r['fid_avg']:.2f})")
        plt.legend()
        plot_path=os.path.join(args.work_dir, "fid_vs_epoch_v3.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Grafico salvato in {plot_path}")
    except Exception as e:
        print(f"[avviso] impossibile generare il grafico: {e}")


def main():
    ap=argparse.ArgumentParser(description="Selezione checkpoint LDM via FID (multi-GPU)")
    ap.add_argument("--config", type=str, default="configs/config_diff_model.json")
    ap.add_argument("--network", type=str, default="configs/config_network.json")
    ap.add_argument("--splits", type=str, default="data/splits/dataset.json")
    ap.add_argument("--models_dir", type=str, default="outputs/models_v3")
    ap.add_argument("--work_dir", type=str, default="outputs/checkpoint_selection_v3",
                    help="dove salvare le immagini temporanee e il JSON dei risultati")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--epochs", type=str, default="100,200,300,400,500,600,700,800,900,1000",
                    help="lista epoche dei checkpoint da testare, separate da virgola")
    ap.add_argument("--n_gpus", type=int, default=0,
                    help="numero di GPU da usare (0 = tutte le disponibili)")
    args=ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    all_epochs=[int(e) for e in args.epochs.split(",")]

    n_avail=torch.cuda.device_count()
    n_gpus=args.n_gpus if args.n_gpus > 0 else n_avail
    n_gpus=max(1, min(n_gpus, n_avail))
    print(f"GPU disponibili: {n_avail}, uso: {n_gpus}")
    print(f"Checkpoint da valutare: {all_epochs}")

    # lock condiviso per la scrittura atomica del JSON unico
    manager=Manager()
    lock=manager.Lock()

    if n_gpus==1:
        worker(0, 1, args, all_epochs, lock)
    else:
        mp.spawn(worker, args=(n_gpus, args, all_epochs, lock), nprocs=n_gpus, join=True)

    rank_and_plot(args, all_epochs)


if __name__=="__main__":
    main()