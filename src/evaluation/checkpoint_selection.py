"""
SELEZIONE DEL CHECKPOINT LDM MIGLIORE VIA FID  (versione MULTI-GPU, v4)

Per i modelli di diffusione la val_loss NON predice la qualita' di generazione.
Questo script confronta piu' checkpoint LDM in modo oggettivo, in DUE fasi:

  FASE 1 - selezione grezza (SENZA autoguidance):
    per ogni checkpoint ldm_unet_epoch{N}.pt genera N_SAMPLES immagini e calcola
    il FID 2.5D vs il test set reale. Scrive fid_by_checkpoint_v4.json e produce
    la curva FID-vs-epoca (via plot_fid_curve.py). Serve a trovare l'ORDINAMENTO
    intrinseco dei checkpoint (quale epoca genera meglio).

  FASE 2 - raffinamento dei top-K (CON autoguidance):
    prende i K checkpoint migliori dalla fase 1 e li rivaluta applicando
    l'autoguidance (v = v_bad + w*(v_good - v_bad)). Il 'bad' e' derivato per
    ciascun candidato come il checkpoint disponibile piu' vicino al ~30% della
    sua epoca; se non esiste un bad valido (candidato troppo precoce), quel
    candidato viene valutato senza guida e marcato autoguidance=false.
    Scrive fid_by_checkpoint_v4_refined.json e un grafico a BARRE che confronta,
    per ogni top-K, il FID senza guida (fase 1) vs con guida (fase 2).

Motivazione metodologica: la selezione grezza serve solo a ordinare i checkpoint,
per cui l'autoguidance (costosa e circolare) non e' necessaria. Il FID finale
"vero" della tesi lo calcola eval.py sulle immagini di sample.py, che usano
l'autoguidance. Il raffinamento dei top-K e' una verifica extra che l'ordine non
cambi sotto autoguidance.

MULTI-GPU: i checkpoint sono distribuiti sulle GPU con mp.spawn. Scrittura del
JSON serializzata con lock (read-modify-write atomico). La fase 2 parte solo dopo
che la fase 1 e' completa (due spawn sequenziali).

Lancio:
    python3 -m src.evaluation.checkpoint_selection --n_samples 100 \
        --refine_top 3 --guidance_scale 2.0
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
from src.evaluation.plot_fid_curve import plot_fid_curve


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


def build_unet(config_net, device):
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
    return net


def load_unet(config_net, ckpt_path, device):
    """Carica UNet + scale_factor + latent_mean (v4)."""
    net=build_unet(config_net, device)
    ckpt=torch.load(ckpt_path, map_location=device, weights_only=False)
    state={k.replace("module.", "", 1): v for k, v in ckpt["unet_state_dict"].items()}
    net.load_state_dict(state, strict=True)
    sf=ckpt["scale_factor"]
    if isinstance(sf, torch.Tensor):
        sf=sf.to(device)
    #v4: latent_mean per-canale (retrocompat: 0 se assente)
    lm=ckpt.get("latent_mean", 0.0)
    if isinstance(lm, torch.Tensor):
        lm=lm.to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad=False
    return net, sf, lm


def load_unet_weights_only(config_net, ckpt_path, device):
    """Carica SOLO i pesi (per il modello 'bad' dell'autoguidance)."""
    net=build_unet(config_net, device)
    ckpt=torch.load(ckpt_path, map_location=device, weights_only=False)
    state={k.replace("module.", "", 1): v for k, v in ckpt["unet_state_dict"].items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad=False
    return net


class ReconModel(torch.nn.Module):
    """De-normalizza (v4): z/scale_factor + latent_mean, poi decode."""
    def __init__(self, autoencoder, scale_factor, latent_mean):
        super().__init__()
        self.autoencoder=autoencoder
        self.scale_factor=scale_factor
        self.latent_mean=latent_mean

    def forward(self, z):
        z=z/self.scale_factor + self.latent_mean
        return self.autoencoder.decode_stage_2_outputs(z)


@torch.inference_mode()
def generate_one(unet, unet_bad, guidance_scale, recon_model, scheduler,
                 latent_shape, steps, device, inferer):
    """
    Genera un volume. Se unet_bad is not None applica l'autoguidance:
        v = v_bad + w*(v_good - v_bad)
    altrimenti usa la sola v_good (selezione grezza).
    """
    noise=torch.randn((1, *latent_shape), device=device)
    image=noise
    scheduler.set_timesteps(num_inference_steps=steps,
                            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])))
    all_t=scheduler.timesteps
    all_next=torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))
    with autocast("cuda", enabled=True):
        for t, nt in zip(all_t, all_next):
            t_in=torch.Tensor((t,)).to(device)
            v_good=unet(x=image, timesteps=t_in)
            if unet_bad is not None:
                v_bad=unet_bad(x=image, timesteps=t_in)
                out=v_bad + guidance_scale*(v_good - v_bad)
            else:
                out=v_good
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
    with lock:
        data=_read_results(results_path)
        if key in data:
            return data[key]
    return None


def _save_result(results_path, key, result, lock):
    with lock:
        data=_read_results(results_path)
        data[key]=result
        tmp=results_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, results_path)


def derive_bad_epoch(good_ep, available_epochs, frac=0.30):
    """
    Deriva l'epoca 'bad' per l'autoguidance: il checkpoint disponibile piu'
    vicino a frac*good_ep, ma STRETTAMENTE precoce rispetto al good (bad < good).
    Ritorna l'epoca del bad, oppure None se non esiste un candidato valido
    (es. good troppo precoce -> nessun checkpoint sotto di esso).
    """
    target=frac*good_ep
    candidates=[e for e in available_epochs if e<good_ep]
    if not candidates:
        return None
    #il piu' vicino al target tra quelli piu' precoci del good
    return min(candidates, key=lambda e: abs(e-target))


def evaluate_one_checkpoint(ep, args, config_net, autoencoder, scheduler,
                            inferer, real_stream, latent_shape, steps, spacing,
                            base_seed, device, results_path, lock,
                            bad_ep=None, guidance_scale=1.0, tag=""):
    """
    Genera i campioni di UN checkpoint, calcola il FID, scrive nel JSON (atomico).
    Se bad_ep is not None -> autoguidance attiva con quel bad.
    'tag' distingue le cartelle/chiavi tra fase1 (grezza) e fase2 (refined).
    """
    key=f"epoch{ep}"

    done=_already_done(results_path, key, lock)
    if done is not None:
        print(f"[{device}] [{tag}{key}] gia' valutato (FID={done['fid_avg']:.3f}), salto.")
        return done

    ckpt_path=os.path.join(args.models_dir, f"ldm_unet_epoch{ep}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[{device}] [{tag}{key}] checkpoint non trovato ({ckpt_path}), salto.")
        return None

    use_ag=bad_ep is not None
    print(f"\n{'='*55}\n[{device}] [{tag}{key}] genero {args.n_samples} campioni"
          f"{' + autoguidance (bad=epoch%d, w=%.2f)' % (bad_ep, guidance_scale) if use_ag else ''}"
          f"\n{'='*55}")

    unet, scale_factor, latent_mean=load_unet(config_net, ckpt_path, device)
    recon=ReconModel(autoencoder, scale_factor, latent_mean).to(device)

    unet_bad=None
    if use_ag:
        bad_path=os.path.join(args.models_dir, f"ldm_unet_epoch{bad_ep}.pt")
        if os.path.exists(bad_path):
            unet_bad=load_unet_weights_only(config_net, bad_path, device)
        else:
            print(f"[{device}] [{tag}{key}] bad {bad_path} non trovato: valuto senza guida.")
            use_ag=False

    synth_dir=os.path.join(args.work_dir, f"synth_{tag}{key}")
    os.makedirs(synth_dir, exist_ok=True)

    for idx in tqdm(range(args.n_samples), desc=f"{device} {tag}{key}"):
        set_determinism(seed=base_seed + idx)
        data=generate_one(unet, unet_bad, guidance_scale, recon, scheduler,
                           latent_shape, steps, device, inferer)
        affine=np.eye(4)
        for i in range(3):
            affine[i, i]=spacing[i]
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine),
                 os.path.join(synth_dir, f"hc_synth_{idx+1:04d}.nii.gz"))

    del unet, recon
    if unet_bad is not None:
        del unet_bad
    torch.cuda.empty_cache()

    print(f"[{device}] [{tag}{key}] calcolo FID...")
    synth_files=sorted(glob.glob(os.path.join(synth_dir, "hc_synth_*.nii.gz")))
    synth_stream=VolumeStream(synth_files, _load_synth)
    fid_res=compute_fid_2p5d(real_stream, synth_stream, device=device,
                             drop_empty=True, batch_size=32, verbose=True)
    result={"epoch": ep, **fid_res, "autoguidance": use_ag}
    if use_ag:
        result["bad_epoch"]=bad_ep
        result["guidance_scale"]=guidance_scale

    _save_result(results_path, key, result, lock)
    print(f"[{device}] [{tag}{key}] FID medio = {fid_res['fid_avg']:.3f}  (salvato in {results_path})")
    return result


def _init_worker_common(args, device):
    """Carica config, VAE, scheduler, inferer e stream reali (comuni ai checkpoint)."""
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
    ae_ckpt=paths.get("trained_autoencoder_path", "./outputs/models_v4/autoencoder_best.pt")
    print(f"[{device}] carico VAE da {ae_ckpt}")
    autoencoder=load_autoencoder(config_net, ae_ckpt, device)

    sched_cfg=config_net["noise_scheduler"]
    scheduler=RFlowScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        use_discrete_timesteps=sched_cfg.get("use_discrete_timesteps", False),
        use_timestep_transform=sched_cfg.get("use_timestep_transform", True),
        loc=sched_cfg.get("loc", 0.0),
        scale=sched_cfg.get("scale", 1.0),
        sample_method=sched_cfg.get("sample_method", "uniform"),
    )
    inferer=SlidingWindowInferer(roi_size=[64, 64, 64], sw_batch_size=1, progress=False,
                                 mode="gaussian", overlap=0.4, sw_device=device, device=device)

    with open(args.splits) as f:
        test_items=json.load(f)["test"]
    real_stream=VolumeStream(test_items, _load_real)
    print(f"[{device}] test set reale: {len(real_stream)} volumi")

    return (config_net, autoencoder, scheduler, inferer, real_stream,
            latent_shape, steps, spacing, base_seed)


def worker_phase1(rank, n_gpus, args, all_epochs, lock):
    """FASE 1: selezione grezza senza autoguidance."""
    torch.cuda.set_device(rank)
    device=f"cuda:{rank}"
    my_epochs=all_epochs[rank::n_gpus]
    if not my_epochs:
        print(f"[{device}] (fase1) nessun checkpoint assegnato.")
        return
    print(f"[{device}] (fase1) checkpoint: {my_epochs}")

    common=_init_worker_common(args, device)
    (config_net, autoencoder, scheduler, inferer, real_stream,
     latent_shape, steps, spacing, base_seed)=common

    results_path=os.path.join(args.work_dir, "fid_by_checkpoint_v4.json")
    for ep in my_epochs:
        evaluate_one_checkpoint(
            ep, args, config_net, autoencoder, scheduler, inferer, real_stream,
            latent_shape, steps, spacing, base_seed, device, results_path, lock,
            bad_ep=None, guidance_scale=1.0, tag="",
        )


def worker_phase2(rank, n_gpus, args, refine_jobs, lock):
    """
    FASE 2: raffinamento con autoguidance. refine_jobs e' una lista di tuple
    (good_ep, bad_ep) gia' calcolate dal main sui top-K. Ogni rank ne prende una
    fetta round-robin.
    """
    torch.cuda.set_device(rank)
    device=f"cuda:{rank}"
    my_jobs=refine_jobs[rank::n_gpus]
    if not my_jobs:
        print(f"[{device}] (fase2) nessun job assegnato.")
        return
    print(f"[{device}] (fase2) job (good,bad): {my_jobs}")

    common=_init_worker_common(args, device)
    (config_net, autoencoder, scheduler, inferer, real_stream,
     latent_shape, steps, spacing, base_seed)=common

    results_path=os.path.join(args.work_dir, "fid_by_checkpoint_v4_refined.json")
    for good_ep, bad_ep in my_jobs:
        evaluate_one_checkpoint(
            good_ep, args, config_net, autoencoder, scheduler, inferer, real_stream,
            latent_shape, steps, spacing, base_seed, device, results_path, lock,
            bad_ep=bad_ep, guidance_scale=args.guidance_scale, tag="ag_",
        )


def plot_refined_bars(raw_path, refined_path, out_path):
    """
    Grafico a BARRE: per ogni top-K, FID senza guida (fase1) vs con guida (fase2).
    Mostra l'effetto dell'autoguidance sui checkpoint migliori.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw=_read_results(raw_path)
    refined=_read_results(refined_path)
    if not refined:
        print("Nessun risultato raffinato, salto il grafico a barre.")
        return

    #ordina i candidati raffinati per epoca
    items=sorted(refined.values(), key=lambda r: r["epoch"])
    epochs=[r["epoch"] for r in items]
    fid_with=[r["fid_avg"] for r in items]
    #valore senza guida dallo stesso checkpoint (fase1)
    fid_without=[raw.get(f"epoch{ep}", {}).get("fid_avg", float("nan")) for ep in epochs]

    x=np.arange(len(epochs))
    w=0.38
    fig, ax=plt.subplots(figsize=(8, 5.5))
    b1=ax.bar(x-w/2, fid_without, w, label="senza autoguidance", color="#888780")
    b2=ax.bar(x+w/2, fid_with, w, label="con autoguidance", color="#1d9e75")

    for bars in (b1, b2):
        for bar in bars:
            h=bar.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.1f}", xy=(bar.get_x()+bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"epoch {e}" for e in epochs])
    ax.set_ylabel("FID 2.5D (più basso=meglio)", fontsize=11)
    ax.set_title("Effetto dell'autoguidance sui checkpoint migliori", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=10)
    fig.tight_layout()

    out_dir=os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Salvato: {out_path}")
    if out_path.lower().endswith(".png"):
        fig.savefig(out_path[:-4]+".pdf", bbox_inches="tight")
        print(f"Salvato: {out_path[:-4]}.pdf")
    plt.close(fig)


def print_ranking(results, title):
    if not results:
        print(f"{title}: nessun risultato.")
        return None
    print(f"\n{'='*55}\n{title}\n{'='*55}")
    ranked=sorted(results.items(), key=lambda kv: kv[1]["fid_avg"])
    for rank, (key, r) in enumerate(ranked, 1):
        ag="  [AG]" if r.get("autoguidance") else ""
        print(f"{rank}. {key:>10}  FID medio = {r['fid_avg']:.3f}  "
              f"(XY {r['fid_xy']:.1f} / YZ {r['fid_yz']:.1f} / ZX {r['fid_zx']:.1f}){ag}")
    best_key, best_r=ranked[0]
    print(f"\nMIGLIORE: {best_key} con FID {best_r['fid_avg']:.3f}")
    return ranked


def main():
    ap=argparse.ArgumentParser(description="Selezione checkpoint LDM via FID (multi-GPU, v4)")
    ap.add_argument("--config", type=str, default="configs/config_diff_model.json")
    ap.add_argument("--network", type=str, default="configs/config_network.json")
    ap.add_argument("--splits", type=str, default="data/splits/dataset.json")
    ap.add_argument("--models_dir", type=str, default="outputs/models_v4")
    ap.add_argument("--work_dir", type=str, default="outputs/checkpoint_selection_v4",
                    help="dove salvare le immagini temporanee e i JSON dei risultati")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--epochs", type=str, default="100,200,300,400,500,600,700,800,900,1000",
                    help="lista epoche dei checkpoint da testare, separate da virgola")
    ap.add_argument("--n_gpus", type=int, default=0, help="numero di GPU (0 = tutte)")
    #FASE 2 (raffinamento con autoguidance)
    ap.add_argument("--refine_top", type=int, default=3,
                    help="quanti checkpoint migliori rivalutare con autoguidance (0 = salta la fase 2)")
    ap.add_argument("--guidance_scale", type=float, default=2.0,
                    help="scala w dell'autoguidance nella fase 2")
    ap.add_argument("--bad_frac", type=float, default=0.30,
                    help="frazione dell'epoca del good da cui derivare il bad")
    args=ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    all_epochs=[int(e) for e in args.epochs.split(",")]

    n_avail=torch.cuda.device_count()
    n_gpus=args.n_gpus if args.n_gpus > 0 else n_avail
    n_gpus=max(1, min(n_gpus, n_avail))
    print(f"GPU disponibili: {n_avail}, uso: {n_gpus}")
    print(f"Checkpoint da valutare: {all_epochs}")

    manager=Manager()
    lock=manager.Lock()

    # ---------- FASE 1: selezione grezza (senza autoguidance) ----------
    if n_gpus==1:
        worker_phase1(0, 1, args, all_epochs, lock)
    else:
        mp.spawn(worker_phase1, args=(n_gpus, args, all_epochs, lock), nprocs=n_gpus, join=True)

    raw_path=os.path.join(args.work_dir, "fid_by_checkpoint_v4.json")
    raw_results=_read_results(raw_path)
    ranked=print_ranking(raw_results, "FASE 1 - CLASSIFICA (senza autoguidance)")

    # grafico 1: curva FID-vs-epoca (via plot_fid_curve.py)
    plot_fid_curve(
        json_path=raw_path,
        save_path=os.path.join("outputs/metrics", "fid_vs_epoch_v4.png"),
        title="Selezione checkpoint LDM v4 - FID vs epoca (senza autoguidance)",
    )

    # ---------- FASE 2: raffinamento top-K (con autoguidance) ----------
    if args.refine_top > 0 and ranked:
        #epoche effettivamente presenti su disco (per derivare i bad)
        available=[e for e in all_epochs
                   if os.path.exists(os.path.join(args.models_dir, f"ldm_unet_epoch{e}.pt"))]
        top_keys=[k for k, _ in ranked[:args.refine_top]]
        top_eps=[raw_results[k]["epoch"] for k in top_keys]

        refine_jobs=[]
        for good_ep in top_eps:
            bad_ep=derive_bad_epoch(good_ep, available, frac=args.bad_frac)
            if bad_ep is None:
                print(f"[fase2] good epoch{good_ep}: nessun bad valido (troppo precoce) "
                      f"-> verra' valutato senza guida.")
            refine_jobs.append((good_ep, bad_ep))  # bad_ep None -> evaluate senza guida

        print(f"[fase2] job di raffinamento (good, bad): {refine_jobs}")

        if n_gpus==1:
            worker_phase2(0, 1, args, refine_jobs, lock)
        else:
            mp.spawn(worker_phase2, args=(n_gpus, args, refine_jobs, lock), nprocs=n_gpus, join=True)

        refined_path=os.path.join(args.work_dir, "fid_by_checkpoint_v4_refined.json")
        refined_results=_read_results(refined_path)
        print_ranking(refined_results, "FASE 2 - CLASSIFICA (con autoguidance)")

        # grafico 2: barre di confronto senza vs con autoguidance
        plot_refined_bars(
            raw_path, refined_path,
            os.path.join("outputs/metrics", "fid_autoguidance_top_v4.png"),
        )


if __name__=="__main__":
    main()