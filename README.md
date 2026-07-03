# 3D Latent Diffusion Model for Synthetic Brain MRI

Unconditional generation of synthetic T1-weighted, skull-stripped brain MRI of
healthy controls (HC), using a two-stage **3D Latent Diffusion Model (LDM)** with
a **Rectified Flow** scheduler. The architecture follows NVIDIA's
**NV-Generate-CTMR / MAISI** design, adapted and trained from scratch on a
brain-MRI dataset.

The goal is to produce realistic 3D brain volumes that can augment data-limited
neuroimaging studies, while preserving the anatomical variability of real scans.

---

## Method

The pipeline is organised in two stages, trained sequentially:

1. **Autoencoder (VAE).** A `AutoencoderKlMaisi` compresses each `256^3` volume
   into a `4 x 64 x 64 x 64` latent representation (4x spatial downsampling per
   axis) and reconstructs it. The VAE defines the latent space in which the
   diffusion model operates, and therefore sets the upper bound on the achievable
   reconstruction fidelity.

2. **Latent Diffusion Model (LDM).** A `DiffusionModelUNetMaisi` is trained
   **unconditionally** in the VAE latent space, using a **Rectified Flow** noise
   scheduler (30 inference steps). Starting from Gaussian noise, the model denoises
   a latent that is then decoded by the VAE into a `256^3` volume.

Generation is fully unconditional: no modality, region, or spacing conditioning is
used. The UNet is built with `with_conditioning=False` and `num_class_embeds=None`.
Any conditioning-related parameters inherited from the MAISI framework (e.g. a
`modality` code in the inference config) are therefore ignored by the model.

### Latent normalisation and diffusion training

Since the diffusion model operates on VAE latents, the latents are normalised
before training. Normalisation is **per-channel** and **centred**: each latent
channel is standardised using its own mean and standard deviation
(`(z - mean) * scale`, both stored per-channel), which conditions the four latent
channels consistently and centres the target on the noise support. The statistics
are computed over the whole training set and saved in the checkpoint, so that
sampling can de-normalise correctly before decoding.

Timesteps are sampled with a **logit-normal** schedule during training, which
concentrates capacity on the mid-range timesteps where the denoising task is
hardest (as opposed to a uniform schedule). A linear learning-rate **warmup** is
applied to both the VAE and the LDM optimiser.

### Sampling with autoguidance

Because generation is unconditional, classifier-free guidance does not apply.
Instead, sampling uses **autoguidance**: the trained model is guided by a *weaker
version of itself* — an earlier checkpoint of the same run. At each denoising step
the velocity is extrapolated as

```
v = v_bad + w * (v_good - v_bad)
```

where `v_good` is the selected (best) checkpoint, `v_bad` an earlier checkpoint of
the same run, and `w` the guidance scale. This sharpens fine detail on
unconditional samples without requiring an EMA of the weights. Autoguidance is on
by default and can be disabled to produce a baseline.

### Data

The training set combines T1 skull-stripped HC volumes from six public
collections (ADNI, NIFD, OASIS1, OASIS2, OASIS3, PPMI), split into training /
validation / test subsets. Raw volumes are reoriented, resized to `256^3` and
intensity-normalised to `[0, 1]`. Large data files (raw volumes, latent
embeddings, synthetic outputs) are tracked with **DVC**, not committed to git.

### Evaluation

Synthetic volumes are compared against real ones using three metrics:

- **FID 2.5D** — Frechet Inception Distance computed slice-wise over the three
  orthogonal planes (XY, YZ, ZX) and averaged. Feature extraction uses an
  ImageNet InceptionV3 backbone (a robust alternative to RadImageNet, supported by
  recent medical-imaging FID literature).
- **MMD** — Maximum Mean Discrepancy, a complementary distributional distance.
- **MS-SSIM** (intra-set) — pairwise multi-scale SSIM among synthetic samples vs
  among real samples, used to detect mode collapse (diversity check).

A **checkpoint-selection** routine selects the best LDM checkpoint by FID in two
phases: (1) a coarse pass evaluates every checkpoint *without* autoguidance to
rank them and produce a FID-vs-epoch curve; (2) the top-K checkpoints are
re-evaluated *with* autoguidance, and a bar chart compares FID with vs without
guidance. The final reported metrics are computed by `eval.py` on the volumes
generated with autoguidance, i.e. the final generation configuration.

---

## Repository structure

```
.
├── configs/                        # JSON configuration files
│   ├── config_vae.json             # VAE hyperparameters
│   ├── config_diff_model.json      # LDM / diffusion + inference settings
│   ├── config_network.json         # Network architecture (VAE + UNet + scheduler)
│   └── environment.json            # Paths
│
├── src/
│   ├── data/
│   │   ├── transforms.py           # MONAI transform pipelines (VAE / encoding)
│   │   ├── dataset.py              # dataset for VAE training (image volumes)
│   │   ├── encode_dataset.py       # encode volumes -> latent embeddings (multi-GPU)
│   │   ├── embeddings_dataset.py   # build the latent split for the LDM
│   │   └── ldm_dataset.py          # dataset for LDM training (latents)
│   ├── training/
│   │   ├── train_vae.py            # stage 1: VAE training (DDP, multi-GPU)
│   │   └── train_ldm.py            # stage 2: LDM training (DDP, RFlow, per-channel scale)
│   ├── inference/
│   │   └── sample.py               # generate synthetic volumes (autoguidance)
│   └── evaluation/
│       ├── metrics.py              # FID 2.5D, MMD, MS-SSIM (lazy VolumeStream)
│       ├── eval.py                 # run evaluation (real vs synthetic)
│       ├── checkpoint_selection.py # FID per checkpoint + top-K autoguidance refine
│       └── plot_fid_curve.py       # plot the FID-vs-epoch curve
│
├── scripts/                        # SLURM launch scripts (activate venv, run a stage)
│   ├── run_train_vae.sh
│   ├── run_encode.sh
│   ├── run_train_ldm.sh
│   ├── run_checkpoint_selection.sh
│   ├── run_sample.sh
│   └── run_eval.sh
│
├── notebooks/                      # Analysis & visualisation
│   ├── 01_dataset_preprocessing.ipynb
│   ├── 02_vae_reconstruction.ipynb
│   ├── 03_ldm_generation.ipynb
│   └── 04_evaluation.ipynb
│
├── tests/                          # Smoke tests & checkpoint inspection
│   ├── check_best_model_vae.py     # inspect a VAE checkpoint
│   ├── check_best_model_ldm.py     # inspect an LDM checkpoint (scale_factor, NaN/Inf)
│   ├── test_dataset.py             # VAE image dataset
│   ├── test_vae.py                 # VAE model
│   ├── test_vae_reconstruction.py  # VAE reconstructions + SSIM/PSNR
│   ├── test_encode.py              # latent encoding
│   ├── test_ldm_dataset.py         # LDM latent dataset
│   └── test_sample.py              # generation smoke test
│
├── data/                           # (DVC-tracked, not in git)
│   ├── raw/                        # original HC volumes per dataset
│   ├── processed/embeddings/       # encoded latents
│   ├── splits/                     # train/val/test splits (JSON)
│   └── synthetic/                  # generated synthetic volumes
│
├── outputs/                        # (mostly git-ignored)
│   ├── models/                     # model checkpoints (.pt)
│   ├── generated/                  # preview images
│   └── metrics/                    # evaluation results & figures
│
├── requirements.txt
└── README.md
```

> **Note on paths.** Paths in the configs are versioned by iteration (e.g. models,
> embeddings and synthetic outputs live in version-suffixed folders). The commands
> below use the generic names for readability; the actual folder names follow the
> current iteration configured in `configs/`.

---

## Installation

```bash
# clone
git clone https://github.com/DanieleSchingaro/tesiSchingaro_NeuroImaging.git
cd tesiSchingaro_NeuroImaging

# virtual environment
python3 -m venv .venv
source .venv/bin/activate

# dependencies
pip install -r requirements.txt
```

**Requirements.** The pipeline expects a CUDA-capable GPU (developed and trained on
NVIDIA H100 GPUs). Multi-GPU training uses PyTorch DDP via `torchrun`. Key
dependencies: PyTorch, MONAI 1.5.2, `monai-generative` 0.2.3, nibabel, torchmetrics.

**Data.** Large files (raw volumes, latent embeddings, synthetic outputs) are
versioned locally with **DVC** and kept out of git. The `.dvc` pointer files are
committed; the data itself is managed through the local DVC cache.

**Experiment tracking.** Training is logged with **MLflow** (`mlruns/`).

---

## Usage

All stages read paths and hyperparameters from the `configs/*.json` files. The
long-running stages are submitted as **SLURM** jobs via the `scripts/run_*.sh`
helpers (each activates the environment and launches one stage). The pipeline is
**sequential**: each stage consumes the output of the previous one.

### 1. Train the VAE (stage 1)

```bash
sbatch scripts/run_train_vae.sh
```

### 2. Encode volumes into latents

Once the VAE is trained, real volumes are encoded into latents (multi-GPU) and a
latent split is built for the LDM:

```bash
sbatch scripts/run_encode.sh
python3 -m src.data.embeddings_dataset
```

### 3. Train the LDM (stage 2)

```bash
sbatch scripts/run_train_ldm.sh
```

### 4. Select the best checkpoint (FID) with autoguidance refinement

Evaluate the FID of every LDM checkpoint, then re-evaluate the top-K with
autoguidance. Produces the FID-vs-epoch curve and a with/without-guidance
comparison chart:

```bash
sbatch scripts/run_checkpoint_selection.sh
```

### 5. Generate synthetic volumes

Set the selected (`good`) and earlier (`bad`) checkpoints in
`scripts/run_sample.sh`, then generate N volumes (default 100) with autoguidance:

```bash
sbatch scripts/run_sample.sh 100
```

Each run saves the NIfTI volumes plus orthogonal-view PNG previews. A baseline
without autoguidance can be produced by disabling it in the launch script.

### 6. Evaluate

```bash
# compare synthetic vs real (test set, hold-out volumes)
sbatch scripts/run_eval.sh test

# compare against the full real dataset (lower-variance FID reference)
sbatch scripts/run_eval.sh all
```

Results are written to `outputs/metrics/`.

---

## Notes

- **Hardware.** Training was performed on a multi-GPU server (4x NVIDIA H100).
  Reduce batch/patch sizes or the number of GPUs in the configs to fit smaller
  hardware.
- **Reproducibility.** Generation uses fixed seeds (`base_seed + index`), so
  repeated runs produce the same samples.
- **Reference.** Architecture and scheduler follow NVIDIA's
  NV-Generate-CTMR / MAISI. See the
  [Latent Diffusion (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf),
  [Rectified Flow (ICLR 2023)](https://arxiv.org/pdf/2209.03003) and
  autoguidance ([Karras et al., NeurIPS 2024](https://arxiv.org/abs/2406.02507))
  papers.

---

*Developed as part of a bachelor's thesis on deep generative models for 3D medical
image synthesis.*

**Author:** DanieleSchingaro
**E-mail:** d.schingaro04@gmail.com