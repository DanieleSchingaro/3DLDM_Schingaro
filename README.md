# 3D Latent Diffusion Model for Synthetic Brain MRI

Synthesis of T1-weighted, skull-stripped brain MRI of healthy controls (HC) with a
two-stage **3D Latent Diffusion Model (LDM)** and a **Rectified Flow** scheduler,
extended with a **3D ControlNet** for *conditional* generation. The architecture
follows NVIDIA's **NV-Generate-CTMR / MAISI** design, adapted and trained from
scratch on a brain-MRI dataset.

The project has two parts:

1. **Unconditional generation** — the LDM samples a brain volume from Gaussian
   noise, with no anatomical constraint.
2. **Conditional generation (ControlNet)** — a ControlNet is trained *on top of the
   frozen LDM* and steers generation to follow a given tissue segmentation mask
   (white matter / grey matter / CSF).

Unlike most ControlNet work in this domain, the base LDM here is **not** a
pre-trained foundation model: it is trained from scratch on 805 volumes. The
ControlNet therefore also probes how well conditional control transfers to a
data-scarce, in-house diffusion backbone.

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

Timesteps are sampled with a **two-phase curriculum**:

- **phase 1** (epoch < `curriculum_switch_epoch`, default 500): **uniform** — the
  model consolidates the *global structure* (position and scale of the brain),
  which is established at the high-noise timesteps.
- **phase 2** (epoch >= switch): **logit-normal** (loc=0, scale=1) — the model
  refines *texture* (the mid-range of the trajectory).

This resolves a trade-off observed in an earlier iteration. A **static
logit-normal** schedule gave the lowest FID but produced ~20% of samples with a
**quantised global position shift** (±32 image voxels = 1 voxel at the UNet
bottleneck): it under-samples the high-noise timesteps that set global structure.
A **static uniform** schedule removed the artefact (0% shifted) but lost texture
sharpness (higher FID). The curriculum keeps a uniform base long enough to fix
the geometry, then switches to logit-normal to recover sharpness — obtaining both.
A linear learning-rate **warmup** is applied to the LDM optimiser.

> **Trade-off, resolved (timestep schedule).** static logit-normal: FID 21.8, but
> MMD 0.019, MS-SSIM 0.895, ~20% mis-positioned. static uniform: FID 27.2, MMD
> 0.009, MS-SSIM 0.947, 0% mis-positioned. **curriculum (final): FID 24.3, MMD
> 0.009, MS-SSIM 0.949, 0% mis-positioned** — it recovers about half of the
> logit-normal sharpness advantage at no geometric cost. The artefact is invisible
> to both FID and MS-SSIM (a global translation leaves the slice set largely
> unchanged), and was found by visual inspection and quantified with a dedicated
> geometric test (`tests/check_degenerate_samples.py`).

> **Note on the VAE.** This iteration also experimented with a linear LR warmup
> for the VAE. It produced worse reconstructions than the previous step-wise
> schedule, so the pipeline reuses the earlier, better-performing VAE checkpoint
> for encoding and decoding, while the diffusion-side improvements above are kept.
> The encoding VAE and the decoding VAE are always the same checkpoint.

### Sampling with autoguidance

Because generation is unconditional, classifier-free guidance does not apply.
Instead, sampling uses **autoguidance**: the trained model is guided by a *weaker
version of itself* — an earlier checkpoint of the same run. At each denoising step
the velocity is extrapolated as

```
v = v_bad + w * (v_good - v_bad)
```

where `v_good` is the selected (best) checkpoint, `v_bad` an earlier checkpoint of
the same run, and `w` the guidance scale (2.0). This sharpens fine detail on
unconditional samples without requiring an EMA of the weights. Autoguidance is on
by default and can be disabled (`--no_autoguidance`) to produce a baseline — the
mode used to attribute the positioning artefact to the training schedule rather
than to sampling.

### Conditional generation with ControlNet (stage 3)

A third stage adds **spatial control** over the generated anatomy. Following
ControlNet (Zhang et al., 2023), the trained LDM is **frozen** and a trainable copy
of its encoder — the ControlNet — receives a *conditioning signal* and injects
residuals into the frozen UNet at every denoising step. The base model is left
untouched: only the ControlNet learns to steer generation.

**Conditioning signal.** A 3-tissue segmentation mask (CSF / grey matter / white
matter, values `{0,1,2,3}` including background) obtained with **FSL-FAST**, run in a
Singularity container. Masks are computed on the raw `181x217x181` volumes and then
zero-padded to `256^3`, so that no interpolation alters the discrete labels. The mask
is bit-plane encoded (`binarize_labels`, 8 channels) before entering the ControlNet.

**Training setup.** The ControlNet is initialised by copying the frozen UNet's encoder
weights (192 of 230 tensors match; the remaining 38 are the ControlNet-specific
zero-convolutions and conditioning embedding). Loss and target are identical to the
LDM (`target = images - noise`, L1). Latents use the same per-channel normalisation
`(z - latent_mean) * scale_factor`, with values **read from the LDM checkpoint** rather
than recomputed, since the frozen UNet is calibrated on exactly those statistics.
Timesteps are sampled **uniformly**: the curriculum is unnecessary here, because the
frozen model already encodes the global structure and the ControlNet only learns to
condition it.

**Evaluation — DSC.** Conditional generation is not measured by FID but by how
faithfully the output respects the requested mask. For each test volume: the real
mask conditions the generation, the synthetic volume is re-segmented with FSL-FAST,
and the resulting mask is compared with the conditioning one via **Dice (DSC)**, both
mean and generalised (`weight_type="square"`).

> **Methodological note — the ceiling of the measurement.** The two masks being
> compared come from different paths: the conditioning mask is FAST on the *raw*
> volume, while the generated mask is FAST on a *VAE-decoded* volume, which has lost
> detail through the `4 x 64^3` bottleneck. To quantify this, real volumes were
> decoded and re-segmented through the synthetic path and compared against their own
> conditioning mask: the resulting **DSC of 0.788 is the ceiling** of this evaluation,
> unreachable by any generative model. Reported DSC values should be read against
> that ceiling, not against 1.0.

> **Intensity range.** Decoded volumes must be saved **without** clipping to `[0,1]`.
> Clipping piles up ~0.5M voxels against the upper bound, which breaks FAST's k-means
> initialisation (`variance nan`) and collapses the segmentation to 2 classes. Saved in
> their native range, volumes are segmented directly, with no denoising or added noise.

**Inference-time optimisation.** With the ControlNet trained and the checkpoint selected, two
inference parameters were tuned on a 25-mask subset of the validation split — **no retraining
involved**:

- **Conditioning scale.** The ControlNet residuals are multiplied by a factor before being
  injected into the frozen UNet. DSC grows monotonically with it (0.678 at 1.0, 0.730 at 4.0)
  with diminishing returns past 2.5; **3.0** was chosen. A time-varying schedule along the
  denoising trajectory was also tested — increasing and decreasing, at equal mean — and neither
  beats the constant factor: what matters is the average intensity, not its distribution over
  steps. This does *not* reproduce, for voxel-level tissue control, the reported prevalence of
  early denoising steps in 2D layout control.
- **Autoguidance.** As for the unconditional LDM: `v = v_bad + w*(v_good - v_bad)`, with
  `bad = epoch 200` and **w = 2.0**. The ControlNet residuals are injected **identically** into
  both UNets, so the difference between the two predictions isolates denoising quality without
  touching the conditioning axis. Autoguidance is designed to improve image quality *without*
  affecting prompt adherence; here it improves the DSC too, plausibly because sharper tissue
  boundaries are re-segmented more accurately.

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
  among real samples, used to detect mode collapse (diversity check). Note: only
  interpretable once samples are correctly positioned — a global translation
  lowers MS-SSIM independently of anatomical diversity.
- **Geometric QC** (`tests/check_degenerate_samples.py`) — measures, per volume,
  the tissue fraction, centroid and bounding-box extent, and flags samples outside
  the (tightly aligned) real distribution. Catches the global-position artefact
  that FID and MS-SSIM miss.

A **checkpoint-selection** routine selects the best LDM checkpoint by FID in two
phases: (1) a coarse pass evaluates every checkpoint *without* autoguidance to
rank them and produce a FID-vs-epoch curve; (2) the top-K checkpoints are
re-evaluated *with* autoguidance, and a bar chart compares FID with vs without
guidance. The final reported metrics are computed by `eval.py` on the volumes
generated with autoguidance, i.e. the final generation configuration.

### Results (final model: two-phase curriculum, autoguidance w=2.0, epoch 800)

| Reference | FID 2.5D | MMD | MS-SSIM (synth / real) | Mis-positioned |
|-----------|---------:|------:|:---------------------:|---------------:|
| test (102 hold-out) | 24.31 | 0.0089 | 0.949 / 0.936 | 0% |
| all (1007)          | 23.86 | 0.0089 | 0.949 / 0.936 | 0% |

Per-plane FID (test): XY 25.5 / YZ 27.3 / ZX 20.2 — the sagittal plane (YZ)
remains the hardest, but the gap is the smallest across all configurations.

**Timestep schedule comparison** (all with autoguidance, w=2.0):

| Schedule | FID | MMD | MS-SSIM synth | Mis-positioned |
|----------|----:|------:|:-------------:|---------------:|
| static logit-normal | 21.8 | 0.019 | 0.895 | ~20% |
| static uniform | 27.2 | 0.0089 | 0.947 | 0% |
| **curriculum** (final) | 24.3 | 0.0089 | 0.949 | 0% |
| v2 baseline | ~39 | — | — | 0% |

The curriculum improves FID by ~38% over the baseline and resolves the trade-off
between the two static schedules: it recovers about half of the logit-normal
sharpness advantage while keeping the uniform's perfect geometry (0% mis-positioned,
verified with and without guidance).

### Results — conditional generation (ControlNet, epoch 100)

The ControlNet is trained for 100 epochs on top of the frozen curriculum LDM
(`models_v5/ldm_unet_epoch800.pt`). The checkpoint is selected by DSC on the
**validation** split; the test split is used **once**, with the selected checkpoint,
for the final number.

| Split | n | Mean DSC | Generalised DSC | std | % of ceiling |
|-------|--:|---------:|----------------:|----:|-------------:|
| validation (checkpoint selection) | 100 | 0.677 | 0.643 | 0.023 | 86% |
| **test (hold-out, final)** | **102** | **0.678** | **0.641** | 0.025 | **86%** |
| *measurement ceiling* (real vs itself) | 10 | *0.788* | *0.747* | *0.020* | *100%* |

Validation and test agree to within 0.0002, confirming that checkpoint selection did
not overfit the validation split.

**Checkpoint selection** (validation, DSC):

| Checkpoint | Mean DSC | Generalised DSC |
|------------|---------:|-------------### Results — conditional generation (ControlNet, v_final)

The ControlNet is trained for 100 epochs on top of the frozen curriculum LDM. The checkpoint is
selected by DSC on the **validation** split; inference parameters are then tuned on a 25-mask
validation subset; the **test** split is used only for the final numbers.

**Final configuration (`v_final`):** checkpoint `controlnet_epoch100.pt`, conditioning scale
**3.0**, autoguidance **w = 2.0** (bad = epoch 200), 30 inference steps.

| Split | n | Mean DSC | Generalised DSC | std | % of ceiling |
|-------|--:|---------:|----------------:|----:|-------------:|
| test — standard inference (`cond_scale=1`, no guidance) | 102 | 0.678 | 0.641 | 0.025 | 86% |
| validation — `v_final` | 100 | 0.734 | 0.699 | 0.020 | 93% |
| **test — `v_final` (final)** | **102** | **0.733** | **0.696** | 0.021 | **93%** |
| *measurement ceiling* (real vs itself) | 10 | *0.788* | *0.747* | *0.020* | *100%* |

Validation and test agree to within 0.001, so tuning the inference parameters on a validation
subset did not overfit. Inference-time optimisation alone is worth **+0.055 DSC**, with no
retraining, and also tightens the per-volume spread.

**Checkpoint selection** (validation, standard inference):

| Checkpoint | Mean DSC | Generalised DSC |
|------------|---------:|----------------:|
| epoch 60 | 0.652 | 0.619 |
| epoch 80 | 0.675 | 0.641 |
| **epoch 100** (selected) | **0.677** | **0.643** |

**Realism and diversity** — same metrics as the unconditional model, 102 conditioned volumes vs
the 102 real test volumes:

| Metric | LDM (unconditional) | ControlNet `v_final` |
|--------|--------------------:|---------------------:|
| FID 2.5D (mean) | 24.31 | **22.55** |
| FID XY / YZ / ZX | 25.5 / 27.3 / 20.2 | **21.7 / 23.3 / 22.6** |
| MMD | 0.0089 | **0.0085** |
| MS-SSIM synth / real | 0.949 / 0.936 | **0.939 / 0.936** |

Conditioned volumes are **more** realistic than unconditional ones: with the anatomy constrained,
the model no longer has to invent global structure and its capacity goes into texture. The
sagittal plane — consistently the hardest across every LDM version — is no longer an outlier: the
three planes span 1.6 FID points instead of 7.1. MS-SSIM sits 0.003 from the real volumes
(against 0.013 unconditionally), so conditioning does not flatten anatomical variety.

**Geometric QC (test, 102 volumes):** 0% mis-positioned samples (largest centroid shift 3.4
voxels), no truncated volumes, tissue fraction 0.1102 against 0.1102 for the real volumes.

The ControlNet reaches **93% of the achievable ceiling**. The remaining 21% between the ceiling
and 1.0 is structural: it stems from the VAE's lossy compression and from comparing masks
produced along two different segmentation paths, not from the conditioning itself. Absolute DSC
values are therefore not directly comparable with work built on pre-trained foundation-model
VAEs, whose reconstruction fidelity — and hence ceiling — is higher. FID, MMD and MS-SSIM depend
on the chosen feature extractor (InceptionV3 2.5D here) and are meaningful only for internal
comparisons within this work.

## Repository structure

```
.
├── configs/                        # JSON configuration files
│   ├── config_vae.json             # VAE hyperparameters
│   ├── config_diff_model.json      # LDM / diffusion + inference settings
│   ├── config_network.json         # Network architecture (VAE + UNet + ControlNet + scheduler)
│   ├── config_controlnet.json      # ControlNet training / inference settings
│   └── environment.json            # Paths
│
├── src/
│   ├── data/
│   │   ├── transforms.py           # MONAI transform pipelines (VAE / encoding)
│   │   ├── dataset.py              # dataset for VAE training (image volumes)
│   │   ├── encode_dataset.py       # encode volumes -> latent embeddings (multi-GPU)
│   │   ├── embeddings_dataset.py   # build the latent split for the LDM
│   │   ├── ldm_dataset.py          # dataset for LDM training (latents)
│   │   ├── controlnet_dataset.py   # dataset for ControlNet (latent + mask pairs)
│   │   ├── binarize.py             # bit-plane encoding of the conditioning mask
│   │   ├── list_volumes.py         # list the raw volumes to segment with FSL-FAST
│   │   ├── pad_masks_to256.py      # pad FAST masks 181^3 -> 256^3 (no interpolation)
│   │   ├── create_controlnet_json.py           # training split (latent <-> mask, folds)
│   │   └── create_controlnet_inference_json.py # inference lists (val / test masks)
│   ├── training/
│   │   ├── train_vae.py            # stage 1: VAE training (DDP, multi-GPU)
│   │   ├── train_ldm.py            # stage 2: LDM training (DDP, RFlow, per-channel scale)
│   │   └── train_controlnet.py     # stage 3: ControlNet on the frozen LDM (DDP, MLflow)
│   ├── inference/
│   │   ├── sample.py               # generate synthetic volumes (autoguidance)
│   │   └── sample_controlnet.py    # mask-conditioned generation (ControlNet)
│   └── evaluation/
│       ├── metrics.py              # FID 2.5D, MMD, MS-SSIM (lazy VolumeStream)
│       ├── eval.py                 # run evaluation (real vs synthetic)
│       ├── checkpoint_selection.py # FID per checkpoint + top-K autoguidance refine
│       ├── plot_fid_curve.py       # plot the FID-vs-epoch curve
│       ├── controlnet_dsc.py       # DSC (mean + generalised) mask vs re-segmented output
│       ├── check_mask_classes.py   # verify a segmentation has all 4 tissue classes
│       └── prepare_for_fast.py     # fallback preparation for volumes FAST cannot segment
│
├── scripts/                        # launch scripts (activate venv, run a stage; torchrun/python3)
│   ├── run_train_vae.sh
│   ├── run_encode.sh
│   ├── run_train_ldm.sh
│   ├── run_checkpoint_selection.sh
│   ├── run_sample.sh               # generation with autoguidance
│   ├── run_sample_noag.sh          # baseline generation without autoguidance
│   ├── run_eval.sh
│   ├── run_fast_segmentation.sh    # FSL-FAST on the raw volumes -> conditioning masks
│   ├── run_train_controlnet.sh     # ControlNet training
│   ├── run_sample_controlnet.sh    # mask-conditioned generation
│   └── run_segment_generated.sh    # FSL-FAST re-segmentation of generated volumes
│
├── notebooks/                      # Analysis & visualisation
│   ├── 01_dataset_preprocessing.ipynb
│   ├── 02_vae_reconstruction.ipynb
│   ├── 03_ldm_generation.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_controlnet_generation.ipynb
│   └── 06_controlnet_evaluation.ipynb
│
├── tests/                          # Smoke tests & checkpoint inspection
│   ├── check_best_model_vae.py     # inspect a VAE checkpoint
│   ├── check_best_model_ldm.py     # inspect an LDM checkpoint (per-channel scale/mean, NaN/Inf)
│   ├── check_degenerate_samples.py # geometric QC: detect globally mis-positioned samples
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
│   ├── synthetic/                  # generated synthetic volumes (unconditional)
│   ├── masks_fsl_prePad_181/       # FSL-FAST masks at raw resolution (intermediate)
│   ├── masks_fsl_postPad_256/      # FSL-FAST conditioning masks (256^3)
│   ├── controlnet_gen_val/         # mask-conditioned volumes, validation (checkpoint selection)
│   └── controlnet_gen_test/        # mask-conditioned volumes, test
│       ├── epoch100/               #   standard inference (baseline)
│       └── v_final/                #   final configuration
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
a single node with 4x NVIDIA H100). Multi-GPU stages use PyTorch DDP via
`torchrun` on the single node (no cluster scheduler required). Key dependencies:
PyTorch, MONAI 1.5.2, `monai-generative` 0.2.3, nibabel, torchmetrics.

**Data.** Large files (raw volumes, latent embeddings, synthetic outputs) are
versioned locally with **DVC** and kept out of git. The `.dvc` pointer files are
committed; the data itself is managed through the local DVC cache.

**Experiment tracking.** Training is logged with **MLflow** (`mlruns/`).

---

## Usage

All stages read paths and hyperparameters from the `configs/*.json` files. Each
stage has a launch script under `scripts/` that activates the environment and
runs the stage (logging to `logs/` via `tee`). The pipeline runs on a **single
node with 4 GPUs** (no cluster scheduler); multi-GPU stages use `torchrun`
internally. It is **sequential**: each stage consumes the output of the previous
one. Run each stage inside a `tmux` session so it survives an SSH disconnect.

```bash
bash scripts/run_<stage>.sh        # or: source the script's command directly
```

### 1. Train the VAE (stage 1)

```bash
bash scripts/run_train_vae.sh
```

> The current iteration reuses the previous, better-performing VAE checkpoint
> (see the note in *Method*), so this stage is not re-run; the script is kept
> for reference. The encoding stage points `--checkpoint` at that VAE.

### 2. Encode volumes into latents

Real volumes are encoded into latents (multi-GPU), then a latent split is built
for the LDM:

```bash
bash scripts/run_encode.sh
python3 -m src.data.embeddings_dataset
```

### 3. Train the LDM (stage 2)

```bash
bash scripts/run_train_ldm.sh
```

### 4. Select the best checkpoint (FID) with autoguidance refinement

Evaluate the FID of every LDM checkpoint, then re-evaluate the top-K with
autoguidance. Produces the FID-vs-epoch curve and a with/without-guidance
comparison chart:

```bash
bash scripts/run_checkpoint_selection.sh
```

### 5. Generate synthetic volumes

Set the selected (`good`) and earlier (`bad`) checkpoints in
`scripts/run_sample.sh`, then generate N volumes (default 100) with autoguidance:

```bash
bash scripts/run_sample.sh 100
```

Each run saves the NIfTI volumes plus orthogonal-view PNG previews. A baseline
without autoguidance can be produced by disabling it in the launch script.

### 6. Evaluate

```bash
# compare synthetic vs real (test set, hold-out volumes)
bash scripts/run_eval.sh test

# compare against the full real dataset (lower-variance FID reference)
bash scripts/run_eval.sh all
```

Results are written to `outputs/metrics/`.

---

### 7. Conditional generation with ControlNet

Requires FSL (used here through a Singularity container, `~/containers/fsl.sif`).

```bash
# 7a. conditioning masks: FSL-FAST on the raw volumes, then pad to 256^3.
#     FAST runs at raw resolution (181^3, ~2.4x fewer voxels than 256^3) and the
#     resulting mask is zero-padded afterwards: padding a label map is exact, while
#     resampling the volume first would interpolate the intensities.
python3 src/data/list_volumes.py                 # list the volumes to segment
bash scripts/run_fast_segmentation.sh            # FSL-FAST -> data/masks_fsl_prePad_181/
python3 src/data/pad_masks_to256.py              # pad       -> data/masks_fsl_postPad_256/

# 7b. build the ControlNet splits
python3 src/data/create_controlnet_json.py            # training  (fold 1 = train, fold 0 = val)
python3 src/data/create_controlnet_inference_json.py  # inference (val / test mask lists)

# 7c. train the ControlNet on the frozen LDM
bash scripts/run_train_controlnet.sh

# 7d. mask-conditioned generation, final configuration (cond_scale 3.0, autoguidance w=2.0)
#     args: <ckpt> <mask_list> <out_dir> [cond_scale] [steps] [cond_scale_end] [bad_ckpt] [w]
bash scripts/run_sample_controlnet.sh \
    outputs/controlnet_v6/controlnet_epoch100.pt \
    data/splits/controlnet_infer_test.json \
    data/controlnet_gen_test/v_final \
    3.0 30 "" outputs/models_v5/ldm_unet_epoch200.pt 2.0

# 7e. re-segment the generated volumes with FSL-FAST
bash scripts/run_segment_generated.sh data/controlnet_gen_test/v_final

# 7f. DSC between the conditioning mask and the re-segmented output
python3 src/evaluation/controlnet_dsc.py \
    --gen_dir data/controlnet_gen_test/v_final --tag test_v_final

# 7g. realism metrics (FID / MMD / MS-SSIM) against the real test volumes
#     "cn" switches on the ControlNet file pattern and percentile normalisation
bash scripts/run_eval.sh test data/controlnet_gen_test/v_final controlnet_v_final cn
```

To measure the **ceiling** of the evaluation (see the methodological note above),
run the same DSC script over decoded *real* volumes:

```bash
python3 src/evaluation/controlnet_dsc.py \
    --gen_dir data/tetto --tag ceiling --gen_suffix _realdec_pveseg.nii.gz
```

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