# Pix2Pix — Image-to-Image Translation with Conditional Adversarial Networks

A PyTorch implementation of **Image-to-Image Translation with Conditional Adversarial Networks** ([Isola et al., CVPR 2017](https://arxiv.org/abs/1611.07004)), applied to the **anime sketch-to-colour** task: given a black-and-white sketch of an anime character, the model learns to produce a plausible full-colour version.

<img width="729" height="392" alt="image" src="https://github.com/user-attachments/assets/21237901-d13e-49f2-b6a9-1c540ce50410" />


| Item | Detail |
|---|---|
| **Paper** | [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (Isola et al., 2017) |
| **Dataset** | [Anime Sketch Colorization Pair](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair) (14 224 train / 3 545 val paired images) |
| **Platform** | Kaggle — 2 × Tesla T4 GPUs |
| **Framework** | PyTorch |
| **Task** | Paired image-to-image translation (sketch → colour) |

---

## Table of Contents

1. [Core Idea](#core-idea)
2. [Architecture Overview](#architecture-overview)
3. [Notebook Walkthrough](#notebook-walkthrough)
4. [Training Process](#training-process)
5. [Results & Analysis](#results--analysis)
6. [How to Run](#how-to-run)
7. [Requirements](#requirements)
8. [References](#references)

---

## Core Idea

Traditional image generation with GANs is *unconditional* — the generator maps random noise **z** to an output image with no guidance. **Pix2Pix** introduces *conditional* adversarial networks: both the generator and discriminator are conditioned on an **input image**, turning the GAN framework into a general-purpose **image-to-image translation** system.

The key insight from Isola et al. is that the standard GAN loss alone is insufficient for structured prediction tasks like image translation. The paper proposes combining:

1. **Adversarial loss (cGAN):** The discriminator sees both the input image and the output (real or generated) and learns to distinguish real pairs from fake pairs. This encourages the generator to produce outputs that are *realistic*.
2. **L1 reconstruction loss:** The generator is also penalised for the pixel-wise difference between its output and the ground-truth target. This encourages the generator to produce outputs that are *accurate* — close to the true answer.

The combined objective is:

```
L_total = L_cGAN + λ · L_L1
```

where λ controls the trade-off between realism and accuracy.

> **Why L1 and not L2?** L1 loss produces sharper images than L2 (MSE) loss, which tends to average over uncertainty and create blurry outputs. The adversarial loss adds high-frequency detail that even L1 alone cannot capture.

### Connection to the Original Paper

| Design Choice | Paper | This Notebook |
|---|---|---|
| Generator | **U-Net** (encoder-decoder with skip connections) | **U-Net** (6 encoder + bottleneck + 6 decoder) |
| Discriminator | **PatchGAN** (70×70 receptive field) | **PatchGAN** (output: 14×14 patch map on 128×128 input) |
| Adversarial loss | cGAN (BCE) | **BCEWithLogitsLoss** |
| Reconstruction loss | L1 | **L1Loss** |
| λ (L1 weight) | **100** | **100** |
| Optimizer | Adam (lr=2e-4, β₁=0.5, β₂=0.999) | **Adam (lr=2e-4, β₁=0.5, β₂=0.999)** |
| Input noise (z) | Dropout in generator | **Dropout (0.5) in first 3 decoder layers** |
| Image size | 256×256 | **128×128** |
| Normalisation | Instance Norm | **BatchNorm** |

> The main differences are scale (128×128 vs 256×256) and normalisation strategy. The paper uses Instance Normalisation for some tasks; this implementation uses BatchNorm throughout. The generator depth (6 encoder layers instead of 8) is adjusted for the smaller image size.

---

## Architecture Overview

### Generator: U-Net with Skip Connections

The U-Net architecture is crucial for image-to-image translation because it preserves **low-level spatial information** through skip connections. A standard encoder-decoder would compress everything into a bottleneck, losing fine details. Skip connections allow the decoder to "borrow" high-resolution features directly from the encoder.

```
Input Sketch (3×128×128)
    │
    ▼
┌──────────────────────────────────────────┐
│  ENCODER                                 │
├──────────────────────────────────────────┤
│  enc1: Conv2d(3→64, 4×4, s2)            │  → 64×64×64     ─┐
│        LeakyReLU(0.2)  (no BatchNorm)    │                   │
├──────────────────────────────────────────┤                   │
│  enc2: Conv2d(64→128, 4×4, s2)           │  → 128×32×32   ─┐│
│        BatchNorm + LeakyReLU(0.2)        │                  ││
├──────────────────────────────────────────┤                  ││
│  enc3: Conv2d(128→256, 4×4, s2)          │  → 256×16×16  ─┐││
│        BatchNorm + LeakyReLU(0.2)        │                 │││
├──────────────────────────────────────────┤                 │││
│  enc4: Conv2d(256→512, 4×4, s2)          │  → 512×8×8   ─┐│││
│        BatchNorm + LeakyReLU(0.2)        │                ││││
├──────────────────────────────────────────┤                ││││
│  enc5: Conv2d(512→512, 4×4, s2)          │  → 512×4×4  ─┐│││││
│        BatchNorm + LeakyReLU(0.2)        │               ││││││
├──────────────────────────────────────────┤               ││││││
│  enc6: Conv2d(512→512, 4×4, s2)          │  → 512×2×2 ─┐│││││
│        BatchNorm + LeakyReLU(0.2)        │              ││││││
└──────────────────────────────────────────┘              ││││││
    │                                                     ││││││
    ▼                                                     ││││││
┌──────────────────────────────────────────┐              ││││││
│  BOTTLENECK                              │              ││││││
│  Conv2d(512→512, 4×4, s2) + ReLU         │  → 512×1×1  ││││││
└──────────────────────────────────────────┘              ││││││
    │                                                     ││││││
    ▼                                                     ││││││
┌──────────────────────────────────────────┐              ││││││
│  DECODER (with skip connections)         │              ││││││
├──────────────────────────────────────────┤              ││││││
│  dec1: ConvT2d(512→512) + BN + Drop(0.5) │  → 512×2×2  ││││││
│        + ReLU, cat(enc6)                 │  → 1024×2×2 ─┘│││││
├──────────────────────────────────────────┤               │││││
│  dec2: ConvT2d(1024→512) + BN + Drop(0.5)│  → 512×4×4   │││││
│        + ReLU, cat(enc5)                 │  → 1024×4×4 ──┘││││
├──────────────────────────────────────────┤                ││││
│  dec3: ConvT2d(1024→512) + BN + Drop(0.5)│  → 512×8×8    ││││
│        + ReLU, cat(enc4)                 │  → 1024×8×8 ───┘│││
├──────────────────────────────────────────┤                 │││
│  dec4: ConvT2d(1024→256) + BN + ReLU     │  → 256×16×16   │││
│        cat(enc3)                         │  → 512×16×16 ───┘││
├──────────────────────────────────────────┤                  ││
│  dec5: ConvT2d(512→128) + BN + ReLU      │  → 128×32×32    ││
│        cat(enc2)                         │  → 256×32×32 ────┘│
├──────────────────────────────────────────┤                   │
│  dec6: ConvT2d(256→64) + BN + ReLU       │  → 64×64×64      │
│        cat(enc1)                         │  → 128×64×64 ─────┘
├──────────────────────────────────────────┤
│  final: ConvT2d(128→3, 4×4, s2) + Tanh  │  → 3×128×128
└──────────────────────────────────────────┘
    │
    ▼
  Colour Image (3×128×128), values in [-1, 1]
```

**Key design elements:**
- **No BatchNorm on first encoder layer** — following the DCGAN convention; the first layer operates directly on raw pixel values.
- **Dropout (0.5) in first 3 decoder layers** — acts as noise injection (replacing the random noise vector **z** used in unconditional GANs). This provides stochasticity and prevents the generator from memorising a deterministic mapping.
- **Skip connections** — each decoder layer receives the concatenation of the upsampled features and the corresponding encoder features. This preserves spatial detail that would otherwise be lost through the bottleneck.
- **Tanh output** — matches the [-1, 1] normalisation of the target images.

### Discriminator: PatchGAN

Instead of classifying the entire image as real or fake (producing a single scalar), PatchGAN classifies **overlapping patches** of the image. Each element in the output feature map corresponds to a local receptive field in the input, and the discriminator outputs a 2D grid of real/fake scores.

```
Concatenated Input: sketch + image (6×128×128)
    │
    ▼
┌──────────────────────────────────────────┐
│  Conv2d(6→64, 4×4, s2) + LeakyReLU(0.2) │  → 64×64×64    (no BatchNorm)
├──────────────────────────────────────────┤
│  Conv2d(64→128, 4×4, s2) + BN + LReLU   │  → 128×32×32
├──────────────────────────────────────────┤
│  Conv2d(128→256, 4×4, s2) + BN + LReLU  │  → 256×16×16
├──────────────────────────────────────────┤
│  Conv2d(256→512, 4×4, s1, p1) + BN+LReLU│  → 512×15×15
├──────────────────────────────────────────┤
│  Conv2d(512→1, 4×4, s1, p1)             │  → 1×14×14
└──────────────────────────────────────────┘
    │
    ▼
  Patch scores (14×14 map of real/fake predictions)
```

> **Why PatchGAN?** Full-image discriminators struggle with high-frequency detail — they tend to push the generator toward globally plausible but locally blurry outputs. PatchGAN focuses on **local texture quality** at the patch level, which complements the L1 loss (which handles global/low-frequency structure). The paper shows that a 70×70 patch is sufficient for high-quality results.

**Conditional discriminator:** The discriminator receives the **input sketch concatenated with the output image** (6 channels total). This is the "conditional" part — the discriminator doesn't just judge whether the image looks real, but whether it's a **plausible colourisation of the given sketch**.

### Model Size

| Component | Parameters |
|---|---|
| Generator (U-Net) | **41,828,995** |
| Discriminator (PatchGAN) | **2,768,705** |
| **Total** | **~44.6M** |

### Weight Initialisation

All `Conv2d` and `ConvTranspose2d` weights are initialised from `N(0, 0.02)`. BatchNorm weights from `N(1.0, 0.02)` with biases set to 0. This follows the DCGAN convention used in the Pix2Pix paper.

---

## Notebook Walkthrough

### 1 — Imports & Setup

Imports PyTorch, torchvision, PIL, matplotlib, and **scikit-image** (for SSIM and PSNR metrics). Sets `matplotlib` backend to `Agg` for non-interactive rendering on Kaggle. Enables `cudnn.benchmark = True` for optimised convolution performance.

### 2 — Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `IMG_SIZE` | **128** | Images resized to 128×128 |
| `BATCH_SIZE` | **16** | |
| `NUM_EPOCHS` | **100** | |
| `LR` | **0.0002** | Per the Pix2Pix paper |
| `BETAS` | **(0.5, 0.999)** | Adam momentum parameters |
| `L1_LAMBDA` | **100** | L1 reconstruction loss weight |
| `VIZ_EVERY` | **20** | Visualise results every N epochs |
| `SAVE_EVERY` | **5** | Save checkpoint every N epochs |

### 3 — Data Paths & Device

Configures dataset paths (`/kaggle/input/...`), checkpoint directory, and CUDA device. The dataset uses a common format for paired image-to-image translation: each image file contains the **colour** (left half) and **sketch** (right half) side-by-side.

### 4 — Dataset & DataLoaders

`AnimeSketchDataset` class:
- Opens each PNG file and splits it into left (colour/target) and right (sketch/input) halves.
- Applies transforms: `Resize(128)`, `ToTensor`, `Normalize([0.5]*3, [0.5]*3)` (maps to [-1, 1]).
- Returns `(sketch, colour)` pairs.

| Split | Samples |
|---|---|
| Train | **14,224** |
| Validation | **3,545** |
| Batches per epoch | **889** |

DataLoaders use `num_workers=2`, `pin_memory=True`, `drop_last=True` for training.

Includes a visualisation verification showing a sample sketch-colour pair.

### 5 — Generator Definition (U-Net)

Defines:
- `encoder_block()` — `Conv2d(4×4, stride 2)` + optional `BatchNorm` + `LeakyReLU(0.2)`.
- `decoder_block()` — `ConvTranspose2d(4×4, stride 2)` + `BatchNorm` + optional `Dropout(0.5)` + `ReLU`.
- `UNetGenerator` class with 6 encoder layers, a bottleneck, 6 decoder layers with skip connections, and a final `ConvTranspose2d + Tanh` output.

Includes a shape verification test confirming `(1, 3, 128, 128)` → `(1, 3, 128, 128)`.

**Generator parameters: 41,828,995**

### 6 — Discriminator Definition (PatchGAN)

`PatchGANDiscriminator` class:
- Input: concatenated `(sketch, image)` → 6 channels.
- 5-layer convolutional network producing a spatial map of patch-level real/fake scores.
- Output on 256×256 test input: `(1, 1, 30, 30)`.

**Discriminator parameters: 2,768,705**

### 7 — Model Setup & Loss Functions

- `DataParallel` wrapping for multi-GPU (2× T4).
- **Adversarial loss:** `BCEWithLogitsLoss` — standard cGAN objective.
- **Reconstruction loss:** `L1Loss` — pixel-wise absolute difference between generated and target images.
- **Optimizers:** Adam for both G and D with `lr=2e-4`, `betas=(0.5, 0.999)`.
- **Mixed precision:** `GradScaler` for both G and D.

### 8 — Checkpoint Resume from HuggingFace

Downloads a pre-trained checkpoint (`checkpoint_epoch_90.pth`) from HuggingFace Hub (`adeelumar17/pix2pix`) to resume training. Loads model weights, optimizer states, and loss history. Handles `DataParallel` module prefix stripping.

Resumes from **epoch 90**, continuing to epoch 100.

### 9 — Pre-Training Visualisation

Quick sanity check: generates a coloured image from a validation sketch using the loaded epoch-90 model. Displays triplet: **Input Sketch | Generated Colour | Ground Truth**.

### 10 — Training Loop

For each epoch (90 → 100):

**Discriminator step:**
1. Generate fake colour images from sketches.
2. Score real pairs `(sketch, real_colour)` → should be high.
3. Score fake pairs `(sketch, fake_colour.detach())` → should be low.
4. `D_loss = BCE(real_scores, 1) + BCE(fake_scores, 0)`.

**Generator step:**
1. Score fake pairs `(sketch, fake_colour)` through D → should be high.
2. `G_loss = BCE(fake_scores, 1) + λ · L1(fake_colour, real_colour)`.

Uses `autocast` + `GradScaler` for mixed-precision training. Tracks and logs D and G losses per epoch. Saves checkpoints every `SAVE_EVERY` epochs.

### 11 — Training Loss Curves

Plots Generator and Discriminator losses over all trained epochs.

### 12 — Qualitative Results Visualisation

`visualize_results()` function:
- Generates colorised outputs for 6 validation samples.
- Displays side-by-side grids: **Sketch | Generated | Ground Truth**.
- Denormalises from [-1, 1] to [0, 1] for display.

### 13 — Quantitative Evaluation (SSIM & PSNR)

`evaluate_model()` function:
- Computes **SSIM** (Structural Similarity Index) and **PSNR** (Peak Signal-to-Noise Ratio) between generated and ground-truth images across the validation set.
- Reports average scores.

### 14 — Final Model Save

Saves the final generator and discriminator weights (`generator_final.pth`, `discriminator_final.pth`) for deployment.

---

## Training Process

### Loss Functions

| Loss | Formula | Purpose |
|---|---|---|
| **Adversarial (D)** | `BCE(D(x, y), 1) + BCE(D(x, G(x)), 0)` | Train D to distinguish real from fake pairs |
| **Adversarial (G)** | `BCE(D(x, G(x)), 1)` | Train G to fool D into thinking fake pairs are real |
| **L1 Reconstruction** | `λ · ‖G(x) - y‖₁` | Train G to produce outputs close to ground truth |
| **Total G loss** | `L_adv + λ · L_L1` | Combined realism + accuracy objective |

### Training Details

| Aspect | Detail |
|---|---|
| **Optimizer** | Adam (`lr=2e-4`, `β₁=0.5`, `β₂=0.999`) for both G and D — per the Pix2Pix paper. |
| **L1 weight (λ)** | **100** — heavily weights reconstruction, ensuring outputs closely match ground truth. The adversarial loss adds realism/detail on top. |
| **Mixed precision** | `autocast` + `GradScaler` for both G and D — reduces memory usage and speeds up training on T4 GPUs. |
| **Multi-GPU** | `DataParallel` across 2× Tesla T4. |
| **D:G ratio** | **1:1** — one D update per G update per batch. |
| **Epochs** | **100** total (trained in stages with checkpoint resume). |
| **Checkpointing** | Every 5 epochs; full state (G, D, optimizers, loss history) pushed to HuggingFace Hub. |

### Training Dynamics

In Pix2Pix, the training dynamics differ from unconditional GANs:

- **D loss** reflects how well the discriminator can tell real from generated pairs. It should stabilise around `~1.0` (roughly 50% accuracy on each class at equilibrium) rather than dropping to zero.
- **G loss** includes the large L1 term, so it starts high and gradually decreases as the generator learns to produce accurate reconstructions.
- The **adversarial component** of G loss is relatively small compared to `λ · L1`, so training is more stable than in unconditional GANs — the L1 loss provides a strong, consistent gradient signal.

---

## Results & Analysis

### Qualitative Results

The model produces coloured anime faces from sketches. Results are shown as triplets:

| Column | Content |
|---|---|
| **Input Sketch** | Black-and-white line drawing |
| **Generated Colour** | Model's colourised output |
| **Ground Truth** | Original colour image |

**Observations:**
1. **Overall structure is well preserved** — the model correctly reproduces facial features, hair boundaries, and eye shapes from the sketch.
2. **Colours are plausible but not exact** — colourisation is inherently ambiguous (a sketch doesn't specify hair colour). The model learns the *distribution* of colours in the training set and produces typical anime colour schemes.
3. **Fine details** — eyes, highlights, and hair gradients are generally well-rendered, thanks to the PatchGAN focusing on local texture quality.
4. **Occasional artefacts** — some outputs show slight colour bleeding at boundaries or muted colours in complex regions.

### Quantitative Metrics

| Metric | Purpose |
|---|---|
| **SSIM** (Structural Similarity) | Measures structural similarity between generated and real images (1.0 = perfect) |
| **PSNR** (Peak Signal-to-Noise Ratio) | Measures pixel-level reconstruction quality in dB (higher = better) |

> **Note:** SSIM and PSNR measure similarity to a *specific* ground truth. For tasks like colourisation, where multiple outputs are equally valid, these metrics underestimate perceived quality. A generated image with different but equally plausible colours may score low on SSIM/PSNR but look perfectly fine to humans.

### Why Pix2Pix Works for This Task

1. **Paired data** — the dataset provides exact sketch-colour pairs, enabling supervised training with L1 loss. This is the key requirement for Pix2Pix (as opposed to unpaired methods like CycleGAN).

2. **U-Net preserves structure** — skip connections ensure that the spatial layout of the sketch is faithfully transferred to the coloured output. The generator doesn't need to "re-invent" the structure — it only needs to add colour.

3. **PatchGAN enforces local consistency** — instead of judging the whole image, the PatchGAN ensures that *local patches* look realistic. This is especially important for anime faces, where local textures (eyes, hair strands, skin shading) define visual quality.

4. **L1 loss prevents mode collapse** — with λ=100, the generator has a strong incentive to produce outputs close to the ground truth, preventing it from generating a single "average" colour for all inputs.

### Connection to the Original Paper's Results

The Pix2Pix paper demonstrates the framework on diverse tasks: aerial photos → maps, edges → shoes, BW → colour, day → night, etc. The anime sketch-to-colour task is a natural fit — it's a paired, pixel-aligned translation problem where:
- The input (sketch) contains structural information.
- The output (colour) adds appearance information.
- The mapping is **one-to-many** (multiple valid colourings exist), which the adversarial loss handles gracefully.

### Potential Improvements

- **Larger image size** — train at 256×256 (matching the paper) for finer detail, at the cost of more GPU memory.
- **Instance Normalisation** — the paper uses Instance Norm for some tasks; it may improve stylistic consistency.
- **Spectral Normalisation** — stabilises discriminator training by constraining its Lipschitz constant, potentially improving output quality.
- **Perceptual loss (VGG loss)** — add a feature-matching loss using a pre-trained VGG network for sharper, more perceptually pleasing outputs.
- **Multi-scale discriminator** — use discriminators at multiple resolutions for better global and local consistency (as in Pix2PixHD).
- **Augmentation** — apply random jittering (resize + random crop) and horizontal flips, as done in the original paper, to improve generalisation.
- **Longer training** — 100 epochs is a reasonable starting point, but more training may further refine colour accuracy and detail.

---

## How to Run

1. **Platform:** Upload the notebook to [Kaggle](https://www.kaggle.com/) and attach the [Anime Sketch Colorization Pair dataset](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair).
2. **GPU:** Enable **GPU T4 × 2** accelerator in the Kaggle notebook settings.
3. **Execute cells in order.** Training takes approximately 100 epochs (~15–20 min/epoch on 2× T4).
4. **Checkpoint resume:** Pre-trained weights can be downloaded from HuggingFace Hub (`adeelumar17/pix2pix`) to skip initial training and resume from epoch 90.

---

## Requirements

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Model definition, transforms, DataLoader |
| `torch.cuda.amp` | Mixed-precision training (`GradScaler`, `autocast`) |
| `PIL` (Pillow) | Image loading and processing |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisation (loss curves, result grids) |
| `scikit-image` (`skimage`) | SSIM and PSNR evaluation metrics |
| `huggingface_hub` | Checkpoint download/upload |
| `os`, `glob` | File path handling and discovery |

All dependencies are pre-installed in the default Kaggle Python 3 Docker image.

---

## References

1. **Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A.** (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR 2017. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004) — The foundational Pix2Pix paper defining the U-Net generator + PatchGAN discriminator + L1 loss framework.

2. **Goodfellow, I., et al.** (2014). *Generative Adversarial Nets.* NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) — The original GAN paper that Pix2Pix builds upon with conditional inputs.

3. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) — The U-Net architecture adopted as the Pix2Pix generator.

4. **Radford, A., Metz, L., & Chintala, S.** (2016). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* ICLR 2016. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434) — DCGAN: architectural conventions (weight init, BatchNorm, activation functions) used in Pix2Pix.

---

## License

This project is for educational purposes (Generative AI course — AI4009 Assignment 02). Feel free to use and adapt with attribution.
