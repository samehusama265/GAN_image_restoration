# 🔬 image-restoration-sr

> AI-based image restoration and super-resolution using CNN autoencoders and SRGAN.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat)

---

## Abstract

This project restores degraded images and enhances resolution using deep convolutional networks and GAN-based super-resolution models. The full pipeline processes noisy low-resolution inputs through a U-Net CNN autoencoder denoiser, then applies SRGAN to reconstruct high-resolution outputs — validated on two datasets: DIV2K (natural scenes) and CelebA (face images).

---

## Pipeline

```
Noisy LR Input → CNN Denoiser (U-Net) → SRGAN ×4 → HR Output (256×256)
```

---

## Key Contributions

- **Image denoising & artifact removal** — U-Net style CNN autoencoder with skip connections removes Gaussian noise (σ = 25/255) from degraded low-resolution inputs.
- **High-resolution reconstruction** — SRGAN with 12 residual blocks and sub-pixel (PixelShuffle) upsampling reconstructs 64×64 → 256×256 images (×4 scale).
- **Perceptual quality enhancement** — VGG19-based perceptual loss (relu3_4 features) combined with adversarial and pixel (MSE) losses for photorealistic output.
- **Dual-domain evaluation** — validated independently on 800 DIV2K natural scene images and 5,000 CelebA face images.

---

## Methodology

1. **CNN Denoising Autoencoder** — U-Net encoder-decoder (channels: 3→32→64→128→256 bottleneck) with MaxPool downsampling and ConvTranspose2d upsampling; skip connections preserve spatial detail; trained with MSE loss and `CosineAnnealingLR` scheduler.
2. **SRGAN** — Generator with 12 residual blocks + 2× PixelShuffle upsamplers; Discriminator with 8 strided Conv→BN→LeakyReLU blocks; adversarial training with `BCEWithLogitsLoss`.
3. **Combined Loss** — `G_loss = 1.0 × pixel_loss + 1.0 × perceptual_loss + 1e-3 × adversarial_loss`; mixed-precision training (AMP) with gradient clipping (`max_norm=1.0`).

---

## Configuration

### Shared (both datasets)

| Parameter | Value |
|-----------|-------|
| LR input size | 64 × 64 |
| HR output size | 256 × 256 |
| Scale factor | ×4 |
| Noise std (σ) | 25 / 255 |
| Random seed | 42 |

### Denoising Autoencoder

| Parameter | DIV2K | CelebA |
|-----------|-------|--------|
| Epochs | 30 | 30 |
| Learning rate | 1e-3 | 1e-3 |
| Batch size | 8 | 4 |
| Optimizer | Adam | Adam |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |

### SRGAN

| Parameter | DIV2K | CelebA |
|-----------|-------|--------|
| Epochs | 50 | 50 |
| Generator LR | 1e-4 | 1e-4 |
| Discriminator LR | 1e-4 | 1e-4 |
| Batch size | 8 | 4 |
| Residual blocks | 16 | 12 |
| Perceptual weight | 1.0 | 1.0 |
| Adversarial weight | 1e-3 | 1e-3 |
| Pixel weight | 1.0 | 1.0 |

> CelebA uses smaller batch sizes (4 vs 8) and fewer residual blocks (12 vs 16) due to 256×256 VGG19 perceptual loss being more VRAM-intensive on face crops.

---

## Datasets

| Dataset | Size used | Description |
|---------|-----------|-------------|
| **DIV2K** | 800 images | High-resolution natural scenes; industry-standard SR benchmark |
| **CelebA** | 5,000 images | Aligned face images (202,599 total); capped for fast iteration |

---

## Sample Results

### 🧹 Denoising — Noisy Input | Ground Truth | Restored

![Denoising Results](results/output.png)

---

### ⬆️ SRGAN Super-Resolution — Low-res | HR Ground Truth | SRGAN Output

![SRGAN Output CelebA](results/output_SRGAN.png)

![SRGAN Output](results/SRGAN_output.png)

---

## Technologies

`Python 3.10+` · `PyTorch` · `torchvision` · `OpenCV` · `NumPy` · `Matplotlib` · `Pillow`

---

## Quick Start

```bash
# clone & install
git clone https://github.com/your-username/image-restoration-sr
cd image-restoration-sr
pip install -r requirements.txt

# train denoiser (DIV2K)
python train_denoiser.py --dataset div2k --epochs 30 --batch 8 --lr 1e-3

# train SRGAN (DIV2K)
python train_srgan.py --dataset div2k --epochs 50 --batch 8 --lr_g 1e-4 --lr_d 1e-4

# train on CelebA
python train_denoiser.py --dataset celeba --epochs 30 --batch 4
python train_srgan.py --dataset celeba --epochs 50 --batch 4 --res_blocks 12

# run full pipeline on a single image
python restore.py --input your_image.jpg --output restored.png
```

---

## Project Structure

```
image-restoration-sr/
├── data/
│   ├── denoise_dataset.py     # noisy/clean pair generator
│   └── sr_dataset.py          # LR/HR bicubic pair generator
├── models/
│   ├── autoencoder.py         # U-Net denoising autoencoder (base=32)
│   ├── generator.py           # SRGAN generator (12–16 residual blocks)
│   └── discriminator.py       # SRGAN discriminator
├── losses/
│   ├── perceptual.py          # VGG19 feature loss (relu3_4)
│   └── combined.py            # pixel + perceptual + adversarial
├── results/
│   ├── output.png             # denoising grid
│   ├── output_SRGAN.png       # SRGAN CelebA results
│   └── SRGAN_output.png       # SRGAN additional results
├── checkpoints/               # saved .pth files per epoch
├── train_denoiser.py
├── train_srgan.py
└── restore.py                 # end-to-end inference script
```

---

## License

MIT
