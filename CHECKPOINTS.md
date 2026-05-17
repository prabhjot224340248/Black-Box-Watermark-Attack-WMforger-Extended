# Checkpoint Setup Guide

Model checkpoint files are **not included in this repository** (too large for GitHub).  
Download them and place them in the exact paths shown below.

---

## Required Folder Structure

After downloading, your `checkpoints/` folder must look like this:

```
checkpoints/
├── WMForger/
│   ├── convnext_pref_model.pth           (126 MB)
│   └── convnext_pref_model_finetuned.pth (126 MB — produced by finetune_preference_model.py)
├── CIN/
│   └── cinNet&nsmNet.pth                 (132 MB)
├── MBRS/
│   └── EC_114.pth                        (22 MB)
├── HiDDeN/
│   └── combined-noise--epoch-400.pyt     (5.6 MB)
└── TrustMark/
    (TrustMark downloads its models automatically — see below)
```

---

## Download Instructions

### 1. WMForger Preference Model (`convnext_pref_model.pth`)
- Source: [Meta Research — WMForger](https://github.com/facebookresearch/watermark-anything)
- Download the `convnext_pref_model.pth` from the WMForger model release
- Place at: `checkpoints/WMForger/convnext_pref_model.pth`
- The fine-tuned version (`convnext_pref_model_finetuned.pth`) is produced by running `finetune_preference_model.py`

### 2. CIN Checkpoint (`cinNet&nsmNet.pth`)
- Source: [CIN — Contrastive Invertible Networks](https://github.com/rmpku/CIN)
- Download the pretrained checkpoint from the CIN GitHub releases
- Place at: `checkpoints/CIN/cinNet&nsmNet.pth`

### 3. MBRS Checkpoint (`EC_114.pth`)
- Source: [MBRS — Mini-Batch Real and Simulated](https://github.com/jzyustc/MBRS)
- Download the `MBRS_Diffusion_128_m30` pretrained model (EC_114.pth, encoder-decoder)
- Place at: `checkpoints/MBRS/EC_114.pth`

### 4. HiDDeN Checkpoint (`combined-noise--epoch-400.pyt`)
- Source: [HiDDeN — Hiding Data with Deep Networks](https://github.com/jbuet/HiDDeN)
- Download the `combined-noise` experiment checkpoint at epoch 400
- Place at: `checkpoints/HiDDeN/combined-noise--epoch-400.pyt`
- The config file (`options-and-config.pickle`) is already included at:
  `victim_models/HiDDeN/experiments/combined-noise/options-and-config.pickle`

### 5. TrustMark (Automatic Download)
TrustMark downloads its model weights automatically on first use via its internal loader.
No manual download needed — just install the package:
```bash
pip install -e victim_models/TrustMark/
```
Models are cached locally after first run.

---

## Total Checkpoint Storage Required

| Model | File | Size |
|---|---|---|
| WMForger (original) | convnext_pref_model.pth | 126 MB |
| WMForger (fine-tuned) | convnext_pref_model_finetuned.pth | 126 MB |
| CIN | cinNet&nsmNet.pth | 132 MB |
| MBRS | EC_114.pth | 22 MB |
| HiDDeN | combined-noise--epoch-400.pyt | 5.6 MB |
| TrustMark | (auto-downloaded) | ~135 MB |
| **Total** | | **~547 MB** |

---

## Dataset

Experiments use **COCO val2017** (5,000 images).

```bash
# Download COCO val2017
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d dataset/
# Result: dataset/val2017/*.jpg
```
