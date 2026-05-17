"""
Fine-Tune Preference Model — Thesis Core Contribution
======================================================
Teaches the WMForger preference model (convnext_pref_model.pth) to
specifically recognise watermarks from CIN, TrustMark, MBRS, HiDDeN.

WHAT THIS DOES (thesis contribution):
  Original model: trained by Meta on VideoSeal watermarks only.
  Fine-tuned model: additionally trained on all 4 thesis watermarks.
  Result: forger attacks more accurately on all 4 models.

TRAINING LOGIC:
  Clean image        → model should output HIGH score (target = 1.0)
  Watermarked image  → model should output LOW  score (target = 0.0)
  Loss = MSE(prediction, target)

OUTPUT:
  convnext_pref_model_finetuned.pth  (drop-in replacement for original)

Usage:
    conda activate videoseal
    python finetune_preference_model.py
"""

import os, sys, time, random, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import omegaconf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wmforger.models import build_extractor

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH  = os.path.join(_REPO_ROOT, "dataset", "val2017")
PREF_CKPT     = os.path.join(_REPO_ROOT, "checkpoints", "WMForger", "convnext_pref_model.pth")
EXTRACTOR_CFG = os.path.join(_REPO_ROOT, "configs", "extractor.yaml")
SAVE_PATH     = os.path.join(_REPO_ROOT, "checkpoints", "WMForger", "convnext_pref_model_finetuned.pth")
LOG_PATH      = os.path.join(_REPO_ROOT, "finetune_log.txt")

NUM_IMAGES    = 300       # images per watermarking model (300×4=1200 watermarked + 300 clean)
IMG_SIZE      = 256       # preference model input size
BATCH_SIZE    = 16        # training batch size
EPOCHS        = 5         # fine-tuning epochs
LR            = 1e-5      # small LR — don't overwrite original knowledge
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_CFG = {
    "CIN":       {"module": "wmforger.cin_model",       "cls": "CIN_MODEL"},
    "TrustMark": {"module": "wmforger.trustmark_model", "cls": "TRUSTMARK_MODEL"},
    "MBRS":      {"module": "wmforger.mbrs_model",      "cls": "MBRS_MODEL"},
    "HiDDeN":    {"module": "wmforger.hidden_model",    "cls": "HIDDEN_MODEL"},
}

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def log(msg=""):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

open(LOG_PATH, "w").close()

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD PREFERENCE MODEL (trainable)
# ─────────────────────────────────────────────────────────────────────────────
def load_pref_model_trainable(ckpt_path, cfg_path, device):
    model_type       = "convnext_tiny"
    state_dict       = torch.load(ckpt_path, weights_only=True, map_location="cpu")["model"]
    extractor_params = omegaconf.OmegaConf.load(cfg_path)[model_type]
    model = build_extractor(model_type, extractor_params, img_size=IMG_SIZE, nbits=0)
    model.load_state_dict(state_dict)
    return model.train().to(device)   # train mode (not eval)

# ─────────────────────────────────────────────────────────────────────────────
#  GENERATE TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
def generate_training_data(image_paths, device):
    """
    Returns list of (tensor [1,3,256,256], label float) pairs.
    label = 1.0 for clean, 0.0 for watermarked.
    """
    data = []
    import importlib

    # ── Clean images (label = 1.0) ────────────────────────────────────────────
    log(f"\nGenerating CLEAN images ({len(image_paths)}) ...")
    for img_path in image_paths:
        try:
            pil    = Image.open(img_path).convert("RGB")
            tensor = torch.from_numpy(
                np.array(pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
            ).float().permute(2,0,1).unsqueeze(0)               # [1,3,H,W]
            data.append((tensor.cpu(), 1.0))
        except:
            continue
    log(f"  Clean samples: {len(data)}")

    # ── Watermarked images (label = 0.0) ─────────────────────────────────────
    for model_name, cfg in MODELS_CFG.items():
        log(f"\nGenerating WATERMARKED images — {model_name} ...")
        try:
            mod    = importlib.import_module(cfg["module"])
            cls    = getattr(mod, cfg["cls"])
            victim = cls()
            victim.to(device)
            victim.eval()
        except Exception as e:
            log(f"  FAILED to load {model_name}: {e}")
            continue

        count = 0
        for img_path in image_paths:
            try:
                pil = Image.open(img_path).convert("RGB")
                msg = np.random.randint(0, 2, victim.msg_bits).astype(np.float32)
                wm_tensor = victim._encode_(pil, msg).cpu()     # [1,3,H,W]
                wm_resized = F.interpolate(wm_tensor, (IMG_SIZE, IMG_SIZE),
                                           mode="bilinear", align_corners=False)
                data.append((wm_resized.cpu(), 0.0))
                count += 1
            except:
                continue

        log(f"  {model_name} watermarked samples: {count}")
        del victim
        torch.cuda.empty_cache()

    random.shuffle(data)
    log(f"\nTotal training samples: {len(data)}  "
        f"(clean={sum(1 for _,l in data if l==1.0)}, "
        f"watermarked={sum(1 for _,l in data if l==0.0)})")
    return data

# ─────────────────────────────────────────────────────────────────────────────
#  FINE-TUNE
# ─────────────────────────────────────────────────────────────────────────────
def finetune(pref_model, training_data, device):
    optimizer = torch.optim.Adam(pref_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.1
    )

    log(f"\n{'='*55}")
    log(f"  Fine-tuning starts — {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}")
    log(f"{'='*55}")

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(training_data)
        total_loss = 0.0
        correct    = 0
        total      = 0
        t0         = time.time()

        for i in range(0, len(training_data), BATCH_SIZE):
            batch = training_data[i : i + BATCH_SIZE]
            tensors = torch.cat([item[0] for item in batch], dim=0).to(device)
            labels  = torch.tensor([item[1] for item in batch],
                                   dtype=torch.float32).to(device)

            optimizer.zero_grad()
            preds = pref_model(tensors).squeeze()          # [B]
            loss  = F.mse_loss(torch.sigmoid(preds), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

            # Accuracy: sigmoid>0.5 = predict clean, <0.5 = predict watermarked
            predicted = (torch.sigmoid(preds) > 0.5).float()
            correct  += (predicted == labels).sum().item()
            total    += len(batch)

        scheduler.step()
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        elapsed  = time.time() - t0

        log(f"  Epoch [{epoch}/{EPOCHS}]  "
            f"Loss={avg_loss:.4f}  Acc={accuracy:.1f}%  "
            f"Time={elapsed:.1f}s  LR={scheduler.get_last_lr()[0]:.2e}")

    return pref_model

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log("=" * 55)
    log("  FINE-TUNING PREFERENCE MODEL — Thesis Contribution")
    log(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Device  : {DEVICE}")
    log(f"  Images  : {NUM_IMAGES} per model")
    log(f"  Epochs  : {EPOCHS}")
    log(f"  LR      : {LR}")
    log(f"  Goal    : Teach forger to recognise CIN/TrustMark/MBRS/HiDDeN watermarks")
    log("=" * 55)

    # Pick images
    all_imgs = sorted(Path(DATASET_PATH).glob("*.jpg"))
    random.seed(42)
    image_paths = random.sample(all_imgs, min(NUM_IMAGES, len(all_imgs)))
    log(f"\nSelected {len(image_paths)} COCO images (seed=42)")

    # Generate training data
    training_data = generate_training_data(image_paths, DEVICE)

    # Load preference model
    log("\nLoading preference model for fine-tuning ...")
    pref_model = load_pref_model_trainable(PREF_CKPT, EXTRACTOR_CFG, DEVICE)
    log("  Loaded.")

    # Fine-tune
    pref_model = finetune(pref_model, training_data, DEVICE)

    # Save fine-tuned checkpoint
    torch.save({"model": pref_model.state_dict()}, SAVE_PATH)
    log(f"\n  Fine-tuned model saved → {SAVE_PATH}")

    log("\n" + "=" * 55)
    log(f"  DONE — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Next: run heatmap_experiment_finetuned.py")
    log("=" * 55)

if __name__ == "__main__":
    main()
