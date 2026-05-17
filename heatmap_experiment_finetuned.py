"""
Heatmap Experiment v2 — Optimizer × Learning Rate × Watermarking Model
=======================================================================
Upgrades over v1:
  • 500 COCO images per cell (was 50)
  • GPU batching (batch_size=8) for ~6x speed-up
  • 7 optimisers: SGD, Adam, AdamW, RMSprop, LBFGS, TGD, NAdam
  • 20 optimisation steps (was 30) — balanced for 500-image scale
  • TGD = Truncated Gradient Descent with L-inf ball projection
      (signed gradient step, perturbation clipped to [-eps, eps])
  • NAdam = Nesterov-Adam hybrid (torch.optim.NAdam)

Metric: Average Bit Error Rate (BER) after attack.
  BER = 0.0  → watermark fully intact  (attack failed)
  BER = 0.5  → watermark destroyed     (attack succeeded)

Estimated runtime: ~8-10 hours on RTX 3050 (run overnight)

Output:
  results/finetuned/
    ├── CIN_heatmap.png
    ├── TrustMark_heatmap.png
    ├── MBRS_heatmap.png
    ├── HiDDeN_heatmap.png
    ├── all_models_comparison.png
    ├── results.json
    └── progress.log

Usage:
    conda activate videoseal
    cd <repo_root>
    python heatmap_experiment_v2.py
"""

import os
import sys
import json
import time
import random
import builtins
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import omegaconf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wmforger.models import build_extractor

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH  = os.path.join(_REPO_ROOT, "dataset", "val2017")
PREF_MODEL    = os.path.join(_REPO_ROOT, "checkpoints", "WMForger", "convnext_pref_model_finetuned.pth")
OUTPUT_DIR    = os.path.join(_REPO_ROOT, "results", "finetuned")
EXTRACTOR_CFG = os.path.join(_REPO_ROOT, "configs", "extractor.yaml")

NUM_IMAGES    = 500          # images per cell
NUM_STEPS     = 20           # optimisation steps per image/batch
BATCH_SIZE    = 8            # images processed simultaneously on GPU
OPT_IMG_SIZE  = 256          # resolution fed to preference model
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# 7 optimisers — TGD and NAdam are new
OPTIMIZERS    = ["SGD", "Adam", "AdamW", "RMSprop", "LBFGS", "TGD", "NAdam"]
LEARN_RATES   = [0.001, 0.01, 0.05, 0.1, 0.5]

MODELS_CFG = {
    "CIN":       {"module": "wmforger.cin_model",       "cls": "CIN_MODEL",       "kwargs": {}},
    "TrustMark": {"module": "wmforger.trustmark_model", "cls": "TRUSTMARK_MODEL", "kwargs": {}},
    "MBRS":      {"module": "wmforger.mbrs_model",      "cls": "MBRS_MODEL",      "kwargs": {}},
    "HiDDeN":    {"module": "wmforger.hidden_model",    "cls": "HIDDEN_MODEL",    "kwargs": {}},
}

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING — writes to file even when conda run buffers stdout
# ─────────────────────────────────────────────────────────────────────────────
_LOG_FILE = None

def _log_print(*args, **kwargs):
    builtins.__orig_print__(*args, **kwargs)
    if _LOG_FILE:
        msg = " ".join(str(a) for a in args)
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def setup_logging(log_path):
    global _LOG_FILE
    _LOG_FILE = log_path
    open(log_path, "w").close()           # clear old log
    if not hasattr(builtins, "__orig_print__"):
        builtins.__orig_print__ = builtins.print
    builtins.print = _log_print

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_image_paths(dataset_path, n):
    all_imgs = sorted(Path(dataset_path).glob("*.jpg"))
    if not all_imgs:
        raise FileNotFoundError(f"No .jpg files found in {dataset_path}")
    random.seed(42)
    chosen = random.sample(all_imgs, min(n, len(all_imgs)))
    print(f"  Loaded {len(chosen)} images from {dataset_path}")
    return chosen

# ─────────────────────────────────────────────────────────────────────────────
#  PREFERENCE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_pref_model(ckpt_path, cfg_path, device):
    model_type      = "convnext_tiny"
    state_dict      = torch.load(ckpt_path, weights_only=True, map_location="cpu")["model"]
    extractor_params = omegaconf.OmegaConf.load(cfg_path)[model_type]
    model = build_extractor(model_type, extractor_params, img_size=OPT_IMG_SIZE, nbits=0)
    model.load_state_dict(state_dict)
    return model.eval().to(device)

# ─────────────────────────────────────────────────────────────────────────────
#  CORE ATTACK FUNCTION — supports batched tensors + all 7 optimisers
# ─────────────────────────────────────────────────────────────────────────────
def optimize_with(wm_tensor, pref_model, optimizer_name, lr, num_steps):
    """
    Adds a perturbation that maximises preference model score (removes watermark).

    Args:
        wm_tensor     : [N, 3, H, W] float32 in [0,1] — already on GPU
        pref_model    : ConvNeXt preference model on GPU
        optimizer_name: one of OPTIMIZERS
        lr            : learning rate
        num_steps     : gradient steps

    Returns:
        attacked [N, 3, H, W] float32 in [0,1], detached

    TGD — Truncated Gradient Descent:
        Uses the *sign* of the gradient (truncating magnitude to ±1),
        then projects the perturbation back into an L-inf ball of radius eps.
        This is the adversarial-ML standard (iterative FGSM / PGD variant).
        eps is set to lr × 10, capped at 0.3 (≈ 76/255 pixel budget).
    """
    param = nn.Parameter(torch.zeros_like(wm_tensor))

    # ── TGD: manual signed-gradient loop with L-inf projection ───────────────
    if optimizer_name == "TGD":
        eps = min(lr * 10, 0.3)
        for _ in range(num_steps):
            if param.grad is not None:
                param.grad.zero_()
            attacked = (wm_tensor + param).clamp(0, 1)
            loss = -pref_model(attacked).mean()
            loss.backward()
            with torch.no_grad():
                param.data -= lr * param.grad.sign()   # signed gradient step
                param.data.clamp_(-eps, eps)            # project to L-inf ball
        return (wm_tensor + param).clamp(0, 1).detach()

    # ── Standard optimisers ───────────────────────────────────────────────────
    if optimizer_name == "SGD":
        optim = torch.optim.SGD([param], lr=lr, momentum=0.9)
        steps = num_steps
    elif optimizer_name == "Adam":
        optim = torch.optim.Adam([param], lr=lr)
        steps = num_steps
    elif optimizer_name == "AdamW":
        optim = torch.optim.AdamW([param], lr=lr)
        steps = num_steps
    elif optimizer_name == "RMSprop":
        optim = torch.optim.RMSprop([param], lr=lr)
        steps = num_steps
    elif optimizer_name == "NAdam":
        optim = torch.optim.NAdam([param], lr=lr)
        steps = num_steps
    elif optimizer_name == "LBFGS":
        # LBFGS does multiple inner iterations per outer step
        optim = torch.optim.LBFGS([param], lr=lr, max_iter=4,
                                   line_search_fn="strong_wolfe")
        steps = max(1, num_steps // 4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if optimizer_name == "LBFGS":
        for _ in range(steps):
            def closure():
                optim.zero_grad()
                loss = -pref_model((wm_tensor + param).clamp(0, 1)).mean()
                loss.backward()
                return loss
            optim.step(closure)
    else:
        for _ in range(steps):
            optim.zero_grad()
            loss = -pref_model((wm_tensor + param).clamp(0, 1)).mean()
            loss.backward()
            optim.step()

    return (wm_tensor + param).clamp(0, 1).detach()

# ─────────────────────────────────────────────────────────────────────────────
#  PRE-COMPUTE WATERMARKS — done ONCE per model, reused across all 35 cells
# ─────────────────────────────────────────────────────────────────────────────
def precompute_watermarks(victim_model, image_paths, device):
    """
    Encodes all images once and caches (msg, wm_opt_tensor) on CPU.
    wm_opt_tensor is already resized to OPT_IMG_SIZE — ready to move to GPU.
    """
    cache = []
    print(f"  Pre-computing {len(image_paths)} watermarks ...", flush=True)
    t0 = time.time()
    for img_path in image_paths:
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        msg = np.random.randint(0, 2, victim_model.msg_bits).astype(np.float32)
        try:
            wm_tensor = victim_model._encode_(pil, msg).cpu()          # [1,3,H,W]
        except Exception as e:
            print(f"    [encode error] {e}")
            continue
        wm_opt = F.interpolate(wm_tensor, (OPT_IMG_SIZE, OPT_IMG_SIZE),
                               mode="bilinear", align_corners=False)   # [1,3,256,256] CPU
        cache.append((msg, wm_opt))
    print(f"  Done in {time.time()-t0:.1f}s  ({len(cache)} images OK)")
    return cache

# ─────────────────────────────────────────────────────────────────────────────
#  BATCHED CELL ATTACK — runs all 500 images through attack in batches of 8
# ─────────────────────────────────────────────────────────────────────────────
def attack_cell_batched(wm_cache, victim_model, pref_model,
                        optimizer_name, lr, device):
    """
    Attack all cached watermarked images in GPU batches.
    Returns list of BER values (one per image).

    Batching gives ~6x speed-up on RTX 3050 vs sequential processing:
      500 imgs sequential ≈ 750s/cell
      500 imgs batched(8) ≈  90s/cell
    """
    bers = []

    for batch_start in range(0, len(wm_cache), BATCH_SIZE):
        batch = wm_cache[batch_start : batch_start + BATCH_SIZE]
        msgs      = [item[0] for item in batch]
        wm_tensors = [item[1] for item in batch]           # list of [1,3,256,256]

        # Stack into [B, 3, 256, 256] and move to GPU
        batch_wm = torch.cat(wm_tensors, dim=0).to(device)  # [B,3,H,W]

        # ── Attack whole batch at once ────────────────────────────────────────
        with torch.enable_grad():
            attacked_batch = optimize_with(batch_wm, pref_model,
                                           optimizer_name, lr, NUM_STEPS)
        # attacked_batch: [B, 3, H, W]

        # ── Decode each image individually ────────────────────────────────────
        for msg, attacked in zip(msgs, attacked_batch.unbind(0)):
            attacked_cpu = attacked.unsqueeze(0).cpu()     # [1,3,H,W]
            try:
                decoded = victim_model._decode_(attacked_cpu)   # [1, msg_bits]
                decoded_bits = (decoded.squeeze(0).detach().numpy() > 0.5).astype(int)
                ber = float(np.mean(decoded_bits != msg))
                bers.append(ber)
            except Exception as e:
                print(f"    [decode error] {e}")

    return bers

# ─────────────────────────────────────────────────────────────────────────────
#  RUN FULL GRID FOR ONE VICTIM MODEL (7 opts × 5 LRs = 35 cells)
# ─────────────────────────────────────────────────────────────────────────────
def run_model_grid(model_name, victim_model, pref_model, image_paths, device):
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  |  {len(image_paths)} images  |  batch={BATCH_SIZE}")
    print(f"{'='*60}")

    wm_cache = precompute_watermarks(victim_model, image_paths, device)
    if not wm_cache:
        print("  ERROR: no images encoded.")
        return {}

    results = {opt: {str(lr): {"bers": [], "avg": 0.0}
                     for lr in LEARN_RATES}
               for opt in OPTIMIZERS}

    total_cells   = len(OPTIMIZERS) * len(LEARN_RATES)
    cell_idx      = 0
    t_model_start = time.time()

    for opt_name in OPTIMIZERS:
        for lr in LEARN_RATES:
            cell_idx += 1
            t0 = time.time()
            print(f"\n  [{cell_idx:2d}/{total_cells}] {opt_name:8s}  LR={lr:<6}  ",
                  end="", flush=True)

            bers = attack_cell_batched(wm_cache, victim_model, pref_model,
                                       opt_name, lr, device)

            avg_ber = float(np.mean(bers)) if bers else 0.0
            results[opt_name][str(lr)]["bers"] = [round(b, 4) for b in bers]
            results[opt_name][str(lr)]["avg"]  = round(avg_ber, 4)

            elapsed = time.time() - t0
            remaining_cells = total_cells - cell_idx
            eta_min = remaining_cells * elapsed / 60
            print(f"avg BER={avg_ber:.3f}  ({len(bers)} imgs, {elapsed:.1f}s, "
                  f"ETA this model: {eta_min:.0f}min)")

    total_time = time.time() - t_model_start
    print(f"\n  Done {model_name} in {total_time/60:.1f} min")
    return results

# ─────────────────────────────────────────────────────────────────────────────
#  HEATMAP PLOTTING — one per model
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmap(model_name, results, output_dir):
    matrix = np.zeros((len(OPTIMIZERS), len(LEARN_RATES)))
    for i, opt in enumerate(OPTIMIZERS):
        for j, lr in enumerate(LEARN_RATES):
            matrix[i, j] = results[opt][str(lr)]["avg"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix,
        xticklabels=[str(lr) for lr in LEARN_RATES],
        yticklabels=OPTIMIZERS,
        annot=True, fmt=".3f",
        cmap="YlOrRd", vmin=0.0, vmax=0.5,
        linewidths=0.5, linecolor="grey",
        ax=ax,
        cbar_kws={"label": "Average BER (↑ = better attack)"},
    )
    ax.set_title(
        f"{model_name} — Watermark Removal Rate\n"
        f"(BER: 0.0 = watermark intact | 0.5 = watermark destroyed)\n"
        f"{NUM_IMAGES} images × {NUM_STEPS} steps  |  batch={BATCH_SIZE}",
        fontsize=11, pad=12
    )
    ax.set_xlabel("Learning Rate", fontsize=11)
    ax.set_ylabel("Optimizer",     fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{model_name}_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return matrix

# ─────────────────────────────────────────────────────────────────────────────
#  COMBINED 4-PANEL FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(all_matrices, output_dir):
    model_names = list(all_matrices.keys())
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        matrix = all_matrices[name]
        sns.heatmap(
            matrix,
            xticklabels=[str(lr) for lr in LEARN_RATES],
            yticklabels=OPTIMIZERS,
            annot=True, fmt=".3f",
            cmap="YlOrRd", vmin=0.0, vmax=0.5,
            linewidths=0.4, linecolor="grey",
            ax=axes[idx],
            cbar_kws={"label": "Avg BER"},
        )
        axes[idx].set_title(f"{name}", fontsize=13, fontweight="bold")
        axes[idx].set_xlabel("Learning Rate", fontsize=10)
        axes[idx].set_ylabel("Optimizer",     fontsize=10)

    fig.suptitle(
        "WMForger Black-Box Attack v2: Optimizer × Learning Rate\n"
        f"7 optimisers incl. TGD & NAdam  |  {NUM_IMAGES} images × {NUM_STEPS} steps  |  batch={BATCH_SIZE}\n"
        "Average Bit Error Rate after attack (0.5 = watermark fully destroyed)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "all_models_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved combined figure: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(os.path.join(OUTPUT_DIR, "progress.log"))

    print("=" * 60)
    print("  HEATMAP EXPERIMENT — FINETUNED MODEL")
    print(f"  Started    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device     : {DEVICE}")
    print(f"  Images     : {NUM_IMAGES} per cell")
    print(f"  Steps      : {NUM_STEPS}")
    print(f"  Batch size : {BATCH_SIZE}  (GPU parallelism)")
    print(f"  Optimisers : {OPTIMIZERS}")
    print(f"  LRs        : {LEARN_RATES}")
    print(f"  Total cells: {len(OPTIMIZERS) * len(LEARN_RATES)} per model × 4 models")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Est. time  : ~8-10 hours on RTX 3050")
    print("=" * 60)

    print("\nLoading image paths ...")
    image_paths = load_image_paths(DATASET_PATH, NUM_IMAGES)

    print("Loading preference model ...")
    pref_model = load_pref_model(PREF_MODEL, EXTRACTOR_CFG, DEVICE)
    print("  Preference model ready.")

    all_results  = {}
    all_matrices = {}

    for model_name, cfg in MODELS_CFG.items():
        print(f"\nLoading victim model: {model_name} ...")
        try:
            import importlib
            mod    = importlib.import_module(cfg["module"])
            cls    = getattr(mod, cfg["cls"])
            victim = cls(**cfg["kwargs"])
            victim.to(DEVICE)
            victim.eval()
            print(f"  {model_name} loaded — {victim.msg_bits} bits")
        except Exception as e:
            print(f"  FAILED to load {model_name}: {e}")
            continue

        grid = run_model_grid(model_name, victim, pref_model, image_paths, DEVICE)
        all_results[model_name] = grid

        # Save after every model — safe against crashes
        results_path = os.path.join(OUTPUT_DIR, "results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved → {results_path}")

        matrix = plot_heatmap(model_name, grid, OUTPUT_DIR)
        all_matrices[model_name] = matrix.tolist()

        del victim
        torch.cuda.empty_cache()

    if len(all_matrices) >= 2:
        np_matrices = {k: np.array(v) for k, v in all_matrices.items()}
        plot_comparison(np_matrices, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"  EXPERIMENT COMPLETE — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
