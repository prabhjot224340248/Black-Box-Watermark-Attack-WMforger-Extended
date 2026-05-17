"""
Heatmap Experiment v4 — Four Targeted Improvements Over v2/Finetuned
=====================================================================
Built on top of heatmap_experiment_finetuned.py (same fine-tuned preference model).
All 4 improvements are motivated directly by the v2/finetuned experimental results.

WHAT WAS WRONG (from data analysis):
  LBFGS     : 10/20 cells failed, avg BER 0.185 — worst optimizer by far
  Zero init : First steps wasted finding direction (especially hurts low LR)
  Fixed LR  : LR=0.001 avg BER 0.168 vs LR=0.5 avg BER 0.542 — 3x gap
  Last iter : Optimizer overshoots at high LR; final param != best param

FOUR IMPROVEMENTS (Changes 4-7, following IMPROVEMENTS.md):
  Change 4: PGD Multi-Step Warm Start (5 steps, L-inf eps=0.03)
            Replaces zero init — gives principled adversarial starting point.
  Change 5: Replace LBFGS with Adadelta
            Adadelta adapts per-parameter, no LR sensitivity like LBFGS.
  Change 6: Cosine Annealing with Warm Restarts (SGDR, T_0=7)
            LR cycles high→low→high throughout 20 steps.
            Makes low LR cells effective; replaces simple cosine from v3.
  Change 7: Best-Iterate Tracking
            Return the iterate with highest preference score, not last.
            Prevents overshoot degradation at LR=0.5.

MODEL USED: convnext_pref_model_finetuned.pth (domain-fine-tuned, from thesis contribution)
OUTPUT:     results/v4/

Usage:
    conda activate videoseal
    cd <repo_root>
    python heatmap_experiment_v4.py
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

# v4 uses the FINE-TUNED preference model (thesis contribution)
PREF_MODEL    = os.path.join(_REPO_ROOT, "checkpoints", "WMForger", "convnext_pref_model_finetuned.pth")
OUTPUT_DIR    = os.path.join(_REPO_ROOT, "results", "v4")
EXTRACTOR_CFG = os.path.join(_REPO_ROOT, "configs", "extractor.yaml")

NUM_IMAGES    = 500          # images per cell  (same as v2 for fair comparison)
NUM_STEPS     = 20           # main optimisation steps per batch
BATCH_SIZE    = 8            # images processed simultaneously on GPU
OPT_IMG_SIZE  = 256          # resolution fed to preference model
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Change 5: LBFGS removed — replaced with Adadelta
OPTIMIZERS    = ["SGD", "Adam", "AdamW", "RMSprop", "Adadelta", "TGD", "NAdam"]
LEARN_RATES   = [0.001, 0.01, 0.05, 0.1, 0.5]

# Change 4: PGD warm-start hyperparameters
PGD_STEPS     = 5            # number of PGD initialisation steps
PGD_EPS       = 0.03         # L-inf ball radius for PGD warm-start (≈7/255 pixels)
PGD_LR        = 0.01         # step size for each PGD step

# Change 6: SGDR hyperparameters
SGDR_T0       = 7            # restart every T0 steps → 2 restarts in 20 steps
SGDR_TMULT    = 1            # keep same restart period (not doubling)
SGDR_ETA_MIN  = 0.0          # LR hits 0 at end of each cycle, not lr*0.01

MODELS_CFG = {
    "CIN":       {"module": "wmforger.cin_model",       "cls": "CIN_MODEL",       "kwargs": {}},
    "TrustMark": {"module": "wmforger.trustmark_model", "cls": "TRUSTMARK_MODEL", "kwargs": {}},
    "MBRS":      {"module": "wmforger.mbrs_model",      "cls": "MBRS_MODEL",      "kwargs": {}},
    "HiDDeN":    {"module": "wmforger.hidden_model",    "cls": "HIDDEN_MODEL",    "kwargs": {}},
}

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
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
    open(log_path, "w").close()
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
    random.seed(42)           # same seed as v2 — identical image set
    chosen = random.sample(all_imgs, min(n, len(all_imgs)))
    print(f"  Loaded {len(chosen)} images from {dataset_path}")
    return chosen

# ─────────────────────────────────────────────────────────────────────────────
#  PREFERENCE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_pref_model(ckpt_path, cfg_path, device):
    model_type       = "convnext_tiny"
    state_dict       = torch.load(ckpt_path, weights_only=True, map_location="cpu")["model"]
    extractor_params = omegaconf.OmegaConf.load(cfg_path)[model_type]
    model = build_extractor(model_type, extractor_params, img_size=OPT_IMG_SIZE, nbits=0)
    model.load_state_dict(state_dict)
    return model.eval().to(device)

# ─────────────────────────────────────────────────────────────────────────────
#  CHANGE 4: PGD MULTI-STEP WARM START
# ─────────────────────────────────────────────────────────────────────────────
def pgd_warm_start(param, wm_tensor, pref_model):
    """
    Run 5 PGD steps to initialise the perturbation before the main optimizer.

    Why PGD over FGSM (v3):
      FGSM = one signed-gradient step. It finds the direction but commits fully.
      PGD  = iterative signed steps with L-inf projection at each step.
             5 steps explore the adversarial neighbourhood much more carefully,
             landing in a region where the main optimizer can converge faster
             and to a better solution.

    eps = 0.03 (~7/255 pixels) — small enough to not dominate the final result,
    large enough to provide a meaningful warm-start signal.
    """
    for _ in range(PGD_STEPS):
        if param.grad is not None:
            param.grad.zero_()
        attacked = (wm_tensor + param).clamp(0, 1)
        loss = -pref_model(attacked).mean()
        loss.backward()
        with torch.no_grad():
            param.data -= PGD_LR * param.grad.sign()   # signed step (like FGSM)
            param.data.clamp_(-PGD_EPS, PGD_EPS)       # project back to L-inf ball
    # Clear grad so the main optimizer starts clean
    if param.grad is not None:
        param.grad.zero_()

# ─────────────────────────────────────────────────────────────────────────────
#  CORE ATTACK — v4: PGD init + SGDR scheduler + best-iterate tracking
# ─────────────────────────────────────────────────────────────────────────────
def optimize_with_v4(wm_tensor, pref_model, optimizer_name, lr, num_steps):
    """
    v4 improved attack function. Incorporates all 4 changes:

    Change 4 — PGD warm start:
        param initialised via 5 PGD steps (not zero, not single FGSM).

    Change 5 — Adadelta replaces LBFGS:
        LBFGS scored avg BER=0.185 with 10/20 cell failures.
        Adadelta adapts per-parameter using a window of accumulated gradients.
        It is essentially LR-insensitive, fixing the LBFGS failure mode.

    Change 6 — SGDR (Cosine Annealing Warm Restarts):
        LR follows cosine decay from configured value to 0 every T0=7 steps,
        then resets. With 20 main steps, this gives 2 restart cycles.
        Effect: even LR=0.001 cells periodically spike up to 0.001 (their cap),
        but crucially the cosine HIGH point is reached multiple times, making
        the attack more aggressive than a single monotone decay.

    Change 7 — Best-Iterate Tracking:
        After each step, record the preference model score. At the end, return
        the parameter state that scored highest, not the last state.
        Effect: at LR=0.5, optimisers often overshoot after step 15-18 and the
        final iterate is worse than the peak. This change captures the peak.
    """
    param = nn.Parameter(torch.zeros_like(wm_tensor))

    # ── Change 4: PGD warm-start before creating the main optimizer ──────────
    pgd_warm_start(param, wm_tensor, pref_model)

    # ── TGD: sign-gradient + L-inf projection (no scheduler, own schedule) ──
    if optimizer_name == "TGD":
        eps = min(lr * 10, 0.3)
        best_score = float("-inf")
        best_param = param.data.clone()

        for _ in range(num_steps):
            if param.grad is not None:
                param.grad.zero_()
            attacked = (wm_tensor + param).clamp(0, 1)
            score = pref_model(attacked).mean()
            loss = -score
            loss.backward()
            with torch.no_grad():
                param.data -= lr * param.grad.sign()
                param.data.clamp_(-eps, eps)
                # Change 7: track best iterate
                s = score.item()
                if s > best_score:
                    best_score = s
                    best_param = param.data.clone()

        return (wm_tensor + best_param).clamp(0, 1).detach()

    # ── Build main optimizer ─────────────────────────────────────────────────
    if optimizer_name == "SGD":
        optim = torch.optim.SGD([param], lr=lr, momentum=0.9)
    elif optimizer_name == "Adam":
        optim = torch.optim.Adam([param], lr=lr)
    elif optimizer_name == "AdamW":
        optim = torch.optim.AdamW([param], lr=lr)
    elif optimizer_name == "RMSprop":
        optim = torch.optim.RMSprop([param], lr=lr)
    elif optimizer_name == "Adadelta":
        # Change 5: replaces LBFGS
        # Adadelta is largely LR-insensitive (default rho=0.9, eps=1e-6)
        # We still pass lr so the grid stays comparable to other optimizers
        optim = torch.optim.Adadelta([param], lr=lr)
    elif optimizer_name == "NAdam":
        optim = torch.optim.NAdam([param], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # ── Change 6: SGDR scheduler ─────────────────────────────────────────────
    # CosineAnnealingWarmRestarts: LR = eta_min + (lr - eta_min) * 0.5 *
    #   (1 + cos(pi * T_cur / T_i))
    # With T_0=7, T_mult=1, the LR restarts every 7 steps.
    # Over 20 steps: restarts at step 7 and step 14 → 3 cosine segments.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim,
        T_0    = SGDR_T0,
        T_mult = SGDR_TMULT,
        eta_min = SGDR_ETA_MIN,
    )

    # ── Change 7: best-iterate tracking ─────────────────────────────────────
    best_score = float("-inf")
    best_param = param.data.clone()

    # ── Main optimisation loop ───────────────────────────────────────────────
    for step in range(num_steps):
        optim.zero_grad()
        attacked = (wm_tensor + param).clamp(0, 1)
        score    = pref_model(attacked).mean()
        loss     = -score
        loss.backward()
        optim.step()
        scheduler.step(step)        # SGDR uses absolute step count

        # Change 7: save best iterate by preference model score
        with torch.no_grad():
            s = score.item()
            if s > best_score:
                best_score = s
                best_param = param.data.clone()

    # Return BEST iterate, not last iterate
    return (wm_tensor + best_param).clamp(0, 1).detach()

# ─────────────────────────────────────────────────────────────────────────────
#  PRE-COMPUTE WATERMARKS — once per model, reused across all 35 cells
# ─────────────────────────────────────────────────────────────────────────────
def precompute_watermarks(victim_model, image_paths, device):
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
            wm_tensor = victim_model._encode_(pil, msg).cpu()
        except Exception as e:
            print(f"    [encode error] {e}")
            continue
        wm_opt = F.interpolate(wm_tensor, (OPT_IMG_SIZE, OPT_IMG_SIZE),
                               mode="bilinear", align_corners=False)
        cache.append((msg, wm_opt))
    print(f"  Done in {time.time()-t0:.1f}s  ({len(cache)} images OK)")
    return cache

# ─────────────────────────────────────────────────────────────────────────────
#  BATCHED CELL ATTACK
# ─────────────────────────────────────────────────────────────────────────────
def attack_cell_batched(wm_cache, victim_model, pref_model,
                        optimizer_name, lr, device):
    bers = []
    for batch_start in range(0, len(wm_cache), BATCH_SIZE):
        batch      = wm_cache[batch_start : batch_start + BATCH_SIZE]
        msgs       = [item[0] for item in batch]
        wm_tensors = [item[1] for item in batch]

        batch_wm = torch.cat(wm_tensors, dim=0).to(device)

        with torch.enable_grad():
            attacked_batch = optimize_with_v4(batch_wm, pref_model,
                                              optimizer_name, lr, NUM_STEPS)

        for msg, attacked in zip(msgs, attacked_batch.unbind(0)):
            attacked_cpu = attacked.unsqueeze(0).cpu()
            try:
                decoded      = victim_model._decode_(attacked_cpu)
                decoded_bits = (decoded.squeeze(0).detach().numpy() > 0.5).astype(int)
                ber          = float(np.mean(decoded_bits != msg))
                bers.append(ber)
            except Exception as e:
                print(f"    [decode error] {e}")
    return bers

# ─────────────────────────────────────────────────────────────────────────────
#  RUN FULL GRID FOR ONE VICTIM MODEL
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
            print(f"\n  [{cell_idx:2d}/{total_cells}] {opt_name:10s}  LR={lr:<6}  ",
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
#  HEATMAP PLOTTING
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
        cbar_kws={"label": "Average BER (higher = better attack)"},
    )
    ax.set_title(
        f"{model_name} — Watermark Removal Rate [v4: PGD+Adadelta+SGDR+BestIter]\n"
        f"(BER: 0.0 = watermark intact | 0.5 = watermark destroyed)\n"
        f"{NUM_IMAGES} images x {NUM_STEPS} steps  |  batch={BATCH_SIZE}",
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
    fig, axes   = plt.subplots(2, 2, figsize=(20, 12))
    axes        = axes.flatten()

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
        "WMForger v4: PGD Warm Start + Adadelta + SGDR + Best-Iterate Tracking\n"
        f"Fine-tuned preference model  |  {NUM_IMAGES} images x {NUM_STEPS} steps  |  batch={BATCH_SIZE}\n"
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
    print("  HEATMAP EXPERIMENT v4 — 4 Targeted Improvements")
    print(f"  Started    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device     : {DEVICE}")
    print(f"  Images     : {NUM_IMAGES} per cell")
    print(f"  Main steps : {NUM_STEPS}  (+{PGD_STEPS} PGD warm-start steps)")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Optimisers : {OPTIMIZERS}")
    print(f"  LRs        : {LEARN_RATES}")
    print(f"  PGD eps    : {PGD_EPS}  PGD steps: {PGD_STEPS}")
    print(f"  SGDR T0    : {SGDR_T0}  (restarts every {SGDR_T0} steps)")
    print(f"  Total cells: {len(OPTIMIZERS) * len(LEARN_RATES)} per model x 4 models")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Pref model : convnext_pref_model_finetuned.pth")
    print("  Changes    : [4] PGD init  [5] Adadelta  [6] SGDR  [7] BestIter")
    print(f"  Est. time  : ~12-14 hours on RTX 3050")
    print("=" * 60)

    print("\nLoading image paths ...")
    image_paths = load_image_paths(DATASET_PATH, NUM_IMAGES)

    print("Loading fine-tuned preference model ...")
    pref_model = load_pref_model(PREF_MODEL, EXTRACTOR_CFG, DEVICE)
    print("  Fine-tuned preference model ready.")

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
        print(f"  Results saved -> {results_path}")

        matrix = plot_heatmap(model_name, grid, OUTPUT_DIR)
        all_matrices[model_name] = matrix.tolist()

        del victim
        torch.cuda.empty_cache()

    if len(all_matrices) >= 2:
        np_matrices = {k: np.array(v) for k, v in all_matrices.items()}
        plot_comparison(np_matrices, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"  EXPERIMENT COMPLETE -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
