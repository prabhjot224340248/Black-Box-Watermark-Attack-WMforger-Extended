# WMForger Attack Improvements — Thesis Evidence Log
**Author:** Prabhjot Singh | SIT724 | Deakin University  
**Baseline:** heatmap_experiment_v2.py (500 images, 7 optimizers, fixed LR, zero init)

---

## Change 1 — FGSM Warm Start
**File:** optimize_image_improved.py, heatmap_experiment_v3.py  
**What changed:**  
Original code initialises the perturbation as all zeros:
```python
# BEFORE (v2)
param = nn.Parameter(torch.zeros_like(wm_tensor))
```
New code does one fast FGSM (Fast Gradient Sign Method) step first,
so the perturbation starts already pointing in the right direction:
```python
# AFTER (v3)
param = nn.Parameter(torch.zeros_like(wm_tensor))
# Warm-start: one FGSM step before main loop
loss = -pref_model((wm_tensor + param).clamp(0, 1)).mean()
loss.backward()
with torch.no_grad():
    param.data = (lr * param.grad.sign())
param.grad.zero_()
```
**Why it improves the attack:**  
Zero initialisation means the first several steps are wasted finding the
right direction. FGSM gives the optimizer a head start — it converges
faster and reaches a better final BER in the same number of steps.

**Expected improvement:** Higher BER in same 20 steps, especially for
models where the watermark is subtle (CIN, MBRS).

---

## Change 2 — Cosine Learning Rate Scheduling
**File:** optimize_image_improved.py, heatmap_experiment_v3.py  
**What changed:**  
Original code uses a fixed learning rate for all steps:
```python
# BEFORE (v2)
optim = torch.optim.Adam([param], lr=0.05)
# LR stays 0.05 from step 1 to step 20
```
New code uses cosine annealing — LR starts at the configured value and
smoothly decays to near-zero by the final step:
```python
# AFTER (v3)
optim = torch.optim.Adam([param], lr=0.05)
scheduler = CosineAnnealingLR(optim, T_max=num_steps, eta_min=lr * 0.01)
# LR: 0.05 → 0.04 → 0.03 → ... → 0.0005  (smooth decay)
```
**Why it improves the attack:**  
High LR early = explore aggressively and escape the watermark basin.  
Low LR late  = fine-tune precisely without overshooting the optimum.  
Our v2 heatmap showed that fixed high LR can overshoot (BER drops at LR=0.5
for some models), and fixed low LR barely moves. Cosine scheduling gets the
best of both.

**Expected improvement:** More consistent BER across LR values — the
"sweet spot" becomes wider and less sensitive to LR choice.

---

## Change 3 — Ensemble Attack (Best-of-3)
**File:** optimize_image_improved.py, heatmap_experiment_v3.py  
**What changed:**  
Original code attacks with one optimizer:
```python
# BEFORE (v2)
attacked = optimize_with(wm_tensor, pref_model, "Adam", lr, steps)
```
New code attacks with the 3 best-performing optimizers from v2
(RMSprop, Adam, NAdam), measures the preference model score for each,
and keeps the best result:
```python
# AFTER (v3) — Ensemble row in heatmap
best_attacked = None
best_score = -inf
for opt_name in ["RMSprop", "Adam", "NAdam"]:
    attacked = optimize_with_improved(wm_tensor, pref_model, opt_name, lr, steps)
    score = pref_model(attacked).mean().item()
    if score > best_score:
        best_score = score
        best_attacked = attacked
```
**Why it improves the attack:**  
Different optimizers find different perturbations. By running 3 and
keeping the strongest, we reduce the chance of getting stuck in a
bad local minimum. This is especially useful for robust models like MBRS
where a single optimizer sometimes fails entirely.

**Expected improvement:** Ensemble row should show higher/more consistent
BER than any single optimizer, especially on MBRS and CIN.

---

## Summary of Changes (v3)

| # | Change | Baseline | Improved |
|---|--------|----------|---------|
| 1 | Perturbation init | zeros | FGSM warm start |
| 2 | Learning rate | fixed | cosine decay |
| 3 | Attack strategy | single optimizer | ensemble of best 3 |

## Files Changed (v3)
- `optimize_image_improved.py` — standalone improved attack function
- `heatmap_experiment_v3.py` — full grid experiment using all 3 improvements

## How to Verify (for thesis)
Compare `heatmap_results_v2/` vs `heatmap_results_v3/`:
- Higher average BER = better attack = improvements work
- More consistent (less LR-sensitive) heatmap = more practical attack

---

# WMForger Attack Improvements — Round 2 (v4)
**Motivated by:** Data analysis of heatmap_results_finetuned/ experiment
**Key findings that drove changes:**
- LBFGS: avg BER=0.185, 10/20 cells failed — worst optimizer
- SGD: avg BER=0.195, 9/20 cells failed
- LR=0.001 avg BER=0.168 vs LR=0.5 avg BER=0.542 — 3x gap
- Optimizer frequently overshoots at LR=0.5 (last iterate ≠ best iterate)

---

## Change 4 — PGD Multi-Step Warm Start
**File:** heatmap_experiment_v4.py
**What changed:**
FGSM warm start (v3) used one signed-gradient step:
```python
# BEFORE (v3 — single FGSM step)
loss = -pref_model((wm_tensor + param).clamp(0, 1)).mean()
loss.backward()
with torch.no_grad():
    param.data = (lr * param.grad.sign())
```
PGD warm start (v4) uses 5 iterative steps with L-inf projection:
```python
# AFTER (v4 — 5 PGD steps, eps=0.03)
PGD_STEPS = 5
PGD_EPS   = 0.03   # L-inf ball (~7/255 pixels)
PGD_LR    = 0.01

for _ in range(PGD_STEPS):
    param.grad.zero_()
    loss = -pref_model((wm_tensor + param).clamp(0,1)).mean()
    loss.backward()
    with torch.no_grad():
        param.data -= PGD_LR * param.grad.sign()
        param.data.clamp_(-PGD_EPS, PGD_EPS)   # project to L-inf ball
```
**Why it improves the attack:**
PGD (Projected Gradient Descent) is the gold standard adversarial initialisation
from adversarial ML research (Madry et al., 2018). Unlike single-step FGSM which
commits to one direction, PGD iteratively refines the starting point while staying
within a small adversarial ball. This gives the main optimizer a principled head
start in the correct adversarial region, especially critical for low LR cells where
the optimizer has limited budget to find the right direction.

**Expected improvement:** Lower failure rate at LR=0.001 and LR=0.01.

---

## Change 5 — Replace LBFGS with Adadelta
**File:** heatmap_experiment_v4.py
**What changed:**
```python
# BEFORE (v2/finetuned)
OPTIMIZERS = ["SGD", "Adam", "AdamW", "RMSprop", "LBFGS", "TGD", "NAdam"]

# AFTER (v4)
OPTIMIZERS = ["SGD", "Adam", "AdamW", "RMSprop", "Adadelta", "TGD", "NAdam"]
```
```python
# Adadelta implementation
optim = torch.optim.Adadelta([param], lr=lr)
# Adadelta accumulates squared gradients over a decaying window (rho=0.9)
# and divides by accumulated squared updates — adapts per-parameter
```
**Why it improves the attack:**
LBFGS showed avg BER=0.185 with 10/20 cell failures. For CIN and MBRS it
produced BER≈0.000 across all LRs — it failed entirely on the most important
watermarking models. LBFGS requires a closure and uses line search, which
does not work well with batch gradient estimates (our batches of 8 images give
noisy gradients that violate LBFGS assumptions).

Adadelta (Zeiler, 2012) adapts the learning rate per-parameter using a window
of accumulated squared gradients. Unlike LBFGS, it handles noisy gradients
robustly and doesn't require careful LR tuning (it's largely LR-insensitive
by design). Expected to perform similarly to Adam/RMSprop.

**Expected improvement:** Adadelta cells should achieve BER ≥ 0.3 vs LBFGS ≈ 0.0.

---

## Change 6 — Cosine Annealing with Warm Restarts (SGDR)
**File:** heatmap_experiment_v4.py
**What changed:**
v2/finetuned had fixed LR:
```python
# BEFORE (v2 — fixed LR for all 20 steps)
optim = torch.optim.Adam([param], lr=0.001)
# LR stays at 0.001 from step 1 to step 20 — barely moves
```
v4 uses CosineAnnealingWarmRestarts (Loshchilov & Hutter, SGDR, 2017):
```python
# AFTER (v4 — LR cycles every 7 steps, 2 restarts in 20 steps)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optim, T_0=7, T_mult=1, eta_min=0.0
)
# Step  0-6:  LR: 0.001 -> 0.0 (cosine decay)
# Step  7:    LR resets to 0.001 (warm restart)
# Step  7-13: LR: 0.001 -> 0.0 (cosine decay again)
# Step 14:    LR resets to 0.001 (second restart)
# Step 14-20: LR: 0.001 -> 0.0
```
**Why it improves the attack:**
The core problem: LR=0.001 cells barely moved in v2 (avg BER=0.168 vs 0.542 for LR=0.5).
With SGDR, even LR=0.001 cells cycle through aggressive → conservative phases.
The warm restarts allow the optimizer to escape local minima multiple times,
making the low-LR regime much more competitive. This directly addresses the
3x BER gap between low and high LR that we observed in the data.

Note: This replaces the simple cosine decay from v3. SGDR is strictly better
because warm restarts allow multiple aggressive exploration phases.

**Expected improvement:** Low LR cells (0.001, 0.01) should improve significantly.
The BER gap between LR=0.001 and LR=0.5 should narrow.

---

## Change 7 — Best-Iterate Tracking
**File:** heatmap_experiment_v4.py
**What changed:**
v2/finetuned always returned the LAST iterate:
```python
# BEFORE (v2 — returns whatever state optimizer ends at)
for step in range(num_steps):
    optim.zero_grad()
    loss = -pref_model((wm_tensor + param).clamp(0,1)).mean()
    loss.backward()
    optim.step()
return (wm_tensor + param).clamp(0, 1).detach()  # last iterate only
```
v4 tracks the BEST iterate by preference model score:
```python
# AFTER (v4 — saves best scoring iterate throughout optimization)
best_score = float("-inf")
best_param = param.data.clone()

for step in range(num_steps):
    optim.zero_grad()
    attacked = (wm_tensor + param).clamp(0, 1)
    score    = pref_model(attacked).mean()   # score computed anyway for loss
    loss     = -score
    loss.backward()
    optim.step()
    scheduler.step(step)

    # No extra forward pass — score already computed above
    with torch.no_grad():
        if score.item() > best_score:
            best_score = score.item()
            best_param = param.data.clone()

return (wm_tensor + best_param).clamp(0, 1).detach()  # BEST iterate
```
**Why it improves the attack:**
At high LR (0.5), optimizers frequently overshoot. The preference score peaks
around step 12-15 then drops as the perturbation becomes too large and
the attacked image no longer looks "clean" to the preference model. Returning
the last iterate misses the peak. Best-iterate tracking captures the peak
automatically with zero extra computation (score is already computed for loss).
This is analogous to model checkpointing in standard training.

**Expected improvement:** Consistent improvement at LR=0.5 where overshoot
is most common. RMSprop, Adam, NAdam at LR=0.5 should improve.

---

## Full Summary of All Changes (v2 -> v3 -> v4)

| # | Change | v2 Baseline | v3 | v4 |
|---|--------|-------------|----|----|
| 1 | Init | zeros | FGSM (1 step) | **PGD (5 steps)** |
| 2 | LR schedule | fixed | cosine decay | **SGDR warm restarts** |
| 3 | Multi-optimizer | single | ensemble top-3 | — (not used in v4) |
| 4 | Warm start | zeros | FGSM | **PGD multi-step** |
| 5 | Optimizer set | LBFGS included | LBFGS included | **Adadelta replaces LBFGS** |
| 6 | LR schedule | fixed | simple cosine | **SGDR (T0=7, warm restarts)** |
| 7 | Return value | last iterate | last iterate | **best iterate by score** |

## Files Changed (v4)
- `heatmap_experiment_v4.py` — full grid using fine-tuned model + all 4 v4 improvements

## How to Verify (for thesis)
Compare `heatmap_results_finetuned/` vs `heatmap_results_v4/`:
- Lower failure rate (cells with BER < 0.1) = warm start + SGDR working
- Adadelta row BER >> LBFGS row BER = replacement successful
- Low LR cells (0.001, 0.01) showing higher BER = SGDR working
- High LR cells (0.5) showing higher BER = best-iterate tracking working
