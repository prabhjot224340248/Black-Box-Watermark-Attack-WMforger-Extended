"""
Quick sanity test — load all 4 watermarking models and run one encode/decode.
Run this BEFORE building the heatmap to confirm all models work.

Usage:
    cd <repo_root>
    python test_all_models.py
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMAGE = "img1.jpg"   # must exist in current folder

# ─────────────────────────────────────────────────────────────────────────────
def compute_bit_accuracy(decoded_tensor, original_bits):
    """Compare decoded bits to original message. Returns accuracy 0-100%."""
    decoded_bits = (decoded_tensor.squeeze(0).cpu().numpy() > 0.5).astype(int)
    original     = np.array(original_bits)
    return float(np.mean(decoded_bits == original)) * 100.0

def run_test(name, model, pil_image):
    """Run one encode→decode cycle and report results."""
    print(f"\n{'─'*55}")
    print(f"  Testing: {name}")
    print(f"{'─'*55}")

    msg_bits = model.msg_bits
    msg      = np.random.randint(0, 2, msg_bits)
    print(f"  Message ({msg_bits} bits): {msg.tolist()}")

    # ── Encode ────────────────────────────────────────────────────────────
    try:
        wm_tensor = model._encode_(pil_image, msg)
        print(f"  Encode: ✓  output shape = {list(wm_tensor.shape)}")
    except Exception as e:
        print(f"  Encode: ✗  ERROR → {e}")
        return

    # ── Image quality ─────────────────────────────────────────────────────
    try:
        orig_tensor = TF.to_tensor(pil_image).unsqueeze(0)
        # Resize if sizes differ
        if orig_tensor.shape != wm_tensor.shape:
            import torchvision.transforms.functional as TFF
            wm_resized = TFF.resize(wm_tensor.squeeze(0),
                                    [orig_tensor.shape[2], orig_tensor.shape[3]])
            wm_resized = wm_resized.unsqueeze(0)
        else:
            wm_resized = wm_tensor

        mse  = torch.mean((orig_tensor - wm_resized.cpu()) ** 2).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        print(f"  PSNR  : {psnr:.2f} dB  (>35 dB = invisible watermark)")
    except Exception as e:
        print(f"  PSNR  : could not compute — {e}")

    # ── Decode ────────────────────────────────────────────────────────────
    try:
        decoded = model._decode_(wm_tensor.to(DEVICE))
        acc     = compute_bit_accuracy(decoded, msg)
        status  = "✓  PASS" if acc > 80 else "✗  FAIL"
        print(f"  Decode: {status}  →  Bit accuracy = {acc:.1f}%  (>80% = success)")
    except Exception as e:
        print(f"  Decode: ✗  ERROR → {e}")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  ALL WATERMARKING MODEL SANITY TEST")
    print("=" * 55)
    print(f"  Device : {DEVICE}")
    print(f"  Image  : {TEST_IMAGE}")

    if not os.path.exists(TEST_IMAGE):
        print(f"\n  ERROR: {TEST_IMAGE} not found in current folder.")
        print("  Please run from the repository root directory.")
        sys.exit(1)

    pil_image = Image.open(TEST_IMAGE).convert("RGB")
    print(f"  Size   : {pil_image.size}")

    # ── CIN ───────────────────────────────────────────────────────────────
    print("\n[1/4] Loading CIN...")
    try:
        from wmforger.cin_model import CIN_MODEL
        CHECKPOINT_CIN = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "checkpoints", "CIN", "cinNet&nsmNet.pth"
        )
        cin = CIN_MODEL(checkpoint_path=CHECKPOINT_CIN)
        cin.to(DEVICE).eval()
        run_test("CIN", cin, pil_image)
    except Exception as e:
        print(f"  CIN load failed: {e}")

    # ── TrustMark ─────────────────────────────────────────────────────────
    print("\n[2/4] Loading TrustMark...")
    try:
        from wmforger.trustmark_model import TRUSTMARK_MODEL
        tm = TRUSTMARK_MODEL()
        run_test("TrustMark", tm, pil_image)
    except Exception as e:
        print(f"  TrustMark load failed: {e}")

    # ── MBRS ──────────────────────────────────────────────────────────────
    print("\n[3/4] Loading MBRS...")
    try:
        from wmforger.mbrs_model import MBRS_MODEL
        mbrs = MBRS_MODEL()
        mbrs.to(DEVICE).eval()
        run_test("MBRS", mbrs, pil_image)
    except Exception as e:
        print(f"  MBRS load failed: {e}")

    # ── HiDDeN ────────────────────────────────────────────────────────────
    print("\n[4/4] Loading HiDDeN...")
    try:
        from wmforger.hidden_model import HIDDEN_MODEL
        hidden = HIDDEN_MODEL()
        hidden.to(DEVICE).eval()
        run_test("HiDDeN", hidden, pil_image)
    except Exception as e:
        print(f"  HiDDeN load failed: {e}")

    print(f"\n{'=' * 55}")
    print("  Test complete. Fix any ✗ errors before running heatmap.")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
