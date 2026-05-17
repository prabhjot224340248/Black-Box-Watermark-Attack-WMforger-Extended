"""
TrustMark Watermarking Model — WMForger API Wrapper
=====================================================
Wraps TrustMark so WMForger can attack it as a victim watermarker.

TrustMark natively works with PIL images and bit strings.
This wrapper converts to/from WMForger's tensor interface.

WMForger calls:
    outputs = embedder.embed(imgs)          # imgs: [B,3,H,W] in [0,1]
    watermarked = outputs["imgs_w"]
    decoded = embedder.decode(imgs)         # returns [B, msg_bits] floats
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import warnings
warnings.filterwarnings("ignore")

# ── Bridge to TrustMark source ─────────────────────────────────────────────
_HERE_T = os.path.dirname(os.path.abspath(__file__))
_ROOT_T = os.path.dirname(_HERE_T)
TRUSTMARK_PATH = os.path.join(_ROOT_T, "victim_models", "TrustMark")
if TRUSTMARK_PATH not in sys.path:
    sys.path.insert(0, TRUSTMARK_PATH)

from trustmark import TrustMark as _TrustMark

# ── Configuration ──────────────────────────────────────────────────────────
MSG_BITS   = 100      # TrustMark checkpoint trained with 100-bit secrets — must match
MODEL_TYPE = 'P'      # P = highest visual quality variant (PSNR ~48 dB)


class TrustMarkEmbedder(nn.Module):
    """
    WMForger-compatible embedder for TrustMark.

    Key notes:
    - TrustMark uses PIL images natively (no raw tensors)
    - Payload is a bit string: '010110...' (string of '0'/'1' chars)
    - use_ECC=False gives exactly MSG_BITS user bits without error correction
    - Professor's PIL evaluation: decode from PIL (not raw float tensor)
    """

    def __init__(self, model_type=MODEL_TYPE):
        super().__init__()
        self.tm = _TrustMark(
            use_ECC     = False,        # No ECC — raw 100 bits matches checkpoint
            secret_len  = MSG_BITS,     # 100 bits — must match checkpoint training
            model_type  = model_type,
            verbose     = False,
            loadRemover = False,        # Don't load remover (not needed for attack)
        )
        self.msg_bits = MSG_BITS

    # ── Bit format helpers ─────────────────────────────────────────────────
    @staticmethod
    def _bits_to_str(bits):
        """[1,0,1,...] → '101...'  (TrustMark payload format)"""
        return ''.join(str(int(b)) for b in bits)

    @staticmethod
    def _str_to_arr(s, length):
        """'101...' → np.float32 array [1,0,1,...]"""
        arr = np.array([int(c) for c in s[:length]], dtype=np.float32)
        if len(arr) < length:
            arr = np.concatenate([arr, np.zeros(length - len(arr), dtype=np.float32)])
        return arr

    # ── PIL-level encode/decode ────────────────────────────────────────────
    def _encode_pil(self, pil_image, bits):
        """Encode bit array into PIL image → returns watermarked PIL image"""
        payload_str = self._bits_to_str(bits)
        wm_pil = self.tm.encode(pil_image, payload_str, MODE='binary')
        return wm_pil

    def _decode_pil(self, pil_image):
        """Decode PIL image → bit array (float32)"""
        secret_str, present, schema = self.tm.decode(pil_image)
        if not present or secret_str is None:
            return np.zeros(self.msg_bits, dtype=np.float32)
        return self._str_to_arr(secret_str, self.msg_bits)

    # ── Tensor-level encode/decode (for attack scripts) ────────────────────
    def _encode_(self, image_pil, message_bits):
        """
        Encode message into PIL image.
        Args:
            image_pil:    PIL Image (any size)
            message_bits: list/array of 0/1 ints, length = MSG_BITS
        Returns:
            tensor [1, 3, H, W] in [0,1]
        """
        wm_pil = self._encode_pil(image_pil, message_bits)
        return TF.to_tensor(wm_pil).unsqueeze(0).clamp(0, 1)

    def _decode_(self, img_tensor):
        """
        Decode watermark from tensor using PIL evaluation (professor's requirement).
        PIL evaluation: quantises float pixels to uint8 [0,255] before decoding —
        this simulates real-world image processing rather than perfect float tensors.

        Args:
            img_tensor: [1, 3, H, W] in [0,1]
        Returns:
            tensor [1, MSG_BITS] of float values (0 or 1)
        """
        pil  = TF.to_pil_image(img_tensor.squeeze(0).clamp(0, 1).cpu())
        bits = self._decode_pil(pil)
        return torch.tensor(bits, dtype=torch.float32).unsqueeze(0)

    # ── WMForger batch interface ────────────────────────────────────────────
    def embed(self, imgs):
        """
        WMForger interface: embed random watermarks into batch.
        Args:
            imgs: [B, 3, H, W] in [0,1]
        Returns:
            dict with key 'imgs_w': [B, 3, H, W]
        """
        results = []
        for i in range(imgs.shape[0]):
            msg    = np.random.randint(0, 2, self.msg_bits)
            pil    = TF.to_pil_image(imgs[i].clamp(0, 1).cpu())
            wm     = self._encode_(pil, msg)
            results.append(wm)
        imgs_w = torch.cat(results, dim=0).to(imgs.device)
        return {"imgs_w": imgs_w}

    def decode(self, imgs):
        """
        WMForger interface: decode watermarks from batch.
        Args:
            imgs: [B, 3, H, W] in [0,1]
        Returns:
            [B, MSG_BITS] float tensor
        """
        results = []
        for i in range(imgs.shape[0]):
            bits = self._decode_(imgs[i:i+1].cpu())
            results.append(bits)
        return torch.cat(results, dim=0).to(imgs.device)

    def forward(self, imgs):
        return self.embed(imgs)

    # ── nn.Module compatibility ────────────────────────────────────────────
    def parameters(self, recurse=True):
        return self.tm.encoder.parameters()

    def eval(self):
        self.tm.encoder.eval()
        self.tm.decoder.eval()
        return self

    def train(self, mode=True):
        return self

    def to(self, device):
        # TrustMark handles device internally
        return self


class TRUSTMARK_MODEL(nn.Module):
    """
    Full TrustMark model — top-level wrapper for attack scripts.
    Mirrors the CIN_MODEL interface for consistency across all attack scripts.

    Usage:
        tm = TRUSTMARK_MODEL()
        wm_tensor = tm._encode_(pil_image, message_bits)
        decoded   = tm._decode_(wm_tensor)
    """

    def __init__(self, model_type=MODEL_TYPE):
        super().__init__()
        self.embedder = TrustMarkEmbedder(model_type=model_type)
        self.msg_bits = MSG_BITS

    def _encode_(self, image_pil, message_bits):
        return self.embedder._encode_(image_pil, message_bits)

    def _decode_(self, img_tensor):
        return self.embedder._decode_(img_tensor)

    def parameters(self, recurse=True):
        return self.embedder.parameters()

    def eval(self):
        self.embedder.eval()
        return self

    def to(self, device):
        return self
