"""
CIN Watermarking Model - WMForger API Wrapper
=============================================
Wraps the CIN (Contrastive Invertible Network) watermarking model so that
WMForger can use it as an embedder (watermarker to attack).

WMForger calls:
    outputs = embedder.embed(imgs)
    watermarked_images = outputs["imgs_w"]

So this class must:
    1. Be a torch.nn.Module (so WMForger can call .parameters(), .eval(), .to(device))
    2. Have an embed() method that accepts a batch tensor [N, 3, H, W] in [0,1]
       and returns {"imgs_w": watermarked_tensor} also [N, 3, H, W] in [0,1]
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# ── Config for CIN ──────────────────────────────────────────────────────────
from wmforger.config import CIN_OPT

# ── Bridge to CIN source code (relative to repo root) ────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))   # wmforger/
_ROOT         = os.path.dirname(_HERE)                        # repo root
CIN_CODE_PATH = os.path.join(_ROOT, "victim_models", "CIN", "codes")
if CIN_CODE_PATH not in sys.path:
    sys.path.insert(0, CIN_CODE_PATH)

from models.CIN import CIN


# ── CIN message length (from config) ──────────────────────────────────────
MSG_LENGTH = 30   # 30-bit binary messages
CIN_SIZE   = 128  # CIN only works on 128x128 images


class CINEmbedder(nn.Module):
    """
    WMForger-compatible wrapper for the CIN watermarking model.

    Usage (how WMForger uses it):
        embedder = CINEmbedder().to(device)
        embedder.eval()
        outputs = embedder.embed(imgs)        # imgs: [N,3,H,W] in [0,1]
        watermarked = outputs["imgs_w"]       # same shape, in [0,1]
    """

    def __init__(
        self,
        checkpoint_path: str = None   # defaults to checkpoints/CIN/cinNet&nsmNet.pth
    ):
        super().__init__()
        if checkpoint_path is None:
            checkpoint_path = os.path.join(_ROOT, "checkpoints", "CIN", "cinNet&nsmNet.pth")

        # Build the CIN network and register it as a submodule
        # (registering as submodule means .to(device) and .parameters() work automatically)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CIN(CIN_OPT, _device)

        # Load pretrained weights
        # The .pth file stores weights under the key "cinNet"
        # The checkpoint was saved with DataParallel (multi-GPU), so every key
        # has a "module." prefix — strip it before loading on single GPU/CPU.
        from collections import OrderedDict
        checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
        raw_state_dict = checkpoint["cinNet"]
        clean_state_dict = OrderedDict()
        for k, v in raw_state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k  # strip "module."
            clean_state_dict[new_key] = v
        # strict=False: skip the one mismatched noise_model key (Rotation param).
        # The noise model is only used during training, not during encode/decode.
        self.net.load_state_dict(clean_state_dict, strict=False)
        self.net.eval()

        # Freeze all CIN weights — WMForger must NOT train CIN
        for param in self.net.parameters():
            param.requires_grad = False

    # ── Core encode (used internally) ───────────────────────────────────────
    def _encode_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images with random CIN watermarks.

        Args:
            imgs: [N, 3, H, W] float tensor in [0, 1], any resolution

        Returns:
            watermarked: [N, 3, H, W] float tensor in [0, 1], same resolution
        """
        B, C, H, W = imgs.shape
        device = imgs.device

        # Step 1: Resize down to 128x128 (CIN requirement)
        imgs_128 = F.interpolate(imgs, size=(CIN_SIZE, CIN_SIZE),
                                 mode='bilinear', align_corners=False)

        # Step 2: Normalize to [-1, 1] (CIN requirement)
        imgs_128_norm = imgs_128 * 2.0 - 1.0

        # Step 3: Generate random 30-bit binary messages (one per image in batch)
        # mod_a: messages in [0, 1] range (matches opt.yml setting)
        messages = torch.from_numpy(
            np.random.choice([0, 1], size=(B, MSG_LENGTH)).astype(np.float32)
        ).to(device)

        # Step 4: Run CIN encoder
        with torch.no_grad():
            wm_128_norm = self.net.encoder(imgs_128_norm, messages)  # [B, 3, 128, 128] in [-1,1]

        # Step 5: Convert back to [0, 1]
        wm_128 = (wm_128_norm + 1.0) / 2.0  # [B, 3, 128, 128] in [0,1]

        # Step 6: Compute the watermark residual at 128x128
        #   residual = what CIN added to the image
        residual_128 = wm_128 - imgs_128  # [B, 3, 128, 128]

        # Step 7: Upsample the residual back to original image size
        #   This keeps the watermark at full resolution (like TrustMark does)
        residual_full = F.interpolate(residual_128, size=(H, W),
                                      mode='bilinear', align_corners=False)

        # Step 8: Apply residual to original full-resolution image
        watermarked = (imgs + residual_full).clamp(0.0, 1.0)

        return watermarked

    # ── The method WMForger calls ────────────────────────────────────────────
    def embed(self, imgs: torch.Tensor, **kwargs) -> dict:
        """
        Required by WMForger. Called as:
            outputs = embedder.embed(imgs, is_video=False)
            watermarked = outputs["imgs_w"]

        Args:
            imgs: [N, 3, H, W] float tensor in [0, 1]

        Returns:
            dict with key "imgs_w": [N, 3, H, W] float tensor in [0, 1]
        """
        watermarked = self._encode_batch(imgs)
        return {"imgs_w": watermarked}

    # ── Decode (used for evaluation / BER measurement) ────────────────────
    def decode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Decode (extract) the watermark message from watermarked images.

        Args:
            imgs: [N, 3, H, W] float tensor in [0, 1], any resolution

        Returns:
            decoded_msgs: [N, MSG_LENGTH] float tensor — bits rounded to 0 or 1
        """
        B, C, H, W = imgs.shape
        device = imgs.device

        # Resize to 128x128 for CIN
        imgs_128 = F.interpolate(imgs, size=(CIN_SIZE, CIN_SIZE),
                                 mode='bilinear', align_corners=False)

        # Normalize to [-1, 1]
        imgs_128_norm = imgs_128 * 2.0 - 1.0

        with torch.no_grad():
            # pre_noise=0 → uses the invertible DEM decoder path (no JPEG noise)
            _, _, _, msg_nsm = self.net.test_decoder(imgs_128_norm, pre_noise=0)

        decoded = msg_nsm.detach().cpu().round().clamp(0.0, 1.0)
        return decoded


# ── Backwards-compatible class name (used in your existing test script) ──────
class CIN_MODEL:
    """
    Simple wrapper kept for backwards compatibility with test_cin_encode_decode.py.
    Use CINEmbedder for WMForger integration.
    """

    def __init__(
        self,
        checkpoint_path: str = None   # defaults to checkpoints/CIN/cinNet&nsmNet.pth
    ):
        self.embedder = CINEmbedder(checkpoint_path=checkpoint_path)
        self.embedder.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = self.embedder.to(self.device)
        self.msg_bits = 30   # CIN uses 30-bit messages

    def to(self, device):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.embedder = self.embedder.to(self.device)
        return self

    def eval(self):
        self.embedder.eval()
        return self

    def _encode_(self, image_pil: Image.Image, message_tensor) -> torch.Tensor:
        """Encode a single PIL image with a specific message. Returns [1,3,H,W] in [0,1]."""
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        # Accept numpy array or torch tensor; ensure shape [1, MSG_LENGTH]
        if isinstance(message_tensor, np.ndarray):
            message_tensor = torch.tensor(message_tensor, dtype=torch.float32)
        if message_tensor.dim() == 1:
            message_tensor = message_tensor.unsqueeze(0)   # (30,) → (1, 30)

        img_tensor = TF.to_tensor(image_pil).unsqueeze(0).to(self.device)  # [1,3,H,W] in [0,1]

        # Use the internal CIN net directly with the specific message
        img_128 = F.interpolate(img_tensor, size=(CIN_SIZE, CIN_SIZE),
                                mode='bilinear', align_corners=False)
        img_128_norm = img_128 * 2.0 - 1.0

        with torch.no_grad():
            wm_128_norm = self.embedder.net.encoder(
                img_128_norm, message_tensor.to(self.device)
            )

        wm_128 = (wm_128_norm + 1.0) / 2.0
        residual_128 = wm_128 - img_128
        residual_full = F.interpolate(residual_128, size=img_tensor.shape[2:],
                                      mode='bilinear', align_corners=False)
        watermarked = (img_tensor + residual_full).clamp(0.0, 1.0)
        return watermarked

    def _decode_(self, image_input) -> torch.Tensor:
        """Decode a single image (PIL or tensor). Returns decoded bits [1, MSG_LENGTH]."""
        if not isinstance(image_input, torch.Tensor):
            img_tensor = TF.to_tensor(image_input).unsqueeze(0).to(self.device)
        else:
            img_tensor = image_input.to(self.device)
            if img_tensor.min() >= 0 and img_tensor.max() <= 1:
                pass  # already [0,1]
            else:
                img_tensor = (img_tensor + 1.0) / 2.0  # convert from [-1,1] to [0,1]

        return self.embedder.decode(img_tensor)
