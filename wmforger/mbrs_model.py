"""
MBRS Watermarking Model — WMForger API Wrapper
===============================================
Wraps MBRS (Mini-Batch Real and Simulated JPEG) so WMForger can attack it.

Checkpoint : MBRS_Diffusion_128_m30/models/EC_114.pth
Image size : 128 × 128  (resizes input, restores original size after encode)
Msg length : 30 bits
Range      : [0, 1]  (no normalisation — MBRS uses raw [0,1] tensors)

WMForger calls:
    outputs = embedder.embed(imgs)
    watermarked = outputs["imgs_w"]
    decoded = embedder.decode(imgs)
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore")

# ── Bridge to MBRS source ──────────────────────────────────────────────────
_HERE_M = os.path.dirname(os.path.abspath(__file__))
_ROOT_M = os.path.dirname(_HERE_M)
MBRS_PATH = os.path.join(_ROOT_M, "victim_models", "MBRS")
if MBRS_PATH not in sys.path:
    sys.path.insert(0, MBRS_PATH)

from network.Encoder_MP_Decoder import EncoderDecoder_Diffusion

# ── Configuration ──────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(_ROOT_M, "checkpoints", "MBRS", "EC_114.pth")
IMG_SIZE = 128
MSG_BITS = 30


class MBRSEmbedder(nn.Module):
    """
    WMForger-compatible embedder for MBRS.

    Key notes:
    - MBRS saved checkpoints WITHOUT the DataParallel 'module.' prefix
      (see Network.save_model → torch.save(self.encoder_decoder.module.state_dict()))
    - Images resized to 128×128 for inference, then restored to original size
    - Messages are float tensors [B, 30] with values in {0.0, 1.0}
    - PIL evaluation applied on decode (professor's requirement)
    """

    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        super().__init__()

        # Build encoder-decoder (no noise layers during inference)
        self.encoder_decoder = EncoderDecoder_Diffusion(
            H              = IMG_SIZE,
            W              = IMG_SIZE,
            message_length = MSG_BITS,
            noise_layers   = [],        # No noise — inference only
        )

        # Load checkpoint — MBRS saves module.state_dict() so no prefix stripping needed
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.encoder_decoder.load_state_dict(state_dict, strict=False)
        self.encoder_decoder.eval()

        self.msg_bits = MSG_BITS
        self.img_size = IMG_SIZE

        # Transform for input to MBRS (resize to 128×128, keep [0,1])
        self.resize_transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
        ])

    # ── Tensor helpers ─────────────────────────────────────────────────────
    def _pil_to_mbrs_tensor(self, pil_image, device):
        """PIL → [1, 3, 128, 128] in [0,1]"""
        return self.resize_transform(pil_image).unsqueeze(0).to(device)

    # ── Core encode/decode ─────────────────────────────────────────────────
    def _encode_(self, image_pil, message_bits, device='cpu'):
        """
        Encode bits into image.
        Args:
            image_pil:    PIL Image (any size — will be resized internally)
            message_bits: list/array of 0/1 ints, length = MSG_BITS
            device:       torch device string or object
        Returns:
            tensor [1, 3, H, W] in [0,1] — restored to original PIL size
        """
        orig_size  = image_pil.size                             # (W, H)
        img_tensor = self._pil_to_mbrs_tensor(image_pil, device)
        msg_tensor = torch.tensor(
            np.array(message_bits, dtype=np.float32)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            # Call encoder directly — bypasses the Noise layer which breaks
            # when noise_layers=[] because nn.Sequential passes a list unchanged
            encoded = self.encoder_decoder.encoder(img_tensor, msg_tensor)

        # Restore original size via PIL (PIL evaluation step)
        encoded_pil = TF.to_pil_image(encoded.squeeze(0).clamp(0, 1).cpu())
        encoded_pil = encoded_pil.resize(orig_size, Image.BICUBIC)
        return TF.to_tensor(encoded_pil).unsqueeze(0).clamp(0, 1)

    def _decode_(self, img_tensor, device='cpu'):
        """
        Decode watermark from [0,1] tensor using PIL evaluation.
        PIL evaluation: tensor → PIL (uint8 quantisation) → back to tensor → decode.
        This simulates real-world image loading rather than perfect floats.

        Args:
            img_tensor: [1, 3, H, W] in [0,1]
        Returns:
            tensor [1, MSG_BITS] of float values
        """
        # PIL evaluation (professor's requirement)
        pil        = TF.to_pil_image(img_tensor.squeeze(0).clamp(0, 1).cpu())
        tensor_128 = self._pil_to_mbrs_tensor(pil, device)

        with torch.no_grad():
            # Call decoder directly — same reason as encoder above
            decoded = self.encoder_decoder.decoder(tensor_128)

        return decoded.clamp(0, 1).cpu()

    # ── WMForger batch interface ────────────────────────────────────────────
    def embed(self, imgs):
        """
        WMForger interface: embed random watermarks into batch.
        Args:
            imgs: [B, 3, H, W] in [0,1]
        Returns:
            dict with key 'imgs_w': [B, 3, H, W]
        """
        device  = imgs.device
        results = []
        for i in range(imgs.shape[0]):
            msg = np.random.randint(0, 2, self.msg_bits).astype(np.float32)
            pil = TF.to_pil_image(imgs[i].clamp(0, 1).cpu())
            wm  = self._encode_(pil, msg, device=device)
            results.append(wm.to(device))
        imgs_w = torch.cat(results, dim=0)
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


class MBRS_MODEL(nn.Module):
    """
    Full MBRS model — top-level wrapper for attack scripts.
    Mirrors the CIN_MODEL interface for consistency across all attack scripts.

    Usage:
        mbrs = MBRS_MODEL()
        mbrs.to(device)
        wm_tensor = mbrs._encode_(pil_image, message_bits)
        decoded   = mbrs._decode_(wm_tensor)
    """

    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        super().__init__()
        self.embedder = MBRSEmbedder(checkpoint_path=checkpoint_path)
        self.msg_bits = MSG_BITS

    def _get_device(self):
        return next(self.embedder.encoder_decoder.parameters()).device

    def _encode_(self, image_pil, message_bits):
        return self.embedder._encode_(image_pil, message_bits, device=self._get_device())

    def _decode_(self, img_tensor):
        return self.embedder._decode_(img_tensor, device=self._get_device())

    def to(self, device):
        self.embedder.encoder_decoder.to(device)
        return self

    def eval(self):
        self.embedder.encoder_decoder.eval()
        return self

    def parameters(self, recurse=True):
        return self.embedder.encoder_decoder.parameters()
