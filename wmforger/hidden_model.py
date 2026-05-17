"""
HiDDeN Watermarking Model — WMForger API Wrapper
=================================================
Wraps HiDDeN (Hiding Data with Deep Networks) so WMForger can attack it.

Checkpoint : experiments/combined-noise/checkpoints/combined-noise--epoch-400.pyt
Config     : experiments/combined-noise/options-and-config.pickle
Image size : 128 × 128
Msg length : 30 bits  (from saved config)
Range      : [-1, 1] internally  (HiDDeN normalises with mean=0.5, std=0.5)
             → converted to [0,1] for external interface

WMForger calls:
    outputs = embedder.embed(imgs)
    watermarked = outputs["imgs_w"]
    decoded = embedder.decode(imgs)
"""

import sys
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore")

# ── Bridge to HiDDeN source ────────────────────────────────────────────────
_HERE_H = os.path.dirname(os.path.abspath(__file__))
_ROOT_H = os.path.dirname(_HERE_H)
HIDDEN_PATH = os.path.join(_ROOT_H, "victim_models", "HiDDeN")
if HIDDEN_PATH not in sys.path:
    sys.path.insert(0, HIDDEN_PATH)

from options import HiDDenConfiguration
from model.encoder_decoder import EncoderDecoder
from noise_layers.noiser import Noiser

# ── Configuration ──────────────────────────────────────────────────────────
EXPERIMENT_FOLDER = os.path.join(HIDDEN_PATH, "experiments", "combined-noise")
CHECKPOINT_PATH = os.path.join(
    EXPERIMENT_FOLDER, "checkpoints", "combined-noise--epoch-400.pyt"
)
OPTIONS_PATH = os.path.join(EXPERIMENT_FOLDER, "options-and-config.pickle")


class HiDDeNEmbedder(nn.Module):
    """
    WMForger-compatible embedder for HiDDeN.

    Key notes:
    - HiDDeN normalises images to [-1, 1] using Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    - This wrapper accepts/returns [0,1] tensors externally and converts internally
    - Messages are float tensors with values ~{0.0, 1.0}; decoded values are
      thresholded at 0.5 for binary accuracy measurement
    - Checkpoint format: {'enc-dec-model': ..., 'discrim-model': ..., 'epoch': ...}
    - Config is stored in options-and-config.pickle as 3 pickled objects:
      train_options, noise_config, hidden_config
    - PIL evaluation applied on decode (professor's requirement)
    """

    def __init__(self, checkpoint_path=CHECKPOINT_PATH, options_path=OPTIONS_PATH):
        super().__init__()

        # ── Load config from pickle ────────────────────────────────────────
        with open(options_path, 'rb') as f:
            _train_options = pickle.load(f)
            _noise_config  = pickle.load(f)
            hidden_config  = pickle.load(f)

        # Backward-compatibility for older checkpoints
        if not hasattr(hidden_config, 'enable_fp16'):
            hidden_config.enable_fp16 = False

        self.hidden_config = hidden_config
        self.msg_bits      = hidden_config.message_length   # 30
        self.img_size      = hidden_config.H                # 128

        # ── Build model (no noise for inference) ───────────────────────────
        device = torch.device('cpu')
        noiser = Noiser([], device)                         # Empty noise layer
        self.encoder_decoder = EncoderDecoder(hidden_config, noiser)

        # ── Load weights ───────────────────────────────────────────────────
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
        self.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
        self.encoder_decoder.eval()

        # ── Image transforms ───────────────────────────────────────────────
        # HiDDeN uses: Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) → maps [0,1] to [-1,1]
        self.to_hidden = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # [0,1] → [-1,1]
        ])

    # ── Internal tensor conversion ─────────────────────────────────────────
    def _pil_to_hidden_tensor(self, pil_image, device):
        """PIL Image → [1, 3, 128, 128] in [-1, 1]"""
        return self.to_hidden(pil_image).unsqueeze(0).to(device)

    def _hidden_tensor_to_pil(self, tensor):
        """[-1, 1] tensor → PIL Image in [0, 1] colour space"""
        img_01 = (tensor.squeeze(0).cpu() + 1.0) / 2.0     # [-1,1] → [0,1]
        return TF.to_pil_image(img_01.clamp(0, 1))

    # ── Core encode/decode ─────────────────────────────────────────────────
    def _encode_(self, image_pil, message_bits, device='cpu'):
        """
        Encode message bits into a PIL image.
        Args:
            image_pil:    PIL Image (any size)
            message_bits: list/array of 0/1 values, length = msg_bits
            device:       torch device
        Returns:
            tensor [1, 3, H, W] in [0, 1] restored to original image size
        """
        orig_size  = image_pil.size                         # (W, H)
        img_tensor = self._pil_to_hidden_tensor(image_pil, device)
        msg_tensor = torch.tensor(
            np.array(message_bits, dtype=np.float32)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            # encoder_decoder returns (encoded, noised, decoded)
            encoded, _, _ = self.encoder_decoder(img_tensor, msg_tensor)

        # Convert [-1,1] back to PIL, restore original size, return as [0,1] tensor
        encoded_pil = self._hidden_tensor_to_pil(encoded)
        encoded_pil = encoded_pil.resize(orig_size, Image.BICUBIC)
        return TF.to_tensor(encoded_pil).unsqueeze(0).clamp(0, 1)

    def _decode_(self, img_tensor, device='cpu'):
        """
        Decode watermark from [0,1] tensor using PIL evaluation.
        PIL evaluation: float tensor → PIL (uint8) → back to [-1,1] tensor → decode.
        This simulates real-world image file loading (no infinite float precision).

        Args:
            img_tensor: [1, 3, H, W] in [0, 1]
        Returns:
            tensor [1, msg_bits] of float values (threshold at 0.5 for binary)
        """
        # PIL evaluation step (professor's requirement)
        pil           = TF.to_pil_image(img_tensor.squeeze(0).clamp(0, 1).cpu())
        hidden_tensor = self._pil_to_hidden_tensor(pil, device)

        with torch.no_grad():
            decoded = self.encoder_decoder.decoder(hidden_tensor)

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
            [B, msg_bits] float tensor
        """
        results = []
        for i in range(imgs.shape[0]):
            bits = self._decode_(imgs[i:i+1].cpu())
            results.append(bits)
        return torch.cat(results, dim=0).to(imgs.device)

    def forward(self, imgs):
        return self.embed(imgs)


class HIDDEN_MODEL(nn.Module):
    """
    Full HiDDeN model — top-level wrapper for attack scripts.
    Mirrors the CIN_MODEL interface for consistency across all attack scripts.

    Usage:
        hidden = HIDDEN_MODEL()
        hidden.to(device)
        wm_tensor = hidden._encode_(pil_image, message_bits)
        decoded   = hidden._decode_(wm_tensor)
    """

    def __init__(self, checkpoint_path=CHECKPOINT_PATH, options_path=OPTIONS_PATH):
        super().__init__()
        self.embedder = HiDDeNEmbedder(
            checkpoint_path = checkpoint_path,
            options_path    = options_path,
        )
        self.msg_bits = self.embedder.msg_bits

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
