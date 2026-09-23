"""Loads the pretrained LDMSeg mask autoencoder (frozen decoder) from
https://github.com/segments-ai/latent-diffusion-segmentation (Van Gansbeke & Van Gool, "A
Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting", ECCV 2024).
Prototyped in notebooks/84_ldmseg_position_precision.ipynb; this factors that loading logic out
for reuse by precompute_latents.py and model.py.

We only ever use this checkpoint's ENCODE/DECODE, frozen -- never its own training code (Part 1
of the paper). We never use Part 2 (RGB-conditioned diffusion) at all: we have no photo of a
closed box's contents to condition on.
"""
import functools
import sys
import types
import importlib.util
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

LDMSEG_CACHE = Path.home() / ".cache" / "ldmseg"
LDMSEG_REPO = LDMSEG_CACHE / "repo"
LDMSEG_CKPT = LDMSEG_CACHE / "ae.pth"

# The released checkpoint's own config (tools/configs/base/base.yaml in the LDMSeg repo).
VAE_KWARGS = dict(
    in_channels=7, int_channels=256, out_channels=128, block_out_channels=[32, 64, 128, 256],
    latent_channels=4, num_latents=2, num_upscalers=2, upscale_channels=256, norm_num_groups=32,
    scaling_factor=0.2, parametrization="gaussian", act_fn="none", clamp_output=False,
    freeze_codebook=False, num_mid_blocks=0, fuse_rgb=False, resize_input=False, skip_encoder=False,
)


def _load_vae_class():
    """ldmseg/models/vae.py needs `ldmseg.utils.OutputDict` (not worth installing the repo's
    detectron2 dependency for) and an old diffusers import path that newer diffusers moved
    (diffusers.models.unet_2d_blocks -> diffusers.models.unets.unet_2d_blocks). Shim both instead
    of pinning an old diffusers version -- the classes still exist, just at the new path."""
    import diffusers.models.unets.unet_2d_blocks as new_path
    sys.modules["diffusers.models.unet_2d_blocks"] = new_path

    class OutputDict(OrderedDict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            super().__setattr__(key, value)

    ldmseg_pkg = types.ModuleType("ldmseg")
    ldmseg_pkg.__path__ = [str(LDMSEG_REPO / "ldmseg")]
    utils_pkg = types.ModuleType("ldmseg.utils")
    utils_pkg.OutputDict = OutputDict
    sys.modules["ldmseg"] = ldmseg_pkg
    sys.modules["ldmseg.utils"] = utils_pkg

    spec = importlib.util.spec_from_file_location(
        "ldmseg.models.vae", str(LDMSEG_REPO / "ldmseg" / "models" / "vae.py"))
    vae_mod = importlib.util.module_from_spec(spec)
    sys.modules["ldmseg.models.vae"] = vae_mod
    spec.loader.exec_module(vae_mod)
    return vae_mod.GeneralVAESeg


def load_frozen_ae(ckpt_path: Path = LDMSEG_CKPT) -> nn.Module:
    """Loads the pretrained mask autoencoder, frozen and in eval mode. Requires the LDMSeg repo
    + checkpoint already cached at ~/.cache/ldmseg (see notebooks/84_ldmseg_position_precision.ipynb
    cell 2 for the one-time clone/download)."""
    if not (LDMSEG_REPO / "ldmseg" / "models" / "vae.py").exists() or not ckpt_path.exists():
        raise FileNotFoundError(
            f"LDMSeg repo/checkpoint not found under {LDMSEG_CACHE} -- run "
            "notebooks/84_ldmseg_position_precision.ipynb's cell 2 once to clone/download them.")

    GeneralVAESeg = _load_vae_class()
    model = GeneralVAESeg(**VAE_KWARGS)
    # The released checkpoint uses an old numpy-scalar pickle format PyTorch>=2.6 refuses under
    # its new default weights_only=True; safe here since this is the paper's own checkpoint.
    orig_load = torch.load
    torch.load = functools.partial(orig_load, weights_only=False)
    try:
        model.load_pretrained(str(ckpt_path))
    finally:
        torch.load = orig_load
    model.eval()
    model.requires_grad_(False)
    return model


def encode_bitmap(ids: torch.Tensor, n_bits: int = 7) -> torch.Tensor:
    """(H,W) int id map -> (n_bits,H,W) float bit-encoding, LDMSeg's own input format."""
    ids = ids.long()
    bits = torch.bitwise_right_shift(ids, torch.arange(n_bits)[:, None, None])
    return torch.remainder(bits, 2).float()


def dominant_class(argmax_map: torch.Tensor, instance_mask: torch.Tensor) -> int:
    """The "target_class" trick: read off whichever of the decoder's 128 fixed output classes it
    assigns to this instance's own footprint (mode of argmax_map over instance_mask's True
    pixels). There's no reason to expect this equals whatever id was bit-encoded at input --
    verified empirically (see the plan): the same shape encoded with 5 different ids decoded to
    5 different, unrelated classes."""
    return int(argmax_map[instance_mask].mode().values.item())
