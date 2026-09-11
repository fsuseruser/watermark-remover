"""
Vendored from simple-lama-inpainting (https://github.com/enesmsahin/simple-lama-inpainting),
MIT licensed. Trimmed to just the inference class we actually use.

Why this is vendored instead of installed from PyPI: the published package
declares dependencies on old pinned versions of pillow/numpy/torchvision/fire
that it doesn't actually need at runtime (those are only used by its CLI
wrapper, which this app never calls). On hosts that use a very new Python
version, pip can't find prebuilt wheels for those old pins and falls back to
building Pillow from source, which fails outright on modern setuptools. Using
this trimmed copy lets the app run on whatever modern Pillow/numpy/opencv the
host already has, with no version conflict.
"""
import os
import sys
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file, get_dir

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)


def get_image(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3
    img = img.astype(np.float32) / 255
    return img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    out_image = get_image(image)
    out_mask = get_image(mask)

    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)
    out_mask = (out_mask > 0) * 1

    return out_image, out_mask


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))
    filename = os.path.basename(parts.path)
    return os.path.join(model_dir, filename)


def download_model(url):
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, None, progress=True)
    return cached_file


class SimpleLama:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = os.environ.get("LAMA_MODEL") or download_model(LAMA_MODEL_URL)
        if os.environ.get("LAMA_MODEL") and not os.path.exists(model_path):
            raise FileNotFoundError(f"lama torchscript model not found: {model_path}")

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image, mask):
        image, mask = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(cur_res)
