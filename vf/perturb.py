"""Controlled image degradations for robustness probing."""
import io

import numpy as np
from PIL import Image, ImageFilter


def perturb(img: Image.Image, kind: str, strength: float, seed: int = 0) -> Image.Image:
    """strength in [0, 1]."""
    if kind == "none" or strength <= 0:
        return img
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=3.0 * strength))
    if kind == "noise":
        a = np.asarray(img).astype(np.float32)
        rng = np.random.default_rng(seed)
        a += rng.normal(0, 60 * strength, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=max(5, int(95 - 85 * strength)))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "rotate":
        return img.rotate(25 * strength, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    if kind == "contrast":
        a = np.asarray(img).astype(np.float32)
        a = 128 + (a - 128) * (1 - 0.8 * strength)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    raise ValueError(kind)


KINDS = ["blur", "noise", "jpeg", "rotate", "contrast"]
