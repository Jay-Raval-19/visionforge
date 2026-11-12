"""Lightweight task API: one function per task returns image + question + computed answer.

    @task("gear_train", application="...", capability="...", knob="...", twin="...")
    def gear_train(rng, d):          # d in [0, 1] is the difficulty knob
        ...                          # raise Reject() to resample an ambiguous scene
        return dict(image=img, question=q, answer=a, atype="choice", choices=[...], spec={...})
"""
from __future__ import annotations

import json

from PIL import ImageDraw

from .core import FAMILIES, Family
from .draw import font

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Reject(Exception):
    """Raised by a generator when a sampled scene is ambiguous; the factory resamples."""


class FuncFamily(Family):
    def __init__(self, name, fn, application, capability, knob, twin, group):
        self.name, self.fn = name, fn
        self.application, self.capability, self.knob, self.twin, self.group = \
            application, capability, knob, twin, group

    def make_spec(self, rng, d, **kw):
        try:
            out = self.fn(rng, d, **kw)
        except Reject:
            return None
        spec = dict(out.get("spec", {}))
        spec["_img"] = out["image"]
        spec["_qa"] = dict(question=out["question"], answer=out["answer"], atype=out["atype"],
                           tol=float(out.get("tol", 0.0)), choices=out.get("choices"))
        spec["_text"] = out.get("text")
        return spec

    def render(self, spec):
        return spec["_img"]

    def qa(self, spec):
        return spec["_qa"]

    def text(self, spec):
        if spec.get("_text"):
            return spec["_text"]
        pub = {k: v for k, v in spec.items() if not k.startswith("_")}
        return "Scene description (JSON): " + json.dumps(pub, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    def meta(self, spec):
        return dict(group=self.group)


def task(name, application, capability, knob="", twin="", group=""):
    def deco(fn):
        FAMILIES[name] = FuncFamily(name, fn, application, capability, knob, twin, group)
        return fn
    return deco


def autocrop(img, margin=24, bg=(255, 255, 255)):
    """Trim uniform background around the drawing (keeps a margin)."""
    import numpy as np
    a = np.asarray(img.convert("RGB")).astype(int)
    m = (np.abs(a - np.array(bg)).sum(-1) > 12)
    if not m.any():
        return img
    ys, xs = np.nonzero(m)
    x0, y0 = max(0, xs.min() - margin), max(0, ys.min() - margin)
    x1, y1 = min(img.width, xs.max() + margin), min(img.height, ys.max() + margin)
    return img.crop((x0, y0, x1, y1))


def seg_dist(a, b, c, d):
    """Minimum distance between 2D segments ab and cd."""
    import numpy as np
    a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))

    def pt(p, q, r):
        t = np.clip((p - q) @ (r - q) / max((r - q) @ (r - q), 1e-9), 0, 1)
        return np.linalg.norm(q + t * (r - q) - p)

    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    if (ccw(a, c, d) > 0) != (ccw(b, c, d) > 0) and (ccw(a, b, c) > 0) != (ccw(a, b, d) > 0):
        return 0.0
    return min(pt(a, c, d), pt(b, c, d), pt(c, a, b), pt(d, a, b))


def lerp(a, b, t):
    return a + (b - a) * t


def ilerp(a, b, t):
    return int(round(a + (b - a) * t))


def tag(img, xy, text, size=16, fg=(0, 0, 0), bg=(255, 255, 255), border=(0, 0, 0)):
    """Draw a small boxed label centred at xy (in output pixels) on a PIL image."""
    dr = ImageDraw.Draw(img)
    f = font(size)
    x0, y0, x1, y1 = dr.textbbox((0, 0), str(text), font=f)
    w, h = x1 - x0, y1 - y0
    x, y = xy
    dr.rounded_rectangle([x - w / 2 - 4, y - h / 2 - 4, x + w / 2 + 4, y + h / 2 + 4], radius=4, fill=bg,
                         outline=border, width=1)
    dr.text((x, y), str(text), font=f, fill=fg, anchor="mm")
    return img
