"""Crowded conjunction counting: crowding (spacing/overlap) is the knob ('VLMs are blind', BedBench)."""
import numpy as np
from PIL import Image, ImageDraw

from ..core import Family, register
from ..draw import Canvas

COLORS = {"red": (220, 40, 40), "blue": (40, 90, 220), "green": (40, 160, 70), "orange": (240, 140, 20),
          "purple": (140, 60, 190), "teal": (20, 160, 160)}
KINDS = ["circle", "square", "triangle"]
W, H, R = 640, 480, 17


def _pts(kind, x, y, r):
    if kind == "square":
        return [(x - r * .9, y - r * .9), (x + r * .9, y - r * .9), (x + r * .9, y + r * .9), (x - r * .9, y + r * .9)]
    return [(x, y - r * 1.1), (x + r * 1.05, y + r * .8), (x - r * 1.05, y + r * .8)]


def _mask(s):
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    if s["kind"] == "circle":
        dr.ellipse([s["x"] - s["r"], s["y"] - s["r"], s["x"] + s["r"], s["y"] + s["r"]], fill=255)
    else:
        dr.polygon(_pts(s["kind"], s["x"], s["y"], s["r"]), fill=255)
    return np.asarray(im) > 0


@register
class CrowdCount(Family):
    name = "crowd_count"
    application = "Inventory / shelf / lab-slide counting (counting & locating in clutter)"
    capability = "counting in clutter"
    toggles = {"conj": [False, True]}

    def make_spec(self, rng, d, conj=True):
        n_t = 3 + round(d * 17)
        n_d = 6 + round(d * 34)
        fac = 1.6 - 1.0 * d
        tk, tc = rng.choice(KINDS), rng.choice(list(COLORS))
        shapes = [dict(kind=tk, color=tc) for _ in range(n_t)]
        for _ in range(n_d):
            u = rng.random()
            if conj and u < .4:
                k, c = rng.choice([k for k in KINDS if k != tk]), tc
            elif conj and u < .8:
                k, c = tk, rng.choice([c for c in COLORS if c != tc])
            else:
                while True:
                    k, c = rng.choice(KINDS), rng.choice(list(COLORS))
                    if (k, c) != (tk, tc):
                        break
            shapes.append(dict(kind=k, color=c))
        rng.shuffle(shapes)
        n = len(shapes)
        f = fac
        while True:
            pos, tries = [], 0
            while len(pos) < n and tries < 6000:
                tries += 1
                x, y = rng.uniform(R + 6, W - R - 6), rng.uniform(R + 6, H - R - 6)
                if all((x - px) ** 2 + (y - py) ** 2 >= (f * 2 * R) ** 2 for px, py in pos):
                    pos.append((x, y))
            if len(pos) == n:
                break
            f += 0.1
        for s, (x, y) in zip(shapes, pos):
            s.update(x=round(x, 1), y=round(y, 1), r=R)
        return dict(shapes=shapes, target=[tk, tc], spacing=round(f, 2))

    def valid(self, spec):
        # every shape must remain >= 55% visible, so the count is unambiguous
        idmap = np.zeros((H, W), dtype=np.int32)
        areas = []
        for i, s in enumerate(spec["shapes"]):
            m = _mask(s)
            idmap[m] = i + 1
            areas.append(m.sum())
        for i, a in enumerate(areas):
            if (idmap == i + 1).sum() < 0.55 * a:
                return False
        return True

    def render(self, spec):
        cv = Canvas(W, H)
        for s in spec["shapes"]:
            col = COLORS[s["color"]]
            if s["kind"] == "circle":
                cv.ellipse(s["x"], s["y"], s["r"], fill=col, outline=(0, 0, 0), width=1)
            else:
                cv.poly(_pts(s["kind"], s["x"], s["y"], s["r"]), fill=col, outline=(0, 0, 0))
        return cv.done()

    def qa(self, spec):
        k, c = spec["target"]
        n = sum(1 for s in spec["shapes"] if s["kind"] == k and s["color"] == c)
        return dict(question=f"How many {c} {k}s are in the image? Count only shapes that match both the color "
                             f"and the shape type.", answer=n, atype="int", tol=0.0)

    def text(self, spec):
        return "Shapes (kind, color, center x, center y, radius), drawn in this order (later ones on top):\n" + \
            "\n".join(f"{s['color']} {s['kind']} at ({s['x']:.0f},{s['y']:.0f}) r={s['r']}" for s in spec["shapes"])

    def meta(self, spec):
        return dict(n_shapes=len(spec["shapes"]), spacing=spec["spacing"])

    def _mutate(self, spec, rng):
        k, c = spec["target"]
        idx = [i for i, s in enumerate(spec["shapes"]) if s["kind"] == k and s["color"] == c]
        if len(idx) < 2:
            return None
        del spec["shapes"][rng.choice(idx)]
        return spec
