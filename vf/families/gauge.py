"""Analog gauge reading: tick density / label sparsity is the knob (MeasureBench-style instruments)."""
import math

import numpy as np

from ..core import Family, register
from ..draw import Canvas

RANGES = [(0, 100), (0, 10), (0, 200), (0, 40), (-50, 50), (0, 1)]
NEEDLE = (200, 30, 30)
S = 520
CX = CY = S // 2
RAD = 200


def _fmt(v, span):
    if span >= 10:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".") if span <= 1 else f"{v:.1f}"


def _angle(f):  # math degrees; 225 deg (lower-left) sweeping clockwise 270 deg
    return 225.0 - 270.0 * f


@register
class Gauge(Family):
    name = "gauge"
    application = "Dashboards, meters and instruments (exact value reading)"
    capability = "exact value reading"
    toggles = {"labels": [True, False]}

    def make_spec(self, rng, d, labels=None):
        lo, hi = rng.choice(RANGES)
        nint = 5 if d < 0.34 else 10 if d < 0.67 else 20
        lab_every = {5: 1, 10: 1 if d < 0.4 else 2, 20: 2 if d < 0.7 else 5}[nint]
        if labels is False:
            lab_every = nint  # only the endpoints
        span = hi - lo
        v = rng.uniform(lo + 0.06 * span, hi - 0.06 * span)
        return dict(lo=lo, hi=hi, nint=nint, lab_every=lab_every, minor=4 if nint < 20 else 1,
                    value=round(v, 2 if span >= 10 else 3), width=round(5 - 2 * d, 1))

    def render(self, spec):
        cv = Canvas(S, S)
        lo, hi, nint = spec["lo"], spec["hi"], spec["nint"]
        span = hi - lo
        cv.ellipse(CX, CY, RAD + 28, fill=(246, 246, 246), outline=(30, 30, 30), width=3)
        for i in range(nint + 1):
            th = math.radians(_angle(i / nint))
            c, s = math.cos(th), -math.sin(th)
            cv.line([(CX + c * (RAD - 20), CY + s * (RAD - 20)), (CX + c * RAD, CY + s * RAD)], (20, 20, 20), 2.5)
            if i % spec["lab_every"] == 0 or i == nint:
                cv.text((CX + c * (RAD - 42), CY + s * (RAD - 42)), _fmt(lo + span * i / nint, span), 15)
            if i < nint:
                for m in range(1, spec["minor"] + 1):
                    th2 = math.radians(_angle((i + m / (spec["minor"] + 1)) / nint))
                    c2, s2 = math.cos(th2), -math.sin(th2)
                    cv.line([(CX + c2 * (RAD - 10), CY + s2 * (RAD - 10)), (CX + c2 * RAD, CY + s2 * RAD)],
                            (60, 60, 60), 1.2)
        th = math.radians(_angle((spec["value"] - lo) / span))
        cv.line([(CX, CY), (CX + math.cos(th) * (RAD - 8), CY - math.sin(th) * (RAD - 8))], NEEDLE, spec["width"])
        cv.ellipse(CX, CY, 11, fill=(40, 40, 40))
        return cv.done()

    def check(self, spec, img):
        # recover the needle angle from pixels and compare with the spec
        a = np.asarray(img).astype(int)
        m = (a[..., 0] > 170) & (a[..., 1] < 90) & (a[..., 2] < 90)
        ys, xs = np.nonzero(m)
        if len(xs) < 20:
            return False
        dist = np.hypot(xs - CX, ys - CY)
        sel = dist > 0.5 * RAD
        if sel.sum() < 5:
            return False
        ang = math.degrees(math.atan2(-(ys[sel] - CY).mean(), (xs[sel] - CX).mean()))
        f = ((225.0 - ang) % 360.0) / 270.0
        span = spec["hi"] - spec["lo"]
        return abs(spec["lo"] + f * span - spec["value"]) <= 0.012 * span

    def qa(self, spec):
        span = spec["hi"] - spec["lo"]
        return dict(question="What value does the red needle of the gauge point to? Give a number.",
                    answer=spec["value"], atype="float", tol=round(0.025 * span, 4))

    def text(self, spec):
        span = spec["hi"] - spec["lo"]
        ang = _angle((spec["value"] - spec["lo"]) / span)
        return (f"A circular gauge. The scale starts at {spec['lo']} (pointing at 225 degrees, measured "
                f"counter-clockwise from the +x axis) and increases clockwise over a 270 degree sweep to "
                f"{spec['hi']} (at -45 degrees). The scale has {spec['nint']} major intervals. The needle points "
                f"at {ang:.1f} degrees.")

    def meta(self, spec):
        return dict(nint=spec["nint"], lab_every=spec["lab_every"])

    def _mutate(self, spec, rng):
        span = spec["hi"] - spec["lo"]
        delta = rng.uniform(0.12, 0.30) * span * rng.choice([-1, 1])
        v = spec["value"] + delta
        if not (spec["lo"] + 0.04 * span <= v <= spec["hi"] - 0.04 * span):
            v = spec["value"] - delta
        spec["value"] = round(v, 2 if span >= 10 else 3)
        return spec
