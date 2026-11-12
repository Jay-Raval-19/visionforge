"""Wire / pipe / cable tracing: follow one wire through crossings (cf. 'VLMs have tunnel vision')."""
import copy

import numpy as np

from ..core import Family, register
from ..draw import Canvas, catmull

LETTERS = "ABCDEFGHIJKLMNOP"
PALETTE = [(220, 50, 50), (40, 110, 220), (40, 160, 80), (230, 140, 20), (140, 60, 190), (20, 170, 170),
           (200, 60, 140), (110, 110, 110), (150, 100, 40), (0, 150, 255), (180, 180, 0), (90, 0, 0)]
MONO = (45, 45, 45)


@register
class WireTrace(Family):
    name = "wire_trace"
    application = "Circuit / cable / pipe inspection (technical diagrams)"
    capability = "structure following"
    toggles = {"mono": [False, True]}

    def make_spec(self, rng, d, mono=None):
        n = 3 + round(d * 9)                 # wires
        n_mid = 1 + round(d * 3)             # waypoints per wire
        W = 900
        H = max(360, 90 + 46 * n)
        top, bot = 55, H - 55
        ys = [top + i * (bot - top) / (n - 1) for i in range(n)]
        perm = list(range(n))
        while perm == list(range(n)):
            rng.shuffle(perm)
        amp = (0.05 + 0.30 * d) * H
        xl, xr = 70, W - 70
        xs = [xl + (xr - xl) * (k + 1) / (n_mid + 1) for k in range(n_mid)]
        wires = []
        for i in range(n):
            ye = ys[perm[i]]
            pts = [[xl, ys[i]]]
            for k, x in enumerate(xs):
                t = (k + 1) / (n_mid + 1)
                y = ys[i] + (ye - ys[i]) * t + rng.uniform(-amp, amp)
                pts.append([round(x, 1), round(min(H - 25, max(25, y)), 1)])
            pts.append([xr, ye])
            wires.append(pts)
        return dict(n=n, W=W, H=H, ys=ys, perm=perm, wires=wires, q=rng.randrange(n),
                    mono=bool(d >= 0.25) if mono is None else bool(mono), xl=xl, xr=xr,
                    order=rng.sample(range(n), n))

    def _curves(self, spec):
        return [catmull([tuple(p) for p in w], 80) for w in spec["wires"]]

    def valid(self, spec):
        # ambiguity filter: reject long near-tangent contact between two wires
        cs = [np.array(c) for c in self._curves(spec)]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                dm = np.sqrt(((cs[i][:, None, :] - cs[j][None, :, :]) ** 2).sum(-1))
                if (dm.min(axis=1) < 6.0).sum() > 40:
                    return False
        return True

    def render(self, spec):
        W, H = spec["W"], spec["H"]
        cv = Canvas(W, H)
        curves = self._curves(spec)
        for i in spec["order"]:
            cv.line(curves[i], MONO if spec["mono"] else PALETTE[i % len(PALETTE)], 3)
        for i, y in enumerate(spec["ys"]):
            for x, lab, dx in ((spec["xl"], LETTERS[i], -30), (spec["xr"], str(i + 1), 30)):
                cv.ellipse(x, y, 9, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
                cv.text((x + dx, y), lab, 20)
        return cv.done()

    def qa(self, spec):
        q = spec["q"]
        return dict(question=("Each wire starts at a lettered terminal on the left and ends at a numbered "
                              f"terminal on the right. Follow the wire that starts at terminal {LETTERS[q]}. "
                              "At which numbered terminal on the right does it end?"),
                    answer=spec["perm"][q] + 1, atype="int", tol=0.0)

    def text(self, spec):
        lines = [f"Left terminals (x={spec['xl']}): " + ", ".join(
            f"{LETTERS[i]} at y={y:.0f}" for i, y in enumerate(spec["ys"])),
            f"Right terminals (x={spec['xr']}): " + ", ".join(
                f"{i + 1} at y={y:.0f}" for i, y in enumerate(spec["ys"])),
            "Each wire is a smooth curve passing through these waypoints, listed from left to right:"]
        for i, w in enumerate(spec["wires"]):
            lines.append(f"Wire from {LETTERS[i]}: " + " -> ".join(f"({x:.0f},{y:.0f})" for x, y in w))
        return "\n".join(lines)

    def meta(self, spec):
        return dict(n=spec["n"], mono=spec["mono"])

    def _mutate(self, spec, rng):
        q, n = spec["q"], spec["n"]
        j = rng.choice([k for k in range(n) if k != q])
        pq, pj = spec["perm"][q], spec["perm"][j]
        spec["perm"][q], spec["perm"][j] = pj, pq
        spec["wires"][q][-1][1] = spec["ys"][pj]
        spec["wires"][j][-1][1] = spec["ys"][pq]
        return spec
