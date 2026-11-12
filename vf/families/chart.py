"""Multi-step grouped-bar chart reading: series count / no value labels / truncated axis / similar hues."""
import numpy as np

from ..core import Family, register
from ..draw import Canvas

SERIES = ["North", "South", "East", "West", "Central", "Online"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
PAL = [(55, 105, 200), (230, 120, 40), (60, 160, 90), (190, 60, 70), (140, 90, 190), (30, 170, 180)]
SHADES = [(30, 60, 140), (60, 100, 175), (95, 135, 205), (130, 165, 225), (165, 190, 240), (200, 215, 250)]
W, H = 860, 520
L, T, B, RM = 70, 40, 90, 30


def layout(spec):
    pw, ph = W - L - RM, H - T - B
    ymin, ymax = spec["ymin"], 100
    ppu = ph / (ymax - ymin)
    ncat, ns = spec["ncat"], spec["ns"]
    gw = pw / ncat
    bw = gw * 0.8 / ns
    ybase = T + ph

    def bar(ci, si):
        x0 = L + ci * gw + gw * 0.1 + si * bw
        return x0, x0 + bw

    return dict(ppu=ppu, ybase=ybase, bar=bar, gw=gw, ymin=ymin, ymax=ymax, ph=ph, pw=pw)


@register
class Chart(Family):
    name = "chart"
    application = "Finance / analytics dashboards (multi-step chart reading)"
    capability = "comparison & multi-step reading"
    toggles = {"value_labels": [True, False], "similar_hues": [False, True]}

    def make_spec(self, rng, d, value_labels=None, similar_hues=None):
        ns = 2 + round(d * 4)
        ncat = 3 + round(d * 5)
        ymin = 40 if d > 0.5 else 0
        qtype = rng.choice(["gap", "rank"]) if ns >= 3 else "gap"
        for _ in range(500):
            vals = [[rng.randrange(ymin + 8, 94) for _ in range(ncat)] for _ in range(ns)]
            spec = dict(ns=ns, ncat=ncat, ymin=ymin, vals=vals, qtype=qtype,
                        names=SERIES[:ns], cats=MONTHS[:ncat],
                        value_labels=(d < 0.3) if value_labels is None else bool(value_labels),
                        grid=bool(d < 0.6),
                        similar=(d > 0.7) if similar_hues is None else bool(similar_hues))
            a, b = rng.sample(range(ns), 2)
            spec["pair"] = [a, b]
            spec["cat"] = rng.randrange(ncat)
            spec["ser"] = rng.randrange(ns)
            if self.valid(spec):
                return spec
        raise RuntimeError("chart spec")

    def valid(self, spec):
        v = spec["vals"]
        if spec["qtype"] == "gap":
            a, b = spec["pair"]
            g = sorted((v[a][c] - v[b][c] for c in range(spec["ncat"])), reverse=True)
            return g[0] - g[1] >= 7
        col = sorted(v[s][spec["cat"]] for s in range(spec["ns"]))
        return all(col[i + 1] - col[i] >= 5 for i in range(len(col) - 1))

    def render(self, spec):
        lay = layout(spec)
        cv = Canvas(W, H)
        cols = SHADES if spec["similar"] else PAL
        ymin, ymax, ppu, yb = lay["ymin"], lay["ymax"], lay["ppu"], lay["ybase"]
        step = 20
        for v in range((ymin // step) * step if ymin % step == 0 else ymin, ymax + 1, step):
            y = yb - (v - ymin) * ppu
            if spec["grid"]:
                cv.line([(L, y), (W - RM, y)], (220, 220, 220), 1)
            cv.text((L - 8, y), v, 13, anchor="rm")
            cv.line([(L - 4, y), (L, y)], (0, 0, 0), 1)
        cv.line([(L, T), (L, yb), (W - RM, yb)], (0, 0, 0), 2)
        for ci in range(spec["ncat"]):
            for si in range(spec["ns"]):
                x0, x1 = lay["bar"](ci, si)
                v = spec["vals"][si][ci]
                cv.rect(x0, yb - (v - ymin) * ppu, x1, yb, fill=cols[si])
                if spec["value_labels"]:
                    cv.text(((x0 + x1) / 2, yb - (v - ymin) * ppu - 9), v, 11)
            cv.text((L + (ci + .5) * lay["gw"], yb + 16), spec["cats"][ci], 14)
        lx = L
        for si in range(spec["ns"]):
            cv.rect(lx, H - 40, lx + 16, H - 24, fill=cols[si])
            cv.text((lx + 22, H - 32), spec["names"][si], 14, anchor="lm")
            lx += 120
        return cv.done()

    def check(self, spec, img):
        # re-read every bar height from pixels and compare with the data
        lay = layout(spec)
        a = np.asarray(img).astype(int)
        cols = SHADES if spec["similar"] else PAL
        for ci in range(spec["ncat"]):
            for si in range(spec["ns"]):
                x0, x1 = lay["bar"](ci, si)
                x = int((x0 + x1) / 2)
                y = int(lay["ybase"]) - 3
                tgt = np.array(cols[si])
                while y > 0 and np.abs(a[y, x] - tgt).max() < 14:
                    y -= 1
                v = spec["ymin"] + (lay["ybase"] - (y + 1)) / lay["ppu"]
                if abs(v - spec["vals"][si][ci]) > 1.6:
                    return False
        return True

    def qa(self, spec):
        v, n = spec["vals"], spec["names"]
        if spec["qtype"] == "gap":
            a, b = spec["pair"]
            gaps = [v[a][c] - v[b][c] for c in range(spec["ncat"])]
            return dict(question=f"For which category is the value of {n[a]} minus the value of {n[b]} the "
                                 f"largest (most positive)?", answer=spec["cats"][int(np.argmax(gaps))],
                        atype="choice", tol=0.0, choices=list(spec["cats"]))
        c, s = spec["cat"], spec["ser"]
        rank = 1 + sum(1 for k in range(spec["ns"]) if v[k][c] > v[s][c])
        return dict(question=f"In {spec['cats'][c]}, what is the rank of {n[s]} among all series by value "
                             f"(1 = highest)?", answer=rank, atype="int", tol=0.0)

    def text(self, spec):
        rows = [f"Grouped bar chart. y-axis from {spec['ymin']} to 100. Categories: {', '.join(spec['cats'])}."]
        for s, nm in enumerate(spec["names"]):
            rows.append(f"{nm}: " + ", ".join(f"{c}={v}" for c, v in zip(spec["cats"], spec["vals"][s])))
        return "\n".join(rows)

    def meta(self, spec):
        return dict(ns=spec["ns"], ncat=spec["ncat"], qtype=spec["qtype"])

    def _mutate(self, spec, rng):
        v = spec["vals"]
        if spec["qtype"] == "gap":
            a, b = spec["pair"]
            gaps = [v[a][c] - v[b][c] for c in range(spec["ncat"])]
            best = int(np.argmax(gaps))
            c2 = rng.choice([c for c in range(spec["ncat"]) if c != best])
            margin = rng.randrange(8, 14)
            raise_c2 = v[b][c2] + gaps[best] + margin          # make the runner-up category win ...
            lower_best = v[b][best] + gaps[c2] - margin        # ... or make the winner lose
            opts = []
            if raise_c2 <= 96:
                opts.append((a, c2, raise_c2))
            if lower_best >= spec["ymin"] + 4:
                opts.append((a, best, lower_best))
            if not opts:
                return None
            s, c, val = rng.choice(opts)
            v[s][c] = val
            return spec
        c, s = spec["cat"], spec["ser"]
        order = sorted(range(spec["ns"]), key=lambda k: v[k][c])
        pos = order.index(s)
        nb = order[pos + 1] if pos + 1 < len(order) and rng.random() < .5 else order[pos - 1 if pos else 1]
        v[s][c], v[nb][c] = v[nb][c], v[s][c]
        return spec
