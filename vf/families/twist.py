"""'Familiar thing, one detail changed': evidence over memory (VLMBias). Control = canonical version."""
import math

from ..core import Family, register
from ..draw import Canvas


@register
class Twist(Family):
    name = "twist"
    application = "Reporting what is in the image even when it looks unusual (QA, inspection)"
    capability = "evidence over memory"
    toggles = {}

    def make_spec(self, rng, d):
        sub = rng.choice(["chess", "clock"])
        canon = 8 if sub == "chess" else 12
        delta = 1 + round(d * 5)
        sign = 1 if d > 0.5 else rng.choice([-1, 1])  # high d: more elements, so counting is also harder
        n = canon + sign * delta
        if sub == "chess":
            n = max(4, n)
        else:
            n = max(5, n)
        if n == canon:
            n += 1
        return dict(sub=sub, n=n, canon=canon, hands=[rng.uniform(0, 360), rng.uniform(0, 360)])

    def render(self, spec):
        n = spec["n"]
        if spec["sub"] == "chess":
            s = 44
            size = n * s
            cv = Canvas(size + 40, size + 40)
            for r in range(n):
                for c in range(n):
                    col = (240, 217, 181) if (r + c) % 2 == 0 else (181, 136, 99)
                    cv.rect(20 + c * s, 20 + r * s, 20 + (c + 1) * s, 20 + (r + 1) * s, fill=col)
            cv.rect(20, 20, 20 + size, 20 + size, outline=(40, 30, 20), width=3)
            return cv.done()
        S, R = 480, 200
        cx = cy = S / 2
        cv = Canvas(S, S)
        cv.ellipse(cx, cy, R + 14, fill=(252, 252, 252), outline=(20, 20, 20), width=4)
        for k in range(n):
            th = 2 * math.pi * k / n - math.pi / 2
            cv.line([(cx + math.cos(th) * (R - 14), cy + math.sin(th) * (R - 14)),
                     (cx + math.cos(th) * R, cy + math.sin(th) * R)], (20, 20, 20), 3)
        for h in range(1, 13):
            th = 2 * math.pi * h / 12 - math.pi / 2
            cv.text((cx + math.cos(th) * (R - 44), cy + math.sin(th) * (R - 44)), h, 22)
        for ang, ln, w in ((spec["hands"][0], .5, 6), (spec["hands"][1], .75, 4)):
            th = math.radians(ang) - math.pi / 2
            cv.line([(cx, cy), (cx + math.cos(th) * R * ln, cy + math.sin(th) * R * ln)], (20, 20, 20), w)
        cv.ellipse(cx, cy, 7, fill=(20, 20, 20))
        return cv.done()

    def qa(self, spec):
        if spec["sub"] == "chess":
            return dict(question="How many squares are there along the bottom row of this board?",
                        answer=spec["n"], atype="int", tol=0.0)
        return dict(question="How many short tick marks are there around the rim of this clock face? "
                             "(Do not count the numerals.)", answer=spec["n"], atype="int", tol=0.0)

    def text(self, spec):
        if spec["sub"] == "chess":
            return f"A {spec['n']} x {spec['n']} board of alternating light and dark squares."
        return f"A clock face with numerals 1 to 12 and {spec['n']} evenly spaced tick marks around the rim."

    def meta(self, spec):
        return dict(sub=spec["sub"], prior=spec["canon"], is_twist=spec["n"] != spec["canon"])

    def _mutate(self, spec, rng):
        spec["n"] = spec["canon"]  # the edit is the canonical (control) version
        return spec
