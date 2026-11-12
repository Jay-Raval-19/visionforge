"""Small supersampled drawing helpers (anti-aliased PIL canvas)."""
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=None)
def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


class Canvas:
    def __init__(self, w, h, ss=3, bg=(255, 255, 255)):
        self.w, self.h, self.ss = w, h, ss
        self.im = Image.new("RGB", (w * ss, h * ss), bg)
        self.dr = ImageDraw.Draw(self.im)

    def _p(self, pts):
        s = self.ss
        return [(x * s, y * s) for x, y in pts]

    def line(self, pts, fill=(0, 0, 0), width=2):
        self.dr.line(self._p(pts), fill=fill, width=max(1, int(round(width * self.ss))), joint="curve")

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1):
        s = self.ss
        self.dr.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], fill=fill, outline=outline,
                          width=max(1, int(round(width * s))))

    def ellipse(self, cx, cy, r, fill=None, outline=None, width=1):
        s = self.ss
        self.dr.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], fill=fill,
                        outline=outline, width=max(1, int(round(width * s))))

    def poly(self, pts, fill=None, outline=None):
        self.dr.polygon(self._p(pts), fill=fill, outline=outline)

    def text(self, xy, s, size=14, fill=(0, 0, 0), anchor="mm"):
        self.dr.text((xy[0] * self.ss, xy[1] * self.ss), str(s), font=font(int(size * self.ss)),
                     fill=fill, anchor=anchor)

    def done(self):
        return self.im.resize((self.w, self.h), Image.LANCZOS)


def catmull(points, n=40):
    """Catmull-Rom spline through `points`; returns a dense list of (x, y)."""
    P = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n
            out.append(tuple(
                0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t
                       + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t * t
                       + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t ** 3)
                for j in (0, 1)))
    out.append(tuple(points[-1]))
    return out
