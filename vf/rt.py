"""Tiny vectorised NumPy ray tracer for procedural 3D scenes (no Blender needed).

Primitives: box (with yaw), sphere, vertical cylinder, ground plane y=0 with a texture function, and
mirror boxes. Features: directional light with true shadows, per-object shadow-direction override (for
'wrong shadow' tasks), per-object offset in reflections (for 'wrong reflection' tasks), edge darkening on
boxes, and an object-id buffer so generators can measure exactly how visible each object is.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

EPS = 1e-4
INF = np.inf


def _norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _rot(yaw):
    return np.cos(yaw), np.sin(yaw)


def _to_local(v, cy, sy):
    out = v.copy()
    out[..., 0] = cy * v[..., 0] - sy * v[..., 2]
    out[..., 2] = sy * v[..., 0] + cy * v[..., 2]
    return out


def _to_world(v, cy, sy):
    out = v.copy()
    out[..., 0] = cy * v[..., 0] + sy * v[..., 2]
    out[..., 2] = -sy * v[..., 0] + cy * v[..., 2]
    return out


class Scene:
    def __init__(self, W=640, H=480, cam=(0, 6, 10), look=(0, 0, 0), fov=50, light=(-0.5, 1.0, 0.4),
                 ambient=0.38, sky=(205, 218, 235), ground=None):
        self.W, self.H = W, H
        self.cam = np.array(cam, float)
        f = _norm(np.array(look, float) - self.cam)
        r = _norm(np.cross(f, [0.0, 1.0, 0.0]))
        self.f, self.r, self.u = f, r, np.cross(r, f)
        self.th = np.tan(np.radians(fov) / 2)
        self.L = _norm(np.array(light, float))
        self.amb = ambient
        self.sky = np.array(sky, float)
        self.ground = ground   # None (no ground) or fn(x, z) -> (N, 3) colours
        self.objs = []

    # ------------------------------------------------------------ scene building
    def _add(self, o):
        o.setdefault("shadow_L", None)
        o.setdefault("refl_shift", None)
        o.setdefault("refl_hide", False)
        o.setdefault("refl_color", None)
        o.setdefault("color_fn", None)       # optional fn(world points (N,3)) -> colours (N,3)
        o.setdefault("mirror", False)
        o.setdefault("edges", False)
        o.setdefault("cast", True)
        self.objs.append(o)
        return len(self.objs) - 1

    def box(self, c, size, color, yaw=0.0, **kw):
        return self._add(dict(kind="box", c=np.array(c, float), h=np.array(size, float) / 2, yaw=float(yaw),
                              color=np.array(color, float), **kw))

    def sphere(self, c, r, color, **kw):
        return self._add(dict(kind="sphere", c=np.array(c, float), r=float(r), color=np.array(color, float), **kw))

    def cyl(self, base, r, h, color, **kw):
        return self._add(dict(kind="cyl", c=np.array(base, float), r=float(r), hh=float(h),
                              color=np.array(color, float), **kw))

    # ------------------------------------------------------------ geometry
    @staticmethod
    def _shift(o, off):
        o2 = dict(o)
        o2["c"] = o["c"] + np.asarray(off, float)
        return o2

    @staticmethod
    def _hit(o, O, D):
        P = O - o["c"]
        k = o["kind"]
        if k == "box":
            cy, sy = _rot(o["yaw"])
            Pl, Dl = _to_local(P, cy, sy), _to_local(D, cy, sy)
            Dl = np.where(np.abs(Dl) < 1e-12, 1e-12, Dl)
            t1, t2 = (-o["h"] - Pl) / Dl, (o["h"] - Pl) / Dl
            tmin, tmax = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
            t = np.where(tmin > EPS, tmin, tmax)
            return np.where((tmax >= tmin) & (t > EPS), t, INF)
        if k == "sphere":
            b = (P * D).sum(1)
            c = (P * P).sum(1) - o["r"] ** 2
            disc = b * b - c
            sq = np.sqrt(np.maximum(disc, 0))
            t = np.where(-b - sq > EPS, -b - sq, -b + sq)
            return np.where((disc >= 0) & (t > EPS), t, INF)
        # vertical cylinder, base at c, height hh
        r, hh = o["r"], o["hh"]
        a = D[:, 0] ** 2 + D[:, 2] ** 2
        b = P[:, 0] * D[:, 0] + P[:, 2] * D[:, 2]
        c = P[:, 0] ** 2 + P[:, 2] ** 2 - r * r
        disc = b * b - a * c
        sq = np.sqrt(np.maximum(disc, 0))
        a_ = np.where(a < 1e-12, 1e-12, a)
        best = np.full(len(D), INF)
        for t in ((-b - sq) / a_, (-b + sq) / a_):
            y = P[:, 1] + t * D[:, 1]
            ok = (disc >= 0) & (t > EPS) & (y >= 0) & (y <= hh)
            best = np.where(ok & (t < best), t, best)
        Dy = np.where(np.abs(D[:, 1]) < 1e-12, 1e-12, D[:, 1])
        for yc in (0.0, hh):
            t = (yc - P[:, 1]) / Dy
            x, z = P[:, 0] + t * D[:, 0], P[:, 2] + t * D[:, 2]
            ok = (t > EPS) & (x * x + z * z <= r * r)
            best = np.where(ok & (t < best), t, best)
        return best

    @staticmethod
    def _normal(o, X):
        P = X - o["c"]
        k = o["kind"]
        if k == "box":
            cy, sy = _rot(o["yaw"])
            q = _to_local(P, cy, sy) / o["h"]
            ax = np.abs(q).argmax(1)
            n = np.zeros_like(q)
            n[np.arange(len(q)), ax] = np.sign(q[np.arange(len(q)), ax])
            return _to_world(n, cy, sy), q
        if k == "sphere":
            return P / o["r"], None
        n = np.zeros_like(P)
        top = P[:, 1] > o["hh"] - 1e-3
        bot = P[:, 1] < 1e-3
        side = ~(top | bot)
        n[top, 1], n[bot, 1] = 1, -1
        n[side, 0], n[side, 2] = P[side, 0] / o["r"], P[side, 2] / o["r"]
        return n, None

    def _trace(self, O, D, objs):
        t_best = np.full(len(D), INF)
        idx = np.full(len(D), -2)
        if self.ground is not None:
            Dy = np.where(D[:, 1] < -1e-9, D[:, 1], -1e-9)
            t = np.where(D[:, 1] < -1e-9, -O[:, 1] / Dy, INF)
            t = np.where(t > EPS, t, INF)
            sel = t < t_best
            t_best[sel], idx[sel] = t[sel], -1
        for i, o in enumerate(objs):
            t = self._hit(o, O, D)
            sel = t < t_best
            t_best[sel], idx[sel] = t[sel], i
        return t_best, idx

    def _shade(self, O, D, objs, allow_mirror=True):
        t, idx = self._trace(O, D, objs)
        col = np.tile(self.sky, (len(D), 1))
        hit = idx > -2
        X = O + D * np.where(hit, t, 0)[:, None]
        N = np.zeros_like(X)
        base = np.zeros_like(X)
        edge = np.zeros(len(D))
        if self.ground is not None:
            g = idx == -1
            if g.any():
                N[g] = (0, 1, 0)
                base[g] = self.ground(X[g, 0], X[g, 2])
        for i, o in enumerate(objs):
            m = idx == i
            if not m.any():
                continue
            n, q = self._normal(o, X[m])
            N[m] = n
            base[m] = o["color"] if o["color_fn"] is None else o["color_fn"](X[m])
            if o["edges"] and q is not None:
                a = np.sort(np.abs(q), 1)
                edge[np.where(m)[0]] = (a[:, 1] > 1 - 0.06 / np.maximum(o["h"].min(), 0.05) * 0.5).astype(float)
        diff = np.clip((N * self.L).sum(1), 0, 1)
        shadow = np.zeros(len(D), bool)
        hs = np.where(hit)[0]
        if len(hs):
            Xs = X[hs] + N[hs] * 2e-3
            for o in objs:
                if not o["cast"] or o["mirror"]:
                    continue
                Lk = self.L if o["shadow_L"] is None else _norm(np.array(o["shadow_L"], float))
                tt = self._hit(o, Xs, np.tile(Lk, (len(hs), 1)))
                shadow[hs] |= tt < INF
        lit = self.amb + (1 - self.amb) * diff * (~shadow)
        col[hit] = base[hit] * lit[hit, None]
        col[edge > 0] *= 0.45
        if allow_mirror:
            mi = [i for i, o in enumerate(objs) if o["mirror"]]
            for i in mi:
                m = np.where(idx == i)[0]
                if not len(m):
                    continue
                n = N[m]
                Dr = D[m] - 2 * (D[m] * n).sum(1)[:, None] * n
                robjs = []
                for o in objs:
                    if o["mirror"] or o["refl_hide"]:
                        continue
                    o2 = self._shift(o, o["refl_shift"]) if o["refl_shift"] is not None else o
                    if o["refl_color"] is not None:
                        o2 = dict(o2)
                        o2["color"] = np.array(o["refl_color"], float)
                    robjs.append(o2)
                rc, _ = self._shade(X[m] + n * 2e-3, Dr, robjs, allow_mirror=False)
                col[m] = 0.88 * rc + 0.12 * objs[i]["color"]
        return col, idx

    # ------------------------------------------------------------ public
    def render(self, ss=2):
        W, H = self.W * ss, self.H * ss
        aspect = W / H
        jj, ii = np.meshgrid(np.arange(W), np.arange(H))
        x = (2 * (jj + 0.5) / W - 1) * self.th * aspect
        y = (1 - 2 * (ii + 0.5) / H) * self.th
        D = _norm(x.reshape(-1, 1) * self.r + y.reshape(-1, 1) * self.u + self.f)
        O = np.tile(self.cam, (len(D), 1))
        col, idx = self._shade(O, D, self.objs)
        img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8).reshape(H, W, 3))
        if ss > 1:
            img = img.resize((self.W, self.H), Image.LANCZOS)
        self.idbuf = idx.reshape(H, W)[ss // 2::ss, ss // 2::ss]
        return img

    def visible_px(self, i):
        return int((self.idbuf == i).sum())

    def project(self, p):
        v = np.asarray(p, float) - self.cam
        zc = v @ self.f
        if zc <= 1e-6:
            return None
        aspect = self.W / self.H
        u = (v @ self.r / zc / (self.th * aspect) + 1) / 2 * self.W
        w = (1 - v @ self.u / zc / self.th) / 2 * self.H
        return float(u), float(w)

    def proj_radius(self, p, r):
        """Approximate on-screen radius (px) of a sphere of radius r at p."""
        zc = (np.asarray(p, float) - self.cam) @ self.f
        return r / zc / self.th * self.H / 2


def floor_tex(c1=(190, 180, 165), c2=(175, 165, 150), s=1.0):
    c1, c2 = np.array(c1, float), np.array(c2, float)

    def f(x, z):
        chk = ((np.floor(x / s) + np.floor(z / s)) % 2)[:, None]
        return c1 * (1 - chk) + c2 * chk
    return f


def flat_tex(c):
    c = np.array(c, float)
    return lambda x, z: np.tile(c, (len(x), 1))
