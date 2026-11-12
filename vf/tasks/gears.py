"""Realistic gear trains: machined steel/brass spur gears with interlocking teeth on a brushed steel plate.

Four independent difficulty knobs (each stored with the sample, and each overridable):
  path_len  meshes between the driving gear and the '?' gear
  n_total   total number of gears in the picture
  n_hang    "hanging" gears: placed next to the train but not meshing with anything
  gap_px    tooth-tip gap between each hanging gear and its nearest gear (big = obvious, ~3 px = near miss)
The answer comes from geometry alone: two gears mesh iff their centre distance equals the sum of pitch radii.
"""
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ..taskapi import Reject, ilerp, lerp, task

M = 7.0                    # module in pixels (tooth size); all gears share it so they can mesh
W, H = 1100, 820
SS = 2                     # supersampling
MATERIALS = {"steel": np.array((176, 182, 190.)), "brass": np.array((196, 158, 82.)),
             "dark steel": np.array((120, 126, 134.))}


def radii(N, m=M):
    rp = m * N / 2
    return rp, rp + m, rp - 1.25 * m           # pitch, tip (addendum), root (dedendum)


def gear_poly(cx, cy, N, phase, s=1.0, m=M):
    rp, ra, rf = radii(N, m)
    pitch = 2 * math.pi / N
    w_root, w_tip = .30 * pitch, .15 * pitch
    pts = []
    for k in range(N):
        a = phase + k * pitch
        for t in np.linspace(a - pitch + w_root, a - w_root, 4):           # root arc before the tooth
            pts.append((cx + rf * math.cos(t), cy + rf * math.sin(t)))
        for t in np.linspace(a - w_tip, a + w_tip, 4):                     # flank up, tip arc
            pts.append((cx + ra * math.cos(t), cy + ra * math.sin(t)))
        pts.append((cx + rf * math.cos(a + w_root), cy + rf * math.sin(a + w_root)))
    return [(x * s, y * s) for x, y in pts]


def _mask(poly, size):
    im = Image.new("L", size, 0)
    ImageDraw.Draw(im).polygon(poly, fill=255)
    return np.asarray(im) > 0


def mesh_phase(parent, child):
    """Phase for `child` so its teeth sit in the parent's gaps (searched numerically: least overlap)."""
    (px, py, pN, pph), (cx, cy, cN) = parent, child
    mx, my = (px + cx) / 2, (py + cy) / 2
    win = int(6 * M)
    ox, oy = mx - win / 2, my - win / 2
    pm = _mask([(x - ox, y - oy) for x, y in gear_poly(px, py, pN, pph)], (win, win))
    pitch = 2 * math.pi / cN
    cands = np.linspace(0, pitch, 36, endpoint=False)
    ov = [(pm & _mask([(x - ox, y - oy) for x, y in gear_poly(cx, cy, cN, ph)], (win, win))).sum() for ph in cands]
    best = min(ov)
    good = [ph for ph, o in zip(cands, ov) if o <= best]
    return float(good[len(good) // 2])


@task("gear_train", "Mechanical engineering / robotics: predicting motion through a gear train",
      "physical reasoning through linkages", knob="path_len, n_total, n_hang, gap_px (independent)",
      twin="text: gear centres, tooth counts and which pairs mesh", group="puzzle")
def gear_train(rng, d, path_len=None, n_total=None, n_hang=None, gap_px=None):
    L = path_len if path_len is not None else ilerp(1, 8, d)
    nh = n_hang if n_hang is not None else ilerp(0, 5, d)
    extra = ilerp(0, 5, d)
    nt = n_total if n_total is not None else L + 1 + extra + nh
    extra = max(0, nt - (L + 1) - nh)
    gap = gap_px if gap_px is not None else lerp(40, 3, d)
    CLEAR = max(12.0, gap + 8)               # any other non-meshing pair is clearly apart
    gears, parent = [], []                   # gear = [x, y, N, phase]

    def ok(x, y, N, mesh_with=None, tip_gap=None):
        _, ra, _ = radii(N)
        if not (ra + 25 < x < W - ra - 25 and ra + 25 < y < H - ra - 25):
            return False
        for k, (gx, gy, gN, _) in enumerate(gears):
            dd = math.hypot(x - gx, y - gy)
            need = radii(gN)[1] + ra + (CLEAR if tip_gap is None else min(CLEAR, tip_gap))
            if k == mesh_with:
                continue
            if dd < need - 1e-6:
                return False
        return True

    def attach(p, N, tries=200):
        """Place a new gear meshing with gear p."""
        gx, gy, gN, gph = gears[p]
        for _ in range(tries):
            a = rng.uniform(0, 2 * math.pi)
            dist = radii(gN)[0] + radii(N)[0]
            x, y = gx + dist * math.cos(a), gy + dist * math.sin(a)
            if ok(x, y, N, mesh_with=p):
                return x, y
        return None
    N0 = rng.randint(14, 30)
    gears.append([rng.uniform(150, 350), rng.uniform(200, H - 200), N0, rng.uniform(0, 1)])
    parent.append(-1)
    for _ in range(L):                       # the driving path, gear 0 -> gear L
        N = rng.randint(12, 30)
        pos = attach(len(gears) - 1, N)
        if pos is None:
            raise Reject
        gears.append([pos[0], pos[1], N, 0.0])
        parent.append(len(gears) - 2)
    for _ in range(extra):                   # extra connected gears branching off the train
        N = rng.randint(12, 26)
        p = rng.randrange(len(gears))
        pos = attach(p, N)
        if pos is None:
            raise Reject
        gears.append([pos[0], pos[1], N, 0.0])
        parent.append(p)
    n_conn = len(gears)
    for i in range(1, n_conn):               # phase each connected gear to interlock with its parent
        gears[i][3] = mesh_phase(gears[parent[i]], gears[i][:3])
    hang_gap = []
    for _ in range(nh):                      # hanging gears: next to a gear, teeth NOT touching
        for _t in range(300):
            q = rng.randrange(len(gears))
            gx, gy, gN, _ = gears[q]
            N = rng.randint(12, 26)
            a = rng.uniform(0, 2 * math.pi)
            dist = radii(gN)[1] + radii(N)[1] + gap
            x, y = gx + dist * math.cos(a), gy + dist * math.sin(a)
            if ok(x, y, N, tip_gap=gap):
                gears.append([x, y, N, rng.uniform(0, 1)])
                parent.append(None)
                break
        else:
            raise Reject
    # minimum tip gap from every hanging gear to any other gear (what a viewer has to resolve)
    for i in range(n_conn, len(gears)):
        hang_gap.append(min(math.hypot(gears[i][0] - g[0], gears[i][1] - g[1]) - radii(gears[i][2])[1] - radii(g[2])[1]
                            for k, g in enumerate(gears) if k != i))
    target = L
    if nh and rng.random() < 0.2:
        target = rng.randrange(n_conn, len(gears))
    cw = rng.random() < .5
    if target >= n_conn:
        ans = "does not turn"
    else:
        same = (L % 2 == 0)
        ans = "clockwise" if cw == same else "counterclockwise"
    img = _render(rng, gears, target, cw)
    q = ("The gear with the red arrow is turned in the direction of the arrow. Gears drive each other only where "
         "their teeth interlock; a gear whose teeth do not touch any other gear does not move. Which way does the "
         "gear marked '?' turn: clockwise, counterclockwise, or does not turn?")
    return dict(image=img, atype="choice", choices=["clockwise", "counterclockwise", "does not turn"], answer=ans,
                question=q,
                spec=dict(module=M, gears=[[round(g[0], 2), round(g[1], 2), g[2], round(g[3], 4)] for g in gears],
                          driver=0, driver_cw=cw, query=target, path_len=L if target < n_conn else None,
                          n_total=len(gears), n_hang=nh, gap_px=round(min(hang_gap), 2) if hang_gap else None))


# ---------------------------------------------------------------- rendering
def _plate(rng):
    h, w = H * SS, W * SS
    r = np.random.default_rng(rng.randrange(1 << 30))
    brushed = ndimage.gaussian_filter(r.normal(0, 1, (h // 2, w // 2)), (0.6, 30))
    brushed = np.kron(brushed, np.ones((2, 2)))[:h, :w]
    base = 78 + 9 * brushed / (brushed.std() + 1e-9)
    yy, xx = np.mgrid[0:h, 0:w]
    vign = 1 - .25 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 2
    img = np.stack([base * vign * 0.98, base * vign, base * vign * 1.04], -1)
    for cx, cy in ((40, 40), (W - 40, 40), (40, H - 40), (W - 40, H - 40)):   # corner screws
        _screw(img, cx * SS, cy * SS, 11 * SS)
    return img


def _screw(img, cx, cy, r):
    y0, y1, x0, x1 = int(cy - r - 2), int(cy + r + 2), int(cx - r - 2), int(cx + r + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    m = d < r
    shade = 150 - 60 * ((xx - cx) + (yy - cy)) / (2 * r)
    sub = img[y0:y1, x0:x1]
    sub[m] = np.stack([shade[m]] * 3, -1)
    slot = m & (np.abs((yy - cy) - (xx - cx) * .3) < r * .18)
    sub[slot] = 50


def _render(rng, gears, target, cw):
    img = _plate(rng)
    h, w = img.shape[:2]
    L_dir = np.array((-0.7, -0.7))                     # light from the top-left
    # drop shadows of all gears
    sh = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(sh)
    for x, y, N, ph in gears:
        sd.polygon(gear_poly(x + 7, y + 9, N, ph, SS), fill=255)
    shadow = ndimage.gaussian_filter(np.asarray(sh).astype(float) / 255, 6 * SS)
    img *= (1 - .55 * shadow)[..., None]
    mats = list(MATERIALS)
    for gi, (x, y, N, ph) in enumerate(gears):
        rp, ra, rf = radii(N)
        mat = MATERIALS[mats[rng.randrange(len(mats))]]
        x0, y0 = int((x - ra - 4) * SS), int((y - ra - 4) * SS)
        x1, y1 = int((x + ra + 4) * SS) + 1, int((y + ra + 4) * SS) + 1
        size = (x1 - x0, y1 - y0)
        poly = [(px - x0, py - y0) for px, py in gear_poly(x, y, N, ph, SS)]
        m = _mask(poly, size)
        yy, xx = np.mgrid[0:size[1], 0:size[0]]
        cx, cy = x * SS - x0, y * SS - y0
        rr = np.hypot(xx - cx, yy - cy)
        th = np.arctan2(yy - cy, xx - cx)
        holes = np.zeros_like(m)
        if rf > 60:                                     # lightening holes in big gears
            nh = 5 if rf > 85 else 4
            for k in range(nh):
                a = ph + 2 * math.pi * (k + .5) / nh
                hx, hy = cx + .62 * rf * SS * math.cos(a), cy + .62 * rf * SS * math.sin(a)
                holes |= np.hypot(xx - hx, yy - hy) < .19 * rf * SS
        bore = rr < .13 * rp * SS + 3
        body = m & ~holes & ~bore
        dt = ndimage.distance_transform_edt(body)
        gy_, gx_ = np.gradient(dt)
        nrm = np.hypot(gx_, gy_) + 1e-9
        facing = -(gx_ * L_dir[0] + gy_ * L_dir[1]) / nrm     # outward normal . light
        bevel = np.clip(1 - dt / (4 * SS), 0, 1)
        rings = .035 * np.sin(rr * .9 + 3 * np.sin(th * 7)) + .02 * np.random.default_rng(gi).normal(0, 1, rr.shape)
        lit = 1 + rings + .45 * bevel * facing - .06 * (rr / (ra * SS))
        hub = rr < .34 * rf * SS
        lit = np.where(hub & body, lit * .9 + .05, lit)
        col = mat[None, None, :] * lit[..., None]
        sub = img[y0:y1, x0:x1]
        sub[body] = col[body]
        # hole walls: darker plate seen through, with an inner shadow
        if holes.any():
            sub[holes & m] *= .55
        shaft = rr < .13 * rp * SS
        sub[bore & ~shaft & m] = 30
        shade = 120 + 60 * np.clip(-(xx - cx + yy - cy) / (.13 * rp * SS * 1.4), -1, 1)
        sub[shaft] = np.stack([shade[shaft]] * 3, -1)
        key = shaft & (np.abs(yy - cy - 0) < .03 * rp * SS + 1) & (xx - cx > .05 * rp * SS)
        sub[key] = 40
    img = np.clip(img, 0, 255).astype(np.uint8)
    out = Image.fromarray(img).resize((W, H), Image.LANCZOS)
    dr = ImageDraw.Draw(out)
    # arrow painted on the driving gear
    x, y, N, _ = gears[0]
    rp = radii(N)[0]
    R = rp * .72
    a0, a1 = (-2.4, -0.7) if cw else (-0.7, -2.4)
    arc = [(x + R * math.cos(t), y + R * math.sin(t)) for t in np.linspace(a0, a1, 30)]
    dr.line(arc, fill=(215, 35, 30), width=6)
    (ex, ey), (px_, py_) = arc[-1], arc[-3]
    dx, dy = ex - px_, ey - py_
    n = math.hypot(dx, dy)
    dx, dy = dx / n, dy / n
    dr.polygon([(ex + dx * 14, ey + dy * 14), (ex - dy * 10, ey + dx * 10), (ex + dy * 10, ey - dx * 10)],
               fill=(215, 35, 30))
    # '?' tag on the target gear's hub
    tx, ty = gears[target][0], gears[target][1]
    dr.ellipse([tx - 17, ty - 17, tx + 17, ty + 17], fill=(250, 250, 245), outline=(30, 30, 30), width=2)
    from ..draw import font
    dr.text((tx, ty), "?", font=font(24), fill=(200, 30, 30), anchor="mm")
    return out
