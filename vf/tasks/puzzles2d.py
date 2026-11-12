"""2D reasoning puzzles (doc tasks 1-13 + shell game). Tangled wires (14) lives in families/wire_trace.py."""
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

from ..draw import Canvas, font
from ..taskapi import LETTERS, Reject, autocrop, ilerp, lerp, seg_dist, tag, task

G = "puzzle"
PAL = [(220, 70, 60), (50, 110, 210), (40, 160, 90), (240, 150, 30), (140, 70, 190), (20, 160, 170),
       (200, 70, 150), (120, 120, 120), (160, 110, 50)]


def _mask(W, H, kind, x, y, r, rot=0.0):
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    dr.polygon(_shape_pts(kind, x, y, r, rot), fill=255) if kind != "circle" else \
        dr.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return np.asarray(im) > 0


def _shape_pts(kind, x, y, r, rot=0.0):
    n = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}[kind]
    return [(x + r * math.cos(rot + 2 * math.pi * k / n), y + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def _draw_shape(cv, s, col, outline=(30, 30, 30)):
    if s["kind"] == "circle":
        cv.ellipse(s["x"], s["y"], s["r"], fill=col, outline=outline, width=1.5)
    else:
        cv.poly(_shape_pts(s["kind"], s["x"], s["y"], s["r"], s["rot"]), fill=col, outline=outline)


# ---------------------------------------------------------------- 1 hidden outline
@task("hidden_outline", "Robot grasping / AR: inferring the hidden part of an occluded object",
      "amodal completion", knob="occluder size, spacing of candidate points", twin="text: coordinates", group=G)
def hidden_outline(rng, d):
    W, H = 560, 440
    P = [(rng.uniform(60, 500), rng.uniform(60, 380)) for _ in range(3)]
    area = abs((P[1][0] - P[0][0]) * (P[2][1] - P[0][1]) - (P[2][0] - P[0][0]) * (P[1][1] - P[0][1])) / 2
    if area < 20000:
        raise Reject
    hv = P[2]
    rad = lerp(55, 95, d)
    cx, cy = hv[0] + rng.uniform(-.35, .35) * rad, hv[1] + rng.uniform(-.35, .35) * rad
    if not (rad + 5 < cx < W - rad - 5 and rad + 5 < cy < H - rad - 5):
        raise Reject
    for v in P[:2]:
        if math.hypot(v[0] - cx, v[1] - cy) < rad + 60:
            raise Reject
    sp = lerp(0.75, 0.32, d) * rad
    cands = [hv]
    for _ in range(400):
        if len(cands) == 4:
            break
        a, rr = rng.uniform(0, 2 * math.pi), rad * math.sqrt(rng.uniform(0, 1)) * 0.85
        p = (cx + rr * math.cos(a), cy + rr * math.sin(a))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > sp for q in cands):
            cands.append(p)
    if len(cands) < 4:
        raise Reject
    rng.shuffle(cands)
    ans = LETTERS[cands.index(hv)]
    cv = Canvas(W, H)
    cv.line([P[0], P[1], P[2], P[0]], (90, 90, 90), 4)
    cv.ellipse(cx, cy, rad, fill=(235, 110, 60))
    for i, p in enumerate(cands):
        cv.ellipse(p[0], p[1], 5, fill=(20, 20, 20))
        cv.text((p[0] + 13, p[1] - 12), LETTERS[i], 16, fill=(255, 255, 255))
    return dict(image=cv.done(), atype="choice", choices=list("ABCD"), answer=ans,
                question="One corner of the gray triangle is hidden behind the orange disc. Which labeled point "
                         "(A, B, C or D) marks exactly where that hidden corner is?",
                spec=dict(triangle=P, disc=[cx, cy, rad], candidates=cands))


# ---------------------------------------------------------------- 2 layer shuffle count
def _stack_scene(rng, n, W, H, spread, rmin, rmax):
    kinds = ["circle", "square", "triangle", "pentagon", "hexagon"]
    shapes = []
    for i in range(n):
        shapes.append(dict(kind=rng.choice(kinds), x=W / 2 + rng.uniform(-spread, spread),
                           y=H / 2 + rng.uniform(-spread * .7, spread * .7), r=rng.uniform(rmin, rmax),
                           rot=rng.uniform(0, 2 * math.pi), col=i))
    return shapes


def _order_visible(shapes, order, W, H):
    """True if, for every overlapping pair, a pixel of their overlap is visible (so their order is readable)."""
    masks = [_mask(W, H, s["kind"], s["x"], s["y"], s["r"], s["rot"]) for s in shapes]
    pos = {s: k for k, s in enumerate(order)}            # order[0] = bottom
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            ov = masks[i] & masks[j]
            if ov.sum() == 0:
                continue
            hi = max(pos[i], pos[j])
            cover = np.zeros_like(ov)
            for k in range(len(shapes)):
                if pos[k] > hi:
                    cover |= masks[k]
            if (ov & ~cover).sum() < 60:
                return None
    return masks


@task("layer_shuffle", "Document/UI compositing and scene editing: reading stacking order",
      "occlusion order", knob="number of shapes", twin="text: two bottom-to-top orders", group=G)
def layer_shuffle(rng, d):
    W, H = 380, 320
    n = ilerp(3, 6, d)
    shapes = _stack_scene(rng, n, W, H, 38, 60, 82)
    o1 = list(range(n))
    rng.shuffle(o1)
    o2 = o1[:]
    for _ in range(ilerp(1, 6, d) + rng.randrange(2)):
        k = rng.randrange(n - 1)
        o2[k], o2[k + 1] = o2[k + 1], o2[k]
    if o1 == o2:
        raise Reject
    for i in range(n):  # every pair must overlap so the full order is determined
        for j in range(i + 1, n):
            if (_mask(W, H, shapes[i]["kind"], shapes[i]["x"], shapes[i]["y"], shapes[i]["r"], shapes[i]["rot"]) &
                    _mask(W, H, shapes[j]["kind"], shapes[j]["x"], shapes[j]["y"], shapes[j]["r"], shapes[j]["rot"])).sum() < 300:
                raise Reject
    if _order_visible(shapes, o1, W, H) is None or _order_visible(shapes, o2, W, H) is None:
        raise Reject
    p2 = {s: k for k, s in enumerate(o2)}
    inv = sum(1 for a in range(n) for b in range(a + 1, n) if p2[o1[a]] > p2[o1[b]])
    cv = Canvas(2 * W + 60, H + 30)
    for off, order, lab in ((0, o1, "Image 1"), (W + 60, o2, "Image 2")):
        for s_idx in order:
            s = dict(shapes[s_idx])
            s["x"] += off
            _draw_shape(cv, s, PAL[s["col"]])
        cv.text((off + W / 2, H + 12), lab, 16)
    cv.line([(W + 10, H / 2), (W + 50, H / 2)], (80, 80, 80), 3)
    cv.poly([(W + 50, H / 2 - 7), (W + 58, H / 2), (W + 50, H / 2 + 7)], fill=(80, 80, 80))
    return dict(image=cv.done(), atype="int", answer=inv,
                question="Image 2 contains the same shapes as Image 1, but their stacking (front-to-back) order "
                         "changed. What is the minimum number of swaps of two adjacent layers that turns Image 1's "
                         "order into Image 2's order?",
                spec=dict(shapes=shapes, order1=o1, order2=o2))


# ---------------------------------------------------------------- 3 order frames (domino chain)
@task("order_frames", "Video understanding: putting events in time order", "temporal ordering",
      knob="number of frames, smaller progress between frames", twin="text: fallen count per frame", group=G)
def order_frames(rng, d):
    F = ilerp(3, 6, d)
    N = ilerp(8, 16, d)
    ms = sorted(rng.sample(range(0, N + 1), F))
    fw, fh = 300, 150
    cols = 3
    rows = math.ceil(F / cols)
    perm = list(range(F))
    rng.shuffle(perm)
    cv = Canvas(cols * fw + 20, rows * (fh + 30) + 10)
    gap = (fw - 40) / N
    for slot, k in enumerate(perm):
        ox, oy = 10 + (slot % cols) * fw, 10 + (slot // cols) * (fh + 30)
        cv.rect(ox + 5, oy, ox + fw - 5, oy + fh, outline=(150, 150, 150), width=1.5)
        base = oy + fh - 20
        cv.line([(ox + 10, base), (ox + fw - 10, base)], (90, 90, 90), 2)
        for i in range(N):
            x = ox + 22 + i * gap
            hgt, wid = 55, 7
            if i < ms[k]:
                ang = math.radians(62 if i < ms[k] - 1 or ms[k] == N else 40)
                pts = [(x, base), (x + wid, base)]
                tx, ty = math.sin(ang) * hgt, math.cos(ang) * hgt
                pts = [(x, base), (x + wid * math.cos(ang), base + wid * math.sin(ang) * 0),
                       (x + wid * math.cos(ang) + tx, base - ty), (x + tx, base - ty - wid * math.sin(ang) * 0.3)]
                cv.poly(pts, fill=(60, 60, 70))
            else:
                cv.rect(x, base - hgt, x + wid, base, fill=(60, 60, 70))
        cv.text((ox + fw / 2, oy + fh + 14), f"Frame {LETTERS[slot]}", 15)
    ans = "".join(LETTERS[perm.index(k)] for k in range(F))
    return dict(image=cv.done(), atype="seq", answer=ans,
                question="These frames show a row of dominoes being knocked over from left to right, but the "
                         "frames are shuffled. List the frame letters in the order in which they happened.",
                spec=dict(fallen_per_frame=[ms[k] for k in perm], n_dominoes=N))


# ---------------------------------------------------------------- 4 missing puzzle piece
def _edge(p0, p1, kind, r):
    if kind == 0:
        return [p0, p1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = uy, -ux
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    pts = [p0]
    for k in range(17):
        t = math.pi * k / 16
        pts.append((mx - ux * r * math.cos(t) + kind * nx * r * math.sin(t),
                    my - uy * r * math.cos(t) + kind * ny * r * math.sin(t)))
    pts.append(p1)
    return pts


def _piece(x, y, s, e):  # e = (top, right, bottom, left), +1 tab out, -1 notch in, 0 flat
    r = s * 0.18
    c = [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    pts = []
    for k in range(4):
        pts += _edge(c[k], c[(k + 1) % 4], e[k], r)[:-1]
    return pts


@task("missing_piece", "Assembly / packing: which part fits a slot", "shape matching",
      knob="how similar the wrong pieces are (differ by one edge or a mirror flip)", twin="text: edge codes", group=G)
def missing_piece(rng, d):
    R = C = 3
    s = 80
    Hk = [[rng.choice([-1, 1]) for _ in range(C - 1)] for _ in range(R)]
    Vk = [[rng.choice([-1, 1]) for _ in range(C)] for _ in range(R - 1)]

    def edges(r, c):
        return (-Vk[r - 1][c] if r > 0 else 0, Hk[r][c] if c < C - 1 else 0,
                Vk[r][c] if r < R - 1 else 0, -Hk[r][c - 1] if c > 0 else 0)
    hr, hc = 1, 1
    true = edges(hr, hc)
    cands = {true}
    mirror = (true[0], -true[3] if true[3] else 0, true[2], -true[1] if true[1] else 0)
    mirror = (true[0], true[3], true[2], true[1])
    pool = []
    if d >= 0.5 and mirror != true:
        pool.append(mirror)
    for k in range(4):
        if true[k]:
            e = list(true)
            e[k] = -e[k]
            pool.append(tuple(e))
    rng.shuffle(pool)
    while len(cands) < 4:
        if d >= 0.5 and pool:
            cands.add(pool.pop())
        else:
            e = tuple(rng.choice([-1, 1]) for _ in range(4))
            if sum(a != b for a, b in zip(e, true)) >= 2:
                cands.add(e)
    cands = list(cands)
    rng.shuffle(cands)
    ans = LETTERS[cands.index(true)]
    cv = Canvas(3 * s + 80 + 4 * (s + 60), 3 * s + 80)
    for r in range(R):
        for c in range(C):
            if (r, c) == (hr, hc):
                continue
            cv.poly(_piece(40 + c * s, 40 + r * s, s, edges(r, c)), fill=(170, 200, 235), outline=(60, 80, 110))
    ox = 3 * s + 110
    for i, e in enumerate(cands):
        x = ox + i * (s + 50)
        cv.poly(_piece(x, 40 + s, s, e), fill=(170, 200, 235), outline=(60, 80, 110))
        cv.text((x + s / 2, 40 + 2 * s + 40), LETTERS[i], 18)
    return dict(image=cv.done(), atype="choice", choices=list("ABCD"), answer=ans,
                question="Which piece (A, B, C or D) fits exactly into the empty slot in the middle of the puzzle, "
                         "without rotating or flipping it?",
                spec=dict(slot_edges=true, candidates=cands))


# ---------------------------------------------------------------- 5 overlapping-circle regions
@task("circle_regions", "Maps / Venn diagrams / layouts: counting enclosed regions", "topological counting",
      knob="number of circles", twin="text: centres and radii", group=G)
def circle_regions(rng, d):
    W, H = 600, 480
    n = ilerp(3, 8, d)
    cs = []
    for _ in range(n):
        for _try in range(200):
            c = (rng.uniform(110, W - 110), rng.uniform(100, H - 100), rng.uniform(55, 95))
            ok = True
            for q in cs:
                dd = math.hypot(c[0] - q[0], c[1] - q[1])
                if abs(dd - (c[2] + q[2])) < 9 or abs(dd - abs(c[2] - q[2])) < 9 or dd < 12:
                    ok = False
                    break
            if ok:
                cs.append(c)
                break
        else:
            raise Reject
    big = Image.new("L", (W * 2, H * 2), 0)
    dr = ImageDraw.Draw(big)
    for x, y, r in cs:
        dr.ellipse([(x - r) * 2, (y - r) * 2, (x + r) * 2, (y + r) * 2], outline=255, width=3)
    free = np.asarray(big) == 0
    lab, k = ndimage.label(free)
    sizes = ndimage.sum(free, lab, range(1, k + 1))
    outer = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    regions = [i + 1 for i in range(k) if (i + 1) not in outer]
    if any(sizes[i - 1] < 300 for i in regions):
        raise Reject                        # a sliver region a human could miss -> ambiguous
    cv = Canvas(W, H)
    for x, y, r in cs:
        cv.ellipse(x, y, r, outline=(40, 100, 210), width=2.5)
    return dict(image=cv.done(), atype="int", answer=len(regions),
                question="The circle outlines divide the plane into areas. How many closed (fully enclosed) "
                         "regions are there? Count every bounded region separately, including lens-shaped overlaps.",
                spec=dict(circles=cs))


# ---------------------------------------------------------------- 6 layer stack
@task("layer_stack", "Bin picking / document layers: what must be removed first", "occlusion reasoning",
      knob="number of shapes", twin="text: shapes + bottom-to-top order", group=G)
def layer_stack(rng, d):
    W, H = 560, 440
    n = ilerp(4, 9, d)
    shapes = _stack_scene(rng, n, W, H, lerp(120, 160, d), 50, 80)
    order = list(range(n))
    rng.shuffle(order)
    masks = _order_visible(shapes, order, W, H)
    if masks is None:
        raise Reject
    pos = {s: k for k, s in enumerate(order)}
    for i in range(n):
        for j in range(i + 1, n):
            a = (masks[i] & masks[j]).sum()
            if 0 < a < 150:
                raise Reject
    vis = []
    for i in range(n):
        cover = np.zeros_like(masks[i])
        for k in range(n):
            if pos[k] > pos[i]:
                cover |= masks[k]
        v = masks[i] & ~cover
        if v.sum() < 0.3 * masks[i].sum():
            raise Reject
        vis.append(v)
    x = rng.choice([i for i in range(n) if pos[i] < n - 2] or [order[0]])
    need, frontier = set(), [x]
    while frontier:
        a = frontier.pop()
        for b in range(n):
            if b not in need and pos[b] > pos[a] and (masks[a] & masks[b]).any():
                need.add(b)
                frontier.append(b)
    cv = Canvas(W, H)
    for s_idx in order:
        _draw_shape(cv, shapes[s_idx], PAL[shapes[s_idx]["col"] % len(PAL)])
    img = cv.done()
    for i in range(n):
        dt = ndimage.distance_transform_edt(vis[i])
        yy, xx = np.unravel_index(dt.argmax(), dt.shape)
        tag(img, (xx, yy), LETTERS[i], 15)
    return dict(image=img, atype="int", answer=len(need),
                question=f"The shapes are flat cut-outs stacked on a table. To lift shape {LETTERS[x]} straight up, "
                         f"every shape lying on top of it (directly or through other shapes) must be removed first. "
                         f"How many shapes must be removed before {LETTERS[x]} can be lifted?",
                spec=dict(shapes=shapes, bottom_to_top=order, target=x))


# 7 gear train: see vf/tasks/gears.py (realistic renderer)


# ---------------------------------------------------------------- 8 mirror laser
@task("mirror_laser", "Optics / beam alignment / line-of-sight planning", "ray tracing by eye",
      knob="grid size and mirror count", twin="text: mirror grid", group=G)
def mirror_laser(rng, d):
    N = ilerp(4, 10, d)
    cell = 400 // N
    grid = [[0] * N for _ in range(N)]
    for _ in range(int(N * N * lerp(0.15, 0.3, d))):
        grid[rng.randrange(N)][rng.randrange(N)] = rng.choice([1, -1])  # 1 = '/', -1 = '\'
    r0 = rng.randrange(N)
    r, c, dr, dc, hits = r0, 0, 0, 1, 0
    for _ in range(4 * N * N):
        m = grid[r][c]
        if m:
            hits += 1
            dr, dc = (-dc, -dr) if m == 1 else (dc, dr)
        r, c = r + dr, c + dc
        if not (0 <= r < N and 0 <= c < N):
            break
    else:
        raise Reject
    if hits < ilerp(1, 5, d):
        raise Reject
    ports = [("top", k) for k in range(N)] + [("right", k) for k in range(N)] + \
            [("bottom", k) for k in range(N)] + [("left", k) for k in range(N) if k != r0]
    exitp = ("top", c) if r < 0 else ("bottom", c) if r >= N else ("right", r) if c >= N else ("left", r)
    ox = oy = 60
    S = cell * N
    cv = Canvas(S + 120, S + 120)
    cv.rect(ox, oy, ox + S, oy + S, outline=(120, 120, 120), width=2)
    for k in range(1, N):
        cv.line([(ox + k * cell, oy), (ox + k * cell, oy + S)], (225, 225, 225), 1)
        cv.line([(ox, oy + k * cell), (ox + S, oy + k * cell)], (225, 225, 225), 1)
    for i in range(N):
        for j in range(N):
            if grid[i][j]:
                x0, y0 = ox + j * cell + cell * .15, oy + i * cell + cell * .15
                x1, y1 = ox + (j + 1) * cell - cell * .15, oy + (i + 1) * cell - cell * .15
                cv.line([(x0, y1), (x1, y0)] if grid[i][j] == 1 else [(x0, y0), (x1, y1)], (20, 20, 20), 3.5)
    ys = oy + (r0 + .5) * cell
    cv.line([(ox - 50, ys), (ox - 8, ys)], (230, 50, 40), 4)
    cv.poly([(ox - 8, ys - 7), (ox, ys), (ox - 8, ys + 7)], fill=(230, 50, 40))
    for k, (side, idx) in enumerate(ports):
        if side == "top":
            p = (ox + (idx + .5) * cell, oy - 16)
        elif side == "bottom":
            p = (ox + (idx + .5) * cell, oy + S + 16)
        elif side == "right":
            p = (ox + S + 18, oy + (idx + .5) * cell)
        else:
            p = (ox - 18, oy + (idx + .5) * cell)
        cv.text(p, k + 1, 11 if N > 7 else 13, fill=(60, 60, 60))
    return dict(image=cv.done(), atype="int", answer=ports.index(exitp) + 1,
                question="A laser enters the grid at the red arrow. Each black line is a two-sided mirror that "
                         "turns the beam by 90 degrees. Through which numbered opening does the beam leave the grid?",
                spec=dict(grid=grid, entry_row=r0))


# ---------------------------------------------------------------- 9 fold and punch
@task("fold_punch", "Packaging / sheet-metal design: predicting the result of folds and cuts",
      "spatial transformation", knob="number of folds", twin="text: fold sequence", group=G)
def fold_punch(rng, d):
    k = ilerp(1, 3, d)
    rect = [0.0, 0.0, 1.0, 1.0]
    folds = []
    for _ in range(k):
        axis = rng.choice("vh")
        side = rng.choice([0, 1])  # which half moves onto the other
        x0, y0, x1, y1 = rect
        if axis == "v":
            m = (x0 + x1) / 2
            folds.append(("v", m))
            rect = [m, y0, x1, y1] if side == 0 else [x0, y0, m, y1]
        else:
            m = (y0 + y1) / 2
            folds.append(("h", m))
            rect = [x0, m, x1, y1] if side == 0 else [x0, y0, x1, m]
        folds[-1] = folds[-1] + (side,)
    x0, y0, x1, y1 = rect
    p = (rng.uniform(x0 + .2 * (x1 - x0), x1 - .2 * (x1 - x0)), rng.uniform(y0 + .2 * (y1 - y0), y1 - .2 * (y1 - y0)))

    def unfold(pts, flist):
        for ax, m, _ in reversed(flist):
            pts = pts | {((2 * m - a) if ax == "v" else a, b if ax == "v" else (2 * m - b)) for a, b in pts}
        return {(round(a, 3), round(b, 3)) for a, b in pts}
    true = unfold({p}, folds)
    cands = [true]
    wrong_axes = unfold({p}, [(("h" if ax == "v" else "v"), m, s) for ax, m, s in folds])
    for w in (wrong_axes, unfold({p}, folds[:-1]) if k > 1 else {(round(p[0], 3), round(p[1], 3))},
              {(round(1 - a, 3), round(b, 3)) for a, b in true}, {(round(a, 3), round(1 - b, 3)) for a, b in true},
              unfold({(p[0], 1 - p[1])}, folds)):
        if w not in cands and all(0.03 < a < 0.97 and 0.03 < b < 0.97 for a, b in w):
            cands.append(w)
        if len(cands) == 4:
            break
    if len(cands) < 4:
        raise Reject
    rng.shuffle(cands)
    ans = LETTERS[cands.index(true)]
    P = 150
    cv = Canvas(max((k + 1) * (P + 40) + 40, 4 * 140 + 40), 2 * P + 140)
    rect = [0.0, 0.0, 1.0, 1.0]
    for i, (ax, m, side) in enumerate(folds):
        ox = 30 + i * (P + 40)
        x0, y0, x1, y1 = rect
        cv.rect(ox + x0 * P, 30 + y0 * P, ox + x1 * P, 30 + y1 * P, fill=(250, 245, 230), outline=(90, 90, 90), width=2)
        if ax == "v":
            cv.line([(ox + m * P, 30 + y0 * P), (ox + m * P, 30 + y1 * P)], (200, 60, 60), 2)
            a, b = ((x0 + m) / 2, (m + x1) / 2) if side == 0 else ((m + x1) / 2, (x0 + m) / 2)
            yy = 30 + (y0 + y1) / 2 * P
            cv.line([(ox + a * P, yy), (ox + b * P, yy)], (200, 60, 60), 2.5)
            cv.poly([(ox + b * P, yy), (ox + b * P - 8 * (1 if b > a else -1), yy - 6),
                     (ox + b * P - 8 * (1 if b > a else -1), yy + 6)], fill=(200, 60, 60))
            rect = [m, y0, x1, y1] if side == 0 else [x0, y0, m, y1]
        else:
            cv.line([(ox + x0 * P, 30 + m * P), (ox + x1 * P, 30 + m * P)], (200, 60, 60), 2)
            a, b = ((y0 + m) / 2, (m + y1) / 2) if side == 0 else ((m + y1) / 2, (y0 + m) / 2)
            xx = ox + (x0 + x1) / 2 * P
            cv.line([(xx, 30 + a * P), (xx, 30 + b * P)], (200, 60, 60), 2.5)
            cv.poly([(xx, 30 + b * P), (xx - 6, 30 + b * P - 8 * (1 if b > a else -1)),
                     (xx + 6, 30 + b * P - 8 * (1 if b > a else -1))], fill=(200, 60, 60))
            rect = [x0, m, x1, y1] if side == 0 else [x0, y0, x1, m]
        cv.text((ox + P / 2, 30 + P + 18), f"Fold {i + 1}", 14)
    ox = 30 + k * (P + 40)
    x0, y0, x1, y1 = rect
    cv.rect(ox + x0 * P, 30 + y0 * P, ox + x1 * P, 30 + y1 * P, fill=(250, 245, 230), outline=(90, 90, 90), width=2)
    cv.ellipse(ox + p[0] * P, 30 + p[1] * P, 6, fill=(30, 30, 30))
    cv.text((ox + P / 2, 30 + P + 18), "Punch", 14)
    q = 100
    for i, c in enumerate(cands):
        bx = 30 + i * (q + 40)
        cv.rect(bx, P + 80, bx + q, P + 80 + q, fill=(250, 245, 230), outline=(90, 90, 90), width=2)
        for a, b in c:
            cv.ellipse(bx + a * q, P + 80 + b * q, 4.5, fill=(30, 30, 30))
        cv.text((bx + q / 2, P + 80 + q + 16), LETTERS[i], 16)
    return dict(image=cv.done(), atype="choice", choices=list("ABCD"), answer=ans,
                question="A square sheet is folded as shown (red arrows show which half folds over), then a hole is "
                         "punched through all layers. Which pattern (A-D) shows the sheet after unfolding?",
                spec=dict(folds=folds, punch=p, candidates=[sorted(c) for c in cands]))


# ---------------------------------------------------------------- 11 string on pegs
@task("string_pegs", "Geometry for packaging / fencing / robot workspace bounds", "convex hull by eye",
      knob="number of pegs, near-collinear pegs", twin="text: peg coordinates", group=G)
def string_pegs(rng, d):
    W, H = 520, 420
    n = ilerp(5, 22, d)
    pl = []
    for _ in range(n):
        for _t in range(300):
            q = (rng.uniform(40, W - 40), rng.uniform(40, H - 40))
            if all(math.hypot(q[0] - o[0], q[1] - o[1]) > 26 for o in pl):   # pegs never touch
                pl.append(q)
                break
        else:
            raise Reject
    pts = np.array(pl)
    hull = ConvexHull(pts)
    hv = set(hull.vertices.tolist())
    for i in range(n):
        if i in hv:
            continue
        for a, b in hull.simplices:
            pa, pb, p = pts[a], pts[b], pts[i]
            e, w = pb - pa, p - pa
            dist = abs(e[0] * w[1] - e[1] * w[0]) / np.linalg.norm(e)
            if dist < 9:
                raise Reject
    vs = hull.vertices
    for k in range(len(vs)):
        a, b, c = pts[vs[k - 1]], pts[vs[k]], pts[vs[(k + 1) % len(vs)]]
        v1, v2 = a - b, c - b
        ang = math.degrees(math.acos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))
        if ang > 168:
            raise Reject
    cv = Canvas(W, H)
    for x, y in pts:
        cv.ellipse(x, y, 6, fill=(25, 25, 25))
    return dict(image=cv.done(), atype="int", answer=len(vs),
                question="Imagine a rubber band stretched around the outside of all the pegs and then released so it "
                         "pulls tight. How many pegs does the rubber band touch?",
                spec=dict(pegs=pts.round(1).tolist()))


# ---------------------------------------------------------------- 12 one-stroke test
@task("one_stroke", "Route planning (inspection tours, plotter paths): Euler paths", "graph structure",
      knob="number of nodes and edges", twin="text: edge list", group=G)
def one_stroke(rng, d):
    W, H = 520, 440
    V = ilerp(4, 10, d)
    pts = []
    for _ in range(500):
        if len(pts) == V:
            break
        p = (rng.uniform(50, W - 50), rng.uniform(50, H - 50))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 70 for q in pts):
            pts.append(p)
    if len(pts) < V:
        raise Reject
    want_yes = rng.random() < 0.5
    E = set()
    order = list(range(V))
    rng.shuffle(order)
    for a, b in zip(order, order[1:]):
        E.add((min(a, b), max(a, b)))
    for _ in range(ilerp(1, V, d) + rng.randrange(3)):
        a, b = rng.sample(range(V), 2)
        E.add((min(a, b), max(a, b)))
    deg = [0] * V
    for a, b in E:
        deg[a] += 1
        deg[b] += 1
    odd = sum(x % 2 for x in deg)
    ans = "yes" if odd in (0, 2) else "no"
    if (ans == "yes") != want_yes:
        raise Reject
    for a, b in E:  # no edge may pass through a non-endpoint node
        pa, pb = np.array(pts[a]), np.array(pts[b])
        for k in range(V):
            if k in (a, b):
                continue
            p = np.array(pts[k])
            t = np.clip((p - pa) @ (pb - pa) / ((pb - pa) @ (pb - pa)), 0, 1)
            if np.linalg.norm(pa + t * (pb - pa) - p) < 14:
                raise Reject
    cv = Canvas(W, H)
    for a, b in E:
        cv.line([pts[a], pts[b]], (150, 150, 140), 3)
    for x, y in pts:
        cv.ellipse(x, y, 7, fill=(20, 20, 20))
    return dict(image=cv.done(), atype="choice", choices=["yes", "no"], answer=ans,
                question="Can this figure be drawn in one continuous stroke, without lifting the pen and without "
                         "going over any line twice? Lines that cross without a dot are not connected.",
                spec=dict(nodes=pts, edges=sorted(E)))


# ---------------------------------------------------------------- 13 ball drop
@task("ball_drop", "Physical prediction for robotics / chutes and conveyors", "physical simulation",
      knob="number of ramps", twin="text: ramp endpoints", group=G)
def ball_drop(rng, d):
    W, H = 560, 600
    nb = ilerp(3, 5, d)
    npath = ilerp(1, 4, d)
    ndec = ilerp(0, 3, d)
    bx = rng.uniform(80, W - 80)
    x, y, ramps = bx, 40.0, []
    for _ in range(npath):
        y += rng.uniform(62, 85)
        L = rng.uniform(110, 210)
        a = rng.uniform(25, L - 25)
        x0 = x - a
        slope = rng.uniform(0.12, 0.3) * rng.choice([-1, 1])
        x0 = min(max(x0, 15), W - 15 - L)
        if not (x0 + 22 < x < x0 + L - 22):
            raise Reject
        yy = y - slope * (x - x0)
        r = (x0, yy, x0 + L, yy + slope * L)
        if any(seg_dist(r[:2], r[2:], q[:2], q[2:]) < 22 for q in ramps):
            raise Reject
        ramps.append(r)
        x, y = (r[0] - 3, r[1]) if r[1] > r[3] else (r[2] + 3, r[3])
        y = max(y, r[1], r[3])
        if not (10 < x < W - 10) or y > H - 110:
            raise Reject
    for _ in range(ndec):
        x0 = rng.uniform(15, W - 180)
        L = rng.uniform(90, 160)
        yy = rng.uniform(120, H - 130)
        slope = rng.uniform(0.12, 0.3) * rng.choice([-1, 1])
        r = (x0, yy, x0 + L, yy + slope * L)
        if all(seg_dist(r[:2], r[2:], q[:2], q[2:]) > 28 for q in ramps):
            ramps.append(r)

    def simulate():
        x, y, used, last = bx, 40, 0, None
        for _ in range(20):
            if any(abs(x - e) < 14 for r in ramps if r is not last for e in (r[0], r[2])):
                return None
            below = []
            for r in ramps:
                if r[0] < x < r[2]:
                    yr = r[1] + (r[3] - r[1]) * (x - r[0]) / (r[2] - r[0])
                    if yr > y + 1:
                        below.append((yr, r))
            if not below:
                return x, used
            _, r = min(below)
            used, last = used + 1, r
            x, y = (r[0] - 3, r[1]) if r[1] > r[3] else (r[2] + 3, r[3])
        return None
    res = simulate()
    if res is None or res[1] < npath:
        raise Reject
    x = res[0]
    bw = W / nb
    if any(abs(x - k * bw) < 16 for k in range(nb + 1)) or not (0 < x < W):
        raise Reject
    ans = LETTERS[int(x // bw)]
    cv = Canvas(W, H + 40)
    for r in ramps:
        cv.line([(r[0], r[1]), (r[2], r[3])], (20, 20, 20), 5)
    cv.ellipse(bx, 40, 11, fill=(235, 100, 50))
    cv.line([(0, H - 10), (W, H - 10)], (120, 120, 120), 2)
    for k in range(nb + 1):
        cv.line([(k * bw, H - 60), (k * bw, H - 10)], (120, 120, 120), 3)
    for k in range(nb):
        cv.text(((k + .5) * bw, H + 12), LETTERS[k], 18)
    return dict(image=cv.done(), atype="choice", choices=list(LETTERS[:nb]), answer=ans,
                question="The orange ball is dropped straight down. When it lands on a black ramp it rolls down to "
                         "the ramp's lower end and falls straight down from there. Which bin does it end up in?",
                spec=dict(ramps=ramps, ball_x=bx))


# ---------------------------------------------------------------- shell game
def _cup(cv, x, y, w=60, h=70, lifted=False):
    yo = -45 if lifted else 0
    cv.poly([(x - w / 2, y + yo), (x + w / 2, y + yo), (x + w * .36, y - h + yo), (x - w * .36, y - h + yo)],
            fill=(190, 60, 50), outline=(90, 30, 25))


@task("shell_game", "Tracking objects through motion (video monitoring, sports, logistics)", "object tracking",
      knob="number of swaps (and cups)", twin="text: swap list", group=G)
def shell_game(rng, d):
    n = ilerp(3, 5, d)
    S = ilerp(1, 9, d)
    ball = rng.randrange(n)
    swaps = []
    for _ in range(S):
        i, j = sorted(rng.sample(range(n), 2))
        swaps.append((i, j))
    pos = ball
    for i, j in swaps:
        pos = j if pos == i else i if pos == j else pos
    cols = 4
    pw, ph = 90 * n + 40, 170
    panels = S + 1
    rows = math.ceil(panels / cols)
    cv = Canvas(cols * pw + 20, rows * (ph + 20) + 50)
    for p in range(panels):
        ox, oy = 10 + (p % cols) * pw, 10 + (p // cols) * (ph + 20)
        cv.rect(ox + 4, oy, ox + pw - 4, oy + ph, outline=(170, 170, 170), width=1.5)
        base = oy + ph - 30
        for c in range(n):
            x = ox + 40 + c * 90
            if p == 0 and c == ball:
                cv.ellipse(x, base - 12, 12, fill=(240, 200, 40))
            _cup(cv, x, base, lifted=(p == 0 and c == ball))
            cv.text((x, base + 16), c + 1, 13, fill=(90, 90, 90))
        if p == 0:
            cv.text((ox + pw / 2, oy + 14), "Start", 14)
        else:
            i, j = swaps[p - 1]
            xa, xb = ox + 40 + i * 90, ox + 40 + j * 90
            arc = [((xa + xb) / 2 + (xb - xa) / 2 * math.cos(t), oy + 60 - 30 * math.sin(t))
                   for t in np.linspace(math.pi, 0, 24)]
            cv.line(arc, (40, 90, 200), 3)
            for (ex, ey), dx in ((arc[0], -1), (arc[-1], 1)):
                cv.poly([(ex, ey + 9), (ex - 7, ey - 3), (ex + 7, ey - 3)], fill=(40, 90, 200))
            cv.text((ox + 16, oy + 14), f"{p}", 13, fill=(90, 90, 90))
    cv.text((cols * pw / 2, rows * (ph + 20) + 25), "Each numbered panel swaps the two cups joined by the arrow.", 15)
    return dict(image=cv.done(), atype="int", answer=pos + 1,
                question="A ball is placed under one cup (Start panel). Then, panel by panel, the two cups joined by "
                         "the blue arrow swap places. Under which position (1, 2, 3, ...) is the ball at the end?",
                spec=dict(start=ball, swaps=swaps))
