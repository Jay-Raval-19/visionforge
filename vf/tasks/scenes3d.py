"""3D indoor-scene tasks (doc tasks 10, 15-19), rendered with the NumPy ray tracer in vf/rt.py."""
import math

import numpy as np
from PIL import Image
from scipy import ndimage

from ..draw import Canvas
from ..rt import Scene, floor_tex, flat_tex
from ..taskapi import LETTERS, Reject, ilerp, lerp, tag, task

G = "indoor3d"
COLORS = {"red": (210, 60, 50), "blue": (50, 100, 210), "green": (50, 160, 80), "yellow": (230, 190, 40),
          "purple": (140, 70, 190), "orange": (235, 130, 40), "white": (235, 235, 235), "gray": (130, 130, 130)}
WOOD = floor_tex((176, 136, 96), (164, 124, 86), 0.6)


def _labels(img, sc, anchors):
    for lab, p in anchors:
        uv = sc.project(p)
        if uv:
            tag(img, uv, lab, 15)
    return img


# ---------------------------------------------------------------- composite furniture
def _parts(kind):
    """Parts in local coordinates: (shape, centre-or-base (x, y, z), dims, colour). Returns (parts, height)."""
    W_, D_ = (139, 94, 60), (90, 60, 40)
    if kind == "table":
        p = [("box", (0, .76, 0), (1.4, .08, .8), W_)]
        p += [("box", (sx * .62, .36, sz * .32), (.07, .72, .07), D_) for sx in (-1, 1) for sz in (-1, 1)]
        return p, .8
    if kind == "chair":
        p = [("box", (0, .46, 0), (.5, .06, .5), W_), ("box", (0, .8, -.22), (.5, .6, .06), W_)]
        p += [("box", (sx * .21, .22, sz * .21), (.05, .44, .05), D_) for sx in (-1, 1) for sz in (-1, 1)]
        return p, 1.1
    if kind == "bed":
        return [("box", (0, .22, 0), (2.0, .44, 1.4), (90, 110, 160)), ("box", (-.78, .5, 0), (.35, .14, 1.1), (240, 240, 240)),
                ("box", (-1.02, .55, 0), (.06, 1.1, 1.4), D_)], 1.1
    if kind == "lamp":
        return [("cyl", (0, 0, 0), (.16, .05), (60, 60, 60)), ("cyl", (0, .05, 0), (.03, 1.35), (60, 60, 60)),
                ("cyl", (0, 1.3, 0), (.24, .3), (245, 230, 180))], 1.6
    if kind == "shelf":
        p = [("box", (0, .9, 0), (.9, 1.8, .35), (120, 85, 55))]
        for k, y in enumerate((.45, .95, 1.45)):
            p += [("box", (-.3 + j * .2, y + .14, .05), (.12, .26, .22), list(COLORS.values())[(k * 3 + j) % 6])
                  for j in range(4)]
        return p, 1.8
    if kind == "mug":
        return [("cyl", (0, 0, 0), (.06, .12), (220, 70, 60))], .12
    if kind == "rug":
        return [("box", (0, .01, 0), (1.6, .02, 1.0), (170, 60, 70))], .02
    if kind == "plant":
        return [("cyl", (0, 0, 0), (.18, .3), (170, 90, 50)), ("sphere", (0, .55, 0), (.32,), (60, 140, 60))], .9
    raise ValueError(kind)


def _place(sc, kind, x, z, y0=0.0, yaw=0.0, scale=1.0, flip=False, wall=False):
    parts, Ht = _parts(kind)
    cy, sy = math.cos(yaw), math.sin(yaw)
    for shp, c, dims, col in parts:
        lx, ly, lz = (v * scale for v in c)
        if flip:
            if shp == "box":
                ly = Ht * scale - ly
            elif shp == "cyl":
                ly = Ht * scale - ly - dims[1] * scale
            else:
                ly = Ht * scale - ly
        if wall:  # rotate the object so it hangs on the back wall: local y -> world +z
            lx, ly, lz = lx, lz + .5, ly
        wx, wz = x + cy * lx + sy * lz, z - sy * lx + cy * lz
        if shp == "box":
            size = [v * scale for v in dims]
            if wall:
                size = [size[0], size[2], size[1]]
            sc.box((wx, y0 + ly, wz), size, col, yaw=yaw)
        elif shp == "cyl":
            sc.cyl((wx, y0 + ly, wz), dims[0] * scale, dims[1] * scale, col)
        else:
            sc.sphere((wx, y0 + ly, wz), dims[0] * scale, col)
    return y0 + Ht * scale


def _room(cam=(4.6, 4.0, 6.0), look=(-.2, .6, -.6), light=(-.4, 1, .55), W=720, H=540):
    sc = Scene(W, H, cam=cam, look=look, ground=WOOD, light=light, fov=58)
    sc.box((0, 1.6, -3.1), (9, 3.2, .2), (225, 222, 210), cast=False)
    sc.box((-4.6, 1.6, 0), (.2, 3.2, 7), (215, 212, 200), cast=False)
    return sc


# ---------------------------------------------------------------- 15 out-of-place objects
@task("out_of_place", "Home robots / AR assistants: noticing things that are physically or functionally wrong",
      "scene common sense", knob="anomaly subtlety, clutter", twin="text: object list with poses", group=G)
def out_of_place(rng, d):
    kinds = ["table", "chair", "bed", "lamp", "shelf", "plant", "rug"]
    foot = {"table": .85, "chair": .4, "bed": 1.25, "lamp": .3, "shelf": .55, "plant": .35, "rug": .95}
    n = ilerp(4, 6, d)
    chosen = rng.sample(kinds, n)
    chosen.sort(key=lambda k: k not in ("bed", "shelf"))      # fixed-position furniture first
    strong = ["upside_down", "on_wall", "giant", "on_bed"]
    subtle = ["floating", "giant_small", "upside_down"]
    anomaly = rng.choice((strong if d < 0.5 else subtle) + ["none"])
    target = None
    sc = _room()
    anchors, placed = [], []          # placed: (x, z, footprint radius) of everything standing on the floor

    def free_spot(rad):
        for _t in range(300):
            x, z = rng.uniform(-3.7, 3.4), rng.uniform(-2.5, 2.4)
            if all(math.hypot(x - px, z - pz) > rad + pr + .2 for px, pz, pr in placed):
                return x, z
        raise Reject
    for i, k in enumerate(chosen):
        yaw = rng.choice([0, math.pi / 2]) if k not in ("bed", "shelf") else 0
        kw = {}
        if anomaly == "upside_down" and target is None and k in ("table", "chair", "lamp"):
            kw["flip"], target = True, i
        elif anomaly == "on_wall" and target is None and k in ("chair", "rug", "table"):
            kw["y0"], kw["wall"], target = 1.0, True, i
        elif anomaly in ("giant", "giant_small") and target is None and k in ("plant", "lamp", "chair"):
            kw["scale"], target = (2.6 if anomaly == "giant" else 1.9), i
        elif anomaly == "floating" and target is None and k in ("table", "chair", "plant"):
            kw["y0"], target = lerp(0.6, 0.35, d), i
        rad = foot[k] * kw.get("scale", 1.0)
        if k == "bed":
            x, z = -2.6, -1.9
        elif k == "shelf":
            x, z = 0.6, -2.7
        elif kw.get("wall"):
            x, z, yaw = rng.uniform(-2, 2.5), -2.95, 0
        else:
            x, z = free_spot(rad)
        if not kw.get("wall"):
            if any(math.hypot(x - px, z - pz) < rad + pr + .2 for px, pz, pr in placed):
                raise Reject                      # nothing may overlap by accident
            placed.append((x, z, rad))
        top = _place(sc, k, x, z, yaw=yaw, **kw)
        anchors.append((LETTERS[i], (x, top + 0.35, z if not kw.get("wall") else z + .3)))
    if anomaly == "on_bed":
        if "bed" not in chosen:
            raise Reject
        i = len(chosen)
        _place(sc, "chair", -2.4, -1.8, y0=.44)   # the deliberate anomaly: a chair standing on the bed
        anchors.append((LETTERS[i], (-2.4, 1.9, -1.8)))
        target = i
    if anomaly != "none" and target is None:
        raise Reject
    img = sc.render(ss=2)
    pts = [sc.project(p) for _, p in anchors]
    if any(p is None for p in pts) or any(math.dist(pts[a], pts[b]) < 30 for a in range(len(pts)) for b in range(a)):
        raise Reject                              # labels must not collide
    _labels(img, sc, anchors)
    ans = LETTERS[target] if target is not None else "none"
    return dict(image=img, atype="choice", choices=[a for a, _ in anchors] + ["none"], answer=ans,
                question="At most one labeled object in this room is placed in a way it would never normally be "
                         "(wrong orientation, wrong support, attached to a wall, floating, or absurd size). Which "
                         "letter is it? Answer the letter, or 'none' if everything is normal.",
                spec=dict(objects=chosen, anomaly=anomaly, target=target, placed=placed))


# ---------------------------------------------------------------- 16 would it fall?
@task("would_fall", "Robotics / warehouse safety: spotting unstable placements", "physical stability",
      knob="overhang margin", twin="text: box extents vs table edge", group=G)
def would_fall(rng, d):
    sc = Scene(640, 480, cam=(0.6, 2.4, 4.6), look=(0, .8, 0), ground=WOOD, light=(-.3, 1, .6), fov=46)
    sc.box((0, 1.6, -2.5), (9, 3.2, .2), (225, 222, 210), cast=False)
    TX, TZ, TY = 1.3, 0.6, 0.8
    sc.box((0, TY - .04, 0), (2 * TX, .08, 2 * TZ), (139, 94, 60))
    for sx in (-1, 1):
        for sz in (-1, 1):
            sc.box((sx * (TX - .1), (TY - .08) / 2, sz * (TZ - .1)), (.08, TY - .08, .08), (90, 60, 40))
    n = ilerp(2, 4, d)
    m = lerp(0.3, 0.06, d)
    unstable = rng.randrange(n + 1) if rng.random() > 0.2 else n  # index n = none
    cols = rng.sample(list(COLORS.values())[:6], n)
    anchors, boxes = [], []
    zs = np.linspace(-.25, .25, n) if n > 1 else [0]
    order = list(range(n))
    rng.shuffle(order)
    for k in order:
        w = rng.uniform(.35, .6)
        h = rng.uniform(.25, .5)
        side = rng.choice([-1, 1])
        if k == unstable:
            cx = side * (TX + m * w + rng.uniform(0, .05))         # COM beyond the edge
        elif rng.random() < 0.7:
            cx = side * (TX - m * w - rng.uniform(0, .05))         # overhangs but COM inside
        else:
            cx = rng.uniform(-TX + w, TX - w)
        z = zs[order.index(k)]
        if any(abs(cx - bx) < (w + bw) / 2 + .05 and abs(z - bz) < .3 for bx, bz, bw, _ in boxes):
            raise Reject
        boxes.append((cx, z, w, LETTERS[k]))
        sc.box((cx, TY + h / 2, z), (w, h, .3), cols[k], edges=True)
        anchors.append((k, (cx, TY + h + .18, z)))
    img = sc.render(ss=2)
    for k, p in anchors:
        uv = sc.project(p)
        tag(img, uv, LETTERS[k], 15)
    ans = LETTERS[unstable] if unstable < n else "none"
    return dict(image=img, atype="choice", choices=list(LETTERS[:n]) + ["none"], answer=ans,
                question="Each labeled box is a solid block of uniform weight resting on the table. Which box is not "
                         "stably supported, i.e. would tip off the table? Answer the letter, or 'none'.",
                spec=dict(table_half_width=TX, boxes=boxes, unstable=unstable))


# ---------------------------------------------------------------- 17 two-view change
@task("two_view_change", "Inventory / security monitoring from moving cameras", "multi-view correspondence",
      knob="camera angle change, object count, move distance", twin="text: object positions before/after", group=G)
def two_view_change(rng, d):
    n = ilerp(3, 7, d)
    shapes = ["cube", "sphere", "cylinder"]
    combos = [(c, s) for c in ["red", "blue", "green", "yellow", "purple", "orange"] for s in shapes]
    objs = rng.sample(combos, n)
    pos = []
    for _ in range(n):
        for _t in range(200):
            p = (rng.uniform(-2.2, 2.2), rng.uniform(-2.2, 2.2))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 1.0 for q in pos):
                pos.append(p)
                break
        else:
            raise Reject
    mv = rng.randrange(n)
    dist = lerp(1.6, 0.7, d)
    for _t in range(200):
        a = rng.uniform(0, 2 * math.pi)
        np_ = (pos[mv][0] + dist * math.cos(a), pos[mv][1] + dist * math.sin(a))
        if abs(np_[0]) < 2.6 and abs(np_[1]) < 2.6 and all(
                math.hypot(np_[0] - q[0], np_[1] - q[1]) > 1.0 for k, q in enumerate(pos) if k != mv):
            break
    else:
        raise Reject
    pos2 = list(pos)
    pos2[mv] = np_
    az0 = rng.uniform(0, 2 * math.pi)
    gap = math.radians(lerp(25, 140, d)) * rng.choice([-1, 1])

    def view(ps, az):
        sc = Scene(480, 360, cam=(7 * math.cos(az), 5, 7 * math.sin(az)), look=(0, 0, 0),
                   ground=floor_tex((200, 200, 195), (188, 188, 182), 1.0), fov=50)
        for (c, s), (x, z) in zip(objs, ps):
            col = COLORS[c]
            if s == "cube":
                sc.box((x, .35, z), (.7, .7, .7), col, edges=True)
            elif s == "sphere":
                sc.sphere((x, .4, z), .4, col)
            else:
                sc.cyl((x, 0, z), .33, .8, col)
        return sc.render(ss=2)
    a, b = view(pos, az0), view(pos2, az0 + gap)
    img = Image.new("RGB", (980, 400), (255, 255, 255))
    img.paste(a, (5, 5))
    img.paste(b, (495, 5))
    tag(img, (245, 385), "View 1", 15)
    tag(img, (735, 385), "View 2", 15)
    names = [f"{c} {s}" for c, s in objs]
    return dict(image=img, atype="choice", choices=names, answer=names[mv],
                question="The two views show the same table-top scene from different camera positions. Between the "
                         "two views exactly one object was moved. Which object moved? Answer with its colour and "
                         "shape, e.g. 'red cube'.",
                spec=dict(objects=names, before=pos, after=pos2, moved=mv))


# ---------------------------------------------------------------- 18 robot path
@task("robot_path", "Mobile robots / wheelchair access: can it get through?", "clearance & path planning",
      knob="clutter, corridor margin", twin="text: furniture rectangles + robot radius", group=G)
def robot_path(rng, d):
    W, H, ppm = 640, 480, 60  # pixels per metre
    Wm, Hm = W / ppm, H / ppm
    r = 0.35
    mg = lerp(0.14, 0.06, d)
    wx = rng.uniform(.42, .58) * Wm            # partition wall with one or two doorways near the robot's width
    gaps = []
    for _ in range(rng.choice([1, 2])):
        for _t in range(100):
            gw = 2 * r + rng.choice([-1, 1]) * rng.uniform(mg + .02, mg + .3)
            gy = rng.uniform(.6 + gw / 2, Hm - .6 - gw / 2)
            if all(abs(gy - g[0]) > (gw + g[1]) / 2 + 1.0 for g in gaps):
                gaps.append((gy, gw))
                break
    rects = []
    for _ in range(ilerp(2, 7, d)):
        w, h = rng.uniform(.5, 1.6), rng.uniform(.4, 1.2)
        x, y = rng.uniform(.3, Wm - w - .3), rng.uniform(.3, Hm - h - .3)
        if x < wx + .2 and x + w > wx - .2:
            continue                               # keep furniture off the partition
        rects.append((x, y, w, h))
    start = (rng.uniform(.6, wx - 1.0), rng.uniform(1, Hm - 1))
    door = (Wm - .1, rng.uniform(1.0, Hm - 1.0))
    res = 0.025
    gx, gy_ = int(Wm / res), int(Hm / res)
    occ = np.zeros((gy_, gx), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    di = int(door[1] / res)
    occ[di - 18:di + 19, -1] = False
    ws = int((wx - .075) / res)
    occ[:, ws:ws + int(.15 / res)] = True
    for gyc, gw in gaps:
        occ[int((gyc - gw / 2) / res):int((gyc + gw / 2) / res), ws:ws + int(.15 / res)] = False
    for x, y, w, h in rects:
        occ[int(y / res):int((y + h) / res) + 1, int(x / res):int((x + w) / res) + 1] = True
    dt = ndimage.distance_transform_edt(~occ) * res

    def reach(rad):
        lab, _ = ndimage.label(dt > rad)
        s_ = lab[int(start[1] / res), int(start[0] / res)]
        g = set(np.unique(lab[di - 6:di + 7, gx - int((rad + .2) / res):gx - 2])) - {0}
        return s_ != 0 and s_ in g
    if dt[int(start[1] / res), int(start[0] / res)] < r + .15:
        raise Reject
    a, b = reach(r - mg), reach(r + mg)
    if a != b:
        raise Reject                      # the answer flips within the margin -> too close to call
    if (rng.random() < 0.5) != a:
        raise Reject                      # keep yes/no balanced
    cv = Canvas(W, H, bg=(245, 243, 238))
    cv.rect(2, 2, W - 2, H - 2, outline=(40, 40, 40), width=4)
    cv.rect((wx - .075) * ppm, 0, (wx + .075) * ppm, H, fill=(40, 40, 40))
    for gyc, gw in gaps:
        cv.rect((wx - .075) * ppm - 1, (gyc - gw / 2) * ppm, (wx + .075) * ppm + 1, (gyc + gw / 2) * ppm,
                fill=(245, 243, 238))
    cv.rect(W - 9, (door[1] - .45) * ppm, W - 1, (door[1] + .45) * ppm, fill=(40, 180, 90))
    for x, y, w, h in rects:
        cv.rect(x * ppm, y * ppm, (x + w) * ppm, (y + h) * ppm, fill=(190, 182, 168), outline=(120, 110, 100), width=1.5)
    cv.ellipse(start[0] * ppm, start[1] * ppm, r * ppm, fill=(50, 100, 210))
    cv.line([(20, H - 20), (20 + ppm, H - 20)], (0, 0, 0), 3)
    cv.text((20 + ppm / 2, H - 34), "1 m", 13)
    return dict(image=cv.done(), atype="choice", choices=["yes", "no"], answer="yes" if a else "no",
                question="Top view of a room. The blue disc is a round robot; it cannot pass through walls (dark) or "
                         "furniture (gray) and cannot move anything. Using the 1 m scale bar, can the robot reach the "
                         "green door on the right?",
                spec=dict(rects=rects, robot_r=r, start=start, door=door, wall_x=wx, gaps=gaps))


# ---------------------------------------------------------------- 19 shadow check
@task("shadow_check", "Image forensics / AR compositing: is the lighting physically consistent?",
      "lighting consistency", knob="angle error of the wrong shadow", twin="text: light dir + shadow dirs", group=G)
def shadow_check(rng, d):
    n = ilerp(3, 6, d)
    az = rng.uniform(0, 2 * math.pi)
    el = math.radians(rng.uniform(35, 55))
    L = (math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az))
    sc = Scene(640, 480, cam=(0.5, 7, 9), look=(0, 0, 0), ground=flat_tex((205, 200, 190)), light=L, fov=45)
    wrong = rng.randrange(n) if rng.random() > 0.2 else None
    err = math.radians(lerp(90, 30, d)) * rng.choice([-1, 1])
    Lw = (math.cos(el) * math.cos(az + err), math.sin(el), math.cos(el) * math.sin(az + err))
    pos, anchors = [], []
    for i in range(n):
        for _t in range(300):
            p = (rng.uniform(-3.4, 3.4), rng.uniform(-2.2, 2.4))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 2.0 for q in pos):
                pos.append(p)
                break
        else:
            raise Reject
        x, z = p
        kind = rng.choice(["box", "cyl", "post"])
        kw = dict(shadow_L=Lw) if i == wrong else {}
        col = list(COLORS.values())[i % 6]
        if kind == "box":
            sc.box((x, .5, z), (.6, 1.0, .6), col, **kw)
            top = 1.0
        elif kind == "cyl":
            sc.cyl((x, 0, z), .3, 1.1, col, **kw)
            top = 1.1
        else:
            sc.box((x, .75, z), (.25, 1.5, .25), col, **kw)
            top = 1.5
        anchors.append((LETTERS[i], (x, top + .45, z)))
    img = sc.render(ss=2)
    _labels(img, sc, anchors)
    return dict(image=img, atype="choice", choices=list(LETTERS[:n]) + ["none"],
                answer=LETTERS[wrong] if wrong is not None else "none",
                question="All objects are lit by the same single distant light (like the sun). Which labeled object "
                         "casts a shadow that is inconsistent with the others? Answer the letter, or 'none'.",
                spec=dict(light=L, wrong=wrong, error_deg=math.degrees(err)))


# ---------------------------------------------------------------- 10 voxel count
@task("voxel_count", "Warehouse / pallet counting: how many units are in a stack", "3D counting",
      knob="grid size, stack height, edge lines removed, lower camera, count vs. how-many-to-complete",
      twin="text: height map", group=G)
def voxel_count(rng, d):
    g = ilerp(3, 6, d)                    # grid side
    mh = ilerp(2, 6, d)                   # tallest column
    # columns never get taller toward the camera (larger i, j), so every column's top stays visible and
    # the hidden cubes are fixed by the no-gaps rule -> the count is unique
    hm = [[0] * g for _ in range(g)]
    for i in range(g):
        for j in range(g):
            cap = mh if i == j == 0 else min(hm[i - 1][j] if i else mh, hm[i][j - 1] if j else mh)
            hm[i][j] = max(0, cap - rng.choice([0, 0, 1, 1, 2]))
    total = sum(map(sum, hm))
    if total < 4:
        raise Reject
    edges = d < 0.5                       # high difficulty: no drawn edges, cube boundaries from shading only
    base = np.array((70, 120, 210), float)
    shade = {}                            # per-cube brightness, neighbours always differ (3D checkerboard)
    for i in range(g):
        for j in range(g):
            for k in range(hm[i][j]):
                shade[(i, j, k)] = 1 + (1 if (i + j + k) % 2 else -1) * rng.uniform(.06, .12)

    def col_fn(i, j):
        def f(X):
            k = np.clip(np.floor(X[:, 1] + 1e-6).astype(int), 0, hm[i][j] - 1)
            fac = np.array([shade[(i, j, kk)] for kk in k])
            c = base[None, :] * fac[:, None]
            if edges:                     # dark seams between stacked cubes
                fr = X[:, 1] - np.round(X[:, 1])
                c[(np.abs(fr) < .025) & (X[:, 1] > .05)] *= .45
            return c
        return f
    cam_y = lerp(g + 4.5, mh + 2.2, d)    # lower camera at high difficulty
    sc = Scene(640, 520, cam=(g + 3.8, cam_y, g + 4.6), look=(g / 2 - .5, min(mh, 3) * .4, g / 2 - .5),
               ground=flat_tex((225, 225, 220)), light=(-.4, 1, .7), fov=44)
    ids = []
    for i in range(g):
        for j in range(g):
            if hm[i][j]:
                ids.append(sc.box((i, hm[i][j] / 2, j), (1, hm[i][j], 1), base, edges=edges, color_fn=col_fn(i, j)))
    img = sc.render(ss=2)
    if any(sc.visible_px(i) < 40 for i in ids):
        raise Reject                     # every column must be visible
    H = max(map(max, hm))
    if d >= 0.5 and rng.random() < 0.5:
        q = (f"How many more cubes would be needed to turn this structure into a solid rectangular block {g} cubes "
             f"wide, {g} cubes deep and {H} cubes tall? Cubes are stacked in columns on a square grid with no gaps "
             f"(every cube rests on the floor or directly on another cube), so some lower cubes are hidden.")
        ans, qtype = g * g * H - total, "complete"
    else:
        q = ("How many cubes are in this structure? Cubes are stacked in columns on a square grid with no gaps "
             "(every cube rests on the floor or directly on another cube), so some lower cubes are hidden behind "
             "others.")
        ans, qtype = total, "count"
    return dict(image=img, atype="int", answer=ans, question=q,
                spec=dict(heights=hm, grid=g, qtype=qtype, block_h=H, edges=edges))
