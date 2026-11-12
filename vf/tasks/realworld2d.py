"""Real-world-flavoured 2D tasks: inspection, medical, engineering drawings, maps, lab, UI and documents.

Each one is the realistic counterpart of an abstract puzzle (see `twin=` on the decorator)."""
import math
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from ..draw import Canvas, catmull, font
from ..taskapi import LETTERS, Reject, autocrop, ilerp, lerp, seg_dist, tag, task

G = "real2d"


def _noise(img, sigma, seed):
    a = np.asarray(img).astype(np.float32)
    a += np.random.default_rng(seed).normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- PCB defect inspection
@task("pcb_defect", "Electronics manufacturing: automated optical inspection of PCBs", "fine-grained comparison",
      knob="component count, defect subtlety (missing -> rotated -> solder bridge)", twin="puzzle: spot the difference",
      group=G)
def pcb_defect(rng, d):
    W, H = 420, 320
    n = ilerp(6, 24, d)
    comps = []
    kinds = ["R", "C", "U", "D"]
    for i in range(n):
        for _t in range(200):
            k = rng.choice(kinds)
            w, h = {"R": (34, 12), "C": (22, 22), "U": (60, 40), "D": (28, 14)}[k]
            if rng.random() < .5 and k != "C":
                w, h = h, w
            x, y = rng.uniform(25, W - 25 - w), rng.uniform(25, H - 35 - h)
            if all(x + w + 22 < c["x"] or c["x"] + c["w"] + 22 < x or y + h + 26 < c["y"] or c["y"] + c["h"] + 26 < y
                   for c in comps):
                num = sum(1 for c in comps if c["k"] == k) + 1
                comps.append(dict(k=k, x=x, y=y, w=w, h=h, ref=f"{k}{num}", flip=rng.random() < .5))
                break
    defect = rng.choice(["missing", "rotated"] if d < 0.5 else ["rotated", "bridge", "shifted"])
    cand = [c for c in comps if (defect != "bridge" or c["k"] == "U") and (defect != "rotated" or c["k"] in "UDC")]
    if not cand:
        raise Reject
    tgt = rng.choice(cand)

    def board(defective):
        cv = Canvas(W, H, bg=(30, 105, 60))
        for c in comps:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            cv.text((x + w / 2, y + h + 11), c["ref"], 10, fill=(235, 235, 220))
            pads = [(x - 4, y + h / 2 - 4, x + 4, y + h / 2 + 4), (x + w - 4, y + h / 2 - 4, x + w + 4, y + h / 2 + 4)] \
                if w >= h else [(x + w / 2 - 4, y - 4, x + w / 2 + 4, y + 4), (x + w / 2 - 4, y + h - 4, x + w / 2 + 4, y + h + 4)]
            for p in pads:
                cv.rect(*p, fill=(200, 190, 150))
            is_t = defective and c is tgt
            if is_t and defect == "missing":
                continue
            ox = (6 if is_t and defect == "shifted" else 0)
            flip = c["flip"] ^ (is_t and defect == "rotated")
            col = {"R": (210, 190, 140), "C": (40, 40, 45), "U": (25, 25, 28), "D": (40, 40, 45)}[c["k"]]
            if c["k"] == "C":
                cv.ellipse(x + w / 2 + ox, y + h / 2 + ox, w / 2, fill=col)
                cv.rect(x + ox + (w * .7 if flip else 0), y + ox + 2, x + ox + (w if flip else w * .3), y + h + ox - 2,
                        fill=(160, 160, 170))
            else:
                cv.rect(x + ox, y + ox, x + w + ox, y + h + ox, fill=col)
                if c["k"] == "U":
                    for k in range(5):
                        px = x + 6 + k * (w - 12) / 4
                        cv.rect(px - 2, y - 4, px + 2, y, fill=(200, 200, 200))
                        cv.rect(px - 2, y + h, px + 2, y + h + 4, fill=(200, 200, 200))
                    cv.ellipse(x + (w - 7 if flip else 7) + ox, y + 7 + ox, 3, fill=(200, 200, 200))
                    if is_t and defect == "bridge":
                        cv.rect(x + 6, y + h, x + 6 + (w - 12) / 4 + 2, y + h + 5, fill=(215, 215, 215))
                elif c["k"] == "D":
                    if w >= h:
                        cv.rect(x + (w - 6 if flip else 2), y, x + (w - 2 if flip else 6), y + h, fill=(220, 220, 220))
                    else:
                        cv.rect(x, y + (h - 6 if flip else 2), x + w, y + (h - 2 if flip else 6), fill=(220, 220, 220))
                else:
                    for b in range(3):
                        bx = x + w * (.3 + .2 * b) if w >= h else x
                        if w >= h:
                            cv.rect(bx, y, bx + 3, y + h, fill=[(150, 60, 40), (40, 40, 40), (200, 140, 40)][b])
        return cv.done()
    img = Image.new("RGB", (2 * W + 30, H + 30), (255, 255, 255))
    img.paste(board(False), (0, 30))
    img.paste(board(True), (W + 30, 30))
    tag(img, (W / 2, 14), "Reference", 14)
    tag(img, (W + 30 + W / 2, 14), "Inspected", 14)
    refs = [c["ref"] for c in comps]
    return dict(image=img, atype="choice", choices=refs, answer=tgt["ref"],
                question="Compare the inspected board with the reference board. Exactly one component on the "
                         "inspected board is defective (missing, rotated/reversed, shifted, or with a solder bridge). "
                         "Which one? Answer its designator, e.g. R3.",
                spec=dict(components=comps, defect=defect, target=tgt["ref"]))


# ---------------------------------------------------------------- crack measurement
@task("crack_measure", "Infrastructure inspection: measuring defects against a ruler", "metric measurement",
      knob="ruler placement/orientation relative to the crack", twin="puzzle: gauge/scale reading", group=G)
def crack_measure(rng, d):
    W, H = 640, 440
    ppm = rng.uniform(2.6, 4.2)  # pixels per mm
    a = np.full((H, W, 3), 170, np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 14, (H, W, 1))
    a = ndimage.gaussian_filter(a, (1.2, 1.2, 0))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    L = rng.uniform(25, 110)
    ang = rng.uniform(0, math.pi) if d > 0.3 else rng.uniform(-.2, .2)
    cx, cy = rng.uniform(200, W - 200), rng.uniform(130, H - 170)
    p0 = (cx - L * ppm / 2 * math.cos(ang), cy - L * ppm / 2 * math.sin(ang))
    p1 = (cx + L * ppm / 2 * math.cos(ang), cy + L * ppm / 2 * math.sin(ang))
    for px_, py_ in (p0, p1):          # both ends must be fully in frame and clear of the ruler strip
        if not (20 < px_ < W - 20 and 20 < py_ < H - 120):
            raise Reject
    dr = ImageDraw.Draw(img)
    pts = [p0]
    for k in range(1, 12):
        t = k / 12
        pts.append((p0[0] + (p1[0] - p0[0]) * t + rng.uniform(-1.5, 1.5), p0[1] + (p1[1] - p0[1]) * t + rng.uniform(-1.5, 1.5)))
    pts.append(p1)
    dr.line(pts, fill=(45, 40, 38), width=3)
    rang = 0 if d < 0.6 else rng.uniform(-0.4, 0.4)
    rl = 150 * ppm
    if rl > W - 40:
        rl = 100 * ppm
    rx, ry = (W - rl) / 2, H - 70
    rul = Image.new("RGBA", (int(rl + 20), 50), (240, 225, 120, 255))
    rd = ImageDraw.Draw(rul)
    mm = int(rl / ppm)
    for m in range(mm + 1):
        x = 10 + m * ppm
        h = 18 if m % 10 == 0 else 12 if m % 5 == 0 else 7
        rd.line([(x, 0), (x, h)], fill=(20, 20, 20), width=1)
        if m % 10 == 0:
            rd.text((x, 30), str(m // 10), font=font(12), fill=(20, 20, 20), anchor="mm")
    rd.text((rl, 42), "cm", font=font(10), fill=(20, 20, 20), anchor="mm")
    rul = rul.rotate(math.degrees(rang), expand=True)
    img.paste(rul, (int(rx), int(ry - rul.height / 2 + 20)), rul)
    if math.hypot(*(np.subtract(p1, p0))) < 40:
        raise Reject
    return dict(image=img, atype="float", answer=round(L, 1), tol=max(2.0, 0.05 * L),
                question="Using the ruler in the photo (numbers are centimetres, small ticks are millimetres), how "
                         "long is the dark crack from end to end, in millimetres?",
                spec=dict(length_mm=L, px_per_mm=ppm))


# ---------------------------------------------------------------- cytology (RIVA-style)
@task("cytology_count", "Cervical cytology screening (RIVA-style): counting abnormal cells", "counting with a rule",
      knob="cell count, overlap, stain variation", twin="puzzle: overlapping-shape counting", group=G)
def cytology_count(rng, d):
    W, H = 640, 480
    n = ilerp(6, 30, d)
    n_ab = rng.randrange(0, max(2, ilerp(3, 8, d)))
    cells = []
    for i in range(n):
        for _t in range(300):
            r = rng.uniform(26, 42)
            x, y = rng.uniform(r, W - r), rng.uniform(r, H - r)
            if all(math.hypot(x - c[0], y - c[1]) > (r + c[2]) * lerp(1.1, 0.6, d) for c in cells):
                cells.append((x, y, r, i < n_ab))
                break
        else:
            raise Reject
    hue = rng.uniform(-1, 1) * lerp(5, 30, d)
    bg = Image.new("RGB", (W, H), (236, 228, 232))
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    nuc = []
    for x, y, r, ab in cells:
        pts = [(x + r * (1 + rng.uniform(-.12, .12)) * math.cos(t), y + r * (1 + rng.uniform(-.12, .12)) * math.sin(t))
               for t in np.linspace(0, 2 * math.pi, 18, endpoint=False)]
        col = (int(200 + hue), int(150 - hue / 2), int(190 - hue), 150)
        if rng.random() < .4:
            col = (int(140 + hue), int(180), int(210 - hue), 150)
        dr.polygon(pts, fill=col, outline=(150, 110, 150, 200))
        nr = r * (rng.uniform(.62, .75) if ab else rng.uniform(.15, .28))
        nuc.append((x + rng.uniform(-.1, .1) * r, y + rng.uniform(-.1, .1) * r, nr))
    bg.paste(lay, (0, 0), lay)
    dr = ImageDraw.Draw(bg)
    for x, y, nr in nuc:
        dr.ellipse([x - nr, y - nr * .85, x + nr, y + nr * .85], fill=(70, 40, 100))
    for i in range(len(nuc)):
        for j in range(i + 1, len(nuc)):
            if math.hypot(nuc[i][0] - nuc[j][0], nuc[i][1] - nuc[j][1]) < nuc[i][2] + nuc[j][2] + 3:
                raise Reject
    img = bg.filter(ImageFilter.GaussianBlur(lerp(0.3, 1.2, d)))
    img = _noise(img, lerp(3, 10, d), rng.randrange(1 << 30))
    return dict(image=img, atype="int", answer=n_ab,
                question="This is a stained cell smear. Rule: a cell counts as ABNORMAL if its dark nucleus is wider "
                         "than half of the cell's own width. How many abnormal cells are in the image?",
                spec=dict(cells=cells))


# ---------------------------------------------------------------- tube tip placement
@task("tube_tip", "Radiology: checking where a catheter/tube tip ends", "thin-line tracing in noise",
      knob="number of tubes, contrast, noise", twin="puzzle: tangled wires", group=G)
def tube_tip(rng, d):
    W, H = 520, 560
    a = np.full((H, W), 40.0)
    yy, xx = np.mgrid[0:H, 0:W]
    for cx, cy, rx, ry, v in ((W * .3, H * .5, 120, 210, 25), (W * .7, H * .5, 120, 210, 25), (W * .5, H * .55, 60, 240, 110)):
        a += v * (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1)
    a = ndimage.gaussian_filter(a, 18)
    for k in range(9):
        y = 120 + k * 40
        a += 30 * (np.abs(yy - y - 15 * np.sin(xx / 60)) < 4)
    a = ndimage.gaussian_filter(a, 2)
    zones = [140, 260, 380, 500]
    nt = ilerp(1, 4, d)
    contrast = lerp(110, 45, d)
    tubes = []
    for i in range(nt):
        x0 = rng.uniform(80, W - 80)
        tip = (rng.uniform(140, W - 140), rng.uniform(90, 520))
        if any(abs(tip[1] - z) < 18 for z in zones):
            raise Reject
        mid = [(x0 + (tip[0] - x0) * t + rng.uniform(-60, 60) * d, 20 + (tip[1] - 20) * t) for t in (.35, .7)]
        pts = catmull([(x0, 0), *mid, tip], 40)
        tubes.append((pts, tip, x0))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    for pts, tip, _ in tubes:
        dr.line(pts, fill=int(min(255, 70 + contrast)), width=3)
    img = _noise(img.filter(ImageFilter.GaussianBlur(.8)), lerp(4, 16, d), rng.randrange(1 << 30)).convert("RGB")
    dr = ImageDraw.Draw(img)
    for k, z in enumerate(zones):
        for x in range(0, W, 14):
            dr.line([(x, z), (x + 7, z)], fill=(240, 200, 60), width=1)
    labs = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5"]
    ys = [70] + [(zones[k] + zones[k + 1]) / 2 for k in range(3)] + [530]
    for lab, y in zip(labs, ys):
        dr.text((W - 8, y), lab, font=font(13), fill=(240, 200, 60), anchor="rm")
    q = rng.randrange(nt)
    tag(img, (tubes[q][2], 14), "T", 14)
    ty = tubes[q][1][1]
    zone = 1 + sum(1 for z in zones if ty > z)
    return dict(image=img, atype="int", answer=zone,
                question="Follow the thin bright tube that enters at the top at the label 'T'. In which numbered zone "
                         "(1-5, separated by the yellow dashed lines) does its tip end?",
                spec=dict(tubes=[(t[2], t[1]) for t in tubes], query=q, zones=zones))


# ---------------------------------------------------------------- ultrasound caliper
@task("caliper_measure", "Ultrasound/medical imaging: measuring between calipers with the depth scale",
      "metric measurement on a scale", knob="scale label sparsity, speckle noise", twin="puzzle: gauge reading", group=G)
def caliper_measure(rng, d):
    W, H = 560, 520
    ppcm = rng.uniform(28, 44)
    rng_np = np.random.default_rng(rng.randrange(1 << 30))
    sp = rng_np.rayleigh(lerp(30, 55, d), (H, W))
    sp = ndimage.gaussian_filter(sp, 1.2)
    yy, xx = np.mgrid[0:H, 0:W]
    ang = np.arctan2(xx - W / 2, yy + 40)
    fan = (np.abs(ang) < .6) & (yy > 20)
    blob = ((xx - W / 2 - rng.uniform(-60, 60)) / rng.uniform(60, 120)) ** 2 + ((yy - rng.uniform(200, 360)) / rng.uniform(40, 90)) ** 2 < 1
    sp = sp * np.where(blob, .25, 1.0)
    img = Image.fromarray(np.clip(sp * fan, 0, 255).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(img)
    depth_cm = int((H - 40) / ppcm)
    lab_every = 1 if d < 0.5 else 5
    for c in range(depth_cm + 1):
        y = 20 + c * ppcm
        dr.line([(12, y), (24 if c % 5 == 0 else 19, y)], fill=(230, 230, 230), width=1)
        if c % lab_every == 0:
            dr.text((30, y), str(c), font=font(12), fill=(230, 230, 230), anchor="lm")
    dr.text((14, 8), "cm", font=font(11), fill=(230, 230, 230))
    for _t in range(100):
        p = (rng.uniform(W * .3, W * .7), rng.uniform(80, H - 60))
        q = (rng.uniform(W * .3, W * .7), rng.uniform(80, H - 60))
        dist = math.hypot(p[0] - q[0], p[1] - q[1]) / ppcm
        if 1.0 < dist < 7:
            break
    for x, y in (p, q):
        dr.line([(x - 7, y), (x + 7, y)], fill=(250, 240, 90), width=2)
        dr.line([(x, y - 7), (x, y + 7)], fill=(250, 240, 90), width=2)
    return dict(image=img, atype="float", answer=round(dist, 2), tol=0.2,
                question="Using the depth scale on the left (in cm), what is the straight-line distance between the "
                         "two yellow '+' calipers, in centimetres?",
                spec=dict(px_per_cm=ppcm, p=p, q=q))


# ---------------------------------------------------------------- floor-plan feasibility
@task("floorplan_path", "Architecture / facilities / accessibility: can equipment move between rooms?",
      "measure + path reasoning", knob="number of rooms, door widths near the threshold", twin="puzzle: robot path",
      group=G)
def floorplan_path(rng, d):
    Wm, Hm, ppm = 14.0, 10.0, 48
    rooms = [(0.0, 0.0, Wm, Hm)]
    for _ in range(ilerp(3, 8, d) - 1):
        rooms.sort(key=lambda r: -(r[2] * r[3]))
        x, y, w, h = rooms.pop(0)
        if w >= h:
            s = rng.uniform(.35, .65) * w
            rooms += [(x, y, s, h), (x + s, y, w - s, h)]
        else:
            s = rng.uniform(.35, .65) * h
            rooms += [(x, y, w, s), (x, y + s, w, h - s)]
    cart = rng.choice([0.8, 0.9, 1.0])
    mg = lerp(0.25, 0.08, d)
    doors = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b = rooms[i], rooms[j]
            for vert in (True, False):
                if vert and (abs(a[0] + a[2] - b[0]) < 1e-6 or abs(b[0] + b[2] - a[0]) < 1e-6):
                    x = b[0] if abs(a[0] + a[2] - b[0]) < 1e-6 else a[0]
                    lo, hi = max(a[1], b[1]), min(a[1] + a[3], b[1] + b[3])
                elif not vert and (abs(a[1] + a[3] - b[1]) < 1e-6 or abs(b[1] + b[3] - a[1]) < 1e-6):
                    x = b[1] if abs(a[1] + a[3] - b[1]) < 1e-6 else a[1]
                    lo, hi = max(a[0], b[0]), min(a[0] + a[2], b[0] + b[2])
                else:
                    continue
                if hi - lo < 1.8:
                    continue
                w = cart + rng.choice([-1, 1]) * rng.uniform(mg, mg + .3)
                c = rng.uniform(lo + w / 2 + .3, hi - w / 2 - .3)
                doors.append((i, j, vert, x, c, w))
    if len(rooms) < 3:
        raise Reject
    s, t = rng.sample(range(len(rooms)), 2)
    adj = {k: [] for k in range(len(rooms))}
    for i, j, _, _, _, w in doors:
        if w >= cart:
            adj[i].append(j)
            adj[j].append(i)
    seen, dq = {s}, deque([s])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                dq.append(v)
    ans = "yes" if t in seen else "no"
    if (rng.random() < .5) != (ans == "yes"):
        raise Reject
    W, H = int(Wm * ppm) + 40, int(Hm * ppm) + 70
    cv = Canvas(W, H, bg=(250, 250, 247))
    o = 20
    for x, y, w, h in rooms:
        cv.rect(o + x * ppm, o + y * ppm, o + (x + w) * ppm, o + (y + h) * ppm, outline=(30, 30, 30), width=4)
    for i, j, vert, x, c, w in doors:
        if vert:
            cv.rect(o + x * ppm - 4, o + (c - w / 2) * ppm, o + x * ppm + 4, o + (c + w / 2) * ppm, fill=(250, 250, 247))
        else:
            cv.rect(o + (c - w / 2) * ppm, o + x * ppm - 4, o + (c + w / 2) * ppm, o + x * ppm + 4, fill=(250, 250, 247))
    img_names = {s: "A", t: "B"}
    cv.line([(o, H - 25), (o + 2 * ppm, H - 25)], (0, 0, 0), 3)
    cv.text((o + ppm, H - 40), "2 m", 13)
    img = cv.done()
    for k, nm in img_names.items():
        x, y, w, h = rooms[k]
        tag(img, (o + (x + w / 2) * ppm, o + (y + h / 2) * ppm), f"Room {nm}", 15)
    return dict(image=img, atype="choice", choices=["yes", "no"], answer=ans,
                question=f"Floor plan (gaps in the walls are doorways; use the 2 m scale bar). A cart {cart:.1f} m wide "
                         f"must be pushed from Room A to Room B through doorways. Can it get there?",
                spec=dict(rooms=rooms, doors=doors, cart_m=cart, room_a=s, room_b=t))


# ---------------------------------------------------------------- P&ID valves
@task("pid_flow", "Process plants (P&ID diagrams): which tanks can receive fluid?", "graph reachability",
      knob="network size, number of valves", twin="puzzle: mirror laser / maze", group=G)
def pid_flow(rng, d):
    gx, gy = ilerp(4, 8, d), ilerp(3, 6, d)
    nodes = [(i, j) for i in range(gx) for j in range(gy)]
    edges = set()
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        u = stack[-1]
        nb = [(u[0] + a, u[1] + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= u[0] + a < gx and 0 <= u[1] + b < gy and (u[0] + a, u[1] + b) not in seen]
        if not nb:
            stack.pop()
            continue
        v = rng.choice(nb)
        edges.add(tuple(sorted((u, v))))
        seen.add(v)
        stack.append(v)
    for _ in range(ilerp(1, 5, d)):
        u = rng.choice(nodes)
        a, b = rng.choice(((1, 0), (0, 1)))
        v = (u[0] + a, u[1] + b)
        if v in seen:
            edges.add(tuple(sorted((u, v))))
    edges = sorted(edges)
    valves = {e: rng.random() < .5 for e in rng.sample(edges, min(len(edges), ilerp(3, 12, d)))}
    deg = {n: 0 for n in nodes}
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    leaves = [n for n in nodes if deg[n] == 1]
    if len(leaves) < 3:
        raise Reject
    rng.shuffle(leaves)
    src, tanks = leaves[0], leaves[1:1 + min(4, len(leaves) - 1)]
    adj = {n: [] for n in nodes}
    for e in edges:
        if valves.get(e, True):
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
    reach, dq = {src}, deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in reach:
                reach.add(v)
                dq.append(v)
    ans = sum(1 for t in tanks if t in reach)
    S = 90
    cv = Canvas(gx * S + 80, gy * S + 80)
    P = lambda n: (60 + n[0] * S, 60 + n[1] * S)
    for e in edges:
        cv.line([P(e[0]), P(e[1])], (60, 60, 70), 4)
    for e, open_ in valves.items():
        (x0, y0), (x1, y1) = P(e[0]), P(e[1])
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        s = 11
        if y0 == y1:
            tri = [[(mx - s, my - s), (mx - s, my + s), (mx, my)], [(mx + s, my - s), (mx + s, my + s), (mx, my)]]
        else:
            tri = [[(mx - s, my - s), (mx + s, my - s), (mx, my)], [(mx - s, my + s), (mx + s, my + s), (mx, my)]]
        for t in tri:
            cv.poly(t, fill=(20, 20, 20) if not open_ else (255, 255, 255), outline=(20, 20, 20))
    for n in nodes:
        if deg[n] >= 3:
            cv.ellipse(*P(n), 5, fill=(60, 60, 70))
    x, y = P(src)
    cv.ellipse(x, y, 16, fill=(60, 140, 220), outline=(20, 20, 20))
    cv.text((x, y), "S", 15, fill=(255, 255, 255))
    for k, t in enumerate(tanks):
        x, y = P(t)
        cv.rect(x - 18, y - 22, x + 18, y + 22, fill=(225, 225, 225), outline=(20, 20, 20), width=2)
        cv.text((x, y), f"T{k + 1}", 13)
    return dict(image=cv.done(), atype="int", answer=ans,
                question="Process diagram: fluid enters at source S and flows through the pipes. A valve drawn as two "
                         "WHITE triangles is open; two BLACK triangles is closed (blocks flow). Pipes only connect "
                         "where they meet. How many of the tanks (T1, T2, ...) can the fluid reach?",
                spec=dict(edges=edges, valves={f"{a}-{b}": v for (a, b), v in valves.items()}, src=src, tanks=tanks))


# ---------------------------------------------------------------- electrical schematic
@task("circuit_leds", "Electrical schematics: which loads are powered?", "circuit tracing",
      knob="branches and switches per branch", twin="puzzle: one-stroke / connectivity", group=G)
def circuit_leds(rng, d):
    nb = ilerp(2, 6, d)
    sw_max = ilerp(1, 3, d)
    main = rng.random() < .85
    branches = [[rng.random() < .55 for _ in range(rng.randint(1, sw_max))] for _ in range(nb)]
    lit = sum(1 for b in branches if main and all(b)) if True else 0
    W = 140 + nb * 110
    H = 420
    cv = Canvas(W, H)
    top, bot = 80, 360
    lx = 60
    cv.line([(lx, top), (lx, 190)], (20, 20, 20), 2.5)
    cv.line([(lx, 230), (lx, bot)], (20, 20, 20), 2.5)
    cv.line([(lx - 18, 190), (lx + 18, 190)], (20, 20, 20), 3)
    cv.line([(lx - 10, 200), (lx + 10, 200)], (20, 20, 20), 3)
    cv.line([(lx - 18, 220), (lx + 18, 220)], (20, 20, 20), 3)
    cv.line([(lx - 10, 230), (lx + 10, 230)], (20, 20, 20), 3)
    cv.text((lx - 30, 210), "+ -", 12)

    def switch(x0, y0, x1, y1, closed):
        cv.ellipse(x0, y0, 3.5, fill=(20, 20, 20))
        cv.ellipse(x1, y1, 3.5, fill=(20, 20, 20))
        if closed:
            cv.line([(x0, y0), (x1, y1)], (20, 20, 20), 2.5)
        elif y0 == y1:
            cv.line([(x0, y0), (x1 - 4, y1 - 16)], (20, 20, 20), 2.5)
        else:
            cv.line([(x0, y0), (x1 - 16, y1 - 4)], (20, 20, 20), 2.5)
    xr = 140 + (nb - 1) * 110
    cv.line([(lx, top), (100, top)], (20, 20, 20), 2.5)
    switch(100, top, 130, top, main)
    cv.line([(130, top), (xr, top)], (20, 20, 20), 2.5)
    cv.line([(lx, bot), (xr, bot)], (20, 20, 20), 2.5)
    for k, b in enumerate(branches):
        x = 140 + k * 110
        ys = np.linspace(top, bot - 90, len(b) + 1)
        y = top
        for s, closed in enumerate(b):
            y0 = top + 20 + s * 55
            cv.line([(x, y), (x, y0)], (20, 20, 20), 2.5)
            switch(x, y0, x, y0 + 30, closed)
            y = y0 + 30
        cv.line([(x, y), (x, 270)], (20, 20, 20), 2.5)
        cv.poly([(x - 14, 270), (x + 14, 270), (x, 296)], fill=(255, 255, 255), outline=(20, 20, 20))
        cv.line([(x - 14, 296), (x + 14, 296)], (20, 20, 20), 2.5)
        cv.line([(x + 16, 276), (x + 26, 268)], (20, 20, 20), 1.5)
        cv.line([(x, 296), (x, bot)], (20, 20, 20), 2.5)
        cv.text((x + 30, 286), f"L{k + 1}", 13)
        cv.ellipse(x, top, 4, fill=(20, 20, 20))
        cv.ellipse(x, bot, 4, fill=(20, 20, 20))
    return dict(image=cv.done(), atype="int", answer=lit,
                question="Circuit schematic: a battery feeds LEDs L1, L2, ... in parallel branches. A switch drawn as "
                         "a straight line between its two dots is CLOSED; one drawn at an angle is OPEN. An LED lights "
                         "only if there is a closed path from the battery through it and back. How many LEDs are lit?",
                spec=dict(main_closed=main, branches=branches))


# ---------------------------------------------------------------- transit map
@task("transit_route", "Navigation: reading a metro map for the route with fewest changes", "route planning on a map",
      knob="number of lines", twin="puzzle: tangled wires / maze", group=G)
def transit_route(rng, d):
    gx, gy, S = 9, 7, 70
    nl = ilerp(2, 6, d)
    cols = [(220, 50, 50), (40, 110, 210), (40, 160, 80), (240, 160, 30), (150, 70, 190), (20, 170, 180)]
    lines, used = [], set()
    for k in range(nl):
        for _t in range(200):
            x, y = rng.randrange(gx), rng.randrange(gy)
            path = [(x, y)]
            dx, dy = rng.choice(((1, 0), (0, 1), (1, 1), (1, -1)))
            for _s in range(rng.randint(4, 8)):
                if rng.random() < .35:
                    dx, dy = rng.choice(((1, 0), (0, 1), (1, 1), (1, -1), (0, -1)))
                nx, ny = path[-1][0] + dx, path[-1][1] + dy
                if not (0 <= nx < gx and 0 <= ny < gy) or (nx, ny) in path:
                    break
                path.append((nx, ny))
            segs = {tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)}
            if len(path) >= 4 and not (segs & used):
                used |= segs
                lines.append(path)
                break
        else:
            raise Reject
    stations = {}
    for li, p in enumerate(lines):
        for n in p:
            stations.setdefault(n, set()).add(li)
    if not any(len(v) > 1 for v in stations.values()):
        raise Reject
    names = {n: LETTERS[i] if i < 26 else LETTERS[i - 26].lower() for i, n in enumerate(sorted(stations))}
    if len(stations) > 50:
        raise Reject
    a, b = rng.sample(sorted(stations), 2)
    INF = 99
    best = {}
    dq = deque()
    for li in stations[a]:
        best[(a, li)] = 0
        dq.append((a, li))
    while dq:
        n, li = dq.popleft()
        c = best[(n, li)]
        p = lines[li]
        i = p.index(n)
        for m in (p[i - 1] if i > 0 else None, p[i + 1] if i < len(p) - 1 else None):
            if m is not None and best.get((m, li), INF) > c:
                best[(m, li)] = c
                dq.appendleft((m, li))
        for lj in stations[n]:
            if best.get((n, lj), INF) > c + 1:
                best[(n, lj)] = c + 1
                dq.append((n, lj))
    ans = min(best.get((b, li), INF) for li in stations[b])
    if ans >= INF:
        raise Reject
    cv = Canvas(gx * S + 60, gy * S + 60)
    P = lambda n: (40 + n[0] * S, 40 + n[1] * S)
    for li, p in enumerate(lines):
        cv.line([P(n) for n in p], cols[li], 7)
    img_tags = []
    for n, ls in stations.items():
        x, y = P(n)
        if len(ls) > 1:
            cv.ellipse(x, y, 10, fill=(255, 255, 255), outline=(20, 20, 20), width=3)
        else:
            cv.ellipse(x, y, 5, fill=(255, 255, 255), outline=cols[next(iter(ls))], width=2)
        cv.text((x + 14, y - 12), names[n], 12)
    return dict(image=cv.done(), atype="int", answer=ans,
                question=f"Metro map: each coloured line is a train line; white circles with a thick black ring are "
                         f"interchange stations where you can change lines. What is the minimum number of line changes "
                         f"needed to travel from station {names[a]} to station {names[b]}?",
                spec=dict(lines=lines, start=a, end=b))


# ---------------------------------------------------------------- aerial change detection
@task("aerial_change", "Satellite/aerial monitoring: counting new buildings between two dates", "change detection",
      knob="building count, registration/brightness shift", twin="puzzle: spot the difference", group=G)
def aerial_change(rng, d):
    W, H = 460, 380
    roads = [("h", rng.uniform(80, H - 80)), ("v", rng.uniform(80, W - 80))]
    blds = []

    def free(x, y, w, h):
        for k, v in roads:
            if k == "h" and y < v + 14 and y + h > v - 14:
                return False
            if k == "v" and x < v + 14 and x + w > v - 14:
                return False
        return all(x + w + 6 < b[0] or b[0] + b[2] + 6 < x or y + h + 6 < b[1] or b[1] + b[3] + 6 < y for b in blds)
    n = ilerp(6, 24, d)
    for _ in range(n + 6):
        for _t in range(100):
            w, h = rng.uniform(16, 42), rng.uniform(16, 42)
            x, y = rng.uniform(5, W - w - 5), rng.uniform(5, H - h - 5)
            if free(x, y, w, h):
                blds.append((x, y, w, h, rng.choice([(170, 160, 150), (150, 90, 70), (120, 120, 130)])))
                break
    if len(blds) < n + 4:
        raise Reject
    new = rng.randrange(0, ilerp(3, 6, d) + 1)
    removed = rng.randrange(0, 2)
    before, after = blds[:n], blds[:n - removed] + blds[n:n + new]
    trees = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(30)]

    def draw(bs, shift, gain):
        cv = Canvas(W, H, bg=tuple(int(c * gain) for c in (120, 150, 90)))
        for k, v in roads:
            if k == "h":
                cv.rect(0, v - 9 + shift[1], W, v + 9 + shift[1], fill=tuple(int(c * gain) for c in (95, 95, 95)))
            else:
                cv.rect(v - 9 + shift[0], 0, v + 9 + shift[0], H, fill=tuple(int(c * gain) for c in (95, 95, 95)))
        for x, y in trees:
            cv.ellipse(x + shift[0], y + shift[1], 6, fill=tuple(int(c * gain) for c in (60, 110, 55)))
        for x, y, w, h, c in bs:
            cv.rect(x + 3 + shift[0], y + 3 + shift[1], x + w + 3 + shift[0], y + h + 3 + shift[1], fill=(60, 60, 60))
            cv.rect(x + shift[0], y + shift[1], x + w + shift[0], y + h + shift[1], fill=tuple(int(v * gain) for v in c))
        return cv.done()
    sh = (rng.uniform(-1, 1) * 10 * d, rng.uniform(-1, 1) * 10 * d)
    img = Image.new("RGB", (2 * W + 20, H + 34), (255, 255, 255))
    img.paste(draw(before, (0, 0), 1.0), (0, 34))
    img.paste(draw(after, sh, lerp(1.0, 0.8, d)), (W + 20, 34))
    tag(img, (W / 2, 16), "Before", 14)
    tag(img, (W + 20 + W / 2, 16), "After", 14)
    return dict(image=img, atype="int", answer=new,
                question="Two aerial images of the same area taken at different dates (the second may be slightly "
                         "shifted or darker). How many NEW buildings appear in the After image that were not there "
                         "Before? (Ignore buildings that were demolished.)",
                spec=dict(n_before=n, new=new, removed=removed))


# ---------------------------------------------------------------- flood connectivity
@task("flood_route", "Disaster response: is a town still reachable by road after flooding?", "connectivity",
      knob="road grid size, flood patches", twin="puzzle: maze", group=G)
def flood_route(rng, d):
    gx, gy, S = ilerp(4, 7, d), ilerp(3, 5, d), 90
    W, H = gx * S + 60, gy * S + 60
    P = lambda i, j: (30 + i * S, 30 + j * S)
    edges = [((i, j), (i + 1, j)) for i in range(gx - 1) for j in range(gy)] + \
            [((i, j), (i, j + 1)) for i in range(gx) for j in range(gy - 1)]
    edges = [e for e in edges if rng.random() < .85]
    floods = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(25, 55)) for _ in range(ilerp(3, 7, d))]
    blocked = set()
    for e in edges:
        a, b = np.array(P(*e[0])), np.array(P(*e[1]))
        for fx, fy, fr in floods:
            pts = a + np.linspace(0, 1, 40)[:, None] * (b - a)
            dist = np.hypot(pts[:, 0] - fx, pts[:, 1] - fy).min()
            if abs(dist - fr) < 5:
                raise Reject
            if dist < fr:
                blocked.add(e)
    A = (0, rng.randrange(gy))
    B = (gx - 1, rng.randrange(gy))
    for fx, fy, fr in floods:
        for n in (A, B):
            if math.hypot(P(*n)[0] - fx, P(*n)[1] - fy) < fr + 12:
                raise Reject
    adj = {}
    for e in edges:
        if e not in blocked:
            adj.setdefault(e[0], []).append(e[1])
            adj.setdefault(e[1], []).append(e[0])
    seen, dq = {A}, deque([A])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    ans = "yes" if B in seen else "no"
    if (rng.random() < .5) != (ans == "yes"):
        raise Reject
    cv = Canvas(W, H, bg=(225, 220, 200))
    for a, b in edges:
        cv.line([P(*a), P(*b)], (110, 110, 110), 9)
    img = cv.done().convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for fx, fy, fr in floods:
        od.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=(60, 120, 220, 170))
    img = Image.alpha_composite(img, ov).convert("RGB")
    tag(img, P(*A), "A", 15)
    tag(img, P(*B), "B", 15)
    return dict(image=img, atype="choice", choices=["yes", "no"], answer=ans,
                question="Road map after a flood: gray lines are roads, blue areas are flooded and impassable (any "
                         "road segment touching blue is blocked). Can you still drive from A to B?",
                spec=dict(edges=edges, floods=floods, A=A, B=B))


# ---------------------------------------------------------------- shelf planogram
@task("shelf_planogram", "Retail: checking a shelf against its planogram", "grid comparison",
      knob="shelf size, product similarity", twin="puzzle: spot the difference", group=G)
def shelf_planogram(rng, d):
    R, C = ilerp(2, 4, d), ilerp(4, 8, d)
    prods = [("A", (220, 60, 50)), ("B", (50, 110, 210)), ("C", (240, 190, 40)), ("D", (50, 160, 80)),
             ("E", (230, 120, 160)), ("F", (130, 80, 50))]
    if d > 0.6:
        prods = [("A", (200, 60, 50)), ("B", (215, 80, 60)), ("C", (190, 70, 70)), ("D", (50, 110, 210)),
                 ("E", (70, 120, 220)), ("F", (60, 100, 190))]
    plan = [[rng.randrange(len(prods)) for _ in range(C)] for _ in range(R)]
    real = [row[:] for row in plan]
    err = rng.random() > .15
    if err:
        r, c = rng.randrange(R), rng.randrange(C)
        if rng.random() < .4:
            real[r][c] = None
        else:
            real[r][c] = rng.choice([k for k in range(len(prods)) if k != plan[r][c]])
        ans = f"{LETTERS[r]}{c + 1}"
    else:
        ans = "none"
    cw, rh = 52, 70
    W, H = 2 * (C * cw + 60) + 20, R * rh + 80
    cv = Canvas(W, H)
    for r in range(R):
        cv.text((14, 50 + r * rh + rh / 2), LETTERS[r], 14)
        for c in range(C):
            x, y = 30 + c * cw, 50 + r * rh
            k = plan[r][c]
            cv.rect(x + 2, y + 2, x + cw - 2, y + rh - 2, fill=tuple(int(v * .35 + 165) for v in prods[k][1]),
                    outline=(120, 120, 120))
            cv.text((x + cw / 2, y + rh / 2), prods[k][0], 15)
    for c in range(C):
        cv.text((30 + (c + .5) * cw, 38), c + 1, 12)
    cv.text((30 + C * cw / 2, 16), "Planogram", 15)
    ox = C * cw + 90
    cv.rect(ox - 8, 44, ox + C * cw + 8, 56 + R * rh, fill=(225, 215, 195))
    for r in range(R):
        y = 50 + r * rh
        cv.rect(ox - 8, y + rh - 6, ox + C * cw + 8, y + rh, fill=(150, 120, 90))
        for c in range(C):
            k = real[r][c]
            if k is None:
                continue
            x = ox + c * cw + rng.uniform(-3, 3)
            hgt = rh - 18 - rng.uniform(0, 8)
            cv.rect(x + 6, y + rh - 6 - hgt, x + cw - 6, y + rh - 6, fill=prods[k][1], outline=(40, 40, 40))
            cv.rect(x + 10, y + rh - 6 - hgt * .6, x + cw - 10, y + rh - 6 - hgt * .35, fill=(250, 250, 250))
            cv.text((x + cw / 2, y + rh - 6 - hgt * .48), prods[k][0], 10)
    cv.text((ox + C * cw / 2, 16), "Shelf photo", 15)
    slots = [f"{LETTERS[r]}{c + 1}" for r in range(R) for c in range(C)]
    return dict(image=cv.done(), atype="choice", choices=slots + ["none"], answer=ans,
                question="Compare the shelf photo (right) with the planogram (left); rows are lettered from the top, "
                         "columns numbered from the left. Which slot does not match the planogram (wrong product or "
                         "empty)? Answer like B3, or 'none'.",
                spec=dict(plan=plan, real=real))


# ---------------------------------------------------------------- liquid comparison
@task("liquid_volume", "Lab / food service: comparing volumes in differently shaped containers",
      "volume estimation", knob="similarity of volumes, decoy (tallest is not largest)", twin="text: profiles", group=G)
def liquid_volume(rng, d):
    n = ilerp(2, 5, d)
    glasses = []
    for _ in range(n):
        Hh = rng.uniform(120, 220)
        rb, rt = rng.uniform(20, 55), rng.uniform(20, 60)
        lvl = rng.uniform(.3, .92) * Hh
        rh = rb + (rt - rb) * lvl / Hh
        V = math.pi / 3 * lvl * (rb * rb + rb * rh + rh * rh)
        glasses.append((Hh, rb, rt, lvl, V))
    vs = sorted(g[4] for g in glasses)
    if vs[-1] < vs[-2] * lerp(1.5, 1.12, d):
        raise Reject
    best = max(range(n), key=lambda i: glasses[i][4])
    if d > 0.5 and max(range(n), key=lambda i: glasses[i][3]) == best and rng.random() < .7:
        raise Reject
    W = n * 150 + 40
    cv = Canvas(W, 300)
    for i, (Hh, rb, rt, lvl, V) in enumerate(glasses):
        cx, base = 90 + i * 150, 250
        rh = rb + (rt - rb) * lvl / Hh
        cv.poly([(cx - rb, base), (cx + rb, base), (cx + rh, base - lvl), (cx - rh, base - lvl)], fill=(120, 180, 230))
        cv.line([(cx - rt, base - Hh), (cx - rb, base), (cx + rb, base), (cx + rt, base - Hh)], (60, 60, 70), 3)
        cv.text((cx, base + 22), LETTERS[i], 16)
    return dict(image=cv.done(), atype="choice", choices=list(LETTERS[:n]), answer=LETTERS[best],
                question="These glasses are round when seen from above (side view shown, same scale). Which glass "
                         "contains the most liquid by volume?",
                spec=dict(glasses=[g[:4] for g in glasses]))


# ---------------------------------------------------------------- graduated cylinder
@task("cylinder_reading", "Lab instruments: reading a graduated cylinder at the meniscus", "exact value reading",
      knob="graduation density, label sparsity", twin="puzzle: gauge reading", group=G)
def cylinder_reading(rng, d):
    cap = rng.choice([50, 100, 250])
    minor = {50: 1, 100: 1 if d > .5 else 2, 250: 5 if d > .5 else 10}[cap]
    lab = {50: 10, 100: 10 if d < .5 else 20, 250: 50}[cap]
    v = round(rng.uniform(.15, .9) * cap / minor) * minor
    if rng.random() < .5 and d > .3:
        v += minor / 2 * 0
    H, top, bot = 560, 50, 520
    ppv = (bot - top) / cap
    cv = Canvas(260, H)
    cx, rw = 120, 34
    lv = bot - v * ppv
    cv.rect(cx - rw, lv, cx + rw, bot, fill=(140, 200, 235))
    for x in np.linspace(cx - rw, cx + rw, 30):
        dy = 7 * (1 - ((x - cx) / rw) ** 2)
        cv.line([(x, lv - 7 + dy), (x, lv)], (140, 200, 235), 3)
    cv.ellipse(cx, lv - 7 + 7, 1, fill=(140, 200, 235))
    cv.line([(cx - rw, top - 10), (cx - rw, bot), (cx + rw, bot), (cx + rw, top - 10)], (60, 60, 70), 3)
    for m in np.arange(0, cap + 1e-9, minor):
        y = bot - m * ppv
        major = abs(m / lab - round(m / lab)) < 1e-9
        cv.line([(cx - rw, y), (cx - rw + (16 if major else 8), y)], (40, 40, 40), 1.5 if major else 1)
        if major:
            cv.text((cx - rw - 8, y), int(m), 12, anchor="rm")
    cv.text((cx, 25), "mL", 13)
    return dict(image=cv.done(), atype="float", answer=float(v), tol=minor / 2 + 1e-6,
                question="Read this graduated cylinder at the bottom of the curved liquid surface (meniscus). "
                         "What volume does it show, in mL?",
                spec=dict(capacity=cap, minor=minor, volume=v))


# ---------------------------------------------------------------- billiards bank shot
@task("billiards_bank", "Sports analytics / robotics: predicting a bounce path", "reflection reasoning",
      knob="number of cushion bounces", twin="puzzle: mirror laser", group=G)
def billiards_bank(rng, d):
    Wt, Ht, S = 2.0, 1.0, 360
    pockets = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    pr = 0.07
    want = ilerp(0, 3, d)
    p = np.array([rng.uniform(.3, 1.7), rng.uniform(.2, .8)])
    a = rng.uniform(0, 2 * math.pi)
    v = np.array([math.cos(a), math.sin(a)])
    pos, bounces, path = p.copy(), 0, [p.copy()]
    for _ in range(4000):
        pos = pos + v * 0.003
        for k, q in enumerate(pockets):
            dist = math.hypot(pos[0] - q[0], pos[1] - q[1])
            if dist < pr:
                path.append(pos.copy())
                if bounces != want:
                    raise Reject
                # ambiguity: path must not graze another pocket
                for j, q2 in enumerate(pockets):
                    if j != k and min(math.hypot(pt[0] - q2[0], pt[1] - q2[1]) for pt in path) < pr + .06:
                        raise Reject
                cv = Canvas(int(Wt * S) + 80, int(Ht * S) + 80, bg=(255, 255, 255))
                o = 40
                cv.rect(o - 18, o - 18, o + Wt * S + 18, o + Ht * S + 18, fill=(110, 70, 40))
                cv.rect(o, o, o + Wt * S, o + Ht * S, fill=(40, 120, 70))
                for kk, (qx, qy) in enumerate(pockets):
                    cv.ellipse(o + qx * S, o + qy * S, pr * S, fill=(20, 20, 20))
                    cv.text((o + qx * S + (0 if qx == 1 else (-28 if qx == 0 else 28)), o + qy * S + (-30 if qy == 0 else 30)),
                            kk + 1, 15)
                bx, by = o + p[0] * S, o + p[1] * S
                cv.line([(bx, by), (bx + v0[0] * 60, by + v0[1] * 60)], (250, 250, 250), 2.5)
                cv.ellipse(bx, by, 9, fill=(250, 250, 250), outline=(30, 30, 30))
                ex, ey = bx + v0[0] * 60, by + v0[1] * 60
                cv.poly([(ex + v0[0] * 10, ey + v0[1] * 10), (ex - v0[1] * 6, ey + v0[0] * 6),
                         (ex + v0[1] * 6, ey - v0[0] * 6)], fill=(250, 250, 250))
                return dict(image=cv.done(), atype="int", answer=k + 1,
                            question="Top view of a pool table. The white ball is hit in the direction of the arrow; "
                                     "it travels in a straight line and bounces off cushions like a mirror (angle in = "
                                     "angle out), with no spin and enough speed. Which numbered pocket does it fall "
                                     "into first?", spec=dict(start=p.tolist(), dir=v0.tolist(), bounces=bounces))
        if pos[0] < 0 or pos[0] > Wt:
            v[0] = -v[0]
            bounces += 1
            path.append(pos.copy())
        if pos[1] < 0 or pos[1] > Ht:
            v[1] = -v[1]
            bounces += 1
            path.append(pos.copy())
        if bounces > want:
            raise Reject
        if len(path) == 1:
            v0 = v.copy()
    raise Reject


# ---------------------------------------------------------------- UI state
@task("ui_state", "Computer-use agents: reading the state of a settings screen", "small-detail reading",
      knob="number of widgets, low contrast, small size", twin="puzzle: small-detail spot the difference", group=G)
def ui_state(rng, d):
    sz = lerp(1.0, 0.75, d)
    W, H = 520, 520
    cv = Canvas(W, H, bg=(246, 247, 250))
    cv.rect(20, 20, W - 20, H - 20, fill=(255, 255, 255), outline=(220, 222, 228))
    cv.text((40, 45), "Notification settings", 18 * sz, anchor="lm")
    opts = ["Email alerts", "SMS alerts", "Weekly digest", "Product news", "Security notices", "Beta features",
            "Mentions only", "Daily summary"]
    nchk = ilerp(3, 8, d)
    checked = [rng.random() < .5 for _ in range(nchk)]
    light = d > .6
    y = 85
    for o, c in zip(opts, checked):
        s = 16 * sz
        cv.rect(40, y - s / 2, 40 + s, y + s / 2, fill=(255, 255, 255) if not c else ((150, 170, 210) if light else (40, 100, 220)),
                outline=(170, 175, 185), width=1.5)
        if c:
            cv.line([(43, y), (40 + s * .42, y + s * .3), (40 + s - 3, y - s * .35)], (255, 255, 255), 2)
        cv.text((40 + s + 12, y), o, 14 * sz, anchor="lm", fill=(40, 40, 50))
        y += 30 * sz
    freq = ["Instantly", "Hourly", "Daily", "Weekly"]
    sel = rng.randrange(4)
    y += 14
    cv.text((40, y), "Frequency", 15 * sz, anchor="lm")
    y += 28 * sz
    for k, f in enumerate(freq):
        cv.ellipse(48, y, 8 * sz, fill=(255, 255, 255), outline=(150, 155, 165), width=1.5)
        if k == sel:
            cv.ellipse(48, y, 4 * sz, fill=(150, 170, 210) if light else (40, 100, 220))
        cv.text((64, y), f, 14 * sz, anchor="lm")
        y += 26 * sz
    save_on = rng.random() < .5
    cv.rect(W - 170, H - 70, W - 50, H - 36, fill=(40, 100, 220) if save_on else (205, 208, 214))
    cv.text((W - 110, H - 53), "Save", 15, fill=(255, 255, 255) if save_on else (245, 245, 245))
    qt = rng.choice(["count", "radio", "save"])
    if qt == "count":
        return dict(image=cv.done(), atype="int", answer=sum(checked),
                    question="How many checkboxes in this settings screen are checked?",
                    spec=dict(checked=checked, freq=sel, save=save_on))
    if qt == "radio":
        return dict(image=cv.done(), atype="choice", choices=freq, answer=freq[sel],
                    question="Which 'Frequency' option is currently selected?",
                    spec=dict(checked=checked, freq=sel, save=save_on))
    return dict(image=cv.done(), atype="choice", choices=["yes", "no"], answer="yes" if save_on else "no",
                question="Is the 'Save' button enabled (clickable) in this screen? A disabled button is grayed out.",
                spec=dict(checked=checked, freq=sel, save=save_on))


# ---------------------------------------------------------------- spreadsheet sort
@task("spreadsheet_rank", "Office/finance documents: reading and ranking table values from a screenshot",
      "table reading + sorting", knob="row count, small font", twin="text: CSV", group=G)
def spreadsheet_rank(rng, d):
    n = ilerp(5, 22, d)
    names = rng.sample(["Avery", "Blake", "Casey", "Drew", "Emery", "Finley", "Gray", "Harper", "Indy", "Jules", "Kai",
                        "Logan", "Morgan", "Noel", "Oakley", "Parker", "Quinn", "Reese", "Sage", "Taylor", "Uma",
                        "Val", "Wren", "Xen"], n)
    sales = rng.sample(range(1200, 9900, 7), n)
    units = [rng.randrange(10, 400) for _ in range(n)]
    regs = [rng.choice(["North", "South", "East", "West"]) for _ in range(n)]
    k = rng.randint(1, min(5, n))
    order = sorted(range(n), key=lambda i: -sales[i])
    fs = lerp(14, 10.5, d)
    rh = fs + 10
    cols = [("", 34), ("Name", 110), ("Region", 90), ("Sales", 90), ("Units", 70)]
    W = sum(c[1] for c in cols) + 20
    H = int((n + 2) * rh + 30)
    cv = Canvas(W, H)
    x = 10
    letters = " ABCD"
    for ci, (nm, w) in enumerate(cols):
        cv.rect(x, 10, x + w, 10 + rh, fill=(235, 236, 240), outline=(200, 200, 205))
        cv.text((x + w / 2, 10 + rh / 2), letters[ci], fs, fill=(90, 90, 90))
        x += w
    for r in range(n + 1):
        y = 10 + (r + 1) * rh
        x = 10
        vals = [str(r + 1), "Name", "Region", "Sales", "Units"] if r == 0 else \
            [str(r + 1), names[r - 1], regs[r - 1], f"{sales[r - 1]:,}", str(units[r - 1])]
        for ci, (nm, w) in enumerate(cols):
            cv.rect(x, y, x + w, y + rh, fill=(235, 236, 240) if ci == 0 else (255, 255, 255), outline=(210, 210, 215))
            anchor = "rm" if ci in (3, 4) and r > 0 else "lm"
            tx = x + w - 6 if anchor == "rm" else x + 6
            cv.text((tx, y + rh / 2), vals[ci], fs, anchor=anchor, fill=(30, 30, 30) if r else (20, 20, 20))
            x += w
    ordn = {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")
    return dict(image=cv.done(), atype="choice", choices=names, answer=names[order[k - 1]],
                question=f"If this table were sorted by Sales from highest to lowest, whose row would be {ordn}? "
                         f"Answer with the Name.", spec=dict(names=names, sales=sales, k=k))


# ---------------------------------------------------------------- Gantt chart
@task("gantt_overlap", "Project management: reading a Gantt chart", "interval reading",
      knob="number of tasks and days, sparse date labels", twin="text: task intervals", group=G)
def gantt_overlap(rng, d):
    nt = ilerp(4, 14, d)
    nd = ilerp(10, 30, d)
    tasks = []
    for i in range(nt):
        s = rng.randrange(0, nd - 2)
        e = min(nd - 1, s + rng.randrange(1, max(2, nd // 2)))
        tasks.append((f"Task {i + 1}", s, e))
    q = rng.randrange(nd)
    ans = sum(1 for _, s, e in tasks if s <= q <= e)
    cw = 26 if nd <= 20 else 20
    W, rh = 110 + nd * cw + 20, 26
    H = 60 + nt * rh + 20
    cv = Canvas(W, H)
    lab_every = 1 if d < .4 else (2 if d < .7 else 5)
    for k in range(nd):
        x = 110 + k * cw
        cv.line([(x, 40), (x, H - 15)], (232, 232, 236), 1)
        if (k + 1) % lab_every == 0 or k == 0:
            cv.text((x + cw / 2, 28), k + 1, 11)
    cv.text((110 + nd * cw / 2, 10), "Day of month (March)", 12)
    for i, (nm, s, e) in enumerate(tasks):
        y = 50 + i * rh
        cv.text((12, y + rh / 2 - 3), nm, 12, anchor="lm")
        cv.rect(110 + s * cw + 1, y + 3, 110 + (e + 1) * cw - 1, y + rh - 7, fill=(70, 130, 210))
    return dict(image=cv.done(), atype="int", answer=ans,
                question=f"Gantt chart: each bar covers whole days (a bar spanning a day's column means the task runs "
                         f"that day). How many tasks are running on March {q + 1}?",
                spec=dict(tasks=tasks, day=q + 1))
