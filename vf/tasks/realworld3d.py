"""Real-world 3D tasks rendered with the NumPy ray tracer: assembly, bin picking, agriculture, sport, forensics."""
import math

import numpy as np
from PIL import Image

from ..draw import Canvas
from ..rt import Scene, flat_tex, floor_tex
from ..taskapi import LETTERS, Reject, ilerp, lerp, tag, task

G = "real3d"
PAL = {"red": (210, 60, 50), "blue": (50, 100, 210), "green": (50, 160, 80), "yellow": (230, 190, 40),
       "purple": (140, 70, 190), "orange": (235, 130, 40)}


# ---------------------------------------------------------------- assembly vs blueprint
@task("assembly_check", "Manufacturing / construction: does the build match the blueprint?",
      "2D-plan to 3D-view correspondence", knob="grid size, similar colours, lower camera",
      twin="puzzle: missing piece / spot the difference", group=G)
def assembly_check(rng, d):
    g = ilerp(3, 6, d)
    names = list(PAL) if d < .6 else ["red", "orange", "yellow", "purple", "blue", "green"][:4]
    plan = [[rng.choice(names + [None]) for _ in range(g)] for _ in range(g)]
    build = [row[:] for row in plan]
    err = rng.random() > .15
    if err:
        r, c = rng.randrange(g), rng.randrange(g)
        opts = [n for n in names + [None] if n != plan[r][c]]
        build[r][c] = rng.choice(opts)
        ans = f"{LETTERS[r]}{c + 1}"
    else:
        ans = "none"
    el = lerp(9, 5, d)
    sc = Scene(560, 460, cam=(g / 2 - .5 + 2.5, el, g + 4), look=(g / 2 - .5, 0, g / 2 - .5),
               ground=floor_tex((215, 215, 210), (200, 200, 195), 1.0), fov=45, light=(-.3, 1, .6))
    for r in range(g):
        for c in range(g):
            if build[r][c]:
                sc.box((c, .45, r), (.9, .9, .9), PAL[build[r][c]], edges=True)
    img3 = sc.render(ss=2)
    for r in range(g):
        tag(img3, sc.project((-1.0, 0, r)), LETTERS[r], 13)
    for c in range(g):
        tag(img3, sc.project((c, 0, -1.0)), c + 1, 13)
    S = 300 // g
    cv = Canvas(g * S + 60, g * S + 60)
    for r in range(g):
        cv.text((14, 40 + (r + .5) * S), LETTERS[r], 14)
        for c in range(g):
            x, y = 30 + c * S, 30 + r * S
            k = plan[r][c]
            cv.rect(x, y, x + S, y + S, fill=PAL[k] if k else (250, 250, 250), outline=(150, 150, 150))
    for c in range(g):
        cv.text((30 + (c + .5) * S, 16), c + 1, 13)
    bp = cv.done()
    img = Image.new("RGB", (bp.width + img3.width + 20, max(bp.height, img3.height) + 30), (255, 255, 255))
    img.paste(bp, (0, 30))
    img.paste(img3, (bp.width + 20, 30))
    tag(img, (bp.width / 2, 14), "Blueprint (top view)", 13)
    tag(img, (bp.width + 20 + img3.width / 2, 14), "Build", 13)
    slots = [f"{LETTERS[r]}{c + 1}" for r in range(g) for c in range(g)]
    return dict(image=img, atype="choice", choices=slots + ["none"], answer=ans,
                question="The blueprint (left, top view) shows which coloured block belongs in each grid cell "
                         "(white = empty). The build (right) uses the same row letters and column numbers. Which cell "
                         "of the build does not match the blueprint? Answer like B3, or 'none'.",
                spec=dict(plan=plan, build=build))


# ---------------------------------------------------------------- bin picking order
@task("pick_order", "Robotic bin picking: which item can be lifted without disturbing others?",
      "3D support reasoning", knob="pile size", twin="puzzle: layer stack", group=G)
def pick_order(rng, d):
    n = ilerp(5, 12, d)
    cell = .25
    hmap = {}
    boxes = []
    for i in range(n):
        for _t in range(40):                   # resample a drop position until the box would rest stably
            w, dpt, h = rng.choice([(.75, .5, .3), (.5, .5, .4), (1.0, .5, .25), (.5, .75, .35)])
            # snap the box's left/back edge to the grid so its footprint covers whole cells exactly
            cx0, cz0 = rng.randrange(-5, 5 - round(w / cell)), rng.randrange(-4, 4 - round(dpt / cell))
            x, z = cx0 * cell + w / 2, cz0 * cell + dpt / 2
            cells = [(cx0 + a, cz0 + b) for a in range(round(w / cell)) for b in range(round(dpt / cell))]
            base = max(hmap.get(cc, (0.0, None))[0] for cc in cells)
            sup = {hmap[cc][1] for cc in cells
                   if cc in hmap and abs(hmap[cc][0] - base) < 1e-6 and hmap[cc][1] is not None}
            cover = sum(1 for cc in cells if cc in hmap and abs(hmap[cc][0] - base) < 1e-6) if base > 0 else len(cells)
            if base == 0 or cover >= len(cells) * .5:
                break
        else:
            raise Reject                       # could not place it without tipping over
        boxes.append(dict(x=x, z=z, w=w, d=dpt, h=h, y=base, on=sorted(sup)))
        for cc in cells:
            hmap[cc] = (base + h, i)
    covered = {j for b in boxes for j in b["on"]}
    free = [i for i in range(n) if i not in covered]
    blocked = [i for i in range(n) if i in covered]
    if not free or len(blocked) < 2:
        raise Reject
    sc = Scene(600, 460, cam=(2.6, 4.2, 4.6), look=(0, .3, 0), ground=flat_tex((150, 150, 155)), fov=45,
               light=(-.4, 1, .5))
    cols = [(196, 160, 110), (180, 140, 95), (210, 175, 125), (170, 130, 90)]
    ids = []
    for i, b in enumerate(boxes):
        ids.append(sc.box((b["x"], b["y"] + b["h"] / 2, b["z"]), (b["w"] - .02, b["h"], b["d"] - .02),
                          cols[i % 4], edges=True))
    img = sc.render(ss=2)
    vfree = [i for i in free if sc.visible_px(ids[i]) >= 120]
    vblock = [i for i in blocked if sc.visible_px(ids[i]) >= 120]
    k = min(len(vblock), ilerp(2, 4, d))
    if not vfree or k < 2:
        raise Reject
    labeled = [rng.choice(vfree)] + rng.sample(vblock, k)
    rng.shuffle(labeled)
    for t, i in enumerate(labeled):
        ys, xs = np.nonzero(sc.idbuf == ids[i])
        tag(img, (xs.mean(), ys.mean()), LETTERS[t], 14)
    ans = LETTERS[labeled.index(next(i for i in labeled if i in free))]
    return dict(image=img, atype="choice", choices=list(LETTERS[:len(labeled)]), answer=ans,
                question="A robot must lift one box straight up without moving any other box, so the box it lifts "
                         "must have nothing resting on top of it. Exactly one of the labeled boxes can be lifted this "
                         "way. Which one?",
                spec=dict(boxes=boxes, labeled=labeled))


# ---------------------------------------------------------------- fruit counting under occlusion
@task("fruit_count", "Agriculture: yield estimation by counting fruit through foliage", "counting under occlusion",
      knob="leaf density, fruit count", twin="puzzle: overlapping-circle counting", group=G)
def fruit_count(rng, d):
    sc = Scene(560, 480, cam=(0, 2.2, 6.5), look=(0, 2.2, 0), ground=flat_tex((110, 150, 80)), fov=45,
               light=(-.3, 1, .6), sky=(200, 220, 240))
    sc.cyl((0, 0, 0), .18, 1.6, (110, 80, 50))
    nl = ilerp(4, 22, d)
    for _ in range(nl):
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, 1.2)
        sc.sphere((r * math.cos(a), 2.4 + rng.uniform(-.8, .8), r * math.sin(a) * .6 + .1), rng.uniform(.35, .6),
                  (50 + rng.randrange(30), 120 + rng.randrange(40), 50))
    nf = ilerp(3, 14, d)
    fr = []
    for _ in range(nf):
        for _t in range(100):
            p = (rng.uniform(-1.5, 1.5), rng.uniform(1.5, 3.3), rng.uniform(-.3, 1.2))
            if all(math.dist(p, q) > .32 for q in fr):
                fr.append(p)
                break
    fid = [sc.sphere(p, .14, (215, 40, 35)) for p in fr]
    img = sc.render(ss=2)
    count = 0
    for i, p in zip(fid, fr):
        full = math.pi * sc.proj_radius(p, .14) ** 2
        f = sc.visible_px(i) / max(full, 1)
        if .12 < f < .45:
            raise Reject                     # partly visible fruit: ambiguous for a human counter
        count += f >= .45
    if count == 0 and rng.random() < .7:
        raise Reject
    return dict(image=img, atype="int", answer=int(count),
                question="How many red fruits can you see on this tree? Count a fruit if at least about half of it "
                         "is visible; ignore fruit that is completely hidden.",
                spec=dict(fruit=fr, leaves=nl))


# ---------------------------------------------------------------- offside
def pitch_tex():
    def f(x, z):
        col = np.tile(np.array((60, 140, 60.0)), (len(x), 1))
        col[np.mod(np.floor(x / 5), 2) == 1] = (70, 155, 70)
        line = (np.abs(z - 10) < .08) | (np.abs(x - 52.5) < .08) | (np.abs(x) < .08) | \
               ((np.abs(x - 36) < .08) & (np.abs(z) < 20)) | ((np.abs(z - 20) < .08) & (x > 36))
        col[line] = (235, 235, 235)
        return col
    return f


@task("offside_call", "Sports officiating (VAR): is the attacker beyond the second-last defender?",
      "relative depth under perspective", knob="margin, camera obliqueness", twin="text: player x-positions", group=G)
def offside_call(rng, d):
    gx = 52.5
    nd = ilerp(3, 6, d)
    margin = lerp(2.5, 0.35, d)
    defs = sorted((rng.uniform(25, 46), rng.uniform(-15, 15)) for _ in range(nd))
    second_last = sorted([gx - .5] + [x for x, _ in defs])[-2]
    off = rng.random() < .5
    ax = second_last + (margin if off else -margin) * rng.uniform(1, 1.4)
    if ax > gx - 2:
        raise Reject
    att = (ax, rng.uniform(-14, 14))
    camx = lerp(38, 26, d)
    sc = Scene(720, 420, cam=(camx, 7, 24), look=(40, 0, 0), ground=pitch_tex(), fov=48, light=(.3, 1, .5),
               sky=(160, 190, 220))

    def player(x, z, col):
        sc.cyl((x, 0, z), .28, 1.2, col)
        sc.sphere((x, 1.45, z), .2, (220, 180, 150))
    for x, z in defs:
        player(x, z, (40, 70, 200))
    player(gx - .5, rng.uniform(-3, 3), (230, 200, 40))
    for _ in range(ilerp(1, 3, d)):
        player(rng.uniform(25, 45), rng.uniform(-15, 15), (210, 40, 40))
    player(*att, (210, 40, 40))
    img = sc.render(ss=2)
    tag(img, sc.project((att[0], 2.4, att[1])), "A", 14)
    return dict(image=img, atype="choice", choices=["yes", "no"], answer="yes" if off else "no",
                question="Broadcast view of a football pitch; the blue team (and yellow goalkeeper) defend the goal on "
                         "the right. Is red attacker A closer to that goal line than the second-to-last defender "
                         "(counting the goalkeeper), i.e. is A in an offside position? Judge by the players' bodies.",
                spec=dict(defenders=defs, attacker=att, second_last=second_last))


# ---------------------------------------------------------------- tennis line call
def court_tex(zb, w):
    def f(x, z):
        col = np.tile(np.array((45, 95, 160.0)), (len(x), 1))
        col[(np.abs(x) > 5.5) | (z < -12)] = (70, 130, 80)
        line = ((z >= zb) & (z <= zb + w) & (np.abs(x) <= 4.2)) | ((np.abs(np.abs(x) - 4.1) < w / 2) & (z < zb + w)) | \
               ((np.abs(z - (zb - 5.5)) < w / 2) & (np.abs(x) < 4.1)) | ((np.abs(x) < w / 2) & (z < zb - 5.5))
        col[line] = (240, 240, 240)
        return col
    return f


@task("line_call", "Sports officiating (Hawk-Eye style): was the ball in or out?", "fine boundary judgement",
      knob="distance of the mark from the line, camera angle", twin="text: mark position vs line", group=G)
def line_call(rng, d):
    zb, w, r = 0.0, .05, .045
    m = lerp(.12, .015, d) * rng.uniform(1, 1.5)
    out = rng.random() < .5
    zc = zb + w + r + m if out else zb + w + r - m - .002
    xc = rng.uniform(-3, 3)
    cam = (xc + rng.uniform(-.6, .6), lerp(.9, .55, d), zb - lerp(1.0, 1.9, d))
    sc = Scene(640, 420, cam=cam, look=(xc, 0, zc), ground=court_tex(zb, w), fov=34, light=(.2, 1, -.4))
    tex = sc.ground

    def g(x, z, tex=tex):
        col = tex(x, z)
        mark = (x - xc) ** 2 + (z - zc) ** 2 < r * r
        col[mark] = col[mark] * .45 + np.array((50, 45, 40)) * .55
        return col
    sc.ground = g
    sc.sphere((xc + .25, .033, zc + .3), .033, (220, 235, 60))
    img = sc.render(ss=2)
    return dict(image=img, atype="choice", choices=["in", "out"], answer="out" if out else "in",
                question="Tennis: the darker oval on the court is the mark where the ball bounced, next to the white "
                         "baseline (the court is on the camera side of the line). A ball whose mark touches any part "
                         "of the line is IN. Was this ball in or out?",
                spec=dict(mark_gap_m=round(m if out else -m, 4)))


# ---------------------------------------------------------------- mirror reflection consistency
@task("mirror_check", "Image forensics: detecting a manipulated reflection", "reflection consistency",
      knob="size of the manipulation", twin="puzzle: shadow check", group=G)
def mirror_check(rng, d):
    sc = Scene(640, 480, cam=(1.6, 2.2, 4.6), look=(0, .8, -1.6), ground=floor_tex((176, 136, 96), (164, 124, 86), .6),
               fov=60, light=(-.3, 1, .6))
    sc.box((0, 1.6, -3.0), (6.0, 3.0, .05), (200, 210, 220), mirror=True, cast=False)
    for c, sz in (((0, 3.15, -2.97), (6.3, .12, .1)), ((0, .05, -2.97), (6.3, .12, .1)),
                  ((-3.08, 1.6, -2.97), (.12, 3.2, .1)), ((3.08, 1.6, -2.97), (.12, 3.2, .1))):
        sc.box(c, sz, (55, 45, 40), cast=False)   # dark frame so the mirror reads as a mirror
    n = ilerp(3, 4, d)
    wrong = rng.randrange(n) if rng.random() > .2 else None
    pos, anchors = [], []
    for i in range(n):
        for _t in range(200):
            p = (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 0.8))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 1.5 for q in pos):
                pos.append(p)
                break
        else:
            raise Reject
        x, z = p
        kw = {}
        col = list(PAL.values())[i % 6]
        if i == wrong:                       # a person spots these at a glance: missing, or wrong colour
            if rng.random() < .5:
                kw["refl_hide"] = True
            else:
                kw["refl_color"] = list(PAL.values())[(i + 3) % 6]
        k = rng.choice(["box", "sphere", "cyl"])
        if k == "box":
            sc.box((x, .4, z), (.6, .8, .6), col, **kw)
            top = .8
        elif k == "sphere":
            sc.sphere((x, .4, z), .4, col, **kw)
            top = .8
        else:
            sc.cyl((x, 0, z), .3, .9, col, **kw)
            top = .9
        anchors.append((LETTERS[i], (x, top + .25, z)))
    img = sc.render(ss=2)
    for lab, p in anchors:
        tag(img, sc.project(p), lab, 14)
    return dict(image=img, atype="choice", choices=list(LETTERS[:n]) + ["none"],
                answer=LETTERS[wrong] if wrong is not None else "none",
                question="A flat mirror covers the back wall. Which labeled object's reflection is wrong (missing, "
                         "or a different colour from the object)? Answer the letter, or 'none' if every reflection is "
                         "correct.",
                spec=dict(positions=pos, wrong=wrong))
