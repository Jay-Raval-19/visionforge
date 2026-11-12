"""Driving tasks (doc 20-25). 3D ones use the NumPy ray tracer; rule/timing ones use top-down or side diagrams."""
import math

import numpy as np
from PIL import Image

from ..draw import Canvas
from ..rt import Scene
from ..taskapi import LETTERS, Reject, ilerp, lerp, tag, task

G = "driving"
LANE = 3.5
CARCOL = [(200, 40, 40), (40, 80, 180), (230, 230, 230), (40, 40, 40), (200, 170, 40), (60, 140, 70)]


def road_tex(n_lanes=3, night=False):
    half = n_lanes * LANE / 2

    def f(x, z):
        ax = np.abs(x)
        col = np.tile(np.array((95, 95, 98.0)), (len(x), 1))
        col[ax > half] = (90, 140, 70)
        col[(ax > half - .15) & (ax < half)] = (235, 235, 235)
        for k in range(1, n_lanes):
            bx = -half + k * LANE
            dash = (np.mod(z, 9.0) < 4.0)
            col[(np.abs(x - bx) < .08) & dash] = (235, 235, 235)
        return col * (0.45 if night else 1.0)
    return f


def _car(sc, x, z, col, truck=False):
    if truck:
        sc.box((x, 1.9, z), (2.4, 2.6, 7.0), col)
        sc.box((x, .55, z), (2.2, .6, 6.6), (30, 30, 30))
        return 3.4
    sc.box((x, .7, z), (1.8, .7, 4.2), col)
    sc.box((x, 1.3, z + .3), (1.6, .55, 2.2), tuple(c * .7 for c in col))
    for sx in (-.75, .75):
        for sz in (-1.3, 1.3):
            sc.cyl((x + sx, 0, z + sz), .33, .4, (25, 25, 25))
    return 1.7


def _ped(sc, x, z, col=(40, 60, 160)):
    sc.cyl((x - .1, 0, z), .08, .85, (40, 40, 50))
    sc.cyl((x + .1, 0, z), .08, .85, (40, 40, 50))
    sc.cyl((x, .85, z), .22, .65, col)
    sc.sphere((x, 1.68, z), .14, (230, 190, 160))
    return 1.85


# ---------------------------------------------------------------- 20 obstacle in lane
@task("obstacle_lane", "Autonomous driving / ADAS: what is in my lane?", "lane-relative localisation",
      knob="distance, clutter, night", twin="text: object list with lane and distance", group=G)
def obstacle_lane(rng, d):
    night = d > 0.7 and rng.random() < 0.6
    sc = Scene(720, 420, cam=(0, 1.5, 0), look=(0, 1.1, -30), ground=road_tex(3, night), fov=55,
               light=(.3, 1, .4), ambient=.25 if night else .4, sky=(40, 45, 70) if night else (190, 210, 235))
    kinds = ["car", "truck", "cone", "pedestrian", "debris"]
    n = ilerp(2, 6, d)
    objs = []
    for _ in range(n):
        for _t in range(100):
            lane = rng.choice([-1, 0, 1])
            dist = rng.uniform(8, lerp(28, 50, d))
            if all(abs(dist - o[2]) > 7 or o[1] != lane for o in objs):
                objs.append((rng.choice(kinds), lane, dist, rng.uniform(-.5, .5)))
                break
    for k, lane, dist, off in objs:
        x, z = lane * LANE + off * (0.3 if k in ("car", "truck") else 1), -dist
        if k in ("car", "truck"):
            _car(sc, x, z, CARCOL[rng.randrange(len(CARCOL))], truck=(k == "truck"))
        elif k == "cone":
            sc.cyl((x, 0, z), .25, .12, (240, 110, 20))
            sc.cyl((x, .12, z), .14, .55, (250, 120, 30))
        elif k == "pedestrian":
            _ped(sc, x, z)
        else:
            sc.box((x, .2, z), (1.0, .4, .7), (120, 90, 60), yaw=rng.uniform(0, 1))
    own = sorted((o for o in objs if o[1] == 0), key=lambda o: o[2])
    ans = own[0][0] if own else "none"
    img = sc.render(ss=2)
    return dict(image=img, atype="choice", choices=kinds + ["none"], answer=ans,
                question="You are driving in the middle lane (camera view from the driver). What is the nearest "
                         "object in YOUR lane? Answer one of: car, truck, cone, pedestrian, debris, none.",
                spec=dict(objects=objs, night=night))


# ---------------------------------------------------------------- 22 hidden pedestrian
@task("hidden_pedestrian", "Autonomous driving: anticipating people emerging from behind parked vehicles",
      "partial-visibility detection", knob="visible fraction of the pedestrian", twin="text: positions", group=G)
def hidden_pedestrian(rng, d):
    parked = []
    for side in (-1, 1):
        for _ in range(rng.randrange(1, 3)):
            parked.append((side, rng.uniform(12, 30), rng.random() < .5, CARCOL[rng.randrange(len(CARCOL))]))
    ans = rng.choice(["left", "right", "left", "right", "none"])   # fixed up front; never re-drawn below

    def build(ped=None):
        sc = Scene(720, 420, cam=(0, 1.5, 0), look=(0, 1.0, -30), ground=road_tex(2), fov=55, light=(-.3, 1, .3))
        for side, dist, truck, col in parked:
            _car(sc, side * (LANE + 1.6), -dist, col, truck=truck)
        ids = []
        if ped:
            pid = len(sc.objs)
            _ped(sc, *ped, col=(200, 50, 50))
            ids = list(range(pid, len(sc.objs)))
        return sc, ids
    if ans == "none":
        sc, _ = build()
        img, vis = sc.render(ss=2), None
    else:
        side = -1 if ans == "left" else 1
        _, dist, truck, _ = rng.choice([p for p in parked if p[0] == side])
        z = -dist + rng.uniform(-1.5, 1.5) - 2.0
        # start as hidden as the difficulty asks, then step out until the person is clearly visible
        for hide in np.linspace(lerp(0.1, 0.6, d), -0.8, 8):   # negative = stepping out past the vehicle
            px = side * (LANE + 1.6) - side * (1.1 - 1.4 * hide)
            sc, ids = build((px, z))
            sc.render(ss=1)                                       # cheap visibility probe
            vis = sum(sc.visible_px(i) for i in ids)
            if vis >= 150 and sc.visible_px(ids[-1]) >= 25:     # head + some body visible
                break
        else:
            raise Reject
        img = sc.render(ss=2)
        vis = sum(sc.visible_px(i) for i in ids)
    return dict(image=img, atype="choice", choices=["left", "right", "none"], answer=ans,
                question="Is a pedestrian partly hidden behind a parked vehicle, about to step into the road? "
                         "Answer 'left', 'right' (which side of the road), or 'none'.",
                spec=dict(parked=[p[:3] for p in parked], visible_px=vis))


# ---------------------------------------------------------------- 21 right of way
@task("right_of_way", "Driving rules at junctions (ADAS, driver training)", "rule application to a scene",
      knob="number of cars, stop-sign mix", twin="text: approaches + signs", group=G)
def right_of_way(rng, d):
    appro = ["N", "E", "S", "W"]
    right_of = {"S": "E", "E": "N", "N": "W", "W": "S"}  # car coming from S (heading north) has E on its right
    n = ilerp(2, 4, d)
    cars = rng.sample(appro, n)
    stops = {a: (rng.random() < lerp(0.2, 0.5, d)) for a in appro}
    cand = [a for a in cars if not stops[a]] or cars
    first = [a for a in cand if right_of[a] not in cand]
    if len(first) != 1:
        raise Reject
    W = 520
    cv = Canvas(W, W, bg=(110, 160, 90))
    c, rw = W / 2, 90
    cv.rect(c - rw, 0, c + rw, W, fill=(90, 90, 95))
    cv.rect(0, c - rw, W, c + rw, fill=(90, 90, 95))
    for k in range(0, W, 30):
        if abs(k - c) > rw:
            cv.line([(c, k), (c, k + 15)], (230, 230, 230), 2)
            cv.line([(k, c), (k + 15, c)], (230, 230, 230), 2)
    pos = {"S": (c + rw / 2, c + rw + 50, 0), "N": (c - rw / 2, c - rw - 50, 180),
           "E": (c + rw + 50, c - rw / 2, 270), "W": (c - rw - 50, c + rw / 2, 90)}
    sign = {"S": (c + rw + 18, c + rw + 18), "N": (c - rw - 18, c - rw - 18),
            "E": (c + rw + 18, c - rw - 18), "W": (c - rw - 18, c + rw + 18)}
    labels = {}
    for i, a in enumerate(cars):
        x, y, rot = pos[a]
        ang = math.radians(rot)
        pts = [(-14, -26), (14, -26), (14, 20), (0, 30), (-14, 20)]
        pts = [(x + px * math.cos(ang) - py * math.sin(ang), y + px * math.sin(ang) + py * math.cos(ang)) for px, py in pts]
        pts = [(2 * x - px, 2 * y - py) for px, py in pts]   # point the arrow toward the junction
        cv.poly(pts, fill=CARCOL[i % 6] if i % 6 not in (2, 3) else (60, 120, 220), outline=(20, 20, 20))
        labels[a] = LETTERS[i]
    for a in appro:
        if stops[a]:
            x, y = sign[a]
            cv.poly([(x + 13 * math.cos(math.pi / 8 + k * math.pi / 4), y + 13 * math.sin(math.pi / 8 + k * math.pi / 4))
                     for k in range(8)], fill=(200, 20, 20), outline=(255, 255, 255))
    img = cv.done()
    for a in cars:
        x, y, _ = pos[a]
        tag(img, (x, y), labels[a], 15)
    return dict(image=img, atype="choice", choices=[labels[a] for a in cars], answer=labels[first[0]],
                question="All cars arrive at the junction at the same moment and want to go straight. Rules: a car "
                         "facing a red STOP sign must give way to every car without one; among cars with the same "
                         "sign status, a car must give way to a car approaching from its right. Which car goes first?",
                spec=dict(cars=cars, stops=stops))


# ---------------------------------------------------------------- 23 stop in time
@task("stop_in_time", "Driving safety: stopping distance at the current speed", "reading + physical reasoning",
      knob="margin between stopping distance and gap; analog speedometer", twin="text: speed and gap", group=G)
def stop_in_time(rng, d):
    v_kmh = rng.choice(range(30, 131, 5))
    v = v_kmh / 3.6
    a, tr = 7.0, 1.0
    stop = v * tr + v * v / (2 * a)
    margin = lerp(0.35, 0.08, d) * stop
    gap = stop + margin * rng.choice([-1, 1]) * rng.uniform(1, 1.3)
    if gap < 8:
        raise Reject
    W, H = 900, 300
    ppm = (W - 160) / (gap + 10)
    cv = Canvas(W, H, bg=(205, 225, 245))
    cv.rect(0, 200, W, H, fill=(95, 95, 98))
    x0 = 60
    cv.rect(x0, 160, x0 + 4.2 * ppm if 4.2 * ppm > 40 else x0 + 40, 195, fill=(50, 100, 210))
    xo = x0 + gap * ppm
    cv.rect(xo, 150, xo + 25, 200, fill=(230, 110, 30))
    step = 10 if gap < 120 else 20
    for m in range(0, int(gap) + 11, step):
        x = x0 + m * ppm
        cv.line([(x, 205), (x, 220)], (240, 240, 240), 2)
        cv.text((x, 232), f"{m} m", 12, fill=(240, 240, 240))
    if d < 0.5:
        cv.text((150, 60), f"Speed: {v_kmh} km/h", 22)
    else:
        cx, cy, R = 140, 95, 70
        cv.ellipse(cx, cy, R, fill=(250, 250, 250), outline=(30, 30, 30), width=3)
        for s in range(0, 161, 10):
            th = math.radians(225 - 270 * s / 160)
            r0 = R - (12 if s % 20 == 0 else 6)
            cv.line([(cx + r0 * math.cos(th), cy - r0 * math.sin(th)), (cx + R * math.cos(th), cy - R * math.sin(th))],
                    (30, 30, 30), 2 if s % 20 == 0 else 1)
            if s % 40 == 0:
                cv.text((cx + (R - 26) * math.cos(th), cy - (R - 26) * math.sin(th)), s, 11)
        th = math.radians(225 - 270 * v_kmh / 160)
        cv.line([(cx, cy), (cx + (R - 10) * math.cos(th), cy - (R - 10) * math.sin(th))], (200, 30, 30), 3)
        cv.text((cx, cy + 35), "km/h", 11)
    return dict(image=cv.done(), atype="choice", choices=["yes", "no"], answer="yes" if gap > stop else "no",
                question="The blue car drives at the speed shown towards the orange obstacle (distance markers are "
                         "measured from the car's front). Assume a 1.0 s reaction time and then constant braking at "
                         "7 m/s^2. Can the car stop before reaching the obstacle?",
                spec=dict(speed_kmh=v_kmh, gap_m=round(gap, 1), stopping_m=round(stop, 1)))


# ---------------------------------------------------------------- 24 lane change gap
@task("lane_change", "ADAS lane-change assist: is the gap safe?", "motion estimation from two frames",
      knob="closing speed and gap margins", twin="text: positions at t=0 and t=1 s", group=G)
def lane_change(rng, d):
    v_ego = rng.uniform(20, 30)
    others = []
    for _ in range(ilerp(1, 3, d)):
        others.append((rng.uniform(-60, 50), v_ego + rng.uniform(-6, 10)))
    mind = min(min(abs((x + (v - v_ego) * t)) for t in np.linspace(0, 3, 31)) for x, v in others)
    thr = 10.0
    mg = lerp(5, 1.5, d)
    if abs(mind - thr) < mg:
        raise Reject
    ans = "yes" if mind >= thr else "no"
    if (rng.random() < .5) != (ans == "yes"):
        raise Reject
    W, H, ppm = 880, 170, 6.0
    img = Image.new("RGB", (W, 2 * H + 30), (255, 255, 255))
    for f, t in enumerate((0.0, 1.0)):
        cv = Canvas(W, H, bg=(110, 150, 90))
        cv.rect(0, 30, W, 140, fill=(95, 95, 98))
        for k in range(0, W, 30):
            cv.line([(k, 85), (k + 15, 85)], (235, 235, 235), 2)
        base = 60 + 200 + v_ego * t * 0  # ego-fixed frame: show road markers moving instead
        for m in range(-80, 120, 10):
            x = W / 2 + (m - v_ego * t) * ppm
            if 0 < x < W:
                cv.line([(x, 136), (x, 142)], (240, 240, 240), 2)
                cv.text((x, 152), f"{m}", 10, fill=(240, 240, 240))
        ex = W / 2
        cv.rect(ex - 13, 98, ex + 13, 124, fill=(50, 100, 210))
        cv.rect(ex - 2.25 * ppm, 100, ex + 2.25 * ppm, 122, fill=(50, 100, 210))
        for x0, v in others:
            x = W / 2 + (x0 + (v - v_ego) * t) * ppm
            cv.rect(x - 2.25 * ppm, 46, x + 2.25 * ppm, 68, fill=(200, 60, 50))
        cv.text((60, 15), f"t = {t:.1f} s", 14, fill=(255, 255, 255))
        img.paste(cv.done(), (0, f * (H + 30)))
    return dict(image=img, atype="choice", choices=["yes", "no"], answer=ans,
                question="Two top-view frames taken 1.0 s apart; the blue car (bottom lane) wants to move into the "
                         "top lane. Road markers show metres along the road. The lane change is safe only if no red "
                         "car will be within 10 m (centre-to-centre distance along the road) of the blue car at any time in the next 3 s, "
                         "assuming constant speeds. Is it safe now?",
                spec=dict(v_ego=v_ego, others=others, min_dist=mind))


# ---------------------------------------------------------------- 25 wrong on the road
@task("wrong_way", "Traffic monitoring: spotting rule violations (wrong-way drivers)", "direction consistency",
      knob="number of cars and lanes, small arrows", twin="text: lane directions + car headings", group=G)
def wrong_way(rng, d):
    nl = ilerp(2, 4, d)
    W, lh = 900, 70
    H = nl * lh + 60
    dirs = [1 if k < nl / 2 else -1 for k in range(nl)]
    if nl == 3:
        dirs = [1, -1, -1]
    cv = Canvas(W, H, bg=(110, 150, 90))
    cv.rect(0, 30, W, 30 + nl * lh, fill=(95, 95, 98))
    for k in range(1, nl):
        y = 30 + k * lh
        solid = dirs[k] != dirs[k - 1]
        for x in range(0, W, 30):
            cv.line([(x, y), (x + (30 if solid else 15), y)], (235, 200, 40) if solid else (235, 235, 235), 2)
    asz = lerp(14, 7, d)
    for k in range(nl):
        y = 30 + (k + .5) * lh
        for x in range(60, W, 260):
            s = dirs[k]
            cv.line([(x - asz * 2 * s, y), (x + asz * s, y)], (235, 235, 235), 3)
            cv.poly([(x + asz * 2 * s, y), (x + asz * s, y - asz * .7), (x + asz * s, y + asz * .7)], fill=(235, 235, 235))
    n = ilerp(3, 9, d)
    wrong = rng.randrange(n) if rng.random() > .2 else None
    cars, labels = [], []
    for i in range(n):
        for _t in range(200):
            k = rng.randrange(nl)
            x = rng.uniform(140, W - 60)
            if all(not (kk == k and abs(x - xx) < 110) for kk, xx, _ in cars):
                break
        else:
            raise Reject
        hd = dirs[k] * (-1 if i == wrong else 1)
        cars.append((k, x, hd))
        y = 30 + (k + .5) * lh
        body = [(x - 34 * hd, y - 14), (x + 22 * hd, y - 14), (x + 36 * hd, y), (x + 22 * hd, y + 14), (x - 34 * hd, y + 14)]
        cv.poly(body, fill=CARCOL[i % 6] if i % 6 != 3 else (60, 120, 220), outline=(20, 20, 20))
        labels.append(((x - 6 * hd, y), LETTERS[i]))
    img = cv.done()
    for p, l in labels:
        tag(img, p, l, 13)
    return dict(image=img, atype="choice", choices=list(LETTERS[:n]) + ["none"],
                answer=LETTERS[wrong] if wrong is not None else "none",
                question="Top view of a road; painted arrows show each lane's direction of travel and the pointed end "
                         "of each car is its front. Which car is driving the wrong way? Answer the letter, or 'none'.",
                spec=dict(lane_dirs=dirs, cars=cars))
