"""Cable matching (TRACE-style task, WireSeg-style physics), rendered photorealistically in Blender.

Scene: two USB hubs with labelled ports on a desk; n cables, each plugged into a left port and a right port,
dropped onto the desk with a rope simulation (vf/ropesim.py) so they settle, cross and rest on each other
physically. Question: "the cable in left port X ends in which right port?"  Answer: from the simulation.

Independent difficulty knobs (all stored with the sample):
  n_cables     number of cables (2-8)
  wiggle       how far cables wander (more crossings, loops)
  mono         all cables the same colour (TRACE's monochrome setting) vs colour-coded
  stiffness    bending stiffness (soft cables curl more)
  extra_ports  empty decoy ports on each hub
  tilt         camera angle from overhead (0) to oblique (1)
Measured per sample: crossings on the target cable, the shallowest crossing angle, how much of the target is
covered by other cables.
"""
import hashlib
import json
import math
import os
import subprocess

import numpy as np
from PIL import Image

from ..draw import catmull
from ..ropesim import crossings_2d, simulate
from ..taskapi import LETTERS, Reject, ilerp, lerp, task

R = 0.003                       # cable radius (m)
HUB_X = 0.44
PORT_DY = 0.032
PORT_Z = 0.008
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "vf_cache", "cables")
SCRIPT = os.path.join(os.path.dirname(__file__), "..", "blender", "cables_scene.py")
PALETTE = [(.75, .08, .06), (.05, .18, .62), (.85, .62, .05), (.08, .45, .15), (.06, .06, .07), (.85, .85, .83),
           (.45, .12, .55), (.9, .35, .05)]


def _resample(pts, n):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], 1)


def _render(spec):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    out = os.path.abspath(os.path.join(CACHE, key))
    spec = dict(spec, out=out)
    if not os.path.exists(out + "_rgb.npy"):
        with open(out + ".json", "w") as f:
            json.dump(spec, f)
        r = subprocess.run(["blenderproc", "run", os.path.abspath(SCRIPT), out + ".json"],
                           capture_output=True, text=True)
        if not os.path.exists(out + "_rgb.npy"):
            raise RuntimeError("blender render failed: " + (r.stdout + r.stderr)[-1500:])
    rgb = np.load(out + "_rgb.npy")
    seg = np.load(out + "_seg.npy")
    Image.fromarray(seg).save(out + "_seg.png")
    with open(out + "_proj.json") as f:
        proj = json.load(f)
    return Image.fromarray(rgb), out + "_seg.png", proj


@task("cable_match", "Data-center / lab cabling audits and robot cable handling: which port is the other end?",
      "tracing deformable cables through crossings", knob="n_cables, wiggle, mono, stiffness, extra_ports, tilt",
      twin="puzzle: wire_trace; text: centrelines + port positions", group="real3d")
def cable_match(rng, d, n_cables=None, wiggle=None, mono=None, stiffness=None, extra_ports=None, tilt=None):
    n = n_cables if n_cables is not None else ilerp(2, 5, d)      # TRACE: models already fail at 2-4
    wg = wiggle if wiggle is not None else lerp(0.05, 0.16, d)
    mono = mono if mono is not None else (d >= 0.4)
    stiff = stiffness if stiffness is not None else lerp(0.7, 0.45, d)
    extra = extra_ports if extra_ports is not None else ilerp(0, 3, d)
    tilt = tilt if tilt is not None else lerp(0.0, 0.25, d)
    K = n + extra
    ys = (np.arange(K) - (K - 1) / 2) * PORT_DY
    lp = sorted(rng.sample(range(K), n))                     # used left ports
    rp = rng.sample(range(K), n)                             # their right ports (a random matching)
    exit_x = HUB_X - 0.025 - 0.028                           # where the cable leaves the plug
    paths = []
    for c in range(n):
        a = np.array((-exit_x, ys[lp[c]]))
        b = np.array((exit_x, ys[rp[c]]))
        way = [tuple(a), (-exit_x + .05, ys[lp[c]])]          # leave each plug straight before bending
        nw = 3 + int(wg * 10)
        for k in range(nw):
            t = (k + 1) / (nw + 1)
            way.append((-exit_x + .12 + (2 * exit_x - .24) * t + rng.uniform(-.03, .03),
                        (1 - t) * a[1] + t * b[1] + rng.uniform(-wg, wg) * 1.2))
        way += [(exit_x - .05, ys[rp[c]]), tuple(b)]
        paths.append(np.array(catmull(way, 50)))
    L = max(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) for p in paths)
    N = int(L / (1.2 * R)) + 1
    order = rng.sample(range(n), n)                          # drop order decides who lands on top
    init = []
    for c, pts in enumerate(paths):
        xy = _resample(pts, N)
        if np.abs(xy[:, 1]).max() > 0.36:
            raise Reject                                     # stay on the desk area in view
        ramp = np.clip(np.minimum(np.arange(N), np.arange(N)[::-1]) / 60, 0, 1)   # gentle drop near plugs
        z = PORT_Z + (0.012 + 0.012 * order.index(c)) * ramp
        init.append(np.column_stack([xy, z]))
    P = simulate(np.array(init), R, stiffness=stiff, iters=8, pinned=8)
    # ---- validity (TRACE's hard cases must stay resolvable for a careful person). Every cable is checked and
    # the target is drawn from the clean ones, so one awkward cable does not waste a whole simulation.
    from scipy.spatial import cKDTree

    def check(t):
        cr = crossings_2d(P, t)
        ang = min((x for _, x, _ in cr), default=90.0)
        if ang < 40:
            return None                                      # too-shallow crossing: tangential ambiguity
        a2 = P[t][:, :2]
        if n == 1:
            return cr, ang, 0.0
        others = [P[c][:, :2] for c in range(n) if c != t]
        dmin, _ = cKDTree(np.concatenate(others)).query(a2)
        run = best = 0
        for v in dmin < 2.6 * R:
            run = run + 1 if v else 0
            best = max(best, run)
        if best * 1.2 * R > 0.03:
            return None                                      # runs side by side with another cable > 3 cm
        near = np.stack([cKDTree(o).query(a2)[0] < 0.009 for o in others]).sum(0)
        if (near >= 2).any():
            return None                                      # 3+ cables meet at one spot on the target
        return cr, ang, float(np.mean(dmin < 2 * R))
    # no knots anywhere in the picture: nowhere may 3+ cables come within ~1 cm of one spot
    if n >= 3:
        trees = [cKDTree(P[c][:, :2]) for c in range(n)]
        for c in range(n):
            near = np.stack([trees[o].query(P[c][:, :2])[0] < 0.011 for o in range(n) if o != c]).sum(0)
            if (near >= 2).any():
                raise Reject
    valid = {t: v for t in range(n) if (v := check(t)) is not None}
    if not valid:
        raise Reject
    tgt = rng.choice(sorted(valid))
    if d >= 0.4:                                             # harder settings ask about the most-crossed clean cable
        tgt = max(sorted(valid), key=lambda t: (len(valid[t][0]), rng.random()))
        if len(valid[tgt][0]) < ilerp(1, 3, d):
            raise Reject                                     # scene too easy for this difficulty
    cr, min_angle, covered = valid[tgt]
    # ---- scene spec for Blender
    hub_size = (0.05, K * PORT_DY + 0.03, 0.016)
    hubs = []
    for side, labels in ((-1, [LETTERS[i] for i in range(K)]), (1, [str(i + 1) for i in range(K)])):
        hx = side * HUB_X
        ports = [dict(pos=(hx - side * 0.025, ys[i], PORT_Z), label=labels[i],
                      label_pos=(hx + side * 0.008, ys[i], 0.0165)) for i in range(K)]
        hubs.append(dict(center=(hx, 0, 0.008), size=hub_size, ports=ports))
    colors = [(.86, .86, .84)] * n if mono else [PALETTE[(i * 3) % len(PALETTE)] for i in range(n)]
    if not mono and n > 4:
        colors = [PALETTE[rng.randrange(5)] for _ in range(n)]       # colour-coded but with repeats, as in practice
    cables = []
    for c in range(n):
        conns = []
        for side, yv in ((-1, ys[lp[c]]), (1, ys[rp[c]])):
            hx = side * (HUB_X - 0.025)
            conns.append(dict(dir=(side, 0), body=(hx - side * 0.0145, yv, PORT_Z),
                              shell=(hx - side * 0.0005, yv, PORT_Z)))
        cables.append(dict(pts=P[c][::2].round(5).tolist() + [P[c][-1].round(5).tolist()], color=colors[c], id=c + 1,
                           connectors=conns))
    wood = rng.random() < 0.5
    table = dict(c1=(.24, .14, .07), c2=(.17, .09, .045), scale=6, stretch=(1, 12, 1)) if wood else \
        dict(c1=(.10, .11, .12), c2=(.07, .07, .08), scale=40, rough=.8)
    cam_y = -0.95 * math.sin(tilt * math.radians(55))
    cam_z = 0.95 * math.cos(tilt * math.radians(55)) + 0.12
    lights = [dict(type="AREA", loc=(rng.uniform(-.4, .4), rng.uniform(-.4, .4), 1.3), energy=rng.uniform(180, 320),
                   size=1.2, color=(1, rng.uniform(.9, 1), rng.uniform(.8, .95))),
              dict(type="POINT", loc=(rng.uniform(-1, 1), rng.uniform(-1, 1), .9), energy=rng.uniform(20, 60))]
    project = [list(p["pos"]) for h in hubs for p in h["ports"]]
    spec = dict(res=(1600, 1080), table=table, hubs=[dict(h, label_size=0.016) for h in hubs], cables=cables, radius=R,
                lights=lights, camera=dict(loc=(0, cam_y, cam_z), look=(0, 0, 0), fov=52), samples=64,
                project=project)
    img, seg_path, proj = _render(spec)
    q = (f"Two USB hubs are connected by cables. Follow the cable plugged into port {LETTERS[lp[tgt]]} of the left "
         f"hub. Into which numbered port of the right hub is its other end plugged?")
    return dict(image=img, atype="int", answer=rp[tgt] + 1, choices=None, question=q,
                spec=dict(n_cables=n, K=K, left_ports=lp, right_ports=rp, target=tgt, mono=bool(mono),
                          wiggle=round(wg, 3), stiffness=round(stiff, 3), extra_ports=extra, tilt=round(tilt, 3),
                          crossings_on_target=len(cr), min_cross_angle=round(min_angle, 1),
                          target_covered_frac=round(covered, 3), seg_path=seg_path, ports_2d=proj))
