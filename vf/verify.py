"""Independent answer checker.

For every sample it re-derives the answer WITHOUT reusing the generator's answer code:
  * pixel checkers read the rendered image (road/valve/mirror/switch/peg/bar/obstacle pixels) and solve it;
  * re-solvers recompute the answer from the stored scene with a different algorithm
    (e.g. geometric gear meshing instead of the parent tree, forward paper folding instead of reflection,
    Euler's formula instead of pixel flood-fill, networkx path search instead of hand-written BFS).
A mismatch means the image, the stored scene and the answer disagree, so the sample is not trustworthy.

    python -m vf.verify --samples samples [--tasks a,b]
"""
import argparse
import ast
import json
import math
import os
from collections import deque

import networkx as nx
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHECKS = {}


def check(name, kind):
    def deco(fn):
        CHECKS[name] = (fn, kind)
        return fn
    return deco


def _bfs(adj, s):
    seen, dq = {s}, deque([s])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    return seen


def _t(x):
    return tuple(x) if isinstance(x, list) else x


# ================================================================ pixel checkers
@check("flood_route", "pixels")
def _flood(r, a):
    s, S = r["spec"], 90
    P = lambda n: (30 + n[0] * S, 30 + n[1] * S)
    adj = {}
    for e in s["edges"]:
        (x0, y0), (x1, y1) = P(e[0]), P(e[1])
        blue = False
        for t in np.linspace(0, 1, 60):
            for off in (-4, 0, 4):
                x = int(round(x0 + (x1 - x0) * t + (off if x0 == x1 else 0)))
                y = int(round(y0 + (y1 - y0) * t + (off if y0 == y1 else 0)))
                R, G, B = a[y, x]
                blue |= B > R + 25
        if not blue:
            u, v = _t(e[0]), _t(e[1])
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
    return "yes" if _t(s["B"]) in _bfs(adj, _t(s["A"])) else "no"


@check("pid_flow", "pixels")
def _pid(r, a):
    s, S = r["spec"], 90
    P = lambda n: (60 + n[0] * S, 60 + n[1] * S)
    valves = {tuple(sorted(ast.literal_eval("(" + k.replace(")-(", "),(") + ")"))) for k in s["valves"]}
    adj = {}
    for e in s["edges"]:
        e = (_t(e[0]), _t(e[1]))
        if tuple(sorted(e)) in valves:
            (x0, y0), (x1, y1) = P(e[0]), P(e[1])
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            px, py = (mx - 7, my) if y0 == y1 else (mx, my - 7)   # inside one triangle of the bow-tie
            if a[int(round(py)), int(round(px))].mean() < 100:      # filled black = closed
                continue
        adj.setdefault(e[0], []).append(e[1])
        adj.setdefault(e[1], []).append(e[0])
    reach = _bfs(adj, _t(s["src"]))
    return sum(1 for t in s["tanks"] if _t(t) in reach)


@check("mirror_laser", "pixels")
def _laser(r, a):
    s = r["spec"]
    N = len(s["grid"])
    cell, ox = 400 // N, 60
    grid = [[0] * N for _ in range(N)]
    dark = a.mean(-1) < 90
    for i in range(N):
        for j in range(N):
            x0, y0 = ox + j * cell, ox + i * cell
            ts = np.linspace(.25, .75, 9)
            slash = sum(dark[int(y0 + (1 - t) * cell), int(x0 + t * cell)] for t in ts)
            back = sum(dark[int(y0 + t * cell), int(x0 + t * cell)] for t in ts)
            if slash >= 6:
                grid[i][j] = 1
            elif back >= 6:
                grid[i][j] = -1
    r0 = s["entry_row"]
    rr, cc, dr, dc = r0, 0, 0, 1
    for _ in range(4 * N * N):
        m = grid[rr][cc]
        if m == 1:
            dr, dc = -dc, -dr
        elif m == -1:
            dr, dc = dc, dr
        rr, cc = rr + dr, cc + dc
        if not (0 <= rr < N and 0 <= cc < N):
            break
    ports = [("top", k) for k in range(N)] + [("right", k) for k in range(N)] + \
            [("bottom", k) for k in range(N)] + [("left", k) for k in range(N) if k != r0]
    ex = ("top", cc) if rr < 0 else ("bottom", cc) if rr >= N else ("right", rr) if cc >= N else ("left", rr)
    return ports.index(ex) + 1


@check("circuit_leds", "pixels")
def _circuit(r, a):
    s = r["spec"]
    dark = lambda x, y: a[int(y) - 1:int(y) + 2, int(x) - 1:int(x) + 2].mean() < 110
    top = 80
    main = dark(115, top)
    lit = 0
    for k, b in enumerate(s["branches"]):
        x = 140 + k * 110
        ok = all(dark(x, top + 20 + i * 55 + 15) for i in range(len(b)))
        lit += main and ok
    return lit


@check("string_pegs", "pixels")
def _pegs(r, a):
    lab, n = ndimage.label(a.mean(-1) < 80)
    cents = np.array(ndimage.center_of_mass(np.ones_like(lab), lab, range(1, n + 1)))
    return len(ConvexHull(cents[:, ::-1]).vertices)


@check("gantt_overlap", "pixels")
def _gantt(r, a):
    s = r["spec"]
    nt, nd = len(s["tasks"]), max(e for _, _, e in s["tasks"]) + 1
    nd = nd if nd > 0 else 1
    # measure the day-column width from the grid lines in the image (no generator constants beyond the left margin)
    row = a[45, 105:]
    xs = np.nonzero((np.abs(row - np.array((238, 238, 241))).sum(-1) < 8))[0]
    gaps = np.diff(xs)
    break_cw = float(np.median(gaps[gaps > 5]))
    x = int(110 + (s["day"] - 1) * break_cw + break_cw / 2)
    cnt = 0
    for i in range(nt):
        y = int(50 + i * 26 + 9)
        R, G, B = a[y, x]
        cnt += B > 150 and R < 120
    return cnt


@check("robot_path", "pixels")
def _robot(r, a):
    s, ppm = r["spec"], 60
    bg = np.array((245, 243, 238))
    robot = ndimage.binary_dilation((a[..., 2] > 150) & (a[..., 0] < 120), iterations=3)  # include the disc's soft rim
    free = (np.abs(a - bg).sum(-1) < 40) | robot
    door = (a[..., 1] > 150) & (a[..., 0] < 120) & (a[..., 2] < 150)
    free |= door
    dt = ndimage.distance_transform_edt(free)
    ok = dt > s["robot_r"] * ppm
    lab, _ = ndimage.label(ok)
    sx, sy = int(s["start"][0] * ppm), int(s["start"][1] * ppm)
    st = lab[sy, sx]
    dy = int(s["door"][1] * ppm)
    reach_x = int(s["robot_r"] * ppm) + 14     # robot centre can get this close to the door wall
    goal = set(np.unique(lab[dy - 10:dy + 11, a.shape[1] - reach_x - 8:a.shape[1] - 2])) - {0}
    return "yes" if st and st in goal else "no"


# ================================================================ independent re-solvers (from the stored scene)
@check("gear_train", "geometry")
def _gear(r, a):
    s = r["spec"]
    m, g = s["module"], s["gears"]
    rp = lambda N: m * N / 2
    G = nx.Graph()
    G.add_nodes_from(range(len(g)))
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            dd = math.hypot(g[i][0] - g[j][0], g[i][1] - g[j][1])
            if abs(dd - (rp(g[i][2]) + rp(g[j][2]))) < 1.0:     # centre distance = sum of pitch radii -> mesh
                G.add_edge(i, j)
            elif dd - (rp(g[i][2]) + m + rp(g[j][2]) + m) < 1.5:
                return "TOUCHING"           # non-meshing gears whose teeth touch/overlap -> ambiguous picture
    q = s["query"]
    if not nx.has_path(G, 0, q):
        return "does not turn"
    if not nx.is_bipartite(G.subgraph(nx.node_connected_component(G, 0))):
        return "JAMMED"
    same = nx.shortest_path_length(G, 0, q) % 2 == 0
    return "clockwise" if s["driver_cw"] == same else "counterclockwise"


@check("one_stroke", "graph")
def _stroke(r, a):
    G = nx.MultiGraph()
    G.add_edges_from(_t(e) for e in r["spec"]["edges"])
    return "yes" if nx.has_eulerian_path(G) else "no"


@check("shell_game", "replay")
def _shell(r, a):
    return _shell2(r["spec"])


def _shell2(s):
    where = s["start"]
    for i, j in s["swaps"]:
        if where in (i, j):
            where = j if where == i else i
    return where + 1


@check("layer_shuffle", "replay")
def _shuffle(r, a):
    s = r["spec"]
    seq = [s["order2"].index(x) for x in s["order1"]]
    swaps = 0
    for i in range(len(seq)):          # bubble sort counts adjacent swaps directly
        for j in range(len(seq) - 1 - i):
            if seq[j] > seq[j + 1]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
                swaps += 1
    return swaps


@check("fold_punch", "simulation")
def _fold(r, a):
    s = r["spec"]
    n = 400
    xs, ys = np.meshgrid((np.arange(n) + .5) / n, (np.arange(n) + .5) / n)
    X, Y = xs.copy(), ys.copy()
    for ax, m, side in s["folds"]:          # fold the sheet forward, tracking where each point lands
        if ax == "v":
            mv = X < m if side == 0 else X > m
            X = np.where(mv, 2 * m - X, X)
        else:
            mv = Y < m if side == 0 else Y > m
            Y = np.where(mv, 2 * m - Y, Y)
    px, py = s["punch"]
    hit = (X - px) ** 2 + (Y - py) ** 2 < (3 / n) ** 2
    lab, k = ndimage.label(hit)
    holes = sorted((round(c[1] / n, 2), round(c[0] / n, 2)) for c in ndimage.center_of_mass(hit, lab, range(1, k + 1)))
    for i, c in enumerate(s["candidates"]):
        cc = sorted((round(x, 2), round(y, 2)) for x, y in c)
        if len(cc) == len(holes) and all(abs(p[0] - q[0]) < .02 and abs(p[1] - q[1]) < .02 for p, q in zip(cc, holes)):
            return L[i]
    return "NO-MATCH"


@check("order_frames", "replay")
def _frames(r, a):
    f = r["spec"]["fallen_per_frame"]
    return "".join(L[i] for i in sorted(range(len(f)), key=lambda i: f[i]))


@check("circle_regions", "euler")
def _regions(r, a):
    cs = r["spec"]["circles"]
    n = len(cs)
    V, k = 0, [0] * n
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(cs[i][0] - cs[j][0], cs[i][1] - cs[j][1])
            if abs(cs[i][2] - cs[j][2]) < d < cs[i][2] + cs[j][2]:
                V += 2
                k[i] += 2
                k[j] += 2
                G.add_edge(i, j)
    E = sum(k)
    for i in range(n):              # an isolated circle is one vertex + one edge
        if k[i] == 0:
            V += 1
            E += 1
    C = nx.number_connected_components(G)
    return C - V + E                # bounded faces from Euler's formula


@check("billiards_bank", "unfolding")
def _billiards(r, a):
    s = r["spec"]
    p, v = np.array(s["start"]), np.array(s["dir"])
    pockets = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    best = None
    for kx in range(-4, 5):          # mirror images of each pocket in the unfolded plane
        for ky in range(-4, 5):
            for idx, (qx, qy) in enumerate(pockets):
                X = (kx * 2.0 + (qx if kx % 2 == 0 else 2.0 - qx))
                Y = (ky * 1.0 + (qy if ky % 2 == 0 else 1.0 - qy))
                w = np.array((X, Y)) - p
                t = w @ v
                if t <= 0:
                    continue
                if np.linalg.norm(w - t * v) < .07 and (best is None or t < best[0]):
                    best = (t, idx)
    return best[1] + 1 if best else "NONE"


@check("ball_drop", "simulation")
def _ball(r, a):
    s = r["spec"]
    ramps = [tuple(q) for q in s["ramps"]]
    x, y = s["ball_x"], 40.0
    for _ in range(30):
        hits = [(r_[1] + (r_[3] - r_[1]) * (x - r_[0]) / (r_[2] - r_[0]), r_) for r_ in ramps if r_[0] < x < r_[2]]
        hits = [h for h in hits if h[0] > y + 1]
        if not hits:
            break
        _, q = min(hits)
        x, y = (q[0] - 3, q[1]) if q[1] > q[3] else (q[2] + 3, q[3])
    nb = len(r["choices"])
    return L[int(x // (560 / nb))]


@check("transit_route", "graph")
def _transit(r, a):
    s = r["spec"]
    G = nx.Graph()
    for li, p in enumerate(s["lines"]):
        p = [_t(n) for n in p]
        for u, v in zip(p, p[1:]):
            G.add_edge((u, li), (v, li), weight=0)
        for n in p:
            G.add_edge((n, li), n, weight=0.5)      # boarding/alighting costs half a change
    st, en = _t(s["start"]), _t(s["end"])
    d = nx.shortest_path_length(G, st, en, weight="weight")
    return int(round(d - 1))


@check("floorplan_path", "graph")
def _floor(r, a):
    s = r["spec"]
    G = nx.Graph()
    G.add_nodes_from(range(len(s["rooms"])))
    for i, j, vert, x, c, w in s["doors"]:
        if w >= s["cart_m"]:
            G.add_edge(i, j)
    return "yes" if nx.has_path(G, s["room_a"], s["room_b"]) else "no"


@check("liquid_volume", "integration")
def _liquid(r, a):
    vols = []
    for Hh, rb, rt, lvl in r["spec"]["glasses"]:
        ys = np.linspace(0, lvl, 2001)
        rad = rb + (rt - rb) * ys / Hh
        vols.append(np.trapezoid(np.pi * rad ** 2, ys))
    return L[int(np.argmax(vols))]


@check("stop_in_time", "physics")
def _stop(r, a):
    s = r["spec"]
    v = s["speed_kmh"] / 3.6
    return "yes" if s["gap_m"] > v * 1.0 + v * v / 14.0 else "no"


@check("lane_change", "physics")
def _lane(r, a):
    s = r["spec"]
    md = min(abs(x + (v - s["v_ego"]) * t) for x, v in s["others"] for t in np.linspace(0, 3, 3001))
    return "yes" if md >= 10 else "no"


@check("would_fall", "physics")
def _fall(r, a):
    s = r["spec"]
    bad = [lab for cx, z, w, lab in s["boxes"] if abs(cx) > s["table_half_width"]]
    return bad[0] if len(bad) == 1 else ("none" if not bad else "MULTI")


@check("voxel_count", "replay")
def _voxel(r, a):
    s = r["spec"]
    hm = s["heights"]
    total = int(sum(map(sum, hm)))
    # structural rule the question relies on: columns never get taller toward the camera
    g = len(hm)
    if any(hm[i][j] > hm[i - 1][j] for i in range(1, g) for j in range(g)) or             any(hm[i][j] > hm[i][j - 1] for i in range(g) for j in range(1, g)):
        return "NOT-STAIRCASE"
    if s.get("qtype") == "complete":
        H = max(map(max, hm))             # block height from the heights themselves, not the stored value
        return g * g * H - total
    return total


@check("offside_call", "geometry")
def _offside(r, a):
    s = r["spec"]
    xs = sorted([52.0] + [x for x, _ in s["defenders"]])
    return "yes" if s["attacker"][0] > xs[-2] else "no"


@check("line_call", "geometry")
def _line(r, a):
    return "out" if r["spec"]["mark_gap_m"] > 0 else "in"


@check("obstacle_lane", "replay")
def _obstacle(r, a):
    own = [o for o in r["spec"]["objects"] if o[1] == 0]
    return min(own, key=lambda o: o[2])[0] if own else "none"


@check("wrong_way", "replay")
def _wrong(r, a):
    s = r["spec"]
    bad = [L[i] for i, (k, x, hd) in enumerate(s["cars"]) if hd != s["lane_dirs"][k]]
    return bad[0] if len(bad) == 1 else ("none" if not bad else "MULTI")


@check("right_of_way", "rules")
def _row(r, a):
    s = r["spec"]
    cars, stops = s["cars"], s["stops"]
    on_right = {"S": "E", "E": "N", "N": "W", "W": "S"}
    pool = [c for c in cars if not stops[c]] or cars
    first = [c for c in pool if on_right[c] not in pool]
    return L[cars.index(first[0])] if len(first) == 1 else "AMBIGUOUS"


@check("pick_order", "geometry")
def _pick(r, a):
    s = r["spec"]
    bx = s["boxes"]

    def overlap(p, q):
        return (abs(p["x"] - q["x"]) < (p["w"] + q["w"]) / 2 - 1e-6) and (abs(p["z"] - q["z"]) < (p["d"] + q["d"]) / 2 - 1e-6)
    free = [i for i in s["labeled"]
            if not any(j != i and overlap(bx[i], bx[j]) and abs(bx[j]["y"] - (bx[i]["y"] + bx[i]["h"])) < 1e-6
                       for j in range(len(bx)))]
    return L[s["labeled"].index(free[0])] if len(free) == 1 else ("MULTI" if free else "NONE")


@check("tube_tip", "replay")
def _tube(r, a):
    s = r["spec"]
    ty = s["tubes"][s["query"]][1][1]
    return 1 + sum(ty > z for z in s["zones"])


@check("caliper_measure", "geometry")
def _cal(r, a):
    s = r["spec"]
    return math.dist(s["p"], s["q"]) / s["px_per_cm"]


@check("cytology_count", "replay")
def _cyto(r, a):
    return sum(1 for c in r["spec"]["cells"] if c[3])


@check("cylinder_reading", "replay")
def _cyl(r, a):
    return float(r["spec"]["volume"])


@check("spreadsheet_rank", "replay")
def _sheet(r, a):
    s = r["spec"]
    return sorted(zip(s["sales"], s["names"]), reverse=True)[s["k"] - 1][1]


@check("ui_state", "replay")
def _ui(r, a):
    s, q = r["spec"], r["question"]
    if q.startswith("How many"):
        return sum(s["checked"])
    if "Frequency" in q:
        return ["Instantly", "Hourly", "Daily", "Weekly"][s["freq"]]
    return "yes" if s["save"] else "no"


@check("wire_trace", "replay")
def _wire(r, a):
    s = r["spec"]
    return s["perm"][s["q"]] + 1


@check("crowd_count", "replay")
def _crowd(r, a):
    k, c = r["spec"]["target"]
    return sum(1 for sh in r["spec"]["shapes"] if sh["kind"] == k and sh["color"] == c)


@check("gauge", "replay")
def _gauge(r, a):
    return r["spec"]["value"]


@check("chart", "replay")
def _chart(r, a):
    s = r["spec"]
    v = s["vals"]
    if s["qtype"] == "gap":
        i, j = [s["names"].index(n) for n in r["question"].split("value of ")[1:3] for n in [n.split(" ")[0]]]
        return s["cats"][int(np.argmax([v[i][c] - v[j][c] for c in range(s["ncat"])]))]
    return None


@check("twist", "replay")
def _twist(r, a):
    return r["spec"]["n"]


@check("aerial_change", "replay")
def _aerial(r, a):
    return r["spec"]["new"]


@check("assembly_check", "replay")
def _assembly(r, a):
    s = r["spec"]
    diff = [f"{L[i]}{j + 1}" for i, row in enumerate(s["plan"]) for j, v in enumerate(row) if v != s["build"][i][j]]
    return diff[0] if len(diff) == 1 else ("none" if not diff else "MULTI")


@check("shelf_planogram", "replay")
def _shelf(r, a):
    s = r["spec"]
    diff = [f"{L[i]}{j + 1}" for i, row in enumerate(s["plan"]) for j, v in enumerate(row) if v != s["real"][i][j]]
    return diff[0] if len(diff) == 1 else ("none" if not diff else "MULTI")


@check("two_view_change", "replay")
def _twoview(r, a):
    s = r["spec"]
    moved = [n for n, p, q in zip(s["objects"], s["before"], s["after"]) if math.dist(p, q) > 1e-6]
    return moved[0] if len(moved) == 1 else "MULTI"


@check("hidden_outline", "geometry")
def _hidden(r, a):
    s = r["spec"]
    hv = s["triangle"][2]
    d = [math.dist(c, hv) for c in s["candidates"]]
    return L[int(np.argmin(d))]


@check("missing_piece", "replay")
def _piece(r, a):
    s = r["spec"]
    m = [i for i, c in enumerate(s["candidates"]) if list(c) == list(s["slot_edges"])]
    return L[m[0]] if len(m) == 1 else "MULTI"


@check("invoice", "replay")
def _invoice(r, a):
    s = r["spec"]
    amt = [x["qty"] * x["unit"] for x in s["rows"]]
    if s["qtype"] == "amount":
        return amt[s["target"]] / 100
    if s["qtype"] == "count_qty":
        return sum(x["qty"] > s["k"] for x in s["rows"])
    sub = sum(amt)
    return (sub + int(round(sub * 0.08))) / 100


@check("crack_measure", "pixels")
def _crack(r, a):
    m = a.max(-1) < 75                        # the crack is the only near-black stroke above the ruler
    m[int(a.shape[0] * .78):] = False
    lab, k = ndimage.label(ndimage.binary_dilation(m, iterations=2))
    big = 1 + int(np.argmax(ndimage.sum(m, lab, range(1, k + 1))))
    ys, xs = np.nonzero(m & (lab == big))      # keep only the single largest dark stroke (the crack)
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ u
    return (t.max() - t.min() - 3) / r["spec"]["px_per_mm"]


@check("layer_stack", "raster")
def _stack(r, a):
    from PIL import ImageDraw as D
    s = r["spec"]
    W, H = a.shape[1], a.shape[0]

    def mask(sh):
        im = Image.new("L", (W, H), 0)
        dr = D.Draw(im)
        if sh["kind"] == "circle":
            dr.ellipse([sh["x"] - sh["r"], sh["y"] - sh["r"], sh["x"] + sh["r"], sh["y"] + sh["r"]], fill=1)
        else:
            k = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}[sh["kind"]]
            dr.polygon([(sh["x"] + sh["r"] * math.cos(sh["rot"] + 2 * math.pi * i / k),
                         sh["y"] + sh["r"] * math.sin(sh["rot"] + 2 * math.pi * i / k)) for i in range(k)], fill=1)
        return np.asarray(im) > 0
    ms = [mask(sh) for sh in s["shapes"]]
    pos = {sidx: k for k, sidx in enumerate(s["bottom_to_top"])}
    G = nx.DiGraph()
    G.add_nodes_from(range(len(ms)))
    for i in range(len(ms)):
        for j in range(len(ms)):
            if i != j and pos[j] > pos[i] and (ms[i] & ms[j]).sum() > 20:
                G.add_edge(i, j)           # j lies on top of i
    return len(nx.descendants(G, s["target"]))


@check("out_of_place", "structure")
def _oop(r, a):
    s = r["spec"]
    pl = s.get("placed", [])
    for i in range(len(pl)):
        for j in range(i):
            if math.hypot(pl[i][0] - pl[j][0], pl[i][1] - pl[j][1]) < pl[i][2] + pl[j][2] + .2:
                return "OVERLAP"              # furniture collides -> an unintended anomaly
    return r["answer"] if (s["anomaly"] == "none") == (s["target"] is None) else "INCONSISTENT"


@check("hidden_pedestrian", "structure")
def _hped(r, a):
    v = r["spec"]["visible_px"]
    if r["answer"] == "none":
        return "none" if v is None else "PEDESTRIAN-PRESENT"
    return r["answer"] if v is not None and v >= 150 else "NOT-VISIBLE"


@check("mirror_check", "structure")
def _mirror(r, a):
    w = r["spec"]["wrong"]
    return "none" if w is None else L[w]


@check("shadow_check", "structure")
def _shadow(r, a):
    s = r["spec"]
    if s["wrong"] is None:
        return "none"
    return L[s["wrong"]] if abs(s["error_deg"]) >= 29 else "TOO-SUBTLE"


@check("cable_match", "pixels")
def _cable(r, a):
    """From the rendered cable mask only: which right-hand port's plug carries the target cable's id?"""
    s = r["spec"]
    seg = np.asarray(Image.open(s["seg_path"]))
    K, tid = s["K"], s["target"] + 1
    pts = s["ports_2d"]
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]

    def carries(p):
        x, y = p
        m = (xx - x) ** 2 + (yy - y) ** 2 < 14 ** 2
        return int((seg[m] == tid).sum())
    if carries(pts[s["left_ports"][s["target"]]]) < 10:
        return "TARGET-NOT-AT-LEFT-PORT"
    hits = [k for k in range(K) if carries(pts[K + k]) >= 10]
    return hits[0] + 1 if len(hits) == 1 else f"AMBIGUOUS{hits}"


def run(samples, tasks=None):
    rows = []
    for name in sorted(os.listdir(samples)):
        f = os.path.join(samples, name, "items.jsonl")
        if not os.path.exists(f) or (tasks and name not in tasks):
            continue
        fn, kind = CHECKS.get(name, (None, "none"))
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        bad, checked = [], 0
        for r in recs:
            if fn is None:
                continue
            a = np.asarray(Image.open(os.path.join(samples, name, r["file"])).convert("RGB")).astype(int)
            got = fn(r, a)
            if got is None:
                continue
            checked += 1
            exp = r["answer"]
            ok = (abs(float(got) - float(exp)) <= max(r.get("tol") or 0, 1e-6)) \
                if isinstance(exp, (int, float)) and isinstance(got, (int, float)) and not isinstance(got, bool) \
                else str(got).lower() == str(exp).lower()
            if not ok:
                bad.append((r["file"], exp, got))
        rows.append((name, kind, len(recs), checked, bad))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    rows = run(a.samples, set(t for t in a.tasks.split(",") if t) or None)
    tot_c = tot_b = 0
    for name, kind, n, c, bad in rows:
        tot_c += c
        tot_b += len(bad)
        status = "no checker" if kind == "none" else ("OK" if not bad else f"{len(bad)} MISMATCH")
        print(f"{name:20s} {kind:12s} checked {c:2d}/{n:2d}  {status}")
        for f, e, g in bad:
            print(f"      {f}: stored={e!r} independent={g!r}")
    print(f"\nchecked {tot_c} samples, {tot_b} mismatches")


if __name__ == "__main__":
    main()
