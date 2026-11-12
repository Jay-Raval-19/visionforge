"""Position-based-dynamics rope simulation for cables lying on a table (NumPy).

Each cable is a chain of particles with fixed segment lengths (inextensible), a bending constraint (stiffness),
gravity, ground contact with friction, and particle-particle collision between and within cables, so cables
come to rest physically on top of each other where they cross. Both ends are pinned at their connectors.
The result gives exact centrelines in 3D, hence exact crossings and over/under order.
"""
import numpy as np


def simulate(init, radius, pinned=2, stiffness=0.4, steps=220, iters=6, dt=1 / 120, friction=0.8, seed=0):
    """init: (C, N, 3) starting positions (metres). Returns final positions (C, N, 3).
    Rest lengths come from the top-view (xy) path, so the drop height adds no extra cable length."""
    P = np.array(init, float)
    C, N, _ = P.shape
    prev = P.copy()
    L1 = np.linalg.norm(P[:, 1:, :2] - P[:, :-1, :2], axis=-1)    # rest lengths (xy path)
    L2 = np.linalg.norm(P[:, 2:, :2] - P[:, :-2, :2], axis=-1)
    pin = np.zeros((C, N), bool)
    pin[:, :pinned] = pin[:, -pinned:] = True
    anchor = P.copy()
    flat_c = np.repeat(np.arange(C), N)
    flat_i = np.tile(np.arange(N), C)
    from scipy.spatial import cKDTree
    g = np.array((0, 0, -9.81))
    for _ in range(steps):
        v = (P - prev) * 0.985
        prev = P.copy()
        P = P + v + g * dt * dt
        for _k in range(iters):
            # stretch (inextensible)
            d = P[:, 1:] - P[:, :-1]
            ln = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
            corr = (ln - L1[..., None]) / ln * d * 0.5
            P[:, :-1] += corr
            P[:, 1:] -= corr
            # bending (distance between i and i+2)
            d = P[:, 2:] - P[:, :-2]
            ln = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
            corr = stiffness * (ln - L2[..., None]) / ln * d * 0.5
            P[:, :-2] += corr
            P[:, 2:] -= corr
            # collisions: particles are spaced closer than the cable diameter, so cables cannot tunnel
            F = P.reshape(-1, 3)
            pairs = cKDTree(F).query_pairs(2 * radius, output_type="ndarray")
            if len(pairs):
                ii, jj = pairs[:, 0], pairs[:, 1]
                keep = ~((flat_c[ii] == flat_c[jj]) & (np.abs(flat_i[ii] - flat_i[jj]) <= 2))
                ii, jj = ii[keep], jj[keep]
                if len(ii):
                    diff = F[ii] - F[jj]
                    dist = np.linalg.norm(diff, axis=1) + 1e-12
                    push = ((2 * radius - dist) / dist * 0.5)[:, None] * diff
                    np.add.at(F, ii, push)
                    np.add.at(F, jj, -push)
                    P = F.reshape(C, N, 3)
            # ground + friction
            low = P[..., 2] < radius
            P[..., 2] = np.maximum(P[..., 2], radius)
            if low.any():
                P[low, :2] = prev[low, :2] + (P[low, :2] - prev[low, :2]) * (1 - friction)
            P[pin] = anchor[pin]
    return P


def _tangents(c, k=8):
    """Smoothed direction of a cable at each particle (±k particles), ignoring kinks where it rides over another."""
    idx = np.arange(len(c))
    t = c[np.minimum(idx + k, len(c) - 1), :2] - c[np.maximum(idx - k, 0), :2]
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)


def crossings_2d(P, target, k=8):
    """2D crossings of cable `target` with every other cable: list of (other, angle_deg, target_on_top)."""
    out = []
    a = P[target]
    ta = _tangents(a, k)
    for c in range(len(P)):
        if c == target:
            continue
        b = P[c]
        tb = _tangents(b, k)
        for i in range(len(a) - 1):
            p, r = a[i, :2], a[i + 1, :2] - a[i, :2]
            for j in range(len(b) - 1):
                q, s = b[j, :2], b[j + 1, :2] - b[j, :2]
                den = r[0] * s[1] - r[1] * s[0]
                if abs(den) < 1e-12:
                    continue
                t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
                u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
                if 0 <= t < 1 and 0 <= u < 1:
                    za = a[i, 2] + t * (a[i + 1, 2] - a[i, 2])
                    zb = b[j, 2] + u * (b[j + 1, 2] - b[j, 2])
                    cosang = abs(ta[i] @ tb[j])
                    out.append((c, float(np.degrees(np.arccos(np.clip(cosang, 0, 1)))), bool(za > zb)))
    return out
