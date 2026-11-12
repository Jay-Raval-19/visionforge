"""Closed-loop failure search.

Instead of a fixed difficulty grid, the pipeline *adapts to the tested model*: after every answer it updates a
Bayesian posterior over that model's breaking point (d50) and generates the next item at the most informative
difficulty (QUEST-style psychophysics). It then explains the failure with a knob ablation at d50.

Psychometric model: P(correct | d) = g + (1 - g - lapse) * sigmoid((d50 - d) / slope)
  g = chance level of the family (majority-answer rate), d50 = difficulty where accuracy is midway between
  chance and ceiling, slope = how abruptly the model collapses.
"""
from __future__ import annotations

import numpy as np

from .core import FAMILIES, make_pair
from .runner import evaluate_one

GRID = np.linspace(0, 1, 101)
SLOPES = np.array([0.04, 0.08, 0.16])
LAPSE = 0.02


def _p_correct(d, g):
    z = (GRID[None, :] - d) / SLOPES[:, None]          # (slope, d50)
    return g + (1 - g - LAPSE) / (1 + np.exp(-z))


def posterior(trials, g):
    """trials: iterable of (d, correct). Returns marginal posterior over d50 on GRID."""
    logp = np.zeros((len(SLOPES), len(GRID)))
    for d, ok in trials:
        p = np.clip(_p_correct(d, g), 1e-6, 1 - 1e-6)
        logp += np.log(p if ok else 1 - p)
    post = np.exp(logp - logp.max())
    marg = post.sum(0)
    return marg / marg.sum()


def summarize(marg):
    cdf = np.cumsum(marg)
    q = lambda a: float(GRID[min(len(GRID) - 1, np.searchsorted(cdf, a))])
    return dict(d50=q(0.5), lo=q(0.05), hi=q(0.95))


def chance_rate(fam_name, n=24):
    """Majority-answer rate over generated items = what a prior-only guesser would score."""
    from collections import Counter
    c, tot = Counter(), 0
    for i in range(n):
        d = (i % 4) / 3
        it = make_pair(fam_name, 90000 + i, d)[0]
        c[str(it.answer)] += 1
        tot += 1
    g = max(c.values()) / tot
    return max(g, 0.05) if it.atype == "float" else max(g, 0.03)


def search(model, fam_name, trials=40, cache=None, seed=0, mode="image", g=None, rng=None):
    rng = rng or np.random.default_rng(seed)
    g = chance_rate(fam_name) if g is None else g
    hist, log = [], []
    for t in range(trials):
        marg = posterior(hist, g)
        med = float(GRID[np.searchsorted(np.cumsum(marg), 0.5)])
        d = float(np.clip(med + rng.normal(0, 0.05 if t < 10 else 0.02), 0, 1))
        item = make_pair(fam_name, 50000 + seed * 1000 + t, d)[0]
        r = evaluate_one(model, item, mode, cache)
        if r["error"]:
            continue
        hist.append((d, r["correct"]))
        log.append(dict(t=t, d=round(d, 3), correct=r["correct"], item_id=item.id, pred=r["pred"],
                        answer=item.answer))
    res = summarize(posterior(hist, g))
    res.update(model=model.name, family=fam_name, mode=mode, chance=g, trials=log,
               n=len(hist), acc=float(np.mean([c for _, c in hist])) if hist else float("nan"))
    return res


def ablate(model, fam_name, d, k=12, cache=None, mode="image", seed=0):
    """Re-run the family at difficulty d with each secondary knob toggled; the biggest accuracy swing
    names the factor that drives the failure (e.g. removing color cues in wire tracing)."""
    fam = FAMILIES[fam_name]
    out = {}
    for knob, values in fam.toggles.items():
        out[knob] = {}
        for v in values:
            oks = []
            for i in range(k):
                it = make_pair(fam_name, 70000 + seed * 1000 + i, d, **{knob: v})[0]
                r = evaluate_one(model, it, mode, cache)
                if not r["error"]:
                    oks.append(r["correct"])
            out[knob][str(v)] = float(np.mean(oks)) if oks else float("nan")
    return out
