"""Generate N samples per task across the difficulty range, plus a contact sheet and an HTML gallery.

    python -m vf.samples --n 10 --out samples [--tasks a,b] [--group puzzle]
"""
import argparse
import html
import json
import os
import time
import traceback

from PIL import Image

from . import families, tasks  # noqa: F401  (register everything)
from .core import FAMILIES, _jsonable, make_pair
from .draw import font


def _save(img, path, tries=6):
    """OneDrive can briefly lock a file it is syncing; retry instead of failing the whole run."""
    for k in range(tries):
        try:
            img.save(path)
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(0.5 * (k + 1))


def sheet(paths, out, thumb=300, cols=5):
    ims = [Image.open(p).convert("RGB") for p in paths]
    rows = (len(ims) + cols - 1) // cols
    S = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))
    for k, im in enumerate(ims):
        im.thumbnail((thumb - 8, thumb - 8))
        S.paste(im, ((k % cols) * thumb + 4, (k // cols) * thumb + 4))
    _save(S, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--group", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args()
    if a.index_only:
        build_index(a.out)
        return
    names = [t for t in a.tasks.split(",") if t] or sorted(FAMILIES)
    if a.group:
        names = [t for t in names if getattr(FAMILIES[t], "group", "") == a.group]
    os.makedirs(a.out, exist_ok=True)
    index = []
    for name in names:
        fam = FAMILIES[name]
        d_dir = os.path.join(a.out, name)
        os.makedirs(d_dir, exist_ok=True)
        t0, recs, paths = time.time(), [], []
        for i in range(a.n):
            d = i / max(1, a.n - 1)
            try:
                it = make_pair(name, a.seed * 1000 + i, d)[0]
            except Exception as e:
                print(f"  ! {name} sample {i}: {e!r}")
                traceback.print_exc(limit=2)
                continue
            p = os.path.join(d_dir, f"{i:02d}.png")
            _save(it.image, p)
            paths.append(p)
            rec = dict(id=it.id, file=f"{i:02d}.png", d=round(d, 2), question=it.question, answer=it.answer,
                       atype=it.atype, choices=it.choices, tol=it.tol, text_twin=it.text_context, spec=it.spec)
            recs.append(rec)
        with open(os.path.join(d_dir, "items.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, default=_jsonable) + "\n")
        if paths:
            sheet(paths, os.path.join(d_dir, "_sheet.png"))
        index.append((name, fam, recs))
        print(f"{name:22s} {len(recs):2d} samples  {time.time() - t0:5.1f}s")
    build_index(a.out)


GROUP_ORDER = ["puzzle", "indoor3d", "driving", "real2d", "real3d", ""]


def build_index(out):
    """Gallery of every task folder on disk (not just the last run)."""
    entries = []
    for name in sorted(os.listdir(out)):
        f = os.path.join(out, name, "items.jsonl")
        if name not in FAMILIES or not os.path.exists(f):
            continue
        with open(f, encoding="utf-8") as fh:
            recs = [json.loads(l) for l in fh]
        fam = FAMILIES[name]
        recs = [dict(r, _v=int(os.path.getmtime(os.path.join(out, name, r["file"])))) for r in recs]
        entries.append((GROUP_ORDER.index(getattr(fam, "group", "")) if getattr(fam, "group", "") in GROUP_ORDER else 9,
                        name, fam, recs))
    entries.sort(key=lambda e: (e[0], e[1]))
    nav, rows, last = [], [], None
    for _, name, fam, recs in entries:
        grp = getattr(fam, "group", "") or "families"
        if grp != last:
            rows.append(f"<h1 id='g-{grp}'>{grp}</h1>")
            nav.append(f"<a href='#g-{grp}'>{grp}</a>")
            last = grp
        ver = max((r["_v"] for r in recs), default=0)
        cards = "".join(
            f"<figure><a href='{name}/{r['file']}?v={ver}' target=_blank><img src='{name}/{r['file']}?v={ver}' loading='lazy'></a>"
            f"<figcaption><b>d={r['d']}</b> {html.escape(r['question'])}<br><span class=a>Answer: "
            f"{html.escape(str(r['answer']))}</span></figcaption></figure>" for r in recs)
        rows.append(f"<section id='{name}'><h2>{name} <small>({len(recs)} samples)</small></h2><p class=m>"
                    f"<b>Real-world use:</b> {html.escape(fam.application)} · <b>Capability:</b> "
                    f"{html.escape(fam.capability)} · <b>Knob:</b> {html.escape(getattr(fam, 'knob', '') or '')}"
                    f"{' · <b>Twin:</b> ' + html.escape(fam.twin) if getattr(fam, 'twin', '') else ''}</p>"
                    f"<div class=g>{cards}</div></section>")
    total = sum(len(e[3]) for e in entries)
    page = ("<!doctype html><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"><meta http-equiv=Cache-Control content=no-store><title>VisionForge samples</title><style>"
            "body{font:14px system-ui;margin:24px;background:#fafafa;color:#222}h1{margin:36px 0 0;font-size:22px;"
            "border-bottom:2px solid #ddd;text-transform:uppercase;letter-spacing:.05em}h2{margin:24px 0 4px}"
            "small{color:#888;font-weight:400}nav a{margin-right:14px}"
            ".m{color:#555;margin:0 0 10px}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}"
            "figure{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}"
            "img{width:100%;height:200px;object-fit:contain;background:#fff}figcaption{font-size:12px;color:#333}"
            ".a{color:#0a6;font-weight:600}</style>"
            f"<h1 style='border:0;text-transform:none'>VisionForge samples — {len(entries)} tasks, {total} images</h1>"
            "<nav>" + "".join(nav) + "</nav>" + "".join(rows))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    main()
