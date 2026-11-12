"""Evaluate models on items: prompt building, on-disk response cache, concurrency, scoring."""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from .core import Item, build_prompt, is_correct, parse_answer
from .perturb import perturb as _perturb


class Cache:
    """Append-only jsonl cache keyed by (model, prompt, image bytes): re-runs never repeat a paid call."""

    def __init__(self, path):
        self.path, self.lock, self.d = path, threading.Lock(), {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.d[r["k"]] = r["v"]
                    except Exception:
                        pass

    @staticmethod
    def key(model, prompt, png):
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"\0" + prompt.encode() + b"\0" + (png or b""))
        return h.hexdigest()

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            if self.path:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": v}) + "\n")


def _image_bytes(item: Item, perturb_spec):
    if not perturb_spec:
        return item.png()
    kind, _, s = perturb_spec.partition(":")
    img = item.image if item.image is not None else Image.open(item.image_path).convert("RGB")
    out = _perturb(img, kind, float(s or 0.5), seed=item.seed)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def evaluate_one(model, item: Item, mode="image", cache: Cache | None = None, perturb_spec=None):
    prompt, png = build_prompt(item, mode)
    if png is not None and perturb_spec:
        png = _image_bytes(item, perturb_spec)
    err, raw = None, ""
    try:
        k = Cache.key(model.name, prompt, png) if (cache and model.cacheable) else None
        hit = cache.get(k) if k else None
        if hit is not None:
            raw = hit
        else:
            raw = model.ask(prompt, png, item)
            if k:
                cache.put(k, raw)
    except Exception as e:  # keep going; errors are excluded from accuracy and reported
        err = repr(e)[:300]
    pred = None if err else parse_answer(raw, item.atype, item.choices)
    return dict(model=model.name, item_id=item.id, pair_id=item.pair_id, family=item.family, d=item.d,
                variant=item.variant, mode=mode, perturb=perturb_spec or "none", answer=item.answer,
                pred=pred, correct=bool(is_correct(item, pred)) if not err else False, error=err,
                raw=raw[-400:], question=item.question, meta=item.meta)


def evaluate(model, items, mode="image", cache=None, workers=8, perturb_spec=None, progress=True):
    out, done = [], [0]

    def work(it):
        r = evaluate_one(model, it, mode, cache, perturb_spec)
        done[0] += 1
        if progress and done[0] % 50 == 0:
            print(f"  {model.name} {mode}: {done[0]}/{len(items)}", flush=True)
        return r

    if workers <= 1 or not model.cacheable:
        return [work(it) for it in items]
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(work, items))
    return out
