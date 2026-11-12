"""Core abstractions: Family (spec -> image/question/answer/text twin), Item, scoring, I/O."""
from __future__ import annotations

import copy
import io
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

FAMILIES: dict[str, "Family"] = {}


def register(cls):
    FAMILIES[cls.name] = cls()
    return cls


def _differs(a, b, tol):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) > max(tol, 1e-9)
    return a != b


class Family:
    """A task family. Everything is derived from a JSON-serialisable `spec`.

    make_spec  -> random spec at difficulty d in [0, 1] (the "knob")
    render     -> PIL image
    qa         -> {"question", "answer", "atype", "tol", "choices"}; answer is computed from the spec
    text       -> text twin: the same information serialised as text (IsoBench-style)
    valid      -> ambiguity / solvability filter on the spec
    check      -> independent pixel-level recheck of the rendered image
    _mutate    -> minimal edit intended to flip the answer (counterfactual pair)
    """

    name = ""
    application = ""
    capability = ""
    toggles: dict[str, list] = {}

    def make_spec(self, rng, d, **kw):
        raise NotImplementedError

    def render(self, spec):
        raise NotImplementedError

    def qa(self, spec):
        raise NotImplementedError

    def text(self, spec):
        raise NotImplementedError

    def meta(self, spec):
        return {}

    def valid(self, spec):
        return spec is not None

    def check(self, spec, img):
        return True

    def _mutate(self, spec, rng):
        return None

    def edit_spec(self, spec, rng):
        qa0 = self.qa(spec)
        for _ in range(25):
            s = self._mutate(copy.deepcopy(spec), rng)
            if s is None:
                return None
            if not self.valid(s):
                continue
            qa1 = self.qa(s)
            if _differs(qa1["answer"], qa0["answer"], 3 * qa1["tol"]):
                return s
        return None


@dataclass
class Item:
    id: str
    family: str
    d: float
    seed: int
    variant: str            # "base" | "edit"
    pair_id: str
    question: str
    answer: Any
    atype: str              # "int" | "float" | "choice"
    tol: float
    choices: list | None
    text_context: str
    meta: dict
    spec: dict
    image: Image.Image | None = field(default=None, repr=False)
    image_path: str | None = None
    _png: bytes | None = field(default=None, repr=False)

    def png(self) -> bytes:
        if self._png is None:
            if self.image is not None:
                buf = io.BytesIO()
                self.image.save(buf, format="PNG")
                self._png = buf.getvalue()
            else:
                with open(self.image_path, "rb") as f:
                    self._png = f.read()
        return self._png

    def to_json(self):
        d = {k: v for k, v in self.__dict__.items() if k not in ("image", "_png")}
        return d


def make_pair(fam_name, seed, d, **kw):
    """Build [base] or [base, edit] items for one seed/difficulty (deterministic)."""
    fam = FAMILIES[fam_name]
    rng = random.Random(f"{fam_name}:{seed}:{round(d, 4)}:{sorted(kw.items())}")
    for _ in range(150):
        spec = fam.make_spec(rng, d, **kw)
        if not fam.valid(spec):
            continue
        img = fam.render(spec)
        if not fam.check(spec, img):
            continue
        break
    else:
        raise RuntimeError(f"could not build a valid {fam_name} item at d={d}")
    pid = f"{fam_name}-{seed}-{int(round(d * 1000)):04d}"

    def mk(sp, im, variant):
        qa = fam.qa(sp)
        pub = {k: v for k, v in sp.items() if not k.startswith("_")}
        meta = dict(fam.meta(sp))
        meta.update(kw)
        return Item(id=f"{pid}-{variant}", family=fam_name, d=float(d), seed=int(seed), variant=variant,
                    pair_id=pid, question=qa["question"], answer=qa["answer"], atype=qa["atype"],
                    tol=qa["tol"], choices=qa.get("choices"), text_context=fam.text(sp), meta=meta,
                    spec=pub, image=im)

    items = [mk(spec, img, "base")]
    es = fam.edit_spec(spec, rng)
    if es is not None:
        eimg = fam.render(es)
        if fam.check(es, eimg):
            items.append(mk(es, eimg, "edit"))
    return items


# ---------------------------------------------------------------- prompting & scoring
FORMAT_HINT = {
    "int": "a single integer",
    "float": "a single number",
    "choice": "exactly one of the listed options",
    "seq": "the letters in order with no spaces, e.g. CADB",
}


def build_prompt(item: Item, mode: str):
    """Return (prompt_text, png_bytes_or_None). mode: 'image' or 'text' (the text twin)."""
    hint = FORMAT_HINT[item.atype]
    if item.atype == "choice":
        hint += f" ({', '.join(map(str, item.choices))})"
    if mode == "image":
        head = "Look carefully at the attached image."
        png = item.png()
    else:
        head = ("No image is attached. Instead, here is a textual description of the figure "
                "that contains the same information:\n\n" + item.text_context)
        png = None
    prompt = (f"{head}\n\nQuestion: {item.question}\n\n"
              f"Reason step by step briefly, then finish with a final line of exactly the form\n"
              f"ANSWER: <value>\nwhere <value> is {hint}.")
    return prompt, png


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_answer(text: str, atype: str, choices=None):
    if not text:
        return None
    m = re.findall(r"ANSWER\s*[:=]\s*(.+)", text, flags=re.I)
    s = (m[-1] if m else text.strip().splitlines()[-1]).strip().strip("*`\"'. ")
    if atype in ("int", "float"):
        nums = _NUM.findall(s.replace(",", ""))
        if not nums:
            return None
        v = float(nums[0])
        return int(round(v)) if atype == "int" and abs(v - round(v)) < 1e-9 else v
    if atype == "seq":
        return "".join(re.findall(r"[A-Za-z]", s)).upper() or None
    low = s.lower()
    for c in sorted(choices or [], key=lambda c: -len(str(c))):
        if low == str(c).lower():
            return c
    for c in sorted(choices or [], key=lambda c: -len(str(c))):
        if re.search(r"\b" + re.escape(str(c).lower()) + r"\b", low):
            return c
    return None


def is_correct(item: Item, pred) -> bool:
    if pred is None:
        return False
    if item.atype == "int":
        return isinstance(pred, (int, float)) and abs(pred - item.answer) < 1e-9
    if item.atype == "float":
        return isinstance(pred, (int, float)) and abs(pred - item.answer) <= item.tol + 1e-9
    if item.atype == "seq":
        return str(pred).upper() == str(item.answer).upper()
    return str(pred).lower() == str(item.answer).lower()


# ---------------------------------------------------------------- dataset I/O
def save_items(items, out_dir):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    with open(os.path.join(out_dir, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            rel = f"images/{it.id}.png"
            it.image.save(os.path.join(out_dir, rel))
            it.image_path = os.path.join(out_dir, rel)
            rec = it.to_json()
            rec["image_path"] = rel
            f.write(json.dumps(rec, default=_jsonable) + "\n")


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def load_items(data_dir, families=None):
    items = []
    with open(os.path.join(data_dir, "items.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if families and r["family"] not in families:
                continue
            r["image_path"] = os.path.join(data_dir, r["image_path"])
            items.append(Item(**r))
    return items
