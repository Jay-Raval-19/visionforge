"""Model adapters. One thin interface: ask(prompt, image_png_or_None, item) -> text.

Model spec strings (see make_model):
  oracle | random | sim:<d50>:<slope>[:<text_acc>]        local, offline, never touch a network
  anthropic:<model-id>[@effort]                            needs ANTHROPIC_API_KEY (or `ant auth login`)
  openai:<model-id>                                        any OpenAI-compatible endpoint; OPENAI_API_KEY, OPENAI_BASE_URL
  gemini:<model-id>                                        needs GEMINI_API_KEY
The network adapters are only constructed when you ask for them on the command line.
"""
from __future__ import annotations

import base64
import json
import math
import os
import random
import time
import urllib.error
import urllib.request


class Model:
    name = "?"
    cacheable = True

    def ask(self, prompt: str, image_png: bytes | None, item=None) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------- offline models
class Oracle(Model):
    name = "oracle"
    cacheable = False

    def ask(self, prompt, image_png, item=None):
        return f"The answer follows from the spec.\nANSWER: {item.answer}"


class RandomModel(Model):
    name = "random"
    cacheable = False

    def ask(self, prompt, image_png, item=None):
        rng = random.Random(f"random:{item.id}:{bool(image_png)}")
        if item.atype == "choice":
            return f"ANSWER: {rng.choice(item.choices)}"
        if item.atype == "int":
            return f"ANSWER: {rng.randrange(1, 15)}"
        return f"ANSWER: {rng.uniform(0, 100):.2f}"


class SimModel(Model):
    """Simulated model with a known psychometric curve; used to validate the pipeline and analysis.

    Image accuracy: chance + (0.98 - chance) * sigmoid((d50 - d) / slope). Text twin accuracy is flat
    (text_acc). On 'twist' items it answers from memory (the canonical value) 70% of the time when wrong.
    """
    cacheable = False

    def __init__(self, d50, slope, text_acc=0.97):
        self.d50, self.slope, self.text_acc = float(d50), float(slope), float(text_acc)
        self.name = f"sim:{d50}:{slope}"

    def ask(self, prompt, image_png, item=None):
        mode = "image" if image_png else "text"
        rng = random.Random(f"{self.name}:{item.id}:{mode}")
        is_twist = item.meta.get("is_twist")
        if mode == "text":
            p = self.text_acc
        elif item.meta.get("prior") is not None and not is_twist:
            p = 0.97  # canonical control items are easy: the model remembers them
        else:
            g = 0.05
            p = g + (0.98 - g) / (1 + math.exp((item.d - self.d50) / self.slope))
        if rng.random() < p:
            ans = item.answer
        elif is_twist and mode == "image" and rng.random() < 0.7:
            ans = item.meta["prior"]
        elif item.atype == "choice":
            ans = rng.choice([c for c in item.choices if c != item.answer])
        elif item.atype == "int":
            ans = max(0, item.answer + rng.choice([-2, -1, 1, 2]))
        else:
            ans = round(item.answer * rng.choice([0.6, 0.8, 1.25, 1.5]) + rng.choice([-3, 3]), 2)
        return f"Looking at it... \nANSWER: {ans}"


# ---------------------------------------------------------------- network adapters (not used unless requested)
def _b64(png):
    return base64.standard_b64encode(png).decode()


def _post(url, headers, payload, timeout=300, retries=5):
    data = json.dumps(payload).encode()
    for k in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (408, 409, 429, 500, 502, 503, 529) and k < retries - 1:
                time.sleep(min(60, 2 ** k + random.random()))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError):
            if k == retries - 1:
                raise
            time.sleep(2 ** k)


class AnthropicModel(Model):
    def __init__(self, model_id, effort=None, max_tokens=16000):
        import anthropic  # lazy: only needed when this adapter is requested

        self.client = anthropic.Anthropic()
        self.model_id, self.effort, self.max_tokens = model_id, effort, max_tokens
        self.name = f"anthropic:{model_id}" + (f"@{effort}" if effort else "")

    def ask(self, prompt, image_png, item=None):
        content = []
        if image_png:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                         "data": _b64(image_png)}})
        content.append({"type": "text", "text": prompt})
        kw = {"output_config": {"effort": self.effort}} if self.effort else {}
        r = self.client.messages.create(model=self.model_id, max_tokens=self.max_tokens,
                                        messages=[{"role": "user", "content": content}], **kw)
        if r.stop_reason == "refusal":
            return "[REFUSAL]"
        return "".join(b.text for b in r.content if b.type == "text")


class OpenAIModel(Model):
    def __init__(self, model_id, max_tokens=8000):
        self.model_id, self.max_tokens = model_id, max_tokens
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ["OPENAI_API_KEY"]
        self.name = f"openai:{model_id}"

    def ask(self, prompt, image_png, item=None):
        parts = [{"type": "text", "text": prompt}]
        if image_png:
            parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(image_png)}})
        out = _post(f"{self.base}/chat/completions", {"Authorization": f"Bearer {self.key}"},
                    {"model": self.model_id, "messages": [{"role": "user", "content": parts}],
                     "max_completion_tokens": self.max_tokens})
        return out["choices"][0]["message"].get("content") or ""


class GeminiModel(Model):
    def __init__(self, model_id, max_tokens=8192):
        self.model_id, self.max_tokens = model_id, max_tokens
        self.key = os.environ["GEMINI_API_KEY"]
        self.name = f"gemini:{model_id}"

    def ask(self, prompt, image_png, item=None):
        parts = [{"text": prompt}]
        if image_png:
            parts.append({"inline_data": {"mime_type": "image/png", "data": _b64(image_png)}})
        out = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent",
                    {"x-goog-api-key": self.key},
                    {"contents": [{"parts": parts}], "generationConfig": {"maxOutputTokens": self.max_tokens}})
        cands = out.get("candidates") or [{}]
        return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))


def make_model(spec: str) -> Model:
    kind, _, rest = spec.partition(":")
    if kind == "oracle":
        return Oracle()
    if kind == "random":
        return RandomModel()
    if kind == "sim":
        a = rest.split(":")
        return SimModel(a[0], a[1], *(a[2:3]))
    if kind == "anthropic":
        mid, _, eff = rest.partition("@")
        return AnthropicModel(mid, eff or None)
    if kind == "openai":
        return OpenAIModel(rest)
    if kind == "gemini":
        return GeminiModel(rest)
    raise ValueError(f"unknown model spec {spec!r}")
