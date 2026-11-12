"""Documents & forms: invoice lookups, counts and totals (closest to real document workflows)."""
from ..core import Family, register
from ..draw import Canvas

NAMES = ["Copper pipe", "Hex bolt M8", "Safety gloves", "LED panel", "Cable tie pack", "Drill bit set",
         "Ball valve", "Steel bracket", "Wire spool", "Thermal paste", "Fuse 10A", "Relay module",
         "PVC elbow", "Silicone seal", "Torque wrench", "Battery AA", "Sensor probe", "Fan 80mm",
         "Solder roll", "Label tape"]
TAX = 0.08


def money(c):
    return f"${c // 100:,}.{c % 100:02d}"


@register
class Invoice(Family):
    name = "invoice"
    application = "Invoices, statements, forms (document understanding)"
    capability = "exact value reading in documents"
    toggles = {"extra_cols": [False, True]}

    def make_spec(self, rng, d, extra_cols=None):
        n = 3 + round(d * 11)
        names = rng.sample(NAMES, n)
        rows = [dict(name=nm, qty=rng.randrange(1, 13), unit=rng.randrange(150, 9900),
                     sku=str(rng.randrange(10000, 99999)), disc=rng.choice(["0%", "5%", "10%"]))
                for nm in names]
        qtype = rng.choice(["amount", "count_qty", "total"])
        return dict(rows=rows, qtype=qtype, target=rng.randrange(n), k=rng.randrange(2, 8),
                    extra=bool(d > 0.4) if extra_cols is None else bool(extra_cols),
                    stripes=bool(d <= 0.5), fsize=int(round(17 - 5 * d)))

    @staticmethod
    def _amount(r):
        return r["qty"] * r["unit"]

    def _totals(self, spec):
        sub = sum(self._amount(r) for r in spec["rows"])
        tax = int(round(sub * TAX))
        return sub, tax, sub + tax

    def render(self, spec):
        fs = spec["fsize"]
        rh = fs + 16
        n = len(spec["rows"])
        W, H = 820, 170 + rh * (n + 1) + 130
        cv = Canvas(W, H, bg=(255, 255, 255))
        cv.text((40, 40), "INVOICE", 30, anchor="lm")
        cv.text((W - 40, 36), "No. 2025-" + str(1000 + n * 37), 15, anchor="rm")
        cv.text((W - 40, 58), "Terms: Net 30", 14, anchor="rm")
        cols = [("Item", 40, "lm"), ("Qty", 440, "rm"), ("Unit price", 570, "rm"), ("Amount", W - 40, "rm")]
        if spec["extra"]:
            cols = [("SKU", 40, "lm"), ("Item", 130, "lm"), ("Disc.", 400, "rm"), ("Qty", 480, "rm"),
                    ("Unit price", 610, "rm"), ("Amount", W - 40, "rm")]
        y0 = 100
        cv.rect(30, y0, W - 30, y0 + rh, fill=(60, 70, 90))
        for nm, x, a in cols:
            cv.text((x, y0 + rh / 2), nm, fs, fill=(255, 255, 255), anchor=a)
        for i, r in enumerate(spec["rows"]):
            y = y0 + rh * (i + 1)
            if spec["stripes"] and i % 2:
                cv.rect(30, y, W - 30, y + rh, fill=(238, 241, 246))
            vals = {"Item": r["name"], "Qty": r["qty"], "Unit price": money(r["unit"]),
                    "Amount": money(self._amount(r)), "SKU": r["sku"], "Disc.": r["disc"]}
            for nm, x, a in cols:
                cv.text((x, y + rh / 2), vals[nm], fs, anchor=a)
        yb = y0 + rh * (n + 1) + 14
        cv.line([(30, yb), (W - 30, yb)], (60, 70, 90), 2)
        sub, tax, tot = self._totals(spec)
        for j, (lab, v) in enumerate((("Subtotal", sub), ("Tax (8%)", tax), ("Total due", tot))):
            cv.text((W - 200, yb + 22 + j * (fs + 12)), lab, fs + (2 if j == 2 else 0), anchor="rm")
            cv.text((W - 40, yb + 22 + j * (fs + 12)), money(v), fs + (2 if j == 2 else 0), anchor="rm")
        return cv.done()

    def qa(self, spec):
        rows = spec["rows"]
        if spec["qtype"] == "amount":
            r = rows[spec["target"]]
            return dict(question=f"What is the Amount (line total) for the item '{r['name']}'? "
                                 f"Give the number in dollars, e.g. 12.50.",
                        answer=round(self._amount(r) / 100, 2), atype="float", tol=0.005)
        if spec["qtype"] == "count_qty":
            return dict(question=f"How many line items have a quantity greater than {spec['k']}?",
                        answer=sum(1 for r in rows if r["qty"] > spec["k"]), atype="int", tol=0.0)
        return dict(question="What is the Total due? Give the number in dollars, e.g. 12.50.",
                    answer=round(self._totals(spec)[2] / 100, 2), atype="float", tol=0.005)

    def text(self, spec):
        lines = ["Invoice table (item | sku | disc | qty | unit price | amount):"]
        for r in spec["rows"]:
            lines.append(f"{r['name']} | {r['sku']} | {r['disc']} | {r['qty']} | {money(r['unit'])} | "
                         f"{money(self._amount(r))}")
        sub, tax, tot = self._totals(spec)
        lines.append(f"Subtotal {money(sub)}; Tax (8%) {money(tax)}; Total due {money(tot)}")
        return "\n".join(lines)

    def meta(self, spec):
        return dict(n_rows=len(spec["rows"]), qtype=spec["qtype"])

    def _mutate(self, spec, rng):
        i = spec["target"] if spec["qtype"] == "amount" else rng.randrange(len(spec["rows"]))
        spec["rows"][i]["qty"] = rng.randrange(1, 13)
        return spec
