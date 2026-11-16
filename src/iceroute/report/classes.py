"""The ice-class table, generated from what the model measured.

A-11 in docs/decisions.md, *Early estimates vs. measurements*: the June 2026
prototype material listed five ice classes, one of which — *Deformed Ice /
Ridges* — does not exist in any AI4Arctic label, and the model actually deploys
three. Replacing a wrong table by hand invites a second wrong table, so this
writes it from ``artifacts/metrics.json``, ``configs/assumptions.yaml`` and
``data/polaris_rv.csv``.

The reason it is worth generating rather than typing is not tidiness. Writing the
replacement by hand, I produced three rows that read as equally trustworthy —
and they are not. **Thin ice scores 0.136 IoU.** A table that hides that is the
same species of error as the one it replaces: a claim the measurement does not
support, printed confidently.

So the table carries what the model does when it is wrong, and the distinction
that matters is not accuracy but *direction*:

* thin ice is mostly called **thick** — an error toward caution, which costs a
  longer route and nothing else;
* thick ice called **open water** is the error that could put a ship somewhere
  it may not go, and it is the number to publish.

That second figure is the honest headline of this whole section, and it is
better stated up front than found by someone else.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger("classes")

METRICS = Path("artifacts/metrics.json")
ASSUMPTIONS = Path("configs/assumptions.yaml")
RV_TABLE = Path("data/polaris_rv.csv")
OUT = Path("artifacts/class_table.md")

# What each deployed class means to a bridge, in plain operational terms. The
# wording is a judgement, so it lives here and not in a config: everything
# numeric on the row is measured, this column is what we choose to say.
MEANING = {
    "Open water": "Navigate freely",
    "Thin ice": "POLARIS decides — depends on concentration",
    "Thick ice": "POLARIS decides — often forbidden at high concentration",
}


@dataclass
class Row:
    name: str
    meaning: str
    polaris_type: str
    iou: float | None
    support_px: int | None
    riv: dict[str, int]
    mistaken_for: str = ""
    mistaken_pct: float | None = None


def load_riv(path: Path, ice_classes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """RIV per ice type, for the vessel classes the demo shows."""
    rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    table: dict[str, dict[str, int]] = {}
    for rec in csv.DictReader(rows):
        vessel = rec.get("ice_class")
        if vessel not in ice_classes:
            continue
        for ice_type, value in rec.items():
            if ice_type in {"ice_class", "source"} or value is None:
                continue
            try:
                table.setdefault(ice_type, {})[vessel] = int(value)
            except ValueError:
                continue
    return table


def confusion_rows(agg: dict) -> tuple[dict[str, tuple[str, float]], float | None]:
    """For each class: what it is most often mistaken for, and how often.

    Also returns the one number that belongs in any safety statement — the share
    of thick ice called open water — because that is the mistake with a
    consequence. Everything else costs a detour.
    """
    cm = agg.get("confusion_matrix")
    names = list((agg.get("per_class") or {}).keys())
    if not cm or len(names) != len(cm):
        return {}, None
    m = np.asarray(cm, dtype=float)
    worst: dict[str, tuple[str, float]] = {}
    for i, name in enumerate(names):
        total = m[i].sum()
        if total <= 0:
            continue
        off = [(m[i][j] / total, names[j]) for j in range(len(names)) if j != i]
        pct, other = max(off)
        worst[name] = (other, pct * 100)

    permissive = None
    if "Thick ice" in names and "Open water" in names:
        i, j = names.index("Thick ice"), names.index("Open water")
        if m[i].sum() > 0:
            permissive = float(m[i][j] / m[i].sum() * 100)
    return worst, permissive


def collect(metrics: dict, assumptions: dict, rv_path: Path) -> dict:
    ev = (metrics.get("evaluation") or {}).get("scenes") or {}
    agg = ev.get("aggregate") or {}
    per_class = agg.get("per_class") or {}
    seg = (assumptions.get("segmentation") or {}).get("classes") or {}
    vessels = tuple((assumptions.get("vessel") or {}).get("classes_shown") or ("PC6",))
    riv = load_riv(rv_path, vessels) if rv_path.exists() else {}
    worst, permissive = confusion_rows(agg)

    rows = []
    for idx in sorted(seg):
        entry = seg[idx] or {}
        name = entry.get("name", "")
        scores = per_class.get(name) or {}
        meaning = MEANING.get(name, "")
        other, pct = worst.get(name, ("", None))
        rows.append(
            Row(
                name=name,
                meaning=meaning,
                polaris_type=entry.get("polaris_type", ""),
                iou=scores.get("iou"),
                support_px=scores.get("support_px"),
                riv=riv.get(entry.get("polaris_type", ""), {}),
                mistaken_for=other,
                mistaken_pct=pct,
            )
        )

    per_scene = [
        r
        for r in (ev.get("per_scene") or [])
        if isinstance((r.get("rio") or {}).get("worst_unsafe_fraction"), float)
    ]
    fr = [r["rio"]["worst_unsafe_fraction"] for r in per_scene]
    return {
        "rows": rows,
        "vessels": vessels,
        "miou": agg.get("miou"),
        "binary": agg.get("binary_accuracy"),
        "n_scenes": ev.get("n_scenes"),
        "permissive_thick_as_water": permissive,
        "unsafe_median": float(np.median(fr)) if fr else None,
        "unsafe_max": max(fr) if fr else None,
        "n_unsafe_scenes": len(fr),
        "scheme": (assumptions.get("segmentation") or {}).get("scheme", ""),
    }


def _n(v, spec="{:.3f}", missing="not measured"):
    return spec.format(v) if isinstance(v, int | float) else missing


def render(c: dict) -> str:
    vessels = c["vessels"]
    head = " | ".join(f"RIV {v}" for v in vessels)
    sep = "|".join(["---"] * len(vessels))

    summary = [f"| Class | What it means to a bridge | {head} |", f"|---|---|{sep}|"]
    detail = [
        "| Class | Measured IoU | pixels | when wrong, it calls it | POLARIS type |",
        "|---|---:|---:|---|---|",
    ]
    for r in c["rows"]:
        rivs = " | ".join(
            f"{r.riv.get(v, '—'):+d}" if isinstance(r.riv.get(v), int) else "—" for v in vessels
        )
        summary.append(f"| **{r.name}** | {r.meaning} | {rivs} |")
        px = f"{r.support_px / 1e6:.0f} M" if isinstance(r.support_px, int) else "—"
        mis = (
            f"{r.mistaken_for.lower()} in {r.mistaken_pct:.0f}% of cases"
            if r.mistaken_pct is not None
            else "—"
        )
        detail.append(f"| {r.name} | {_n(r.iou)} | {px} | {mis} | `{r.polaris_type}` |")

    perm = c["permissive_thick_as_water"]
    return f"""# The ice classes, as they actually perform

> **Written by `make report`.** Do not edit by hand: the sources are
> `artifacts/metrics.json`, `configs/assumptions.yaml` and `data/polaris_rv.csv`.
> Closes **A-11** in docs/decisions.md, *Early estimates vs. measurements*: the June 2026
> prototype listed five classes, one of which — *Deformed Ice / Ridges* — does not exist
> in any AI4Arctic label.

Active scheme: **{c["scheme"]}**. mIoU **{_n(c["miou"])}**, ice/water
**{_n(c["binary"])}**, on **{c["n_scenes"]}** scenes of the official test set.

## Summary table

{chr(10).join(summary)}

`RIO = C × RIV(type) + (10 − C) × RIV(open water)` — IMO MSC.1/Circ.1519.

The RIV is the reason the table cannot stop at the ice type: the same class is normal
navigation at low concentration and forbidden at high concentration. That is the lesson of
docs/decisions.md D-031, and it is also why the meaningful question is not the *number of
classes* but *implementing an IMO circular*.

## What the summary table does not show

{chr(10).join(detail)}

**The three classes are not equally reliable, and that is worth stating up front.**

But the right question is not how often it is wrong: it is **in which direction**. Thin ice
is poorly recognised, and it is almost always called **thick** — that is, more dangerous than
it is. That costs a longer route and nothing else.

The error that matters is the opposite one: **thick ice called open water, {_n(perm, "{:.1f}%")}
of its pixels.** It is the only one that could send a ship where it may not go, and it is the
number to publish.

Over the whole scene, the fraction of the map where we are **more permissive** than the
analyst's chart has median **{_n(c["unsafe_median"], "{:.1%}")}** and worst case
**{_n(c["unsafe_max"], "{:.1%}")}** over {c["n_unsafe_scenes"]} scenes.

**And the worst case is in the demo, marked.** It is not a figure tucked away in a footnote:
it is one of the scenes that can be selected. Showing your own worst case is what makes
everything else credible — and it leaves nobody the chance to find it first.

## Why three classes and not six?

Because six is past the point where the labelled data holds up. The six-class model stopped
at **mIoU 0.272** while being confidently wrong (recorded in docs/decisions.md D-018; the
artefact of that evaluation is no longer in the repository, the entry is). With three
classes it scores {_n(c["miou"])}, and on the distinction that decides safety — ice or no
ice — it scores **{_n(c["binary"])}**.

The fall-back to three classes was planned from the start, for the case where the six SOD
classes did not hold up, and is documented in docs/decisions.md D-020. **Caution:** these
numbers must not be compared with the 0.60 mIoU acceptance threshold, which was defined on
the six SOD classes.

## What about ridges?

They do not exist as a label. AI4Arctic annotates concentration (SIC), stage of development
(SOD) and floe size (FLOE): **none of the three encodes deformation**. It is not a class the
model gets wrong, it is one it cannot learn.

The right response is not to drop the row and say nothing: it is to **state it as the next
problem**. "No public dataset labels deformation — it is the next problem, not a result of
ours" shows where the limit is, which is more convincing than one extra class.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the ice-class table for the documentation.")
    ap.add_argument("--metrics", type=Path, default=METRICS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics.exists():
        log.warning("%s missing: the class table is not written.", args.metrics)
        return 0
    metrics = json.loads(args.metrics.read_text())
    assumptions = yaml.safe_load(ASSUMPTIONS.read_text()) if ASSUMPTIONS.exists() else {}

    c = collect(metrics, assumptions, RV_TABLE)
    if not c["rows"]:
        log.warning("No classes in configs/assumptions.yaml: nothing to write.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(c), encoding="utf-8")
    log.info(
        "%s — %d classes, permissive %s",
        args.out,
        len(c["rows"]),
        _n(c["permissive_thick_as_water"], "{:.1f}%"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
