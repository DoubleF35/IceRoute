# The ice classes, as they actually perform

> **Written by `make report`.** Do not edit by hand: the sources are
> `artifacts/metrics.json`, `configs/assumptions.yaml` and `data/polaris_rv.csv`.
> Closes **A-11** in docs/decisions.md, *Early estimates vs. measurements*: the June 2026
> prototype listed five classes, one of which — *Deformed Ice / Ridges* — does not exist
> in any AI4Arctic label.

Active scheme: **3 classes (fallback, docs/decisions.md D-020)**. mIoU **0.532**, ice/water
**0.906**, on **20** scenes of the official test set.

## Summary table

| Class | What it means to a bridge | RIV PC6 | RIV PC7 | RIV IA |
|---|---|---|---|---|
| **Open water** | Navigate freely | +3 | +3 | +3 |
| **Thin ice** | POLARIS decides — depends on concentration | +1 | +1 | +0 |
| **Thick ice** | POLARIS decides — often forbidden at high concentration | -3 | -3 | -5 |

`RIO = C × RIV(type) + (10 − C) × RIV(open water)` — IMO MSC.1/Circ.1519.

The RIV is the reason the table cannot stop at the ice type: the same class is normal
navigation at low concentration and forbidden at high concentration. That is the lesson of
docs/decisions.md D-031, and it is also why the meaningful question is not the *number of
classes* but *implementing an IMO circular*.

## What the summary table does not show

| Class | Measured IoU | pixels | when wrong, it calls it | POLARIS type |
|---|---:|---:|---|---|
| Open water | 0.846 | 168 M | thick ice in 6% of cases | `ice_free` |
| Thin ice | 0.136 | 25 M | thick ice in 74% of cases | `thin_fyi_2nd_stage` |
| Thick ice | 0.615 | 109 M | open water in 14% of cases | `heavy_multi_year_ice` |

**The three classes are not equally reliable, and that is worth stating up front.**

But the right question is not how often it is wrong: it is **in which direction**. Thin ice
is poorly recognised, and it is almost always called **thick** — that is, more dangerous than
it is. That costs a longer route and nothing else.

The error that matters is the opposite one: **thick ice called open water, 13.7%
of its pixels.** It is the only one that could send a ship where it may not go, and it is the
number to publish.

Over the whole scene, the fraction of the map where we are **more permissive** than the
analyst's chart has median **0.9%** and worst case
**41.3%** over 20 scenes.

**And the worst case is in the demo, marked.** It is not a figure tucked away in a footnote:
it is one of the scenes that can be selected. Showing your own worst case is what makes
everything else credible — and it leaves nobody the chance to find it first.

## Why three classes and not six?

Because six is past the point where the labelled data holds up. The six-class model stopped
at **mIoU 0.272** while being confidently wrong (recorded in docs/decisions.md D-018; the
artefact of that evaluation is no longer in the repository, the entry is). With three
classes it scores 0.532, and on the distinction that decides safety — ice or no
ice — it scores **0.906**.

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
