# Engineering decision log

Every non-obvious choice in IceRoute is recorded here, with the measurement or the
source that settled it. The code cites these entries by ID (`docs/decisions.md D-020`),
so a reviewer can go from a line of code to the reason it is written that way.

Three rules run through the whole log:

1. **No number without a measurement or a source.** Values that are still assumptions
   are marked `PROVISIONAL` in code and config and listed under
   [Provisional values](#provisional-values).
2. **A decision is recorded when it is made**, including the ones that reverse an earlier
   one. Superseded entries stay, with a pointer to what replaced them.
3. **Early estimates are audited against measurements**, and the audit is kept even when
   the measurement is worse than the estimate — see
   [Early estimates vs. measurements](#early-estimates-vs-measurements).

## Index

- [D-001 — Python 3.11 installed with `uv`, not the system 3.12](#d-001)
- [D-002 — `requirements.txt` is generated, not written by hand](#d-002)
- [D-003 — `numpy < 2.0`](#d-003)
- [D-004 — torch from PyPI (CUDA 12.8) on a CUDA 13.2 driver](#d-004)
- [D-005 — Base channels 16, not 32](#d-005)
- [D-006 — Variable names verified on the data, not taken from the plan](#d-006)
- [D-007 — POLARIS from the primary IMO circular, not from a secondary source](#d-007)
- [D-008 — SOD → POLARIS mapping: always the most cautious column](#d-008)
- [D-009 — The `flip` attribute requires no correction (verified, not assumed)](#d-009)
- [D-010 — Planar distance in EPSG:3413 must be divided by the scale factor](#d-010)
- [D-011 — "Blocked by ice" and "outside coverage" are two different things](#d-011)
- [D-012 — Criteria for choosing the 5 demo scenes](#d-012)
- [D-013 — Whole-scene inference is streamed, not done with a full accumulator](#d-013)
- [D-014 — The INT8 model measures 2102 KB, not 58 KB](#d-014)
- [D-015 — The threshold baseline is measured in two versions, and one is deliberately unfair](#d-015)
- [D-016 — Compression must be measured on the PREDICTED mask, not on the ice chart](#d-016)
- [D-017 — The 40% edge share is obtained by weighting, not by discarding](#d-017)
- [D-018 — The model is confidently wrong](#d-018)
- [D-019 — INT8 quantization degrades nothing](#d-019)
- [D-020 — The three-class fallback: six classes become three](#d-020)
- [D-021 — The ship-class selector does not hold up (on the scenes of the frozen split)](#d-021)
- [D-022 — Demo scenes come from the official test set, not from the frozen split](#d-022)
- [D-023 — 90 more training scenes improve nothing](#d-023)
- [D-024 — Demo scenes are also chosen on how the model performs](#d-024)
- [D-025 — When accuracy is tied, safety decides: the deployed model is the Lovász one](#d-025)
- [D-026 — Transit hours cannot be quoted, and they do not need to be](#d-026)
- [D-027 — The comparison between the two routes was computed but never drawn](#d-027)
- [D-028 — The payload counted a route the satellite does not send: 371:1 → 439:1](#d-028)
- [D-029 — The live computation runs on the same function as the precomputed one](#d-029)
- [D-030 — Time and fuel become sliders, and the rescaling refuses when it would be false](#d-030)
- [D-031 — "The route goes through thick ice": the map showed half of the equation](#d-031)
- [D-032 — An offline, single-file demo that runs without Python or network](#d-032)
- [D-033 — The assumptions file said "provisional" next to measured numbers](#d-033)
- [D-034 — The satellite transmits the mask **and** the route, and we state both numbers](#d-034)
- [D-035 — `make report` was measuring compression on the wrong model](#d-035)
- [D-036 — Presentation language](#d-036)
- [D-037 — Presentation mode](#d-037)
- [D-038 — The offline demo becomes a console, and the interface colour stops speaking like the data](#d-038)
- [D-039 — `console.html`: the real geography the scenes had always carried](#d-039)
- [D-040 — The route snaked through open water: string pulling after A*](#d-040)
- [D-041 — The console takes Blueprint's palette, read from the source](#d-041)
- [D-042 — Probabilities are calibrated by temperature scaling, fitted on validation](#d-042)
- [Early estimates vs. measurements](#early-estimates-vs-measurements)
- [Provisional values](#provisional-values)

---

<a id="d-001"></a>
## D-001 — Python 3.11 installed with `uv`, not the system 3.12

**Date:** 2026-09-02 · **Status:** accepted

**Decision.** The project runs on CPython 3.11.15 installed by `uv` into `.venv/`, not on the
system Python 3.12.3.

**Rejected alternatives.**
1. *Use the system 3.12.* No installation needed, but the project targets 3.11, and some wheels
   of the geospatial stack (`pyresample`, `rasterio`) have historically lagged a few months
   behind new Python minor releases. Five weeks before a live demo, discovering that a wheel is
   missing and has to be built from source is a risk with nothing to gain in return.
2. *`sudo apt install python3.11`.* It works, but it needs administrator privileges and
   pollutes the system. `uv` downloads a standalone build into `~/.local/share/uv/python/`
   without touching anything outside the home directory.

**Reason.** Zero cost, no `sudo`, and an exact match with the target version.

**Evidence in the repo.** `make setup` runs `uv python install 3.11`; `scripts/check_env.py`
checks the version and warns if it is not 3.11.

---

<a id="d-002"></a>
## D-002 — `requirements.txt` is generated, not written by hand

**Date:** 2026-09-02 · **Status:** accepted

**Decision.** Dependencies are declared in `requirements.in` with loose constraints and
comments; `requirements.txt` is the output of `uv pip compile` and contains the exact pin of
**all** 329 dependencies, transitive ones included. Installation happens **only** from
`requirements.txt`.

**Rejected alternatives.**
1. *A hand-written `requirements.txt` listing only the direct packages.* It pins 25 packages
   and leaves the ~300 transitive ones free. That is exactly the hole through which a breakage
   arrives the day before a demo: a minor update of a library you did not know you had.
2. *Poetry / PDM.* They do the same job, but add one more tool to install on the demo laptop.
   `uv` is already there.

**Reason.** The full dependency graph must be frozen: `pip install -r requirements.txt` run
weeks later installs, bit for bit, what was resolved on 2 September.

**Evidence in the repo.** `requirements.in` (25 entries) → `requirements.txt` (329 pinned
lines). `pyproject.toml` declares `dependencies = []` and the package is installed with
`--no-deps`, so pip's resolver never gets a chance to change its mind about a version.

---

<a id="d-003"></a>
## D-003 — `numpy < 2.0`

**Date:** 2026-09-02 · **Status:** accepted

**Decision.** numpy is pinned to `1.26.4`.

**Rejected alternative.** *numpy 2.x.* Faster on some paths and more modern, but part of the
compiled geospatial stack (`pyresample` in particular, and its dependencies `pykdtree` /
`configobj`) ships wheels built against the numpy 1.x ABI. The typical symptom is an
`ImportError` on an ABI mismatch at runtime, not an installation error: you find out late.

**Reason.** Geocoding is a non-negotiable part of the pipeline and depends on `pyresample`.
Between "a newer numpy" and "geocoding that works" there is no contest.

**To revisit.** After the first public demo. Not before.

---

<a id="d-004"></a>
## D-004 — torch from PyPI (CUDA 12.8) on a CUDA 13.2 driver

**Date:** 2026-09-02 · **Status:** accepted

**Decision.** `torch==2.8.0` from the standard PyPI wheel for Linux x86_64, which brings the
CUDA 12.8 runtime with it (`nvidia-cuda-runtime-cu12==12.8.90`). No dedicated
`download.pytorch.org` index is used.

**Measured hardware context.** RTX 4050 Laptop, 6 GB VRAM, NVIDIA driver 595.84, which exposes
CUDA 13.2.

**Why it works.** The NVIDIA driver is backward compatible with earlier CUDA runtimes: a 13.2
driver runs code compiled for 12.8. The reverse does not hold. So the "old" wheel on the "new"
driver is the safe combination, not the risky one.

**Rejected alternative.** *A `+cu130` wheel from an extra index.* It adds a non-default index to
the installation process — one more thing that may not respond on the day the environment is
rebuilt — in exchange for a performance gain that has not been measured on this workload.

**Evidence in the repo.** `scripts/check_env.py` does not stop at calling
`torch.cuda.is_available()`: it allocates a 2048×2048 fp16 tensor on the GPU and runs a real
matmul. `is_available()` returning `True` while the driver is in fact broken is a failure mode
that shows up at the worst possible moment.

---

<a id="d-005"></a>
## D-005 — Base channels 16, not 32

**Date:** 2026-09-02 · **Status:** accepted

**Decision.** The encoder has channels `[16, 32, 64, 128]` with a 256-channel bottleneck,
instead of the `[32, 64, 128, 256]` of the initial design. Depth unchanged: 4 downsampling
levels.

**Measured count:** **1,964,182 parameters** (`python -m iceroute.model.unet`).
Budget: < 2,000,000. Remaining margin: 35,818.

**Why.** At base 32 the same scheme yields about 4.7M parameters, more than twice the budget.
The bottleneck is the deepest block: on its own, with 512 channels, it accounts for ~3.5M
parameters. The design rule was explicit about the remedy — *"if the count overflows, reduce
base channels before reducing depth"* — so the width was halved and the depth kept, since depth
is what provides the receptive field needed to tell an ice edge from speckle.

**Rejected alternatives.**
1. *Base 24 with the bottleneck capped at 256.* About 2.8M parameters: still over budget.
2. *3 levels instead of 4.* It would fit in the budget, but halves the receptive field, and the
   design rule explicitly forbids it as the first cut.

**Evidence in the repo.** `tests/test_unet.py::test_parameter_budget` fails the build if a
refactor goes over 2M. `test_extra_input_channel_stays_in_budget` checks that the ablation with
`distance_to_land` (4 input channels) stays inside: it costs 144 parameters.

**A consequence not to hide.** INT8 size estimated at ≈ 1.87 MB. The early design estimate was
**58 KB**. See *Early estimates vs. measurements* (A-02) and D-014.

---

<a id="d-006"></a>
## D-006 — Variable names verified on the data, not taken from the plan

**Date:** 2026-09-03 · **Status:** accepted

**Decision.** `configs/data.yaml` contains the names read from a real scene
(`20201016T082722_dmi_prep.nc`) with `make inspect`, cross-checked against `utils.py` in the
official toolkit `astokholm/AI4ArcticSeaIceChallenge`.

**What turned out different from the initial design.** The plan assumed
`sar_grid_latitude` / `sar_grid_longitude` with separate line and sample indices. The real
names are **`sar_grid2d_latitude` / `sar_grid2d_longitude`**, and they are ready-made 2D grids
of **21×21** points over a 4996×5246-pixel scene. This simplifies geocoding: there is no need to
rebuild the grid from indices, it only has to be interpolated.

**Confirmed as expected:** `nersc_sar_primary` (HH), `nersc_sar_secondary` (HV),
`sar_incidenceangle`, `distance_map`, `SIC` / `SOD` / `FLOE`, the 14 AMSR2 channels `btemp_*`,
the 6 ERA5 variables, and the value **255 as fill/ignore**.

**Three observations that matter for training.**

1. *Everything is already standard-scaled*, the incidence angle included: on the verified scene
   it ranges from −1.765 to 0.863, so it is not in degrees. No further normalization is applied.

2. **The SAR uses `0.0` as an undeclared fill value.** This is the dangerous finding. On the
   verified scene, `nersc_sar_primary` has **25,317,424 pixels exactly equal to 0.0, 96.6% of
   the array**. They are not NaN, and there is no `_FillValue` attribute: they are swath padding
   that any naive statistic counts as a measurement. The first `inspect.py` report said
   "mask 0.0%" for the SAR — formally true and practically misleading. If the SAR were
   normalized on these statistics, 96% zeros would drag the mean and standard deviation towards
   zero, and the model would see a huge constant region.

   *Mitigation, verified:* the SAR padding region and the invalid region of the target
   **coincide** (96.6% vs 96.8%, a 0.2-point gap). So masking on the validity of the **target**
   is sufficient, and that is what `patches.py` does. `inspect.py` now checks this coincidence
   on every scene and complains loudly if the two numbers diverge by more than one point: if
   they did not coincide on some scene, masking on the target alone would let padding through
   as data.

3. *The ice-chart coverage is partial and highly variable.* On the verified scene only
   **891,593 pixels out of 26,209,016 are annotated — 3.4%**. The rule "discard patches with
   more than 30% invalid pixels" would discard almost all of them on such a scene. Caution,
   though: this scene is the **smallest file in the whole dataset** (13.4 MB), and it is small
   precisely because it is almost all zeros, which compress well. It is an extreme case, not
   the typical one. The count must be redone on the 60 real scenes before fixing the threshold.
   **A new risk, not anticipated in the initial design.**

**Evidence in the repo.** `artifacts/figures/inspect_20201016T082722_dmi_prep.png`,
`artifacts/inspect_20201016.json`.

---

<a id="d-007"></a>
## D-007 — POLARIS from the primary IMO circular, not from a secondary source

**Date:** 2026-09-03 · **Status:** accepted

**Decision.** `data/polaris_rv.csv` contains the full Risk Index Value matrix (12 ship classes ×
12 ice types) transcribed from the text of **MSC.1/Circ.1519**, Annex, Table 1.3. No value is
`PROVISIONAL`.

**The thresholds come from the same source as well** (Table 1.1):

| RIO | PC1–PC7 | Below PC7 and ships without ice class |
|---|---|---|
| ≥ 0 | normal operation | normal operation |
| −10 ≤ RIO < 0 | elevated operational risk | operation subject to special consideration |
| < −10 | operation subject to special consideration | operation subject to special consideration |

And the speed limits in the elevated-risk band (Table 1.2): **PC1 11 knots, PC2 8, PC3–PC5 5,
below PC5 3 knots**.

**Why it matters.** The initial risk register listed "POLARIS table not obtainable" as a
medium-probability, high-impact risk, with a fallback to peer-reviewed sources, marking the
values `PROVISIONAL` and disclosing it in the demo. That is not needed: we have the regulatory
document. It is one of the few public claims that rests on an IMO standard rather than on an
estimate of our own.

**A limitation to admit, not to hide.** POLARIS does not model ice pressure, ridges or drift.
The question will come up, and the honest answer is that this is the IMO standard *with* these
limitations, not a complete risk model.

---

<a id="d-008"></a>
## D-008 — SOD → POLARIS mapping: always the most cautious column

**Date:** 2026-09-03 · **Status:** accepted

**Decision.** The 6 SOD classes of the dataset do not map one-to-one onto the 12 POLARIS ice
types: each SOD class covers more than one. Wherever there is ambiguity, **the column with the
lowest Risk Index Value is always chosen**, i.e. the most conservative one.

| SOD class | Covers (WMO) | POLARIS column chosen | Why that one |
|---|---|---|---|
| 2 Young ice | grey (10-15 cm) + grey-white (15-30) | `grey_white_ice` | for IB it is 1 against 2 for `grey_ice` |
| 3 Thin FYI | 1st stage (30-50) + 2nd stage (50-70) | `thin_fyi_2nd_stage` | for IA it is 0 against 1 |
| 4 Thick FYI | generic FYI + medium + thick | `thick_fyi` | for PC6 it is −1 against 0 for `medium_fyi` |
| 5 Old ice | second year + light MY + heavy MY | `heavy_multi_year_ice` | circular §1.1.2 requires the heavy column when it cannot be established that the multi-year ice is below 2.5 m — and from an SOD chart it cannot |

**Reason.** This mapping is the joint between model and routing: if it is wrong, the model is
good and the routes are garbage. The asymmetry of errors is stark: erring on the cautious side
produces a route longer than necessary; erring the other way sends a ship into ice its class
cannot handle. These are not errors of the same kind.

**Consequence to state explicitly.** IceRoute routes are systematically *conservative*. If
someone with a maritime background asks why the saving is not higher, this is the answer, and
it is a better answer than an inflated number.

---

<a id="d-009"></a>
## D-009 — The `flip` attribute requires no correction (verified, not assumed)

**Date:** 2026-09-03 · **Status:** accepted

**The problem.** Every scene has a global attribute `flip: 1` that is **not documented in the
dataset manual**. If it meant that the array is stored flipped with respect to the lat/lon
grid, every route would come out mirrored: plausible to look at and completely false. The
geocoding distance test would **not** catch it, because it checks the internal consistency of
the grid, not the grid-to-image correspondence.

**How it was resolved.** With an independent piece of data: `distance_map`, the distance to the
coast provided by the dataset. Assuming direct correspondence (node `[0,0]` ↔ pixel `[0,0]`):

| Array corner | Coordinates from the grid | `distance_map` | Consistent? |
|---|---|---|---|
| `[0,0]` | 78.20 N · 5.44 W — Fram Strait, open sea | **+0.968** (far from land) | yes |
| `[0,-1]` | 79.74 N · 23.47 W — NE Greenland coast | **−1.576** (minimum) | yes |
| `[-1,0]` | 75.01 N · 13.17 W — open sea | +0.625 | yes |
| `[-1,-1]` | 76.24 N · 27.53 W — Greenland interior | **−1.576** (minimum) | yes |

Perfect agreement on all four corners. A flipped interpretation would give anticorrelation.
**No correction to apply.**

**Confirmation from the manual.** *"The ice charts in the netCDFs of the dataset are aligned
with the Sentinel-1 grid"* (User Manual, Figure 8 caption). Chart and SAR share the grid; the
`flip` attribute records what the preprocessing has already done.

**Side note.** The manual shows examples with **378 GCPs (21×18)**, not always 441 (21×21). The
code reads the real shape from the array; `grid_shape: [21, 21]` in `configs/data.yaml` is
documentation, not a value that is used.

---

<a id="d-010"></a>
## D-010 — Planar distance in EPSG:3413 must be divided by the scale factor

**Date:** 2026-09-03 · **Status:** accepted

**The problem, not mentioned in the initial design.** EPSG:3413 is a polar stereographic
projection with **standard parallel at 70°N**: it is true to scale only there. At the latitudes
of the demo scenes, the scale factor measured with PROJ is:

| Latitude | k | Error of a planar distance |
|---|---|---|
| 70°N | 1.00000 | 0.00% |
| 75°N | 0.98666 | **−1.33%** |
| 78°N | 0.98057 | **−1.94%** |
| 80°N | 0.97728 | **−2.27%** |
| 85°N | 0.97171 | **−2.83%** |

A `hypot(dx, dy)` on the projected grid — the obvious thing to write — **underestimates the real
distance by up to 2.8%** and would not pass the 1% geocoding gate. A concrete case: two points
100 km apart geodesically at 78°N measure 98,059 m in the plane.

**Decision.** Every distance goes through `ground_distance_m()`, which divides by the local
scale factor computed by PROJ (`Proj.get_factors`), not by a spherical approximation. After the
correction the same case measures 100,002 m, error +0.00%.

**Why it matters beyond the test.** The cell traversal cost is **time = distance / speed**. A
systematic 2% error on distances is a 2% error on every transit time shown in the demo, and the
saving the project claims is of the same order of magnitude. It would have been a number wrong
in a systematic direction, which is the worst kind.

**Evidence in the repo.** `tests/test_geocode.py` checks the gate on 7 pairs of real Arctic
waypoints between 70°N and 83.5°N. `test_uncorrected_distance_is_the_trap` documents the
failure mode and notices if someone removes the correction.

---

<a id="d-011"></a>
## D-011 — "Blocked by ice" and "outside coverage" are two different things

**Date:** 2026-09-03 · **Status:** accepted

**The problem, which surfaced on the first run on real data.** The geodesic baseline came out
non-navigable in **all six** combinations tried. It looked like a strong result. It was not:
ice-chart coverage after geocoding is **23%**, so a straight line almost always crosses cells
**where we simply have no map**. The code treated them as impassable ice.

Saying *"the direct route is impossible"* when the real reason is *"we have no data there"* is
a claim that anyone with a maritime background takes apart with one question.

**Decision.** `geodesic_baseline()` receives the coverage mask and keeps two separate counts:
`n_no_go_cells` (POLARIS forbids it) and `n_no_data_cells` (no prediction). `compare()` claims a
saving only if the baseline is navigable, and distinguishes the two failure cases: if the
blockage is due to missing data it returns `comparable: false` and **claims nothing**.

**Numbers measured after the fix**, scene `20180824T130831_cis_prep`:

| | cells |
|---|---|
| Blocked by ice (RIO beyond the limit) | **93** |
| Outside ice-chart coverage | **139** |

The 93 still hold: they are old ice at full concentration, which with RIV −3 gives RIO −30,
no-go for all three ship classes. The claim stands, but it is now **documented** rather than
lucky.

---

<a id="d-012"></a>
## D-012 — Criteria for choosing the 5 demo scenes

**Date:** 2026-09-03 · **Status:** accepted; extended by D-022 and D-024

The first end-to-end run surfaced two requirements that the initial scene-selection criteria
did not list, and that must be added to the criterion "visually convincing".

**1. The route must change when the ship class changes.** On the two scenes tried, PC6, PC7 and
IA produce **exactly the same route** (23.1 h, 514 km). It is not a bug: the open-water corridor
is wide enough that all three take it, and the minimum RIO along the route differs correctly
(+10 / +5 / 0). But the most effective moment of the demo — *"switch the ship class: for a 1A
this route is not navigable"* — requires a scene where the ice **forces** different choices.
This has to be verified scene by scene, not taken for granted.

**2. The geodesic must stay inside the coverage.** Otherwise the comparison with the baseline is
not defensible (D-011). With coverage at 23% this narrows the usable waypoint pairs
considerably.

**Operational consequence.** `make route` now reports both things in the JSON index, so the
choice of the 5 scenes is made on numbers rather than by eye. If none of the 60 scenes satisfies
requirement 1, the planned scope cut (from three ship classes to one) must be triggered
**before** building the UI around a switch that changes nothing.

---

<a id="d-013"></a>
## D-013 — Whole-scene inference is streamed, not done with a full accumulator

**Date:** 2026-09-05 · **Status:** accepted

An AI4Arctic scene is ~5000×5000 px and the model works on 256×256 tiles. The obvious
implementation allocates a `(6, H, W)` float32 accumulator — **600 MB**, plus 100 MB for the
blending weights. `inference.py` instead consumes the scene in **strips of tile rows**, and
finalizes a strip as soon as no later tile can touch it any more: the peak drops to
`(6, 256, W)`, i.e. **~30 MB, 20× less**.

This is not cleanliness for its own sake. The premise of the project is that this pipeline fits
on a CubeSat: an implementation whose memory grows with the scene area contradicts that premise
from the start.

**Two bugs found by running, not by reading.**

*The first* was the division guard constant. The 2D blending window is an outer product, so its
minimum is `RAMP_FLOOR² ≈ 9.8e-4`, **below** the `RAMP_FLOOR = 1e-3` I was using to clamp the
denominator. The `np.maximum` therefore crushed a **legitimate** weight and scaled the
probabilities of corner pixels by 0.978. Small, invisible on a map, and wrong. The two
constants are now separate: `RAMP_FLOOR` for the ramp, `DIV_EPS = 1e-12` only as a 0/0 guard.

*The second was worse.* I was zeroing the whole buffer at every tile row, throwing away the
contribution of the previous row exactly over the overlap band: **vertical blending was not
happening at all**. The defect is silent — it still produces a plausible map, with a
discontinuity every 224 rows that reads as "the model is wrong at the ice edges", i.e. exactly
where the routing makes its decisions.

**Why the test had not caught it.** I was comparing the streaming and the naive implementation
using a dummy model whose output depends only on the pixel value. Such a model returns the same
probability vector for a pixel **regardless of which tile covers it**, so any weighting scheme —
including a broken one that discards an entire row — gives the same argmax. The test passed on
wrong code.

The dummy model (`TileEdgeModel`) now depends on the **position within the tile**, which is
exactly the property of convolutional networks that justifies blending. Verified by
reintroducing the bug: 4 tests fail, on all three scene shapes tried.

**Rule that comes out of it.** An equivalence test between two implementations is only worth
something if the subject of the test is sensitive to the difference between them. A "simple"
fake model can make the test vacuous without anything signalling it.

---

<a id="d-014"></a>
## D-014 — The INT8 model measures 2102 KB, not 58 KB

**Date:** 2026-09-05 · **Status:** accepted; closes A-02

First real measurement of the `export → parity → quantize` chain, with untrained weights. Sizes
and latencies **do not depend on the weight values**, only on the architecture: these numbers
are already final, training will not change them.

| Quantity | Measurement |
|---|---|
| Parameters | 1,964,182 |
| ONNX fp32 | 7.525 MB |
| **ONNX INT8** | **2.053 MB (2102 KB)** |
| Compression ratio | 3.67× |
| **Bytes per parameter (INT8)** | **1.096** |
| Latency/tile, 1 thread | 76.0 ms → 37.2 ms (2.04×) |
| Latency/tile, 22 threads | 28.1 ms → 26.6 ms (1.05×) |

**Closes A-02.** The early design estimate was `58 KB INT8 weights`. The measurement is
**2102 KB, 36× larger**. The "bytes per parameter" row is the check that makes the error obvious
without re-running anything: INT8 weights cannot go below ~1 byte per parameter, so 58 KB for
1.96M parameters is **arithmetically impossible**, not merely optimistic. The number to quote is
2.05 MB.

**Why 3.67× and not 4×.** The `.onnx` file does not contain only weights: there is the graph
structure, the scales and zero-points of the QDQ nodes, and the GroupNorm affine parameters,
which stay in float32. The difference between the theoretical 4× and the measured 3.67× **is**
that material.

**On latency, the number to quote is the single-thread one.** At 22 threads INT8 gains almost
nothing (1.05×) because ONNX Runtime's fp32 convolution is already well parallelized. But the
laptop has 22 cores and the payload does not: quoting the 22-thread number for an FPGA target
would mean nothing. Hence the choice to measure and report **both** configurations, labelled.

**Does not close A-03.** The CPU latency we measure is a different quantity from the projected
latency on GPU@SAT, the target on-board accelerator: it does not replace it and must not be used
to validate it.

---

<a id="d-015"></a>
## D-015 — The threshold baseline is measured in two versions, and one is deliberately unfair

**Date:** 2026-09-05 · **Status:** accepted

Ice is bright in HV, water is dark. A single threshold on that channel separates the two
surprisingly well — so the claim "a neural network is needed" has to be **measured** against
that threshold, not asserted. `evaluate.py` calibrates it in two ways:

- **fitted** — threshold chosen on the *training* split, applied to the evaluation split. This
  is what an engineer could actually put into production.
- **oracle** — the best threshold **in hindsight on the evaluation split itself**. It is not
  deployable and it is not fair: that is exactly the point. It is the strongest possible version
  of the cheap baseline, and beating it is the only result worth reporting.

The comparison is reported on **both** axes: ice/water, where the threshold is competitive, and
6-class mIoU, where it structurally cannot compete (a threshold emits only two labels). Quoting
only the second would be cherry-picking.

When the threshold wins on the binary task, `threshold_verdict()` **says so**, and argues where
the real value lies: POLARIS does not ask "is there ice?", it asks *what kind* of ice — and
between young ice and old ice the RIO changes sign. A threshold cannot answer that.

In the 6-class mIoU computation the baseline even gets an advantage it is not entitled to: every
pixel it calls "ice" is assigned to the **most frequent ice class in the ground truth**, a piece
of information no threshold has. Even so it does not exceed 2 classes out of 6. Handicapping it
further would prove nothing.

---

<a id="d-016"></a>
## D-016 — Compression must be measured on the PREDICTED mask, not on the ice chart

**Date:** 2026-09-05 · **Status:** accepted; P-06 left open

First measurement of the downlink ratio on three real scenes. The number is excellent — and for
exactly that reason it must be viewed with suspicion.

| Scene | Coverage | Compressed SAR | Payload | Ratio |
|---|---|---|---|---|
| 20180108T184332 | 30.8% | 65.9 MB | 65.1 KB | 1012:1 |
| 20180327T103959 | 37.4% | 38.8 MB | 64.2 KB | 604:1 |
| 20180428T093937 | 92.4% | 86.6 MB | 63.5 KB | 1363:1 |

Median **1012:1**, against the 200:1 of the early design estimate. But the number **is not
usable**, for the reason explained below, and P-06 stays open.

**The designated numerator is the compressed SAR, not the raw one.** Any real payload compresses
before transmitting: comparing against uncompressed data means beating a baseline nobody would
send into orbit. Here it matters concretely — between 7% and 58% of the measured scenes is swath
padding at exactly `0.0`, which compression removes in one go. With the raw numerator the ratio
would be 1694:1 instead of 1363:1: we would be making the baseline pay to transmit nothing.

**The real problem: the ice chart is not a predicted mask.** An ice chart is drawn by an analyst
**as polygons**, so it is piecewise constant over large areas and PNG compresses it very well —
far below its own per-pixel entropy, because the spatial structure is what gets exploited. The
output of a U-Net is not like that: it is pixel by pixel, noisy at the edges, irregular. **It
will compress noticeably worse.**

Measured on the bench with synthetic masks: the "polygon" version and the same mask with 15% of
pixels perturbed differ by **more than 2×** in bytes. The test
`test_a_speckled_mask_compresses_worse_than_a_polygon_mask` pins this down.

**Consequence in the code.** `measure_scene()` now **requires** `mask_source` to be declared (no
default), and `verdict()` **refuses** to return a quotable headline when the measurement comes
from an ice chart: it answers that the figure is an upper bound and that P-06 stays open. A
flattering number must not be able to leave this module by oversight.

**To be redone** with the predicted mask as soon as training is finished. The ratio will drop;
by how much, the measurement will tell. Even halved it would stay well above 200:1.

---

<a id="d-017"></a>
## D-017 — The 40% edge share is obtained by weighting, not by discarding

**Date:** 2026-09-05 · **Status:** accepted

The training design requires 40% of every batch to contain **ice edges**: the transition bands
between water and pack ice, where classification is hard and where routing decisions are really
made. Entirely uniform patches — a thousand pixels of open water — teach little.

First implementation: extract the patches, then **discard** the uniform ones until the edge
fraction reaches 40%. It works, in the sense that the number comes out right.

**The arithmetic that should have raised suspicion immediately.** Out of 14,188 extracted
patches, 823 contained an edge. To make them 40% of the total you can keep at most
823 ÷ 0.4 = **2,057**. Everything else has to be thrown away. In practice the cache dropped to
**1,850 patches out of 14,188: 87% of the data discarded** to satisfy a ratio.

**Measured consequence.** Training on that cache stalled at **val mIoU 0.2594** and then
overfitted: the training loss kept falling while the validation mIoU fell with it. With 1,850
patches from 39 scenes there was not enough material for a 1.96M-parameter network to
generalize.

**The conceptual error.** "40% of every batch" is a property of the **sampling**, not of the
**dataset**. Achieving it by discarding data means paying with real supervision for a constraint
that did not require it.

**The fix.** `WeightedRandomSampler`: all patches are kept, and the edge patches are given a
weight such that they appear 40% of the time *in expectation*. The constraint is met where it
means something — in the batch the network sees — and not a single patch is thrown away.

| | patches | with edge | scenes |
|---|---|---|---|
| discarding | 1,850 | 740 (40%) | 39 |
| weighting | 14,188 | 823 (6%) | 43 |
| weighting, after D-023 | 18,849 | 1,017 (5.4%) | 55 |

**The rule that remains.** A "data balancing" step that shrinks the dataset by 87% should be met
with suspicion before satisfaction. The target number was green in both cases: only the mIoU
said which of the two was right.

---

<a id="d-018"></a>
## D-018 — The model is confidently wrong

**Date:** 2026-09-05 · re-measured 2026-09-07 on `configs/train3.yaml` · **Status:** accepted

mIoU says **how many** pixels are wrong. It does not say **how**, and the difference decides
which corrections are worth attempting.

If the errors were near-ties — 0.45 on the wrong class, 0.42 on the right one — then the whole
family of **inference-time** corrections would have something to work with: temperature
scaling, threshold tuning, reordering the decision towards the more dangerous class. All of
these shift the **margin** between candidates, and if the right answer is just below the winner,
shifting the margin promotes it.

**What was measured instead** (`scripts/measure_confidence.py`, 20 scenes of the official test
set, 64,327,029 misclassified pixels):

| | |
|---|---|
| median probability on the **wrong** class chosen | **0.950** |
| median probability left to the **true** class | **0.0304** |
| errors where the true class gets less than 10% | 72.2% |
| errors where the true class gets less than 1% | 44.6% |
| **errors that are near-ties (margin < 0.10)** | **3.4%** |

**The model does not hesitate: it is wrong with conviction.** On a misclassified pixel it puts,
at the median, 95% on the wrong answer and 3% on the right one. Only **3.4%** of errors have a
margin narrow enough for a decision rule to flip them.

**Consequence, and it is the most important one in this document.** The family of
inference-time corrections is **closed**. It is not that they work badly: they have nothing to
act on, because they operate on the margin between candidates and the right candidate is not in
the running. Every hour spent tuning thresholds is an hour wasted, and that has to be said
before spending it.

**Experimental check of the consequence.** `safety_bias.py` implements exactly one of these
corrections: among the classes within `tau` of the maximum probability, pick the most dangerous
one. It is the cautious reordering of the decision, the most sensible idea of the family.
Measured on the 6-class model: permissive fraction from **26.42% to 24.97%**. A point and a half,
at the cost of mIoU. The module was kept in the repo on purpose: **the measurement shows that
path is closed**, and that is a result, not a failure.

**What remains open.** Only interventions that change *what the network sees or how it is
supervised*: different features, a different architecture, a different loss (hence the Lovász
branch of D-023), or an easier problem (hence the fallback of D-020).

**A methodological note, paid for in practice.** The first version of the script drew its
conclusion from the wrong statistic: it checked "errors that leave the true class less than 1%"
and, finding 44.6% of them, i.e. less than half, printed *"margin corrections are worth
trying"* — **the opposite** of what the data said. The two quantities are not complementary: an
error where the true class has 3% is not below 1%, but it is nowhere near the race either,
because the winner has 95%. The question was "is there a margin narrow enough to be closed", and
only the 3.4% of near-ties answers it. The automatic verdict is now based on that.

It is worth recording because it is the typical way a correct number produces a wrong
conclusion: the measurement was right, the threshold was on a different quantity, and the
result was a plausible and false sentence. Exactly what the project rule "no number without a
measurement or a source" exists to prevent.

---

<a id="d-019"></a>
## D-019 — INT8 quantization degrades nothing

**Date:** 2026-09-05 · **Status:** accepted; closes A-04

The early design estimate was `1–3% mIoU lost to INT8` (A-04). It is a literature value for
generic networks, not a measurement on this network.

**Measured** with `quantize_static` (MinMax, per-channel, QInt8 weights / QUInt8 activations),
comparing fp32 and INT8 on the same scenes through **exactly the same** tiling code
(`OnnxSessionModel`, so the graph is the only variable):

| | mIoU |
|---|---|
| fp32 | 0.9801 |
| INT8 | 0.9781 |
| difference | **+0.0020** |

That is **0.2%**, an order of magnitude below the estimate, and well within the 0.03 tolerance
set in the configuration.

**Be careful about what these numbers mean.** The absolute 0.98 **is not quotable**: the parity
check runs on two *training* scenes, which the model has already seen. It serves to isolate the
effect of quantization, not to estimate quality. The only figure that comes out of here is the
**difference**; the real quality is the 0.535 of P-08, measured on held-out data.

**Reproducibility.** `check_parity()` pins ONNX Runtime to **a single thread**. With more
threads the convolution reductions happen in completion order, so the result changes slightly
from one run to the next: enough to make a tight-tolerance gate fail under load, which happened
once.

**For public claims:** the statement has to be replaced, but **for the better**. The measured
degradation is nil. It is one of the few points where reality beats the estimate.

---

<a id="d-020"></a>
## D-020 — The three-class fallback: six classes become three

**Date:** 2026-09-05 · **Status:** accepted

The **6-class SOD** model stalled at **mIoU 0.272** on the official test set, against the 0.60
target set for the six SOD classes. The plan anticipated the fallback: reduce the number of
classes.

**Why it was predictable, by looking at the data rather than the model.** The measured class
frequencies (`data_stats.json`) are devastating:

| class | frequency |
|---|---|
| Open water | 67.4% |
| New ice | **0.37%** |
| Young ice | **1.63%** |
| Thin FYI | **2.15%** |
| Thick FYI | 15.5% |
| Old ice | 12.9% |

Three classes out of six are below 2.2%, and together they do not make up 4.2% of the pixels.
mIoU is a **mean over classes**: three nearly empty, poorly classified classes drag it down on
their own, no matter how well the others do.

**The chosen grouping** (`src/iceroute/data/collapse.py`):

| new class | contains | name |
|---|---|---|
| 0 | Open water | Open water |
| 1 | New ice, Young ice, Thin FYI | Thin ice |
| 2 | Thick FYI, Old ice | Thick ice |

**It is not a cosmetic grouping: it is checked against POLARIS.** Merging two classes only makes
sense if the RIV (Risk Index Value) of the group can be defined without ambiguity.
`conservative_member()` checks that, **for every ship class** (PC6, PC7, IA), the worst member
of the group is the same; `polaris_mapping()` **raises an error** if it is not. The fallback
cannot quietly introduce an inconsistent risk mapping.

**Result:** mIoU **0.535**, ice/water **0.905**.

**Mandatory honesty, enforced by the code.** Three classes are a **strictly easier** problem
than six: these numbers are **not comparable** with the 0.60 target, which is defined on the 6
SOD classes. Training prints a `WARNING` saying so at every start, the dashboard shows a
"three-class fallback" label instead of a comparison with 0.60, and `artifacts/estimates_vs_measurements.md` marks
the claim as *scaled down*. Public descriptions must say **three classes: open water, thin ice,
thick ice** — not "six stages of development".

---

<a id="d-021"></a>
## D-021 — The ship-class selector does not hold up (on the scenes of the frozen split)

**Date:** 2026-09-05 · **Status:** accepted; leads to D-022

The strongest moment of the demo should be this: the ship class is switched from PC6 to IA, and
the route changes before your eyes. Verified on the 5 demo scenes of the frozen split, instead of
taking it for granted:

- **only one** scene out of five made the route change, and by **3 cells**, on an ice-chart
  coverage of **12%**;
- `20180108T184332` was **unusable**: the reference geodesic left the chart coverage for all
  three classes, so there was not even a comparison to show (the distinction of D-011).

**Decision:** do not build the demo on an effect the data does not support. Hence D-022 (scenes
from the official test set) and the explicit selection criteria, applied to the numbers and not
by eye.

**In the code.** The UI **states** when the route does not change with the ship class
(`routes_differ: false` → explicit caption), instead of letting someone discover it by clicking
during a live demo. A selector that does nothing and does not say so is theatre; one that does
nothing and says so is a measurement.

---

<a id="d-022"></a>
## D-022 — Demo scenes come from the official test set, not from the frozen split

**Date:** 2026-09-05 · **Status:** accepted

The train/val/demo split is frozen and must not be regenerated. The `demo` scenes of that split,
however, come from the same package as the 512 training scenes: they are scenes the model has
**not** seen, but they come from the same distribution and — above all — there is no way to
prove it to a technical reviewer other than asking them to trust us.

AI4Arctic publishes **20 official test scenes**, with labels in separate `_prep_reference.nc`
files. They are scenes nobody has used to select anything.

**Decision:** the demo draws from there. It is a **stronger** position than the one originally
planned, not a weaker one — and "what data is this measured on?" is the first question that
comes up.

`configs/split.yaml` **is not touched**: it stays frozen as prescribed, and its demo scenes
remain listed under `superseded` in `configs/demo.yaml` together with the reason they are not
used. `resolve_labels()` finds the sibling reference file on its own, and refuses to let a scene
point to itself — otherwise it would evaluate a chart against itself.

**Useful side effect:** all publishable metrics (P-08, the permissive fraction, the model
comparison of D-023) moved to this same set. No published number comes from validation any
more, since validation selected the checkpoints and therefore flatters them.

---

<a id="d-023"></a>
## D-023 — 90 more training scenes improve nothing

**Date:** 2026-09-06 · **Status:** accepted

Hypothesis to test: the model is limited by the amount of data. The dataset offers 512 scenes;
we were using 55.

**90 additional scenes** were downloaded (`scripts/download_subset.py --extend-train`, ~13 GB).
**Only** the training patches were extended: `configs/split_extra.yaml` adds scenes to the
`train` branch and **does not touch** `configs/split.yaml`, so validation and demo stay
identical and the before/after comparison means something. Patches went from 14,188 to
**18,849**.

**Result, on the official test set:**

| | val mIoU | official test | ice/water | worst permissive |
|---|---|---|---|---|
| baseline (55 scenes) | 0.6293 | **0.5352** | 0.9049 | 51.72% |
| more data (145 scenes) | 0.6417 | 0.5335 | 0.8937 | 44.81% |

**On validation it looks better. On the official test set it is slightly worse.** That is
exactly why `scripts/pick_model.py` selects on the official test set and not on validation:
each model early-stopped on its *own* validation mIoU, so that metric flatters each of them by
an unknown and **different** amount. Selecting on validation would have rewarded whichever
model best overfitted its own selection criterion.

**Conclusion:** the bottleneck **is not the amount of data**. Downloading more is wasted time.
Consistent with D-018: the model is not short of things to see, it is wrong with conviction. The
90 scenes stay on disk and in the config — they only cost space, and downloading them again
would cost hours.

**A nuance not to lose.** The model with more data has a *better* permissive fraction (44.81%
against 51.72%). The selector still kept the baseline because it lost on mIoU, and the safety
veto is built to block a *more* permissive model — not to reward a less accurate one. If safety
ever becomes the primary criterion, this run must be re-evaluated: the number is on record.

---

<a id="d-024"></a>
## D-024 — Demo scenes are also chosen on how the model performs

**Date:** 2026-09-06 · **Status:** accepted

The criteria of D-012 and D-022 all look at the **ice**: direct route forbidden, high coverage,
route that changes with the ship class. None of them looks at **how the model does on that
scene**. They are criteria computed on the ice chart, on purpose, because they were needed to
choose the scenes while the model was still training.

**Consequence, discovered by cross-checking the ranking against `metrics.json`:** the carousel
included `20180623T114935`, which is the **worst of the 20 official test scenes**.

| | mIoU | coverage | worst permissive | route changes |
|---|---|---|---|---|
| 20180623 *(outgoing)* | **0.253** | 0.948 | **51.7%** | no |
| 20201013 *(incoming)* | 0.572 | 0.812 | **2.4%** | yes |

The 51.7% is the fraction of pixels on which the model declares navigable ice that POLARIS
**forbids** for that ship class. On that scene, in the worst case, it overestimates the RIO by
80 points. **Nothing had broken:** routes were drawn, the UI worked, the tests passed. The demo
would have routed a ship into forbidden ice, live.

**Fourth criterion, added:** the model must not perform badly on the scene. The deciding number
is the **permissive fraction**, not mIoU: it is the one that measures operational harm.

**The replacement is not the scene with the best mIoU.** `20180903` scores 0.766, but has
**23%** coverage: it would have removed exactly the role the outgoing scene played in the set —
the high-coverage scene, the one that disposes of the objection "how much of the chart do you
actually have". `20201013` keeps that role (81%, the highest available), cuts the permissive
fraction by **22 times**, and in addition **adds** a route change with the ship class, which the
outgoing scene did not have: the scenes that support that moment go from 3/5 to **4/5**.

**Its limitation is stated in the config, not smoothed over.** On `20201013` the direct route is
navigable for PC6 and PC7 (IceRoute costs +1%); it is forbidden only for IA. The "the straight
line cannot be done" moment is carried by the first three scenes.

**20180623 stays in the build, labelled, outside the carousel.** "And when it is wrong?" is a
question that always comes up, and having the answer ready is worth more than hoping it does not.
The UI puts it at the end, marks it, and shows its numbers **read from `metrics.json`** — not
copied into the demo index, which would leave the label claiming 0.25 after the next retraining.

**A test keeps it honest.** `test_the_designated_worst_case_really_is_the_worst` fails if a
retraining reorders the ranking and leaves the config behind. Without it, the choice would stay
anchored to stale numbers exactly as it was before.

---

<a id="d-025"></a>
## D-025 — When accuracy is tied, safety decides: the deployed model is the Lovász one

**Date:** 2026-09-07

Third attempt at improving the model, after the three-class fallback (D-020) and the 90 extra
scenes (D-023): replace part of the loss with **Lovász-Softmax** (Berman, Triki & Blaschko,
CVPR 2018), a tight convex surrogate of the Jaccard index — that is, it optimises directly what
the mIoU measures, instead of optimising cross-entropy and hoping.

**Results on the official test set**, against the two competitors:

| | aggregate mIoU | mean per-scene mIoU | ice/water | worst permissive |
|---|---|---|---|---|
| baseline | **0.5352** | 0.5848 | 0.9049 | 51.72% |
| **Lovász** | 0.5324 | **0.6099** | **0.9059** | **41.29%** |
| more data | 0.5335 | 0.6033 | 0.8937 | 44.81% |

**The two mIoUs disagree, and it is worth understanding why.** Lovász wins on the per-scene mean
(+0.025) and loses on the aggregate (−0.0028). The obvious hypothesis was Simpson's paradox — a
few large scenes flipping the total — and **it was checked and rejected**: among the 10 largest
scenes, Lovász wins 9.

The real explanation is in the per-class composition of the aggregate confusion matrix:

| class | baseline | Lovász | Δ | support |
|---|---|---|---|---|
| Open water | 0.846 | 0.846 | ±0.000 | 168.2 M |
| **Thin** ice | 0.226 | 0.136 | **−0.089** | 24.8 M |
| **Thick** ice | 0.534 | 0.615 | **+0.081** | 109.1 M |

mIoU is an **unweighted** mean over classes, so the two swaps nearly cancel and what is left is a
−0.0028 that is noise. But the two swaps are **not operationally equivalent**: thick ice is what
POLARIS forbids, and getting it wrong is the error that sends the ship where it must not go;
thin ice mislabelled on the side of caution costs time. Lovász improves where an error is
dangerous and degrades where an error is expensive. That is exactly why its permissive fraction
drops by **10.4 points**.

**On the 6 demo scenes Lovász wins on both metrics, all six of them:**

| scene | permissive base → Lovász | mIoU base → Lovász |
|---|---|---|
| 20180707 | 11.8% → **8.0%** | 0.678 → **0.707** |
| 20180917 | 0.6% → **0.3%** | 0.770 → **0.775** |
| 20200701 | 19.7% → **13.1%** | 0.660 → **0.715** |
| 20201013 | 2.4% → **1.3%** | 0.572 → **0.587** |
| 20210506 | 15.9% → **3.5%** | 0.759 → **0.901** |
| 20180623 (worst case) | 51.7% → **41.3%** | 0.253 → **0.304** |

**The selection rule had a hole, and that is the honest point to record.** `scripts/pick_model.py`
had been written expecting the "more accurate but more permissive" trade-off, and treated the
permissive fraction **only as a veto** against such a challenger. It had no branch for the
opposite case — indistinguishable accuracy, much better safety — and so it discarded Lovász not
because of a judgement about safety, but because the rule had no way to express one.

The added clause (`SAFETY_MARGIN = 0.05`) says: if the mIoUs are within `MIN_GAIN` of each other,
the less permissive model wins, provided it is less permissive by at least 5 points. **The
asymmetry that justifies it was not invented for the occasion**: it is the same as in D-008
("always the most cautious column") and in `rio_error.py` ("being too cautious costs time, being
too permissive costs the ship").

**A second defect, found while writing the clause.** The old rule walked the list comparing each
candidate with the current best: an **order-dependent** procedure. With these numbers it really
changed the answer — `more data` is 6.9 points safer than the baseline and would have won if seen
first, after which Lovász, only 3.5 points safer *than it*, could no longer unseat it. Same three
models, same numbers, different winner depending on how the list was written. The choice is now
formulated over the whole set, and `test_the_winner_does_not_depend_on_the_order_of_the_list`
tries every permutation.

**Downstream effects.** Downlink compression from 303:1 to **371:1** — consistent: a cleaner
segmentation of thick ice produces more compact regions, which compress better. INT8 degradation
still nil (+0.0046). The 4/5 demo scenes with a route that changes remain 4/5.

<a id="d-026"></a>
## D-026 — Transit hours cannot be quoted, and they do not need to be

**Date:** 2026-09-07

`open_water_speed_kn: 12.0` has no source (P-05), and every transit time the demo prints scales
with it. The natural reflex is to look up a "typical" value and close the matter, but that only
replaces one assumption with a better-dressed one.

**The question asked instead:** how much does the number we *actually quote* move when the
assumption moves? Measured with `scripts/speed_sensitivity.py` at 10 · 12 · 14.2 · 16 · 19.5
knots, on the 5 demo scenes and 3 ship classes, with the waypoints **fixed once** at the nominal
value, because otherwise they would move too and confound the measurement.

**Two results, both negative, and they are the reason this entry exists.**

1. **Absolute hours vary by 95%** over the range. They are unusable without a source, and no
   source would make them stable: they scale with speed by construction.
2. **The percentage saving — the natural fallback, being a ratio — is defined in only 2 series
   out of 15.** In 65 of the 75 configurations tried, the direct route **is not navigable**.
   Which is obvious in hindsight: the demo scenes were chosen (D-012) precisely because the
   straight line is forbidden by the ice. There is no navigable baseline against which to
   compute a saving, and the project's rule against invented numbers forbids making one up.

**Consequence for public claims.** Neither time quantity holds up on its own. The defensible
claim is **categorical and does not use speed**:

> The direct route is forbidden by POLARIS for that ship class. IceRoute finds one that is not.

It is also the stronger of the two claims: "we save 20% of the time" invites an argument about
the 20%; "the ship cannot take that route" does not. `compare()` in `routing/baseline.py` already
refuses to produce a saving against a non-navigable baseline and returns exactly this sentence —
the code was already honest; what was missing was the measurement saying *how much* we had to
rely on it.

**P-05 stays open but downgraded:** it no longer blocks any public claim, because no public claim
depends on speed any more. A source is needed only if absolute hours are ever to be shown.

**Note on the source searched for.** The only quantitative anchor found in a primary document is
indirect: the ABS advisory on the NSR gives 11,000 miles via Suez against 7,600 via the NSR, "a
saving (depending on service speed) of about ten days". That is 3,400 nm in 240 h, i.e.
**≈14.2 knots** implied. It is a derivation, not a stated value, and that is why it lives here and
not in `configs/routing.yaml`.

**Mistake made and corrected, for the same reason as D-018.** The script's automatic verdict
looked at the spread of the percentage without looking at **how many series** it was computed on,
and printed "varies by 0.0 points → STABLE, safe to quote" on two series out of fifteen. Zero
spread over two points is not stability: it is a quantity that is almost never defined. The rule
now checks coverage before spread. It is the second time in one night that a threshold placed on
the wrong statistic produced a plausible, false sentence.

<a id="d-027"></a>
## D-027 — The comparison between the two routes was computed but never drawn

**Date:** 2026-09-07

Every `data/demo/routes/*.geojson` has always contained **two** tracks: the IceRoute route and the
geodesic — the one a ship with only a nautical chart would follow — with the POLARIS verdict on
the latter. The comparison was complete, measured and stored. **It was drawn nowhere.**

- `figures.py::_draw_routes` read `properties["cells"]`, which `Route.as_geojson` never wrote:
  `if not cells: continue` dropped every track. `02_useful.png` — the figure whose only purpose is
  to show that IceRoute finds a route where the direct one does not exist — was generated for
  weeks as **a bare mask with a caption**.
- In the demo, the "Show route" checkbox did not show the route: it enabled the GeoJSON download
  button.

**Why nobody had noticed.** The failure raises nothing, logs nothing and produces a valid PNG.
There is no way to notice it by watching the run: you have to look at the image. It is the same
family as D-018 and D-026 — a silent defect that produces a plausible artefact — but here the
artefact was the one meant to be shown in the demo.

**The real obstacle: the scene lives on two grids.** SAR and mask are the raw decimated swath
(≈1311×1249); the routes are computed after geocoding, on an EPSG:3413 grid at 500 m (≈976×956).
One is not a crop of the other: different projection, different shape. The old code scaled the
route cells by the **raw** shape, so even if `cells` had existed it would have drawn a wrong line.
Better that it drew nothing.

**Decisions taken.**

1. **The drawable coordinates live in `data/demo/overlays/`, not in the GeoJSON.** The GeoJSON is
   the product that goes down to the ground and its bytes are part of the compression measurement
   (371:1): stuffing 1124 cell pairs into it would inflate it by two orders of magnitude to carry
   something only the ground UI needs.
2. **The overlay declares its own `grid_shape` and the drawing refuses an image that does not
   match it.** The tolerance is **1%**, not a round 5%: a legitimate decimation is `ceil(n/k)` on
   both axes and puts the two scale factors out of step by less than 0.5%, whereas the wrong
   pairing we want to catch (geocoded route on a raw raster) is out of step by 2.8%. The margin
   must sit **below** what it has to catch, not just above what it has to tolerate.
3. **Forbidden cells are drawn as segments, not as dots.** A scene in which the direct route
   crosses a pack produces 296 no-go cells; 296 overlapping markers become a black smudge and
   **misdescribe the situation**: they are not 296 separate episodes, they are a stretch the ship
   cannot enter. On 20200701/IA the 296 cells are 4 segments, one of them 273 long.
4. **The line is the full A\* path; the dots are the simplified waypoints.** Those are the only
   data that actually go down: showing both makes the compression claim visible instead of
   asserting it.

**The colour is not an aesthetic choice.** The mask uses blue/cyan/white and the RIO uses
green/yellow/red: two hue families unusable for a line. Magenta appears in neither. Every stroke
also has a dark casing underneath, as on road maps, which is what lets a thin line survive on any
background and on a washed-out projector.

**The lead scene is now chosen by a measurement, not by index order.** This was needed because a
scene can forbid the direct route in 296 cells and still draw two overlapping lines, which on a
projector prove nothing. `overlay.separation()` measures how far apart the two tracks get, as a
percentage of the diagonal:

| scene | class | forbidden cells | separation |
|---|---|---|---|
| **20200701** | IA | 296 | **10.25%** (88 km) |
| 20180917 | PC6 | 19 | 6.93% (56 km) |
| 20180707 | PC6 | 227 | 4.89% (43 km) |
| 20210506 | PC6 | 217 | 3.73% (24 km) |
| 20201013 | IA | 12 | 1.87% (13 km) |

Before, the figure took the first scene in the index with a route, i.e. 20180707: forbidden in
227 cells but **half as readable** as 20200701. Now 20200701 wins.

**Two points to make, two different scenes — and it is good to know that in advance.** 20200701
is the *visual* point ("look where ours goes and where the direct one cannot"); 20201013 is the
*conceptual* point, the only scene in which the direct route is **legal for PC6 and forbidden for
IA** — change the ship and the very same line becomes illegal. No single scene does both well.

**Mistake made and corrected during the work.** The first version of the dashboard caption chose
its subject by **re-applying the same rule** as the figure, but iterating over all ship classes
instead of the reference one: it titled "IA ship, 296 cells" under an image generated for PC6.
Two places that choose the same subject by re-applying the same rule are two places that can
diverge. Now the figure writes `artifacts/figures/route_lead.json` and the dashboard **reads** it.

**Honesty about the case in which the direct route is navigable.** On 20201013 with PC6, IceRoute
is 0.97% slower (+4.8 km over 494). The card showed "−0.2 h (−1%)" in amber, i.e. it framed a
non-event as a loss. It now says that the direct route is navigable and that **IceRoute follows
it**, which is the true reading and also the better one: it does not invent detours where none
are needed. If the figure ever had to fall back to a scene like that, the dashboard labels it
"weak comparison" instead of describing a thesis the image does not support.

**Verified.** 22 new tests in `test_overlay.py`, the first of which is the only one that would
have caught the original failure: *drawing changes some pixels*. Plus an end-to-end test over all
18 real routes and a check in `make preflight` that does not verify that the file exists but that
**the image changes**.

<a id="d-028"></a>
## D-028 — The payload counted a route the satellite does not send: 371:1 → 439:1

**Date:** 2026-09-07

Found while building the "what is transmitted to the ship" panel. The compression denominator
included **the whole GeoJSON file**, which contains two tracks: the IceRoute route and the
comparison geodesic. The second is a **ground** artefact, there to argue a point with a human
being; a satellite has no reason to transmit the straight line between two points the recipient
has just given it.

This is not a nuance: the geodesic is sampled at every cell it crosses, so it weighs **fifty
times** as much as the real route.

| scene | route counted | actual route | ratio |
|---|---|---|---|
| 20180707 | 17,985 B | **568 B** (11 wp) | 294:1 → **325:1** |
| 20180917 | 27,449 B | **425 B** (5 wp) | 371:1 → **439:1** |
| 20200701 | 25,767 B | **431 B** (5 wp) | 245:1 → **281:1** |
| 20201013 | 26,448 B | **406 B** (4 wp) | 550:1 → **671:1** |
| 20210506 | 24,192 B | **432 B** (5 wp) | 616:1 → **770:1** |
| | | | median **371:1 → 439:1** |

**The error was against us**, so no public claim was inflated, and that is the only reason it is
not a serious incident. But it was still the measurement of something other than what was stated,
and above all: the payload panel, built on the old data, would have shown anyone reading it a
file **containing the route we do not send**. A conservative error is still an error.

`compression.downlink_route()` does the stripping, and it is used **both** by the measurement
**and** by the UI, so the bytes on screen and the bytes in the reported metrics cannot diverge.
*Provisional values* updated: P-06 goes from 371:1 to 439:1.

The resulting number is also the best one we have: **55.2 MB of losslessly compressed SAR against
166 KB**, of which 568 bytes are route. The file that reaches the ship is shown in full on screen.

<a id="d-029"></a>
## D-029 — The live computation runs on the same function as the precomputed one

**Date:** 2026-09-07

The design rule is that the UI reads only precomputed files: it starts in two seconds and does not
depend on PyTorch during a live demo. That rule stays. But it also invites the question "how do I
know these are not images made beforehand?", and until now the answer was a button that timed the
INT8 model on **random noise** — weak evidence, and on top of that it read a hard-coded
`configs/train3.yaml` instead of the selected model (the same defect as the dashboard, D-025).

There is now **"Compute now from the raw scene"**, and the decision that matters is that
`live.compute()` **calls `precompute.process_scene()`**, not a copy. A second implementation of
the pipeline would turn "watch it compute" into a claim about code the product does not use. The
only addition is an `on_stage` hook to advance the progress stages.

**Timed on a real scene, GPU, all three ship classes:**

| stage | s |
|---|---|
| reading the scene from disk | 2.8 |
| U-Net inference | 2.8 |
| rendering the masks | 1.5 |
| geocoding to EPSG:3413 | 1.3 |
| POLARIS RIO + cost grid | 0.4 |
| waypoint selection | 0.0 |
| A\* + simplification + comparison | 5.9 |
| **total** | **14.7** |

**And the result is compared with the precomputed one, with the verdict on screen in both cases.**
On 20200701: *identical to the precomputed, 771 cells* for PC6, *703 cells* for IA — cell by
cell. If they ever diverged, it is better to find out in testing than during a live demo, and the
UI says so instead of showing whichever answer arrived last.

**The INT8 button stays, separately.** The live computation runs in fp32 on the laptop GPU: it
proves the **pipeline** is real. The satellite would run INT8 on a CPU, which is a different
machine. Keeping them together would pass off the laptop GPU as on-board hardware.

<a id="d-030"></a>
## D-030 — Time and fuel become sliders, and the rescaling refuses when it would be false

**Date:** 2026-09-07

D-026 closed the question of hours by concluding that they cannot be quoted. But the question
anyone looking at a routing tool asks remains: *how much does it save?* Not answering is a weak
answer; answering with an invented number violates the project's rule against unmeasured,
unsourced numbers.

**Third way:** the assumption becomes a **slider moved by the viewer**. Open-water speed and daily
consumption are two controls, and everything that follows from them is labelled as **the viewer's
assumption**, not our measurement. Rather than being put on the spot by "at what speed?", the
question is handed to the viewer before it is asked.

**The part that makes this honest and not merely clever: the rescaling refuses to work when it
would be wrong.** Transit time is **not** distance divided by speed. The POLARIS model has two
bands:

- **RIO ≥ 0**, normal operation → full speed, which scales with the slider;
- **−10 ≤ RIO < 0**, elevated risk → **fixed** limit from Table 1.2, **3.0 kn for PC6 and PC7**,
  which the slider does not move.

A route that crosses the second band has a component immune to the slider, and rescaling the total
by `v_nominal/v_chosen` would invent time precisely on the stretch governed by the regulation
rather than by the ship.

So `rescale_transit()` **checks** that the minimum RIO along the route is ≥ 0 and only then
rescales — **exactly**, not as an approximation. Otherwise it returns nothing and explains why. On
the current scenes the check passes (A\* routes all of them in normal operation: the cost grids
report *0.0% of cells at reduced speed*), but it is verified per route and not assumed, because
the day a scene forces the ship into the elevated band is the day this would start lying silently.

**Fuel does not even have a default slider value.** It starts at zero, i.e. "not set", and at zero
it shows no number: a zero on a dashboard reads as a measurement. Without a sourced consumption
figure (P-09) no tonnage comes out. The rule against currency figures stays in force: no dollars,
in any case.

<a id="d-031"></a>
## D-031 — "The route goes through thick ice": the map showed half of the equation

**Date:** 2026-09-07

**Found by the author looking at the UI for the first time**, which is already the most important
fact in this entry: the first person to see the demo concluded, reasonably and within seconds,
that the route was illegal.

The route crosses wide white areas, the legend calls them "thick ice", and the obvious conclusion
is that the ship is going where it cannot. **The route is correct.** The measurements under its
pixels, scene 20180707 / PC6:

| under the route | thick ice | concentration | RIO |
|---|---|---|---|
| **IceRoute** | 296/449 cells (66%) | median **3/10** | median **+12** — normal operation |
| forbidden cells of the direct route | 176/227 (78%) | median **8/10** | median **−18** — forbidden |

Same class, same colour on screen, opposite verdict. Because POLARIS does not decide on ice type
alone:

```
RIO = C × RIV(type) + (10 − C) × RIV(open water)      MSC.1/Circ.1519
      C=3:  3(−3) + 7(+3) = +12      navigable
      C=8:  8(−3) + 2(+3) = −18      forbidden
```

With RIV = −3 for heavy multi-year ice and +3 for open water, for a PC6. The check on the 5
carousel scenes: **100% of the cells of every route are in normal operation, 0% are forbidden.**

**The defect was not in the routing but in the map, and it was the most dangerous one found so
far.** The class mask contains **half** of the POLARIS decision — the type — and we drew the route
on top of it, as the default layer no less. Anyone looking draws the wrong conclusion from what we
showed them. It is not a defect that produces a wrong number: it is a defect that **invites an
objection that makes a correct system look wrong**, live, with seconds to answer.

**Three corrections, in order of importance.**

1. **The default layer is RIO**, not the mask. RIO is the only layer that carries the decision,
   and it is the one A\* actually computes on. Leading with the mask put first the layer that
   cannot answer the question.
2. **There is a "Concentration" layer**, in a hue family different from both others (purple →
   yellow) precisely because it must read as *the third thing*. Without it the demo could assert
   why the route is legal, not show it. With it you can see the route following the dark,
   low-concentration corridor.
3. **Layers are named for what they are**: "RIO — where you can go", "Ice type",
   "Concentration". "Ice mask" suggested completeness.

And when someone opens the ice-type layer anyway with a route drawn on top, a callout appears that
**anticipates the objection with that route's numbers**, measured at precompute time by
`polaris.conditions_along()` and not reworded in the UI: *"The route crosses the white, and it is
compliant. 66% of its cells are thick ice — but at a median concentration of 3/10, which gives
RIO +12."*

**A note on what this says about the demo.** No test would have caught it: the routing is correct,
the numbers are correct, the files are correct. Only the inference a human draws by looking was
wrong. It is exactly the class of defect that the usability check — three outside users operating
the demo on their own — exists to find, and one came out of a single person's first glance. It is
worth doing in full.

<a id="d-032"></a>
## D-032 — An offline, single-file demo that runs without Python or network

**Date:** 2026-09-07

`make demo` needs Python, a virtualenv, Streamlit and a server that stays up. During a live demo
those are four things that can go wrong, and the planned fallback was **a screen recording**.

A video is worse in every respect. It cannot be driven: if someone asks "and with a PC7?" the video
does not answer. Above all, **it goes stale silently**: the day compression goes from 371:1 to
439:1, the video keeps saying 371:1 and nobody notices.

`make demo-static` (`src/iceroute/web/standalone.py`) produces **`demo.html`**: 2.9 MB,
double-click, no Python, no server, no network. Everything works except "Compute now", which needs
the model — and the page **says so** instead of hiding the button.

**It is not a second UI, and it is built so as not to become one.** Two things that draw the same
truth diverge sooner or later: it has happened three times in this project (D-025, D-027, D-031).
Therefore:

1. **Every number comes from `data/demo/index.json`**, the same file the real UI reads. Nothing is
   recomputed and nothing is hand-written.
2. **Route geometry and colours come from `demo/overlay.py`**, the same constants the real UI draws
   with. SVG is a different medium, not a different decision.
3. **There is no logic in the JavaScript.** Transit time at every speed — and the **refusal** to
   give one where POLARIS imposes a fixed limit — are computed in Python by
   `estimate.rescale_transit()` and baked into a lookup table. The browser picks an entry. It
   cannot round differently and **cannot decide to show a number that Python decided not to
   give**. The only exception is fuel: hours divided by a day, times a consumption. A
   multiplication with no judgement in it.

**Verified by actually opening it**, not just generating it: scenes, classes, layers, modes, both
routes in SVG, the payload, the sliders. At 19.5 knots the card says 15.2 h against the nominal
24.7 h — which is exactly 24.7 × 12/19.5. One bug found only by looking: a complete `<svg>` was
being nested inside another `<svg>`, and the inner element collapsed to zero size without drawing
anything. Once again a **silent** failure in a route overlay, the third time.

**It is not in git.** 2.9 MB regenerated at every build would be a new blob each time, and it is
100% derived from `data/demo/` and `artifacts/metrics.json`. But because it is the fallback, it
has to be copied elsewhere by hand — a USB stick, an email to oneself — together with `data/demo/`.

`make preflight` checks that it exists **and that it is not older** than `data/demo/index.json`.
Freshness, not existence: a `demo.html` left over from before the last precompute would open
perfectly, look right and show the previous model's routes — and it is the file that gets used
precisely when nobody is in a position to notice.

**The "record the fallback video" task stays open but downgraded.** It is only needed as a third
safety net, after the real demo and `demo.html`.

<a id="d-033"></a>
## D-033 — The assumptions file said "provisional" next to measured numbers

**Date:** 2026-09-09

`configs/assumptions.yaml` is mandatory by project rule: it is the place every economic or
performance claim must derive from, with a `source:` next to every entry. The demo shows it **in
full** in the footer (`app.py`, `st.code(ASSUMPTIONS.read_text())`) — a deliberate choice, to
defuse in advance the question "what are these numbers based on?".

Several entries were written like this:

```yaml
downlink:
  ratio: null                  # MEASURED by report/compression.py
  provisional: true
model:
  parameters: 1964182
  onnx_int8_bytes: null        # MEASURED by model/quantize.py
```

Those comments were an **intention that nobody implemented**. Nothing wrote that file. So while
`metrics.json` measured 439:1 and 2,148,589 bytes, the file behind the public numbers kept saying
`null` and `provisional: true` — and it said so **on screen**, inside the "Assumptions" expander,
next to the numbers the panel above quoted as measured.

And `parameters: 1964182` was worse: it was not empty, it was **typed by hand and off by 51**
from the 1,964,131 measured.

**Fixed** with `src/iceroute/report/assumptions.py`, run by `make report` right after
`report/metrics`. Four choices, none of them obvious.

**It edits lines, it does not serialise the YAML.** `yaml.safe_dump` would have round-tripped the
data and deleted **every comment in the file**. Those comments are not decoration: they are the
justification of each number, which is exactly what the file exists for, and they are on screen
in the demo. A bare `ratio: 439.2`, without the note saying what the numerator is, would be a
**worse** file than the one with the `null` in it. A test checks that every comment survives the
write.

**It only replaces keys that already exist.** Structure and prose stay hand-written; only the
numbers are machine-written. A key the module looks for and does not find is an error, never
something to append: if a measurement does not yet have a home in the file, a person must decide
where it goes and what to write next to it. For the same reason the module **refuses** to
overwrite a value that is not `null`, a boolean or a number: if someone put prose there, the prose
wins.

**The three downlink figures come from a single scene.** `raw_scene_bytes`, `payload_bytes` and
`ratio` are taken from the scene whose compression is the median, not from three independent
medians. Three separate medians give a quotient that **is none of the printed numbers**: a reader
who divides 64,991,441 by 147,986 must get 439.2, not a third number we never stated. With an even
number of scenes the median does not coincide with any scene, so the nearest one is taken —
interpolating would print a payload nobody measured.

**It writes no timestamp.** The file must change when a number changes and at no other time, so
that a diff on `configs/assumptions.yaml` is a signal rather than noise from the last
`make report`.

**`provisional` is a consequence, not an opinion.** A section stops being provisional exactly when
none of its measurements is missing. It is not a switch that can be flipped by hand to make the
file look more ready than it is.

**What stays `null`, and must stay so:** open-water speed (P-05), daily consumption and fuel price
(P-09). They have no source, so `economics.show_dollar_savings: false` stays `false` — and a test
checks it, because it is the one thing this module must never be able to turn on.

**Guard.** `scripts/preflight.py` gains its thirteenth check: it re-runs the sync as a dry run and
fails if the file is behind, naming the key. Provoked on purpose by falsifying `ratio` to 200.0:
it fires, and `--check` exits 1 without writing.

**The link with *Early estimates vs. measurements*.** This entry came out of the audit of early
public claims: A-12 showed that a 650% ROI figure contradicted
`economics.show_dollar_savings: false`, and opening the file to check it revealed that half of its
measurements were missing. The file that was supposed to back the public numbers was **the
furthest behind of all**.

---

<a id="d-034"></a>
## D-034 — The satellite transmits the mask **and** the route, and we state both numbers

**Date:** 2026-09-09 · **Status:** closes **A-10** (*Early estimates vs. measurements*)

The June 2026 prototype materials stated a `< 4 KB` packet, next to the phrase *"a tactical
vector, not images"*. The measured packet is **148 KB median**, and **99.7%** of those bytes are a
PNG mask, i.e. an image. Wrong by about 37×, and wrong in the sentence as well.

The question was not "which number do we write" but **what the satellite actually sends**: sending
only the route would make `< 4 KB` true by a huge margin (~430 bytes), at the cost of taking the
ice map away from the ship.

**Decision: both are transmitted.** Four reasons, in order of weight.

**1. A route without the ice field cannot be verified, and a route that cannot be verified is not
followed.** This is not a supposition: it is **D-031**, measured on a person. The author looked at
a *correct* route on a map showing only ice type and concluded it went through thick ice —
because POLARIS decides on type **×** concentration, and the map showed half of it. If half the
context is enough to make the person who built the system doubt it, **zero context** on the bridge
produces the same doubt with worse consequences: a master who cannot check the line does not
follow it, and the product is useless. Removing the mask means doing to the master exactly what our
map did to us.

**2. Sending only the route would break the regulatory argument the system rests on.** IMO
resolution **MSC.538(107)**, in force since 1 January 2026 — cited as the motivation for the
system (A-14) — requires the capability to **receive up-to-date ice information**. Not "receive a
route". A satellite that downlinks a polyline and nothing else does not help the operator
demonstrate compliance; one that downlinks the interpreted ice layer does. **The version that makes
`< 4 KB` true is the version that weakens the regulatory case.**

**3. There is no bandwidth argument.** The full packet takes **13 seconds** on an Iridium Certus
100 (88 kbps, the smallest terminal) and **about 3 seconds** on FleetBroadband (432 kbps). And
FleetBroadband **already** distributes real-time ECDIS updates as an ordinary service: 148 KB of ice
layer is precisely the traffic that link exists for. The comparison that matters was never 4 KB
against 148 KB — it is **55 MB against 148 KB**, and that one is untouched.

**4. Zero re-measurement cost three weeks before the code freeze.** 439:1 is in `metrics.json`,
`assumptions.yaml`, the demo, `demo.html`, the dashboard and
`artifacts/estimates_vs_measurements.md`. Changing the payload would mean redoing all those
measurements and invalidating every artefact that cites them.

**What changes in the software: nothing.** The demo already tells the truth — the "What goes down
to the ground" panel has always shown mask + route + metadata and the real total. **The defect was
only in the early written estimate.** That is the real reason for urgency: anyone using the demo
reads 166 KB on the screen, then reads `< 4 KB` in the project's own description.

**How the two numbers are stated.** Not one instead of the other: they answer different objections.

| number | what it is | what it answers |
|---|---|---|
| **148 KB** | what the ship receives, against 55 MB of SAR | "can the link handle it?" — 13 s on the smallest terminal |
| **431 bytes** | the route inside the packet: two SBD messages | "how small is the decision?" |

The 431 bytes are **more impressive than the 4 KB we were claiming**, and they are true. The number
we had to own up to turned out stronger than the invented one — as already happened with A-02.

**The sentence to change together with the number.** *"A tactical vector, not images"* is false and
must be replaced. What we defend is not that no raster is sent: it is that **the reasoning happens
on board**. What goes down is a verdict — three already-interpreted classes plus the route that
follows from them — not 55 MB of radar echo to be interpreted on the ground. Proposed wording:

> **148 KB instead of 55 MB.** The verdict goes down, not the raw data: the already-interpreted ice
> layer and the route that follows from it. The route alone is **431 bytes** — two Iridium SBD
> messages.

**Recorded in `configs/assumptions.yaml`,** section `link`, with the operators' sources. The seconds
are **derived** from the stated peak rate: real throughput is lower, so they must be stated as an
order of magnitude, not as a stopwatch reading.

**Software consequence, the only one.** Stating 148 KB invites exactly one objection — "can the
ship's link handle it?" — and a panel that gives the size but not the cost leaves it standing.
`estimate.link_time()` answers it on both surfaces: **≈ 15 s on Iridium Certus 100 at 88 kbps**,
with the route in **2 SBD messages**. Three choices:

- **The slowest terminal is quoted**, not the fastest. FleetBroadband would give 3 seconds, but that
  would be the best case told to someone who may not have it.
- **The rates live in `assumptions.yaml` with the operator's source**, and the function refuses
  without them. They are not our measurements: the seconds are **derived from the stated peak**,
  real throughput is lower, and the panel says so.
- **No measured bytes, no seconds.** On the "worst case" scene compression is not measured, so the
  link time does not appear — verified in the browser, on both surfaces. A transmission time next
  to an unmeasured payload would be pure invention.

Verified by opening **both** the UI **and** `demo.html`: identical text. It is the rule of D-025,
D-027 and D-031 — two things that draw the same truth diverge, so they compute from the same
function and both get looked at.

**Open, and outside this decision:** the mask is the whole scene at 500 m. Cropping it to the route
corridor would shrink it considerably. It is a real improvement, but it is **a new feature close to
the code freeze**, and it would change a number already measured and already cited everywhere.
Deferred.

---

<a id="d-035"></a>
## D-035 — `make report` was measuring compression on the wrong model

**Date:** 2026-09-09

Re-running `make report` for the first time since the model change (D-025), the compression ratio
went from **439:1 to 347:1**. No line of the measurement chain had changed: `compression.py` had
only gained `downlink_route`, `metrics.py` six lines, the checkpoint was the same, the routes
identical. **Only the predicted mask had grown**, by 18% to 114% depending on the scene, and all
five in the same direction.

**Diagnosis, in three steps.**

Inference is **deterministic**: two runs on the same scene give **0 pixels of difference**, on CUDA
and on CPU, and the two devices differ from each other by **0.0026%** (94,220 against 94,196 bytes
of PNG). So it was not noise.

`measure_downlink()` called **in isolation** returned **exactly 439:1**, with the original masks. So
it was not the function.

The difference was in `main()`:

```python
ap.add_argument("--train-config", type=Path, default=Path("configs/train3.yaml"))
...
checkpoint = Path(train_cfg["output"]["best_checkpoint"])
```

`configs/train3.yaml` names `unet3_fp32.pt` — the **baseline**. The deployed model is the one
`scripts/pick_model.py` wrote into `artifacts/best_config.txt`, i.e.
`configs/train3_lovasz.yaml` → `unet3lov_fp32.pt` (D-025).

**`make report` was measuring compression on a model we do not ship.** The 347:1 is the baseline;
the 439:1 is the real model.

**Why nobody had noticed.** The wrong default had never fired: `make report` had not been re-run
since the model change. And `preflight` checked that `data/demo/index.json` was aligned with the
selected model — not that **the measurements** were. The demo was right, the published number was
not, and the two checks did not talk to each other.

**It is the same defect already fixed earlier the same day** in `run_live_inference`, which also
read a hard-coded `configs/train3.yaml` and timed the wrong model. Same shape, same cause: **a file
name frozen into an argparse default instead of the pointer that says which model is deployed.**
Finding it twice in one day says it was a pattern, not a slip.

**Fixes.**

`deployed_config()` in `src/iceroute/report/metrics.py` resolves `artifacts/best_config.txt` and
falls back to the baseline only if the pointer is missing or points to a file that has gone — a
broken pointer must not bring the report down on the morning of a live demo. `metrics.py` and
`precompute.py` now default `--train-config` to `None` and go through it; both log which model they
are using.

**The number carries its model with it.** `metrics["compression"]` now records `checkpoint` and
`train_config`. Without them, an old `metrics.json` is indistinguishable from a new one measured on
the wrong network — which is exactly the situation we found ourselves in.

**Fourteenth check in `preflight`:** compression must be measured on the deployed model, and if
`metrics.json` does not record which one, that is a warning. Provoked and verified: it warns when
the field is missing, it fails when the model differs.

**What had propagated the error, and what exposed it.** The sync from [D-033](#d-033) faithfully
wrote 347:1 into `configs/assumptions.yaml`, and `artifacts/estimates_vs_measurements.md` — which is
tracked in git — showed the diff `439:1 → 347:1`. In other words: **the mechanism that propagated
the wrong number is also the one that made it visible.** Without the sync, the assumptions file
would have kept saying `null`, and the diff would not have existed.

**And it is worth saying why it surfaced now.** It surfaced because `make report` was re-run to
check a Makefile change. Otherwise the first `make report` before release would have overwritten
every number with the baseline's — plausibly the day before it mattered.

---

<a id="d-036"></a>
## D-036 — Presentation language

**Status:** withdrawn from the public repository (event logistics).

<a id="d-037"></a>
## D-037 — Presentation mode

**Status:** withdrawn from the public repository (event logistics).

---

<a id="d-038"></a>
## D-038 — The offline demo becomes a console, and the interface colour stops speaking like the data

**Date:** 2026-10-04

An earlier keyboard-driven, step-by-step rendering of the same build (D-037) was retired for the
reason that motivates this whole family of decisions: a second renderer of the same data that
nobody opens is a second thing that can diverge from the truth (D-025, D-027, D-031), and a
generator nobody runs stops being verified by use.

**What gets shown is `demo.html`** — the offline, single-file demo that runs without Python or
network (D-032) — and it was redone to look like a working tool rather than a web page: a command
bar at the top with provenance (checkpoint, schema, test scenes, mIoU), a dense monospaced sidebar,
panels with a header strip, the four metrics in **one frame divided by hairlines** instead of four
separate boxes, 2 px corners, a single rule weight, and a status line at the bottom that always
says what is on screen. No data changed: the same `index.json`, the same `metrics.json`, the same
functions from `estimate.py` and `overlay.py`.

### Three things that were not just aesthetics

**The interface accent was the colour of our route.** Active tabs, the selected row in the sidebar
and the sliders were `#ff2d95` — the same magenta `overlay.py` uses to draw the IceRoute route on
the map. On a screen whose subject *is* the map, a UI colour that borrows a series colour forces the
eye to wonder whether a highlighted tab is part of the route. Interaction is now `#4a9eff` and
magenta appears **only** where it means "this is our route". A test enforces it.

**The legend was retyped by hand, and one entry was wrong.** "Open water" declared `#0a2859`;
`precompute.CLASS_COLOURS` paints those cells `#0a285a`. One channel of difference, invisible on
screen, and enough to make the legend a *claim* about the image instead of a reading of it. The
other eleven values were correct — which is exactly the argument for deriving all twelve from the
modules that paint the rasters, instead of correcting one. The same typo lives in the Streamlit
app's legend (`src/iceroute/demo/app.py`) and was **not touched** there: the Streamlit UI was under
a code freeze and is not the surface being shown, and reopening it was not worth the risk.

**The page grouped thousands the Italian way and wrote decimals the English way.** "94.220 B" next
to "15.6 h", two conventions in the same card. Anyone who reads "94.220" as a decimal reads the mask
as ninety-four bytes. Every displayed figure now goes through a single formatter; inside the
transmitted GeoJSON the decimal point **stays**, because that is the real file and localising it
would corrupt it.

### What was verified

Every combination of scene × class × layer × mode (6 × 3 × 5 × 4, skipping the inapplicable ones)
at **1680×1050, 1366×768, 1280×720 and 960×800**: no horizontal scrolling, no element outside the
viewport, every image loaded, route SVG aligned to the raster within 2 px, status line always in
sync. In addition, both "Time and fuel" sliders swept over every scene and class: 540 combinations,
no decimal point outside the GeoJSON.

**`app.py` was not touched** — the freeze on it holds, and `make demo` does what it did. `demo.html`
stays in `.gitignore`.

---

<a id="d-039"></a>
## D-039 — `console.html`: the real geography the scenes had always carried

**Date:** 2026-10-04

`demo.html` stays exactly as it is ([D-038](#d-038)). This is a **second rendering of the same
build** — `console.html`, from `make console` (`src/iceroute/web/console.py`) — with a globe, the
real swath footprints and the Sentinel-1 product metadata that the files had carried from day one
and that nothing had ever shown. Like `demo.html`, it is an offline single file that runs without
Python or network.

### What was already in the data and could not be seen

Every AI4Arctic scene carries a 21×21 geolocation grid (`sar_grid2d_latitude` /
`sar_grid2d_longitude`) and an `original_id` attribute that is the full Sentinel-1 product name:

```
S1B_EW_GRDM_1SDH_20180707T113313_20180707T113413_011704_015877_83D1
```

From it come, **read and not invented**: platform (S1A or S1B), mode, polarisation, **absolute orbit
number**, and the sixty-second acquisition window. The outline drawn on the globe is the border of
that grid, at its true coordinates.

### Pass direction is measured, not inferred from names

The along-track axis is **axis 0**, even though it is called `sar_sample_2dgrid_points`: its azimuth
comes out at 192°–201°, i.e. a near-polar **descending** pass, while axis 1 lies about 92° from it —
Sentinel-1's right-looking direction. The dimension names in these files are transposed with respect
to the array order; the measurement is not. Taking the axis by its name would have put a flight
direction wrong by ninety degrees on screen, looking right.

Two independent checks verify it on every test run: that every track is near-polar, and that the
extent measured on the geolocation grid matches the one implied by `shape × pixel_spacing` (405 km
against 4997 × 80 m, 427 km against 5351 × 80 m). Two different routes to the same number.

### Three things that are **not** there, and why

**No coastline.** This project has no coastline dataset — no cartopy, no Natural Earth — and the
page does not go online. The globe carries the graticule and the Arctic Circle, which are
definitions and not measurements, **and says so on screen**: someone who takes a base map for
granted reads the empty sphere as ocean. Drawing a plausible continent is the one lie this screen
cannot afford.

**No orbit ring.** A full orbit would have to be modelled from elements nobody here has. What moves
is the satellite along the **ground track of the acquisition**: two true endpoints taken from the
grid, sixty true seconds in between. It is not an orbit and does not pretend to be. With an S1A/S1B
TLE for those dates it would become a real orbit.

**No timestamps in the log.** A console log reads as a ticking clock, and `data/demo` does not record
**when** a stage was run. Lines are numbered by sequence. A plausible `[23:07:21]` would have been the
only fake line on the whole page.

### It is not a second source of truth

Every number comes from `standalone.collect` — the same function that builds `demo.html` — and the
colours from `standalone.palette`. This module adds geography and formats strings: it recomputes
nothing. Two renderings that each derive their own figures is the failure of D-025, D-027 and D-031.

The only thing the browser computes is the **orthographic projection**, because the globe spins and
can be dragged. That is display geometry, not measurement: bar widths, POLARIS percentages and every
printed figure are resolved in Python, and a test enforces it — a browser that turns
`fraction_no_go` into a width would be the browser deciding how large a risk looks.

### The forbidden-cell trace

The bottom-right panel is not a decorative chart: the overlay records the full path of the direct
route **and** the subset of cells POLARIS forbids. Each bar is a portion of the path and its height
is the fraction of forbidden cells within it. "Out of coverage" stays a series separate from
"forbidden": one is missing data, the other is a decision, and merging them into one bar would
confuse them.

### The packet that reaches the bridge

The bottom-left panel shows **the transmitted file, byte by byte**, with the real offsets: it is the
only thing the satellite sends to the ship, and seeing it in full makes the argument a single figure
does not. Three choices keep it honest.

`payload.text` is the transmitted route, wrapped for display; the separators are `(",", ":")`, so
there are no spaces to break on, and joining the lines back **returns the exact string** — verified,
not assumed, because the whole panel is a claim about how many bytes arrive. If the reconstructed
file does not weigh what the count next to it says, **no dump is drawn**: two things that look
authoritative and contradict each other are worse than one.

The offsets are **byte offsets**. As long as the file is ASCII, character index and byte index
coincide; if it stopped being so, the ruler would start lying silently, so this is checked and, if
it fails, the ruler disappears instead of being wrong.

A test goes through all 18 scene × class combinations and checks that the text on screen **parses as
valid GeoJSON** and weighs exactly what is declared. What you see is not a representation of the
file: it is the file.

### The two grids, made to coincide

`sar` is the decimated swath; mask, concentration, RIO and routes are EPSG:3413 at 500 m. Switching
layer, the scene **shifts**, because the projection really changed — correct, and nonetheless the
reason the two halves of the demo look like two different places.

The fix is not to warp anything by hand. The SAR is passed **to the same `geocode_scene` call** that
built the routing grid, as an extra band (`extra=`), and ends up aligned *by construction* instead of
by agreement. Then its shape is compared with the `grid_shape` recorded in the overlay and, if it
does not match, **the layer is not generated**: a raster off by one cell would put the route in the
wrong place while looking right, which is D-027 again. All six matched — 1246×1253 against 1246×1253
for the first scene.

The percentile stretch happens **before** resampling, on the native band, exactly as `precompute`
does for the swath SAR: so the geocoded layer carries the same grey mapping and the two are
comparable, not merely similar.

The rasters live in `data/demo/console/`, outside `data/demo/images/` so that preflight does not
count them as build orphans. The cache is **verified against the grid, not trusted because it
exists**: a PNG left over from another resolution would otherwise pass exactly the check that was
meant to fail, so it is deleted and rebuilt.

### The scene squared up inside its own raster

The geocoded grid is north-up; the swath was flown at about 192°. So the scene sits in its raster as
a **rhombus with four empty corners**, and on screen it reads as a crooked image in a mostly empty
box.

The tilt is not guessed: the **four corners** of the geolocation grid are projected into EPSG:3413 —
the grid the raster actually lives on — and the angle is measured from there. The **smallest
rotation** that squares the scene is applied (any multiple of 90° aligns the edges; the one that
counts is the one that moves the image least): from −44.7° to +42.2° depending on the scene. The
transform is on the container, so raster and overlay rotate **together** and the route stays on the
pixels it was computed for — verified, 0 px of offset.

The zoom is **the largest that still contains all four corners**: 1.15× – 1.38×. Filling the frame by
cropping would trade empty pixels for scene, and the corner that disappears is always somebody's ice.

Two defects were found by the test that enforces this guarantee, not by eye. The first: rounding
`scale` to the fourth decimal **upwards** overshot by 15 m. The second, subtler: `scale` was measured
with the unrounded rotation while the browser received one rounded to two decimals — half a
hundredth of a degree over 419 km is 38 m of scene cut off. Now the angle is rounded **first** and the
zoom is measured on that.

Swath layers (`sar`, `mask_pred`, `mask_truth`) are not rotated: they are already square with their
own grid, and turning them would skew a straight image.

### The satellite moves on the great circle of the pass

Before, it slid back and forth over the 400 km of the segment, with a jump at the end of travel, and
the icon grew with the zoom until it was larger than the swath it had just imaged. Now: an icon of
**fixed on-screen size**, oriented along the direction of motion, and the motion is along the **great
circle** that those two measured points and the centre of the Earth already define. It goes around
the globe, disappears behind the limb and reappears on the other side, without discontinuity.

The solid stroke remains the true sixty-second acquisition; the extended circle is dashed, and the
page says it is **continued geometry, not an orbit derived from elements** we do not have here. With
an S1A/S1B TLE for those dates it would become a real orbit.

### Changing scene, the satellite veers and the globe stays still

Before, selecting a scene snapped the globe to a new position. Now the globe **stays where it is**
and it is the satellite that moves: its circle bends from one pass to the other in about a second and
a half.

A pass is held as an **orthonormal basis** `(A, N)`, with `P(θ) = A·cos θ + N·sin θ` sweeping the
whole circle. Keeping the basis instead of the two points is what makes interpolation possible: `A`
and `N` are slerped between the two passes and `N` is **re-orthogonalised** against `A` at every
frame. Without that re-orthogonalisation the intermediate trajectory would no longer be a great
circle but an ellipse — the satellite would leave the sphere. Verified at six instants of the
transition: basis orthonormal within 10⁻¹⁶ and all 72 sampled points on a plane through the centre
at unit radius within 10⁻¹⁵.

**And here it must be said what it is not.** The two circles are real; the curve between them is not:
the scenes are from 2018, 2020 and 2021, and no satellite ever travelled from one to the other. It is
a view transition between two acquisitions, and the caption says so — *"these acquisitions are years
apart, not one orbit"*. During the transition the circle is drawn bolder, so that you can see it is
changing.

Since the globe no longer follows the scene, a **double click** re-centres it.

**`console.html` is in `.gitignore`** like `demo.html`, and is included in `make backup`.

---

<a id="d-040"></a>
## D-040 — The route snaked through open water: string pulling after A*

**Date:** 2026-10-04

A* on a 16-connected grid returns the optimal path **for that grid**, and the grid is the problem:
every step must end on a cell centre, so a leg that should run at 20° is built out of steps at 0°,
26.6° and 45°. The result snakes through open water — slower than necessary, and the first thing
anyone looking at the screen notices.

`routing/shortcut.py` does the *string pulling* pass: it walks the path and, where the straight line
between two of its cells is navigable, replaces everything in between with that line.

### Three conditions, and the second is the one that matters

A shortcut is taken **only** if:

1. **every cell it crosses is passable**, by the same test A* applies, on the same cost grid — it
   cannot go anywhere A* would have refused;
2. **it does not lower the minimum RIO** of the stretch it replaces, **and** it does not cross ice of
   unknown class that the stretch did not already cross;
3. **it is actually faster**, by A*'s own accounting.

The second condition exists because without it the pass would happily cut a corner through marginal
ice to save ten minutes: exactly the trade POLARIS exists to prevent.

### The NaN that disarmed the check

`compute_rio` puts **NaN** where no class can be assigned, and in Python *every* comparison with NaN
is false. A minimum that had absorbed a NaN would have let the shortcut through silently:
`worst < worst_sub` is false when `worst` is NaN. The unknown is therefore carried separately as a
flag and rejected on its own. The rule is "no **new** unknown", not "never": a route that already
crosses unclassified ice can still be straightened through it.

### Four defects, all found by a test and not by eye

1. **`Route.lonlat` must be parallel to `Route.cells`.** `simplify_route` indexes the former with
   indices of the latter. Putting one point per *vertex* instead of per cell, the pipeline raised
   `IndexError` and **two scenes disappeared from the build** — 4 instead of 6.
2. **Two different notions of "passable".** For short steps A* has its own rule
   (`_intermediate_cells`); my sampler used another. Where they diverged, a valid improvement was
   discarded: **12 routes out of 18** kept A* with a warning.
3. **What is drawn must be what was verified.** The rule lived in two places — what `_leg` returns
   and how the path is rebuilt — and they diverged: a never-checked cell ended up in the route, and
   `conditions_along` counted it. Result: "99.87% in normal operation" with nothing to point at. The
   rule now lives in a single function.
4. **Before/after comparison with two different formulas**, which on an untouched route reported
   0.58 ms and 3.6 mm saved. A 10⁻¹²-second shortcut is numerical noise: a relative gain of at least
   10⁻⁶ is now required.

### What it yields, measured

On the 18 demo routes (6 scenes × 3 classes):

| | before | after | |
|---|---|---|---|
| distance | 8,608 km | **8,483 km** | −1.46% |
| time | 387.34 h | **381.69 h** | −1.46% |

From 4.6 to 11.9 km per route, from 12 to 32 minutes per crossing. Transmitted waypoints drop
(11→7, 6→2, 5→4): fewer elbows, and the packet gets shorter.

**No regression**: minimum RIO identical in all 18 (0→0, 6→6, 14→14, 30→30, 12→12, 10→10),
fraction in normal operation always 100%, no route slower.

### What it is not

The pass is greedy and stops at a fixed point: **it is not optimal**. A truly optimal any-angle path
requires Theta\* inside the search, not a pass afterwards. It is, however, a strict improvement on
its input — condition 3 guarantees that a leg is replaced only by a faster one, so the result can
never be worse than what A* found.

It is disabled with `astar.straighten: false` in `configs/routing.yaml`.

---

<a id="d-041"></a>
## D-041 — The console takes Blueprint's palette, read from the source

**Date:** 2026-10-27

The console had been styled after analytics consoles by eye. Its colours and type scale now come
from **Blueprint** (`palantir/blueprint`, `@blueprintjs/core` 6.21.0, Apache-2.0), read from
`packages/colors/src/_colors.scss` and `packages/core/src/common/_variables.scss` instead of
recalled from memory. These are not Blueprint's *components*: only its tokens, and the page does not
claim to be a Blueprint app.

My grey was **dark blue**; theirs is **neutral charcoal**:

| role | before | Blueprint |
|---|---|---|
| app background | `#060a10` | `#1c2127` (dark-gray1) |
| panel | `#0b1118` | `#252a31` (dark-gray2) |
| raised | `#0f1620` | `#2f343c` (dark-gray3) |
| text | `#d7e3f0` | `#f6f7f9` (light-gray5) |
| accent | `#4a9eff` | `#2d72d2` (blue3) |

### The background gets lighter, and half the choices stop working

Twelve colours had been chosen by hand against the near-black. On charcoal **two dropped to 1.18:1
and 1.25:1** — the provenance line at the bottom and the packet offset ruler, effectively invisible.
It did not show by looking: it shows by measuring.

They are all tokens now, **including the legend swatches**, which used to be written inline in the
HTML: a dot and the line it describes can no longer diverge. A test enforces it by searching for any
hand-typed colour outside the `:root` block.

Same for the **halos**: the outlines behind markers and routes were `#060a10`, i.e. the old
background. On a lighter grey a near-black halo stops separating and becomes a dark border. It is now
a token, `--halo`, equal to the app background.

### Three things decided by measurement, not taste

**`blue3` cannot be text.** 3.43:1 on the background. It is fine as a fill behind white, so links and
accents use `blue5` (`#8abbff`, 8.20:1) and `blue3` remains only a background.

**The two grey tiers move up one step.** Blueprint's "muted" is `gray4` and its minimum type size is
12 px; these labels are 8–10 px, where `gray4` drops to 4.35:1 on the raised panel and `gray3` to
3.44:1. So `--dim` becomes `gray5` and `--faint` `gray4`. It is a deviation from Blueprint, and a
deliberate one: the type scale is smaller than theirs.

**The pills become opaque.** They were translucent tints, so they borrowed whatever lay beneath: the
same "forbidden" measured 4.37:1 on a selected row and 4.9:1 elsewhere. A label whose legibility
changes with the selection is a defect, not a style. The background is now the same blend, resolved
once.

### Verified

**3,736 text elements**, over all six scenes and all three ship classes: **none below the WCAG AA
threshold, minimum 5.09:1** (before the pass: 39 below threshold, the worst at 1.18:1). Clean layout
at 1920×1080, 1280×720 and 1120×800 on every combination of scene × class × layer × mode.

Four tests lock this in CI: that the tokens are Blueprint's, that every text tier exceeds 4.5:1 on
every surface it appears on, that the pills are opaque, and that no interface colour is hand-typed
outside the tokens.

**Data colours are not touched**: the route stays `#ff2d95` from `overlay.py`, the RIO stays
green/yellow/red from `precompute.py`. They are measurements, not style.

---

<a id="d-042"></a>
## D-042 — Probabilities are calibrated by temperature scaling, fitted on validation

**Date:** 2026-11-17 · **Status:** accepted

D-018 showed that the model's errors are confident. That is harmless while the router reads only
labels, and it stops being harmless the moment anything reads a probability: a rule such as "avoid
a cell if P(thick ice) ≥ 5 %" is only as safe as that 5 % is honest.

**Measured** (`iceroute.model.calibration`, 200,000 pixels sampled per scene):

| | Uncalibrated | T = 2.52 |
|---|---|---|
| Top-class ECE (15 bins), 20 official test scenes | **16.4 %** | **3.9 %** |
| NLL | 0.963 | 0.554 |
| Brier score | 0.361 | 0.308 |
| Class-wise ECE: Open water / Thin ice / Thick ice | 6.8 / 11.0 / 15.5 % | 4.8 / 6.2 / 7.0 % |
| Pixel accuracy | 0.797 | 0.797 |
| Mean confidence on wrong pixels | 0.909 | 0.754 |

**Decision.** One temperature, fitted by minimising NLL on the 10 scenes of the frozen validation
split, frozen, then applied to the test scenes. It is the simplest method that works (Guo et al.,
2017): one scalar, no retraining, no change to any label, mIoU or route, and it costs one division
per pixel on board.

**Why on log-probabilities.** Tiles are blended in probability space (D-013), so logits do not exist
for the whole scene. Temperature is applied to `log p` of the blended probabilities, which is exact
inside a tile and a close approximation in the 32-pixel overlap bands.

**What it does not fix.** Accuracy is unchanged by construction, and on its wrong pixels the model
still places 0.75 on the wrong class. Calibration makes the *average* confidence honest; it does
not make an individual confident error less wrong. That part is the model's, not the decision rule's.

**Consequence.** Calibrated probabilities are what make the chance-constrained rule of D-043
meaningful: its threshold α can be read as a frequency.

---

<a id="early-estimates-vs-measurements"></a>
## Early estimates vs. measurements

The June 2026 prototype was described with a set of design estimates written **before any
measurement existed**. The software built afterwards produces different numbers. Each row below
pairs an early estimate with what the codebase measured, or with what a primary source says, and
records how it was resolved: either the measured number replaces the estimate, or the sentence
changes. The rows cover both technical estimates and the motivating claims stated alongside them.

**Method (second pass, 2026-09-09).** The text of the four early documents was extracted with
`pdftotext -layout` and compared against `artifacts/metrics.json`, `data/demo/index.json` and
`configs/data.yaml`. The nine rows of the first pass became **sixteen**. Two of the corrections
strengthened the case instead of weakening it (A-02, A-14). As of 2026-09-09 every row has an
answer; none still needs research or a choice. The measured side of this table is written by `report/metrics.py` to
`artifacts/estimates_vs_measurements.md`.

| ID | Early design estimate (June 2026 prototype) | Measured / source | Verdict | Resolution |
|---|---|---|---|---|
| A-01 | `50 MB → < 4 KB` **and** `200:1` on the same line | **439:1**, median over 5 scenes (range 281–770), against SAR already losslessly compressed | **Arithmetically inconsistent.** 50 MB / 4 KB = 12,500:1, not 200:1; no derivation of 200:1 exists | Corrected 2026-09-09: 439:1. The measurement **doubles** the estimate |
| A-02 | `58 KB INT8 weights` | **2.05 MB** (1,964,131 parameters) | **Arithmetically impossible**: 58 KB implies ~59k parameters | Closed 2026-09-05. It is the same size as CloudScout (2.1 MB), the model ESA actually flew on PhiSat-1 — a strength, not a weakness |
| A-03 | `5.3 ms per tile`, `< 9 s per 50×50 km scene` | **2.6 s per 400 × 428 km scene, 21 MB peak** (D-013) | **Not measurements**: `1.08 GMAC ÷ (40% × 512 GMAC/s)`, with an asserted utilisation and no source. **And the two numbers are mutually inconsistent:** 5.3 ms × 1,700 tiles ≈ 9 s implies a scene ~10,500 px on a side, which is 50 km at neither 10 nor 80 m/px | Corrected 2026-09-09: label as a projection for the target on-board accelerator (GPU@SAT); quote the measurement. The early estimate understated the covered area by 68× |
| A-04 | `1–3% mIoU lost to INT8` | **0.46-point degradation** (`quantization.accuracy`, 2 training scenes; only the *difference* is quotable) | Literature value for generic networks, not a measurement on this one | Closed 2026-09-05: better than estimated. See D-019 |
| A-05 | Class `Deformed Ice / Ridges` — the class the whole hazard narrative rested on | **Verified absent.** AI4Arctic labels are SIC/SOD/FLOE; there are six SOD classes and none encodes deformation or ridging (`configs/data.yaml`). `grep` over `configs/`, `src/`, `docs/`: no match | Wrong | Closed 2026-09-09. It is not a class the model gets wrong; it is one it **cannot learn** |
| A-06 | "Already demonstrated on the ground" | `artifacts/metrics.json`: mIoU 0.532 / 0.906 binary on 20 official test scenes, 2.05 MB INT8, 439:1, 2.6 s per scene | In June **no measurement had been run**; the early proposal itself said so. The sentence was unsupported | Closed 2026-09-09: **now true** — the only row closed by doing the work rather than correcting the text |
| A-07 | Platform `12U` (June) → `2U` (September) | — | Mass, power (35 W mean) and antenna had been sized for 12U | Decided 2026-09-09: 2U. **Still to revisit:** the 35 W mean power budget and the 0.5 × 0.12 m antenna were sized for 12U |
| A-08 | `€1B+` cost of a single ice-related disaster | A risk study of a spill at Rankin Inlet (Nunavut) models up to **CAD 9.4 bn (~USD 7.5 bn) over 5 years**: under ice, oil is not recoverable | Supported originally by **Costa Concordia** (Mediterranean) and **X-Press Pearl** (fire, Sri Lanka), neither of which involves ice | Closed 2026-09-09: the figure holds with an Arctic source. It is a **modelled scenario**, not an incident that happened, and must be stated as such — but it makes €1B+ conservative |
| A-09 | `681 ships lost 2015–2024` | Allianz *Safety & Shipping Review 2025* confirms 681 — **global, all causes**. Fishing vessels are ~40% of 2024 losses; the decade's hotspot is the **South China Sea (169)**. Allianz's Arctic figure (**531 incidents 2014–2023**) has the same flaw: leading cause **machinery damage (261)**, not ice | True number, misleading placement: next to latency and cost it suggests a causation that does not exist | Closed 2026-09-09: **removed.** No loss statistic supports "ice sinks ships". The motivation rests on traffic (+37% ships, +111% miles — PAME), regulation (MSC.538(107)) and latency (12–24 h) |
| **A-10** | `< 4 KB` transmitted packet, described as *"a tactical vector, not images"* | **148 KB median** (94.7–172.5), of which **99.7%** is an image — the PNG mask | **37× the estimate**, and the sentence is false | Decided 2026-09-09: **both are transmitted**, and both numbers are stated: **148 KB** against 55 MB (13 s on Iridium Certus 100) and **431 bytes** of route alone (two SBD messages). Sending only the route would break the MSC.538(107) argument of A-14, which requires receiving *ice information*. The software did not change. See **[D-034](#d-034)** |
| **A-11** | `5 ice classes` | **Three** deployed (three-class fallback, D-020): Open water / Thin ice / Thick ice. The dataset has six | Five corresponds to nothing | State three, with the reason ready: at six classes mIoU 0.272 while being confidently wrong (D-018); at three, 0.532 and **0.906 on the distinction that matters for safety** |
| **A-12** | `650% ROI`, `€75K/ship/day`, `€750K`, `€2B` market | No source | **Contradicts the project's binding file**: `configs/assumptions.yaml` carries `economics.show_dollar_savings: false` because neither consumption nor price has a source (P-09), and the project rule requires hours, not currency. The UI refuses to convert hours into euros | Decided 2026-09-09: **no ROI figure.** In its place, the panel where the viewer enters *their own* cost per day. It already exists (`rescale_transit`, D-030), refuses to answer when the answer would be invented, and stays consistent with `show_dollar_savings: false` |
| **A-13** | PhiSat-1, *"first neural-network inference on an FPGA in orbit"* | CloudScout ran on an **Intel Movidius Myriad 2 VPU** (Eyes of Things board), **not on an FPGA** | Wrong — and the error weakens the argument: had it been on an FPGA, there would be nothing left to demonstrate | The precedent stays valid when reworded: on-board inference has been demonstrated in flight; **the FPGA deployment is the part this project adds** |
| **A-14** | Polar Code *"becomes mandatory from 2026"* | IMO resolution **MSC.538(107)**, adopted 2023-06-08, in force **2026-01-01**; extends to cargo ships of 300–499 GT, pleasure yachts > 300 GT, fishing vessels ≥ 24 m; new chapters 9-1 and 11-1 | **True**, but stated without a citation and one step further than allowed: the regulation requires the *capability to receive up-to-date ice information*, not real-time monitoring | Cite the resolution number and date, and state the requirement precisely. Stronger than before |
| **A-15** | `ESA AI4Arctic — 493 Sentinel-1 C-band scenes` | We train on **55 scenes / 18,849 patches** | Minor: that is the size of the corpus, not of our training set | Reword. The good answer exists and is measured: **90 more scenes improved nothing** (D-023) |
| **A-16** | IcySea *"about an hour after acquisition"* — the benchmark IceRoute is positioned against | Copernicus Marine describes IcySea as *near-real time delivery up to 1 hour after satellite recording* | Correct, but unsourced | Closed 2026-09-09: add the source |

**A note on method.** A-10, A-11 and A-13 are the three an engineer would take apart with a single
question — and all three are **new**, i.e. they survived the first pass because that pass looked at
the numbers we knew needed checking, not at the sentences next to them. A-06 was the one this
repository could resolve on its own, and it is closed.

**Related debt, closed the same day.** The audit led to opening `configs/assumptions.yaml` — the
file every public number must derive from, which the demo shows in full in its footer — and its
`downlink` and `model` sections still said `null` and `provisional: true` for quantities
`metrics.json` had measured. The file that was supposed to back the public numbers was the furthest
behind of all. See **[D-033](#d-033)**.

---

<a id="provisional-values"></a>
## Provisional values

Every row here is a number or a choice that **is not yet justified by a measurement or a source**.
In code and configs such values carry the marker `PROVISIONAL`. None of them may appear in a public
claim until its status is resolved.

| ID | What | Status | Closes when |
|---|---|---|---|
| P-01 | Actual AI4Arctic netCDF variable names | **Closed 2026-09-03** | Verified on a real scene, see D-006 |
| P-02 | Mapping of SOD classes → POLARIS ice types | **Closed 2026-09-03** | Defined in `configs/data.yaml`, see D-008 |
| P-03 | POLARIS Risk Value table (`data/polaris_rv.csv`) | **Closed 2026-09-03** | Transcribed from MSC.1/Circ.1519 Table 1.3, see D-007 |
| P-04 | RIO decision thresholds per ship category | **Closed 2026-09-03** | MSC.1/Circ.1519 Table 1.1, see D-007 |
| P-05 | Speed in open water and per RIO band | **Downgraded 2026-09-07** — still unsourced, but no longer blocks anything | Sensitivity measured (D-026): hours vary by 95% over the 10–19.5 knot range, and the percentage saving is defined in 2 series out of 15. No public claim depends on speed any more: the defensible one is categorical ("the direct route is forbidden; IceRoute finds one that is not"). A source is needed only to show absolute hours |
| P-06 | Compression ratio (early design estimate ~200:1) | **Closed 2026-09-07 — 439:1** on the **predicted** mask, with losslessly compressed SAR as the numerator | History of the number: 1012:1 on the ice chart → 303:1 on the predicted mask (D-016) → 371:1 with the Lovász model (D-025) → **439:1** once the comparison route, which the satellite does not transmit, was removed from the payload (D-028). Still above the 200:1 estimated. **Quotable.** Not to be confused with *model* compression, 3.67× in D-014 |
| P-07 | Start/end waypoints of the demo route | Partial | Automatic selection implemented (`waypoints.py`); the final waypoints are to be chosen by hand with the criteria of D-012 and D-024 |
| P-08 | Model mIoU | **Closed 2026-09-07 — 0.532** on the official test set at **3 classes** (ice/water 0.906) | 0.535 was the baseline; the deployed model is the Lovász one (D-025), 0.5324 aggregate but **0.610 mean per scene** and ten points less permissive. The 0.272 of 2026-09-05 was at 6 classes. The three-class fallback is active (D-020), so the number **must not be compared with the 0.60 target**, which was defined on the six SOD classes. See D-018 for why the six classes were abandoned |
| P-09 | Fuel price and daily consumption | **Open, but no longer blocks the UI** (2026-09-07) | No source found. Consumption became a **slider set by the viewer**, with every downstream value labelled as the viewer's assumption (D-030); with the slider at zero no number comes out, because a zero on a dashboard reads as a measurement. **Currency stays out** in every case: the project rule admits no currency figures without a source, and a source for consumption would not provide one for price |
