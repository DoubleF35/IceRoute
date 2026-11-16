"""Inspect one AI4Arctic netCDF scene and report what is actually inside it.

SPEC §5.4 is explicit: *do not trust the variable names in the spec*. This module
exists to find the real ones. It assumes nothing about naming — it enumerates
every variable, classifies each by heuristic, and prints statistics that ignore
fill values.

Usage
-----
    make inspect SCENE=data/raw/train/<file>.nc
    python -m iceroute.data.inspect <file.nc> --figure out.png --json out.json

The output feeds ``configs/data.yaml``. SPEC rule: if a name differs from what
the spec assumed, update the config, never the code.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# AI4Arctic marks pixels to be ignored with 255. Verifying this is one of the
# jobs of this script, so it is a candidate to check, not an assumption to apply.
CANDIDATE_FILL_VALUES = (255, -32767, -9999, 0)

# Heuristics for classifying variables by name. Deliberately loose: the point is
# to catch the real name whatever it turns out to be, then report it.
ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "sar": ("sar_primary", "sar_secondary", "nersc_sar", "_hh", "_hv", "sigma"),
    "incidence": ("incidence", "inc_angle", "incang"),
    "target": ("sic", "sod", "floe", "ice_chart", "polygon_icechart"),
    "geometry": ("latitude", "longitude", "lat", "lon", "grid", "sample", "line"),
    "passive": ("btemp", "amsr", "tb_"),
    "weather": ("era5", "u10m", "v10m", "t2m", "skt", "tcwv", "tclw", "wind"),
    "distance": ("distance_map", "distance_to_land", "dist"),
}

GREEN, YELLOW, DIM, BOLD, RESET = "\033[32m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"


def classify(name: str) -> str:
    low = name.lower()
    for role, pats in ROLE_PATTERNS.items():
        if any(p in low for p in pats):
            return role
    return "other"


@dataclass
class VarReport:
    """Everything worth knowing about one netCDF variable."""

    name: str
    role: str
    dims: list[str]
    shape: list[int]
    dtype: str
    n_total: int
    n_valid: int
    pct_masked: float
    pct_exact_zero: float
    vmin: float | None
    vmax: float | None
    vmean: float | None
    vstd: float | None
    n_unique: int | None
    unique_values: list[float] | None
    attrs: dict[str, str]


def _to_numpy(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if np.ma.isMaskedArray(values):
        arr = np.ma.filled(values.astype("float64"), np.nan)
    return arr


def _valid_mask(arr: np.ndarray, fill_hint: float | None) -> np.ndarray:
    """Boolean mask of usable pixels: not NaN, and not the fill value."""
    mask = np.isfinite(arr) if arr.dtype.kind in "fc" else np.ones(arr.shape, dtype=bool)
    if fill_hint is not None:
        mask &= arr != fill_hint
    return mask


def describe_variable(name: str, var: Any, *, max_unique: int = 32) -> VarReport:
    """Compute statistics for one variable, ignoring fill values."""
    role = classify(name)
    arr = _to_numpy(var[:])

    # Prefer the fill value the file declares over guessing.
    fill_hint: float | None = None
    for attr in ("_FillValue", "missing_value"):
        if hasattr(var, attr):
            with suppress(TypeError, ValueError):
                fill_hint = float(getattr(var, attr))
    # Targets are the ones where 255 matters; only assume it for those, and
    # only when the file did not declare a fill value itself.
    if fill_hint is None and role == "target" and arr.dtype.kind in "iu" and (arr == 255).any():
        fill_hint = 255.0

    mask = _valid_mask(arr, fill_hint)
    valid = arr[mask]
    n_total = int(arr.size)
    n_valid = int(valid.size)

    if n_valid:
        vmin, vmax = float(np.min(valid)), float(np.max(valid))
        vmean, vstd = float(np.mean(valid)), float(np.std(valid))
    else:
        vmin = vmax = vmean = vstd = None

    # Categorical variables are the interesting ones for class mapping, so
    # enumerate their levels — but only when there are few enough to be levels.
    n_unique: int | None = None
    unique_values: list[float] | None = None
    if n_valid and (arr.dtype.kind in "iu" or role == "target"):
        uniq = np.unique(valid)
        n_unique = int(uniq.size)
        if uniq.size <= max_unique:
            unique_values = [float(u) for u in uniq]

    attrs = {}
    for a in getattr(var, "ncattrs", lambda: [])():
        # An unreadable attribute is not a reason to abort the whole report.
        with suppress(Exception):
            attrs[a] = str(getattr(var, a))[:200]

    return VarReport(
        name=name,
        role=role,
        dims=list(var.dimensions),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
        n_total=n_total,
        n_valid=n_valid,
        pct_masked=100.0 * (1.0 - n_valid / n_total) if n_total else 0.0,
        # Tracked separately from pct_masked because AI4Arctic pads the SAR
        # swath with a literal 0.0 that is neither NaN nor a declared
        # _FillValue. It therefore counts as "valid" everywhere unless you look
        # for it on purpose. See check_coherence().
        pct_exact_zero=100.0 * float((arr == 0).sum()) / n_total if n_total else 0.0,
        vmin=vmin,
        vmax=vmax,
        vmean=vmean,
        vstd=vstd,
        n_unique=n_unique,
        unique_values=unique_values,
        attrs=attrs,
    )


def inspect_scene(path: Path, *, max_unique: int = 32) -> dict:
    """Open a netCDF scene and describe every variable in it."""
    import netCDF4  # imported lazily so --help works without the dep

    report: dict[str, Any] = {
        "file": str(path),
        "size_mb": path.stat().st_size / 1024**2,
        "global_attrs": {},
        "dimensions": {},
        "variables": [],
    }

    with netCDF4.Dataset(path, "r") as ds:
        for a in ds.ncattrs():
            with suppress(Exception):
                report["global_attrs"][a] = str(getattr(ds, a))[:300]
        report["dimensions"] = {k: int(v.size) for k, v in ds.dimensions.items()}
        for name, var in ds.variables.items():
            try:
                report["variables"].append(
                    asdict(describe_variable(name, var, max_unique=max_unique))
                )
            except Exception as exc:  # noqa: BLE001 — one bad variable must not abort
                report["variables"].append({"name": name, "error": str(exc)})

    return report


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def print_report(rep: dict) -> None:
    print(f"\n{BOLD}{Path(rep['file']).name}{RESET}   {rep['size_mb']:.1f} MB")
    print("=" * 100)

    if rep["global_attrs"]:
        print(f"\n{BOLD}Attributi globali{RESET}")
        for k, v in rep["global_attrs"].items():
            print(f"  {k:<28} {DIM}{v}{RESET}")

    print(f"\n{BOLD}Dimensioni{RESET}")
    print("  " + "  ".join(f"{k}={v}" for k, v in rep["dimensions"].items()))

    by_role: dict[str, list[dict]] = {}
    for v in rep["variables"]:
        by_role.setdefault(v.get("role", "error"), []).append(v)

    order = [
        "sar",
        "incidence",
        "target",
        "distance",
        "geometry",
        "passive",
        "weather",
        "other",
        "error",
    ]
    for role in order:
        items = by_role.get(role)
        if not items:
            continue
        print(f"\n{BOLD}{role.upper()}{RESET}  {DIM}({len(items)}){RESET}")
        print(
            f"  {'variabile':<30} {'shape':<18} {'dtype':<9} {'mask%':>7} "
            f"{'min':>10} {'max':>10} {'mean':>10}"
        )
        print("  " + "-" * 96)
        for v in items:
            if "error" in v:
                print(f"  {v['name']:<30} {YELLOW}errore: {v['error']}{RESET}")
                continue
            shape = "×".join(str(s) for s in v["shape"])
            fmt = lambda x: f"{x:>10.3f}" if isinstance(x, float) else f"{'—':>10}"  # noqa: E731
            masked = v["pct_masked"]
            mcol = YELLOW if masked > 20 else ""
            mrst = RESET if masked > 20 else ""
            print(
                f"  {v['name']:<30} {shape:<18} {v['dtype']:<9} "
                f"{mcol}{masked:>6.1f}%{mrst} "
                f"{fmt(v['vmin'])} {fmt(v['vmax'])} {fmt(v['vmean'])}"
            )
            if v.get("unique_values") is not None:
                vals = ", ".join(f"{u:g}" for u in v["unique_values"])
                print(f"    {DIM}livelli ({v['n_unique']}): {vals}{RESET}")
            elif v.get("n_unique") is not None:
                print(f"    {DIM}valori distinti: {v['n_unique']}{RESET}")

    if not by_role.get("target"):
        print(f"\n  {YELLOW}Nessuna variabile classificata come target.{RESET}")


def check_coherence(rep: dict) -> list[str]:
    """Cross-checks that catch fill values pretending to be data.

    The one that matters: AI4Arctic pads the SAR swath with a literal ``0.0``.
    It is not NaN and there is no ``_FillValue`` attribute, so every naive
    statistic counts it as a measurement. On the verified scene that is 96.6%
    of the array, which would drag any normalisation to zero and would train
    the model on a huge constant region.

    Returns a list of human-readable warnings.
    """
    warnings: list[str] = []
    by_name = {v["name"]: v for v in rep["variables"] if "error" not in v}

    sar = [v for v in by_name.values() if v["role"] in ("sar", "incidence")]
    targets = [v for v in by_name.values() if v["role"] == "target"]

    for v in sar:
        if v["pct_exact_zero"] > 5.0:
            warnings.append(
                f"{v['name']}: {v['pct_exact_zero']:.1f}% dei pixel è esattamente 0.0 e NON è "
                f"marcato come mancante. È padding dello swath, non backscatter. "
                f"Va escluso con la maschera del target, mai usato per statistiche."
            )

    # If the SAR padding and the target's invalid region coincide, then masking
    # on the target is sufficient — which is the assumption patches.py makes.
    if sar and targets:
        sar_zero = max(v["pct_exact_zero"] for v in sar)
        tgt_masked = max(v["pct_masked"] for v in targets)
        if sar_zero > 5.0:
            delta = abs(sar_zero - tgt_masked)
            if delta < 1.0:
                warnings.append(
                    f"Padding SAR ({sar_zero:.1f}%) e invalidi del target ({tgt_masked:.1f}%) "
                    f"coincidono: mascherare sul target è sufficiente."
                )
            else:
                warnings.append(
                    f"ATTENZIONE: padding SAR {sar_zero:.1f}% ma invalidi del target "
                    f"{tgt_masked:.1f}% — NON coincidono (scarto {delta:.1f} punti). "
                    f"Mascherare sul solo target lascerebbe passare padding come dato."
                )

    for v in targets:
        if v["pct_masked"] > 90.0:
            warnings.append(
                f"{v['name']}: solo il {100 - v['pct_masked']:.1f}% della scena è annotato. "
                f"Poche patch utilizzabili."
            )

    return warnings


def _percent_bar(frac: float, width: int = 40) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def class_histogram(path: Path, var_names: list[str]) -> dict[str, dict[int, int]]:
    """Exact per-class pixel counts for the named categorical variables."""
    import netCDF4

    out: dict[str, dict[int, int]] = {}
    with netCDF4.Dataset(path, "r") as ds:
        for name in var_names:
            if name not in ds.variables:
                continue
            arr = _to_numpy(ds.variables[name][:])
            finite = arr[np.isfinite(arr)] if arr.dtype.kind in "fc" else arr.ravel()
            vals, counts = np.unique(finite.astype("int64"), return_counts=True)
            out[name] = {int(v): int(c) for v, c in zip(vals, counts, strict=True)}
    return out


def print_histograms(hists: dict[str, dict[int, int]]) -> None:
    for name, counts in hists.items():
        total = sum(counts.values())
        print(f"\n  {BOLD}{name}{RESET}  {DIM}{total:,} pixel{RESET}")
        for value, n in sorted(counts.items()):
            frac = n / total if total else 0.0
            tag = f"{DIM}(fill/ignore){RESET}" if value == 255 else ""
            print(f"    {value:>4}  {_percent_bar(frac)}  {100 * frac:6.2f}%  {n:>12,}  {tag}")


# ---------------------------------------------------------------------------
# Figure — Gate 0.2 requires HH, HV and the SOD chart side by side
# ---------------------------------------------------------------------------


def make_figure(path: Path, rep: dict, out_png: Path, downsample: int = 4) -> bool:
    """Render SAR channels next to the SOD chart. Returns False if it could not."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import netCDF4

    sar = [v["name"] for v in rep["variables"] if v.get("role") == "sar"]
    targets = [v["name"] for v in rep["variables"] if v.get("role") == "target"]
    # Prefer SOD, then SIC, then whatever target exists.
    targets.sort(key=lambda n: 0 if "SOD" in n.upper() else 1 if "SIC" in n.upper() else 2)
    panels = sar[:2] + targets[:2]
    if not panels:
        return False

    with netCDF4.Dataset(path, "r") as ds:
        arrays = {}
        for name in panels:
            a = _to_numpy(ds.variables[name][:])
            if a.ndim != 2:
                continue
            arrays[name] = a[::downsample, ::downsample]

    if not arrays:
        return False

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6), facecolor="white")
    if n == 1:
        axes = [axes]

    for ax, (name, arr) in zip(axes, arrays.items(), strict=True):
        is_target = name in targets
        plot = arr.astype("float64")
        if is_target:
            plot[plot == 255] = np.nan
            cmap, vmin, vmax = "viridis", np.nanmin(plot), np.nanmax(plot)
        else:
            cmap = "gray"
            finite = plot[np.isfinite(plot)]
            # Percentile stretch: SAR backscatter has a long tail and a raw
            # min/max scaling renders the scene almost uniformly black.
            vmin, vmax = (
                (np.percentile(finite, 2), np.percentile(finite, 98)) if finite.size else (0, 1)
            )
        im = ax.imshow(plot, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(Path(rep["file"]).name, fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene", type=Path, help="percorso del file netCDF")
    ap.add_argument("--figure", type=Path, default=None, help="salva un PNG di confronto")
    ap.add_argument("--json", type=Path, default=None, help="salva il report in JSON")
    ap.add_argument("--max-unique", type=int, default=32)
    args = ap.parse_args(argv)

    if not args.scene.exists():
        print(f"File non trovato: {args.scene}", file=sys.stderr)
        return 1

    rep = inspect_scene(args.scene, max_unique=args.max_unique)
    print_report(rep)

    issues = check_coherence(rep)
    rep["coherence_warnings"] = issues
    if issues:
        print(f"\n{BOLD}Coerenza{RESET}")
        for w in issues:
            colour = YELLOW if w.startswith("ATTENZIONE") else DIM
            print(f"  {colour}·{RESET} {w}")

    targets = [v["name"] for v in rep["variables"] if v.get("role") == "target"]
    if targets:
        hists = class_histogram(args.scene, targets)
        print_histograms(hists)
        rep["class_histograms"] = {k: {str(a): b for a, b in v.items()} for k, v in hists.items()}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=2))
        print(f"\n{GREEN}Report JSON:{RESET} {args.json}")

    if args.figure:
        if make_figure(args.scene, rep, args.figure):
            print(f"{GREEN}Figura:{RESET} {args.figure}")
        else:
            print(f"{YELLOW}Nessuna variabile 2D adatta alla figura.{RESET}")

    print(
        f"\n{DIM}Prossimo passo: riportare questi nomi in configs/data.yaml. "
        f"Se un nome differisce dalla SPEC, si aggiorna la config, non il codice.{RESET}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
