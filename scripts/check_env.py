#!/usr/bin/env python3
"""Environment check.

Verifies the machine can actually run the project before anyone wastes a day
discovering otherwise. Exits non-zero on a hard failure, prints a warning and
continues on a soft one.

Run: ``make setup`` (implicit) or ``make check-cuda``.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Free space below which downloading the dataset subset is not worth starting.
# 60 train scenes + ~20 test scenes, plus the patch cache, plus room to breathe.
MIN_FREE_GB = 60.0

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"

_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str, detail: str = "") -> None:
    print(f"  {YELLOW}!{RESET} {msg}")
    if detail:
        print(f"    {DIM}{detail}{RESET}")
    _warnings.append(msg)


def fail(msg: str, detail: str = "") -> None:
    print(f"  {RED}✗{RESET} {msg}")
    if detail:
        print(f"    {DIM}{detail}{RESET}")
    _failures.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")


def check_python() -> None:
    section("Python")
    v = sys.version_info
    if (v.major, v.minor) == (3, 11):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        # Not fatal, but the pins in requirements.txt were resolved for 3.11.
        warn(
            f"Python {v.major}.{v.minor}.{v.micro} (expected 3.11)",
            "The pins in requirements.txt are resolved for 3.11. Regenerate them if you change version.",
        )
    print(f"    {DIM}{sys.executable}{RESET}")


def check_torch() -> None:
    section("PyTorch and GPU")
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        fail("torch cannot be imported", str(exc))
        return

    ok(f"torch {torch.__version__}")

    if not torch.cuda.is_available():
        fail(
            "CUDA not available — torch.cuda.is_available() is False",
            "Training on CPU is not practical in any reasonable time. "
            "Check the NVIDIA driver and that the torch wheel is the CUDA one, not +cpu.",
        )
        return

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    ok(f"CUDA available — {name}")
    print(
        f"    {DIM}CUDA runtime {torch.version.cuda} · capability {props.major}.{props.minor}{RESET}"
    )

    if vram_gb < 5.0:
        warn(
            f"VRAM {vram_gb:.1f} GB",
            "Below expectations: lower the batch size in configs/train.yaml.",
        )
    else:
        ok(f"VRAM {vram_gb:.1f} GB")

    # A real allocation + matmul. `is_available()` returning True while the
    # driver is actually broken is a failure mode that shows up at the worst
    # possible moment, so we force the issue here.
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = (a @ a).float().sum().item()
        del a
        torch.cuda.empty_cache()
        if b != b:  # NaN check
            warn("fp16 matmul on GPU produced NaN")
        else:
            ok("fp16 allocation and matmul on GPU succeeded")
    except Exception as exc:  # noqa: BLE001 — we really do want any exception here
        fail("The GPU cannot run a matmul", str(exc))

    # AMP is mandatory given 6 GB of VRAM (see configs/train.yaml).
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        ok("torch.amp available (AMP is mandatory with 6 GB)")
    else:
        fail("torch.amp not available")


def check_imports() -> None:
    section("Dependencies")
    # (module, why it matters if missing)
    required = [
        ("numpy", "array"),
        ("scipy", "geographic grid interpolation"),
        ("xarray", "netCDF reading"),
        ("netCDF4", "backend netCDF"),
        ("skimage", "mask morphology"),
        ("sklearn", "confusion matrix, baseline threshold"),
        ("pyproj", "CRS and geodesic distance (geocoding test)"),
        ("pyresample", "resampling onto EPSG:3413"),
        ("shapely", "route simplification"),
        ("onnx", "export"),
        ("onnxruntime", "INT8 inference"),
        ("streamlit", "demo"),
        ("matplotlib", "figures"),
        ("yaml", "config"),
        ("requests", "dataset download"),
        ("psutil", "peak RAM"),
    ]
    missing = []
    for mod, why in required:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            ok(f"{mod} {ver}".rstrip())
        except ImportError:
            missing.append((mod, why))

    for mod, why in missing:
        fail(f"{mod} missing", f"needed for: {why}")


def check_disk() -> None:
    section("Disk")
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / 1024**3
    total_gb = usage.total / 1024**3
    msg = f"{free_gb:.0f} GB free of {total_gb:.0f} GB"
    if free_gb < MIN_FREE_GB:
        fail(
            f"Not enough space: {msg}",
            f"At least {MIN_FREE_GB:.0f} GB is needed for the dataset subset and the patch cache.",
        )
    else:
        ok(msg)

    raw = REPO_ROOT / "data" / "raw"
    if raw.exists():
        n = len(list(raw.rglob("*.nc")))
        if n:
            size_gb = sum(f.stat().st_size for f in raw.rglob("*.nc")) / 1024**3
            ok(f"data/raw/: {n} scenes, {size_gb:.1f} GB")
        else:
            print(f"    {DIM}data/raw/ empty — run 'make download'{RESET}")


def check_optional_tools() -> None:
    section("Optional tools")
    for tool, why in [
        ("ffmpeg", "backup video (make video)"),
        ("git", "version control"),
    ]:
        if shutil.which(tool):
            ok(tool)
        else:
            warn(f"{tool} not installed", why)


def main() -> int:
    print("\nIceRoute — environment check")
    print("=" * 60)

    check_python()
    check_torch()
    check_imports()
    check_disk()
    check_optional_tools()

    print("\n" + "=" * 60)
    if _failures:
        print(f"{RED}Environment check FAILED{RESET} — {len(_failures)} blocking problem(s):")
        for f in _failures:
            print(f"  · {f}")
        print()
        return 1

    if _warnings:
        print(
            f"{GREEN}Environment check passed{RESET} with {len(_warnings)} non-blocking warning(s)."
        )
    else:
        print(f"{GREEN}Environment check passed.{RESET}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
