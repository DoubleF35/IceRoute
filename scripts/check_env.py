#!/usr/bin/env python3
"""Environment gate (SPEC Gate 0.1).

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
            f"Python {v.major}.{v.minor}.{v.micro} (atteso 3.11)",
            "I pin in requirements.txt sono risolti per 3.11. Rigenerali se cambi versione.",
        )
    print(f"    {DIM}{sys.executable}{RESET}")


def check_torch() -> None:
    section("PyTorch e GPU")
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        fail("torch non importabile", str(exc))
        return

    ok(f"torch {torch.__version__}")

    if not torch.cuda.is_available():
        fail(
            "CUDA non disponibile — torch.cuda.is_available() è False",
            "Il training su CPU non è praticabile nei tempi previsti. "
            "Verifica driver NVIDIA e che la ruota torch sia quella CUDA, non la +cpu.",
        )
        return

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    ok(f"CUDA disponibile — {name}")
    print(
        f"    {DIM}CUDA runtime {torch.version.cuda} · capability {props.major}.{props.minor}{RESET}"
    )

    if vram_gb < 5.0:
        warn(f"VRAM {vram_gb:.1f} GB", "Sotto le attese: riduci batch size in configs/train.yaml.")
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
            warn("matmul fp16 su GPU ha prodotto NaN")
        else:
            ok("allocazione e matmul fp16 su GPU riuscite")
    except Exception as exc:  # noqa: BLE001 — vogliamo davvero qualsiasi eccezione qui
        fail("La GPU non esegue una matmul", str(exc))

    # AMP is mandatory per SPEC §7.2 given 6 GB of VRAM.
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        ok("torch.amp disponibile (AMP obbligatoria con 6 GB)")
    else:
        fail("torch.amp non disponibile")


def check_imports() -> None:
    section("Dipendenze")
    # (module, why it matters if missing)
    required = [
        ("numpy", "array"),
        ("scipy", "interpolazione griglia geografica"),
        ("xarray", "lettura netCDF"),
        ("netCDF4", "backend netCDF"),
        ("skimage", "morfologia mask"),
        ("sklearn", "matrice di confusione, soglia baseline"),
        ("pyproj", "CRS e distanza geodetica (test geocodifica)"),
        ("pyresample", "ricampionamento su EPSG:3413"),
        ("shapely", "semplificazione rotta"),
        ("onnx", "export"),
        ("onnxruntime", "inferenza INT8"),
        ("streamlit", "demo"),
        ("matplotlib", "figure"),
        ("yaml", "config"),
        ("requests", "download dataset"),
        ("psutil", "RAM di picco"),
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
        fail(f"{mod} mancante", f"serve per: {why}")


def check_disk() -> None:
    section("Disco")
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / 1024**3
    total_gb = usage.total / 1024**3
    msg = f"{free_gb:.0f} GB liberi su {total_gb:.0f} GB"
    if free_gb < MIN_FREE_GB:
        fail(
            f"Spazio insufficiente: {msg}",
            f"Servono almeno {MIN_FREE_GB:.0f} GB per il sottoinsieme del dataset "
            "e la cache delle patch.",
        )
    else:
        ok(msg)

    raw = REPO_ROOT / "data" / "raw"
    if raw.exists():
        n = len(list(raw.rglob("*.nc")))
        if n:
            size_gb = sum(f.stat().st_size for f in raw.rglob("*.nc")) / 1024**3
            ok(f"data/raw/: {n} scene, {size_gb:.1f} GB")
        else:
            print(f"    {DIM}data/raw/ vuota — lancia 'make download'{RESET}")


def check_optional_tools() -> None:
    section("Strumenti opzionali")
    for tool, why in [
        ("ffmpeg", "video di backup (make video) — richiesto entro il 1 ottobre"),
        ("git", "versionamento"),
    ]:
        if shutil.which(tool):
            ok(tool)
        else:
            warn(f"{tool} non installato", why)


def main() -> int:
    print("\nIceRoute — verifica ambiente (Gate 0.1)")
    print("=" * 60)

    check_python()
    check_torch()
    check_imports()
    check_disk()
    check_optional_tools()

    print("\n" + "=" * 60)
    if _failures:
        print(f"{RED}Gate 0.1 NON superato{RESET} — {len(_failures)} problema/i bloccante/i:")
        for f in _failures:
            print(f"  · {f}")
        print()
        return 1

    if _warnings:
        print(f"{GREEN}Gate 0.1 superato{RESET} con {len(_warnings)} avviso/i non bloccante/i.")
    else:
        print(f"{GREEN}Gate 0.1 superato.{RESET}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
