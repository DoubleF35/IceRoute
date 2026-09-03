#!/usr/bin/env python3
"""Project status: which gates are green, what is missing, how much time is left.

Run with ``make status``. Reads the filesystem only — it never re-runs anything,
so it is safe to call at any moment, including five minutes before the pitch.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FEATURE_FREEZE = date(2026, 10, 1)
PITCH_DAY = date(2026, 10, 7)

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)


def _exists(rel: str) -> bool:
    return (REPO / rel).exists()


def _count(pattern: str, root: str = ".") -> int:
    return len(list((REPO / root).glob(pattern)))


def _size_gb(rel: str, pattern: str = "**/*") -> float:
    base = REPO / rel
    if not base.exists():
        return 0.0
    return sum(f.stat().st_size for f in base.glob(pattern) if f.is_file()) / 1024**3


def _metrics() -> dict:
    p = REPO / "artifacts" / "metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def mark(done: bool, partial: bool = False) -> str:
    if done:
        return f"{GREEN}✓{RESET}"
    if partial:
        return f"{YELLOW}◐{RESET}"
    return f"{RED}·{RESET}"


def main() -> int:
    m = _metrics()
    n_train = _count("*.nc", "data/raw/train")
    n_test = _count("*.nc", "data/raw/test")
    n_patch = _count("*.npz", "data/processed")
    raw_gb = _size_gb("data/raw")

    print(f"\n{BOLD}IceRoute — stato{RESET}")
    print("=" * 62)

    # --- Calendario ---------------------------------------------------------
    today = date.today()
    to_freeze = (FEATURE_FREEZE - today).days
    to_pitch = (PITCH_DAY - today).days
    freeze_col = RED if to_freeze <= 7 else YELLOW if to_freeze <= 14 else DIM
    print(f"\n{BOLD}Calendario{RESET}")
    print(f"  Oggi                {today.isoformat()}")
    print(f"  Feature freeze      {FEATURE_FREEZE.isoformat()}  {freeze_col}fra {to_freeze} giorni{RESET}")
    print(f"  Wave / OGR Torino   {PITCH_DAY.isoformat()}  {DIM}fra {to_pitch} giorni{RESET}")

    # --- Gate ---------------------------------------------------------------
    print(f"\n{BOLD}Gate{RESET}")

    g01 = _exists(".venv/bin/python") and _exists("requirements.txt")
    print(f"  {mark(g01)} 0.1  ambiente + CUDA")

    g02 = _exists("configs/data.yaml") and _count("inspect_*.png", "artifacts/figures") > 0
    print(f"  {mark(g02, partial=n_train > 0)} 0.2  ispezione dati, nomi variabile verificati")

    g1 = n_patch > 0 and _exists("artifacts/data_stats.json")
    print(f"  {mark(g1, partial=n_patch > 0)} 1    pipeline patch")

    has_miou = "miou" in str(m).lower()
    g2 = _exists("artifacts/models/unet_int8.onnx") and has_miou
    print(f"  {mark(g2, partial=_exists('artifacts/models/unet_fp32.pt'))} 2    modello addestrato + INT8")

    n_geojson = _count("*.geojson", "data/demo")
    g3 = n_geojson >= 15  # 5 scene demo × 3 ice class
    print(f"  {mark(g3, partial=n_geojson > 0)} 3    geocodifica + POLARIS + rotte")

    g4 = _exists("data/demo/scene_meta.json") and _count("*.mp4", "artifacts/video") > 0
    print(f"  {mark(g4, partial=_exists('data/demo/scene_meta.json'))} 4    demo + video di backup")

    # --- Dati ---------------------------------------------------------------
    print(f"\n{BOLD}Dati{RESET}")
    print(f"  Scene train         {n_train:>4}  {DIM}(obiettivo 60){RESET}")
    print(f"  Scene test          {n_test:>4}")
    print(f"  Patch in cache      {n_patch:>4}")
    print(f"  data/raw/           {raw_gb:>4.1f} GB")
    n_part = _count("*.part", "data/raw/train") + _count("*.part", "data/raw/test")
    if n_part:
        print(f"  {YELLOW}Download parziali   {n_part:>4}{RESET}  {DIM}(riprendi con 'make download'){RESET}")

    # --- Numeri misurati ----------------------------------------------------
    print(f"\n{BOLD}Numeri misurati{RESET}")
    if not m:
        print(f"  {DIM}artifacts/metrics.json non esiste ancora — 'make report'{RESET}")
    else:
        def show(label: str, *keys: str, fmt: str = "{}") -> None:
            cur: object = m
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    print(f"  {label:<20}{DIM}—{RESET}")
                    return
                cur = cur[k]
            print(f"  {label:<20}{fmt.format(cur)}")

        show("mIoU (val)", "model", "miou", fmt="{:.3f}")
        show("Acc. ghiaccio/acqua", "model", "binary_accuracy", fmt="{:.3f}")
        show("Modello INT8", "onboard", "int8_size_mb", fmt="{:.2f} MB")
        show("Latenza scena", "onboard", "latency_median_s", fmt="{:.2f} s")
        show("Compressione", "downlink", "ratio", fmt="{:.1f}:1")

    # --- PROVVISORI aperti --------------------------------------------------
    dec = REPO / "DECISIONS.md"
    if dec.exists():
        open_items = [
            ln for ln in dec.read_text().splitlines()
            if ln.startswith("| P-") and "| Aperta |" in ln
        ]
        colour = RED if open_items else GREEN
        print(f"\n{BOLD}Voci PROVVISORIE aperte{RESET}  {colour}{len(open_items)}{RESET}")
        for ln in open_items:
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 3:
                print(f"  {DIM}{parts[1]}{RESET}  {parts[2]}")
        if open_items:
            print(f"  {DIM}Nessuna di queste può finire nel deck finché è aperta.{RESET}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
