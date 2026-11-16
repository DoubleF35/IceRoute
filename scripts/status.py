#!/usr/bin/env python3
"""Project status: which pipeline stages are done and what is missing.

Run with ``make status``. Reads the filesystem only — it never re-runs anything,
so it is safe to call at any moment, including right before a live demo.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

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

    print(f"\n{BOLD}IceRoute — status{RESET}")
    print("=" * 62)

    # --- Pipeline stages ----------------------------------------------------
    print(f"\n{BOLD}Stages{RESET}")

    g01 = _exists(".venv/bin/python") and _exists("requirements.txt")
    print(f"  {mark(g01)} 0.1  environment + CUDA")

    g02 = _exists("configs/data.yaml") and _count("inspect_*.png", "artifacts/figures") > 0
    print(f"  {mark(g02, partial=n_train > 0)} 0.2  data inspection, variable names verified")

    g1 = n_patch > 0 and _exists("artifacts/data_stats.json")
    print(f"  {mark(g1, partial=n_patch > 0)} 1    patch pipeline")

    best_cfg = REPO / "artifacts" / "best_config.txt"
    stem = "unet3lov"
    if best_cfg.exists():
        import yaml

        cfg = yaml.safe_load((REPO / best_cfg.read_text().strip()).read_text())
        stem = Path(cfg["output"]["onnx_int8"]).name.replace("_int8.onnx", "")
    has_eval = bool(m.get("evaluation", {}).get("scenes", {}).get("aggregate"))
    g2 = _exists(f"artifacts/models/{stem}_int8.onnx") and has_eval
    print(
        f"  {mark(g2, partial=_exists(f'artifacts/models/{stem}_fp32.pt'))} 2    "
        f"trained model + INT8 ({stem})"
    )

    g_cal = _exists("artifacts/calibration.json") and _exists("artifacts/risk_curve.json")
    print(f"  {mark(g_cal, partial=_exists('artifacts/calibration.json'))} 2.5  calibration + risk")

    n_geojson = _count("*.geojson", "data/demo/routes")
    g3 = n_geojson >= 15  # 5 demo scenes x 3 ice classes
    print(f"  {mark(g3, partial=n_geojson > 0)} 3    geocoding + POLARIS + routes")

    g4 = _exists("data/demo/index.json") and _exists("console.html") and _exists("demo.html")
    print(f"  {mark(g4, partial=_exists('data/demo/index.json'))} 4    demo bundle + pages")

    # --- Data ---------------------------------------------------------------
    print(f"\n{BOLD}Data{RESET}")
    print(f"  Train scenes        {n_train:>4}  {DIM}(target 60){RESET}")
    print(f"  Test scenes         {n_test:>4}")
    print(f"  Cached patches      {n_patch:>4}")
    print(f"  data/raw/           {raw_gb:>4.1f} GB")
    n_part = _count("*.part", "data/raw/train") + _count("*.part", "data/raw/test")
    if n_part:
        print(
            f"  {YELLOW}Partial downloads   {n_part:>4}{RESET}  {DIM}(resume with 'make download'){RESET}"
        )

    # --- Measured numbers ---------------------------------------------------
    print(f"\n{BOLD}Measured numbers{RESET}")
    if not m:
        print(f"  {DIM}artifacts/metrics.json does not exist yet — 'make report'{RESET}")
    else:

        def show(label: str, *keys: str, fmt: str = "{}") -> None:
            cur: object = m
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    print(f"  {label:<20}{DIM}—{RESET}")
                    return
                cur = cur[k]
            print(f"  {label:<20}{fmt.format(cur)}")

        show("mIoU (test)", "evaluation", "scenes", "aggregate", "miou", fmt="{:.3f}")
        show(
            "Ice/water accuracy",
            "evaluation",
            "scenes",
            "aggregate",
            "binary_accuracy",
            fmt="{:.3f}",
        )
        show("INT8 model", "quantization", "size", "int8_mb", fmt="{:.2f} MB")
        show(
            "INT8 tile latency",
            "quantization",
            "latency",
            "single_thread",
            "int8",
            "median_ms_per_tile",
            fmt="{:.1f} ms (1 thread)",
        )
        show("Downlink ratio", "compression", "ratio_median", fmt="{:.0f}:1")

    # --- Open provisional values --------------------------------------------
    dec = REPO / "docs" / "decisions.md"
    if dec.exists():
        open_items = [
            ln
            for ln in dec.read_text().splitlines()
            if ln.startswith("| P-") and "Closed" not in ln.split("|")[3]
        ]
        colour = RED if open_items else GREEN
        print(f"\n{BOLD}Open PROVISIONAL values{RESET}  {colour}{len(open_items)}{RESET}")
        for ln in open_items:
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 3:
                print(f"  {DIM}{parts[1]}{RESET}  {parts[2]}")
        if open_items:
            print(f"  {DIM}None of these may appear in a public claim while it is open.{RESET}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
