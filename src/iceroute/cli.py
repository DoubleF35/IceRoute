"""``iceroute``: one entry point for every stage of the pipeline.

Each subcommand is a thin dispatch to the module that owns the stage, so the
CLI never becomes a second implementation of anything. Arguments after the
subcommand are passed through unchanged; ``iceroute <cmd> --help`` shows the
stage's own options.

    iceroute run data/raw/test/20180707T113313_cis_prep.nc --out out/
    iceroute train --config configs/train3_lovasz.yaml
    iceroute calibrate
    iceroute console
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# name -> (module with a ``main(argv)``, one-line help)
STAGES: dict[str, tuple[str, str]] = {
    "inspect": ("iceroute.data.inspect", "report on one AI4Arctic scene"),
    "patches": ("iceroute.data.patches", "extract and cache training patches"),
    "train": ("iceroute.model.train", "train the U-Net"),
    "eval": ("iceroute.model.evaluate", "evaluate on the official test set"),
    "export": ("iceroute.model.export", "export the checkpoint to ONNX"),
    "quantize": ("iceroute.model.quantize", "INT8 quantization + size/latency/RAM"),
    "calibrate": ("iceroute.model.calibration", "temperature scaling + reliability (ECE)"),
    "risk": ("iceroute.model.risk", "safety vs. accuracy curves for risk-aware rules"),
    "route": ("iceroute.routing.run", "geocode + POLARIS + A* on scenes"),
    "precompute": ("iceroute.demo.precompute", "build the demo bundle in data/demo/"),
    "report": ("iceroute.report.metrics", "consolidate every measurement"),
    "figures": ("iceroute.report.figures", "render the result figures"),
    "console": ("iceroute.web.console", "build console.html"),
    "demo-static": ("iceroute.web.standalone", "build the single-file demo.html"),
}


def _run_scene(argv: list[str]) -> int:
    """Full pipeline on one scene: SAR -> mask -> POLARIS -> route -> payload."""
    ap = argparse.ArgumentParser(
        prog="iceroute run",
        description="Segment one Sentinel-1 scene, compute POLARIS-constrained routes "
        "for every configured ship class, and write the downlink payload.",
    )
    ap.add_argument("scene", type=Path, help="an AI4Arctic *_prep.nc scene")
    ap.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--train-config", type=Path, default=None)
    ap.add_argument("--routing-config", type=Path, default=Path("configs/routing.yaml"))
    args = ap.parse_args(argv)

    if not args.scene.exists():
        ap.error(f"scene not found: {args.scene}")
    from iceroute.demo import precompute

    fwd = [
        "--scenes",
        args.scene.name,
        "--raw-dir",
        str(args.scene.parent),
        "--out",
        str(args.out),
        "--config",
        str(args.routing_config),
    ]
    if args.device:
        fwd += ["--device", args.device]
    if args.train_config:
        fwd += ["--train-config", str(args.train_config)]
    return int(precompute.main(fwd) or 0)


def _streamlit(argv: list[str]) -> int:
    app = Path(__file__).parent / "demo" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.call(cmd + argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    width = max(len(k) for k in [*STAGES, "run", "demo"])
    usage = "\n".join(
        [
            "usage: iceroute <command> [args...]",
            "",
            "commands:",
            f"  {'run':<{width}}  full pipeline on one scene (SAR -> route -> payload)",
            f"  {'demo':<{width}}  launch the Streamlit demo",
            *(f"  {k:<{width}}  {h}" for k, (_, h) in STAGES.items()),
            "",
            "`iceroute <command> --help` shows the options of each stage.",
        ]
    )
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(usage)
        return 0
    if argv[0] in {"-V", "--version"}:
        from importlib.metadata import PackageNotFoundError, version

        try:
            print(f"iceroute {version('iceroute')}")
        except PackageNotFoundError:
            print("iceroute (not installed)")
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        return _run_scene(rest)
    if cmd == "demo":
        return _streamlit(rest)
    if cmd not in STAGES:
        print(f"iceroute: unknown command '{cmd}'\n\n{usage}", file=sys.stderr)
        return 2
    module = importlib.import_module(STAGES[cmd][0])
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
