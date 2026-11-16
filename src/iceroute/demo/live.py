"""Run the real pipeline while somebody is watching: the one exception to precompute.

The demo reads precomputed files so it starts instantly and cannot fail during
a live run. That is the right default and it stays the default. It also invites
the one question this module answers: *how do I know these aren't just pictures
you made earlier?*

So: the scene opens from ``data/demo/`` immediately, and a button re-runs the
entire chain — read, infer, geocode, RIO, A\\*, simplify, compare — on the raw
NetCDF, into a scratch directory, with the stages ticking over as they finish.

Two decisions hold this together
--------------------------------
**It calls ``precompute.process_scene``, not a copy of it.** A second
implementation for the live path would make "watch it compute" a statement
about code the product does not ship. Progress display is the only thing added,
through the ``on_stage`` hook, and the output lands in the same layout the UI
already reads — so the live result is displayed by the same code that displays
the precomputed one.

**The result is compared against the precomputed route, and the verdict is
shown either way.** If the live run disagrees, that is worth knowing during
development rather than discovering in the middle of a demo. Measured, not
asserted: :func:`agreement` reports the cells that differ.

Roughly fifteen seconds on a GPU for all three vessel classes, most of it in
A\\*. Slower on CPU. Whatever it costs, it costs it in front of whoever is
watching, which is the point.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("live")


@dataclass
class Stage:
    name: str
    seconds: float = 0.0
    done: bool = False


@dataclass
class LiveRun:
    """What one live computation produced, and how long each part took."""

    assets: object = None
    out_dir: Path | None = None
    stages: list[Stage] = field(default_factory=list)
    total_seconds: float = 0.0
    device: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.assets is not None and not self.error


def load_model(train_config: Path, device: str | None = None):
    """The deployed checkpoint, loaded once. Cache this in the caller."""
    import torch

    from iceroute.model.evaluate import load_checkpoint

    cfg = yaml.safe_load(Path(train_config).read_text())
    ckpt = Path(cfg["output"]["best_checkpoint"])
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(ckpt, dev)
    return model, dev


def compute(
    scene_path: Path,
    *,
    model,
    device: str,
    out_dir: Path,
    train_config: Path,
    data_config: Path = Path("configs/data.yaml"),
    routing_config: Path = Path("configs/routing.yaml"),
    downscale: int = 4,
    on_stage=None,
) -> LiveRun:
    """The whole chain, on the raw scene, now.

    Failures are returned rather than raised. This runs behind a button in a
    live demo; a traceback across the screen is a worse outcome than a red box
    saying what broke, and the precomputed result is still on screen behind it.
    """
    from iceroute.data.loader import load_config
    from iceroute.demo.precompute import process_scene
    from iceroute.routing.polaris import PolarisTable

    run = LiveRun(out_dir=out_dir, device=device)
    clock = {"t0": time.monotonic()}

    def stage(name: str) -> None:
        now = time.monotonic()
        if run.stages:
            run.stages[-1].seconds = now - clock["t0"]
            run.stages[-1].done = True
        clock["t0"] = now
        run.stages.append(Stage(name))
        if on_stage:
            on_stage(run.stages)

    started = time.monotonic()
    try:
        run.assets = process_scene(
            Path(scene_path),
            model=model,
            data_cfg=load_config(data_config),
            routing_cfg=yaml.safe_load(Path(routing_config).read_text()),
            train_cfg=yaml.safe_load(Path(train_config).read_text()),
            out_dir=Path(out_dir),
            table=PolarisTable.load(),
            device=device,
            downscale=downscale,
            on_stage=stage,
        )
    except Exception as exc:  # noqa: BLE001 — must never break the demo
        run.error = f"{type(exc).__name__}: {exc}"
        log.exception("live computation failed")

    if run.stages:
        run.stages[-1].seconds = time.monotonic() - clock["t0"]
        run.stages[-1].done = not run.error
    run.total_seconds = time.monotonic() - started
    return run


def agreement(live_overlay: dict, stored_overlay: dict) -> dict:
    """Does the live route match the precomputed one, cell for cell.

    Reported rather than assumed. The two runs execute the same function on the
    same input, so they should agree exactly — and if they ever stop agreeing,
    the honest thing is a line on screen saying so, not a demo that quietly
    shows whichever answer came last.
    """
    a = [tuple(c) for c in ((live_overlay or {}).get("iceroute") or {}).get("path", [])]
    b = [tuple(c) for c in ((stored_overlay or {}).get("iceroute") or {}).get("path", [])]
    if not a or not b:
        return {
            "comparable": False,
            "identical": False,
            "note": "one of the two routes does not exist: no comparison",
        }
    if a == b:
        return {
            "comparable": True,
            "identical": True,
            "n_cells": len(a),
            "note": f"identical to the precomputed route, {len(a)} cells",
        }
    same = sum(1 for x, y in zip(a, b, strict=False) if x == y)
    return {
        "comparable": True,
        "identical": False,
        "n_cells": len(a),
        "n_matching": same,
        "note": (
            f"DIFFERENT from the precomputed route: {len(a)} cells against {len(b)}, "
            f"{same} match. This must be understood before the demo is shown."
        ),
    }
