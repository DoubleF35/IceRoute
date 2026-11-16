"""Write the measured numbers into ``configs/assumptions.yaml``.

That file is the one place every economic and performance figure is supposed to
come from, and it carries a ``source:`` beside each value so anyone reading it can
ask "based on what?" and get an answer. Several of its entries were written as
``null`` with a comment saying *MEASURED by report/compression.py* — an intention
nobody implemented. So the file kept saying ``provisional: true`` for quantities
``metrics.json`` had already measured, and the demo footer renders the file
verbatim: the person reading the console was being shown "provisional" next to
the very numbers being presented to them as measured.

Three things about how this writes.

**It edits lines, it does not dump YAML.** ``yaml.safe_dump`` would round-trip
the data and delete every comment in the file. Those comments are not
decoration — they are the justification for each number, which is the entire
point of keeping every assumption in one sourced file, and they are on screen
whenever the demo runs. A naked
``ratio: 439.2`` with no note saying what the numerator is would be a worse file
than the one with the ``null`` in it.

**It only replaces keys that already exist.** The structure and the prose stay
hand-written; only the numbers are machine-written. A key this module wants and
cannot find is an error, never something to append: if a measurement has no home
in the file yet, a person should decide where it goes and what to say about it.

**It writes no timestamp.** The file should change when a number changes and at
no other time, so that a diff in ``configs/assumptions.yaml`` is a signal rather
than noise from the last ``make report``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("assumptions")

METRICS = Path("artifacts/metrics.json")
ASSUMPTIONS = Path("configs/assumptions.yaml")

# A line we are allowed to rewrite. Anchored to an indented key, because the
# sections themselves sit at column zero.
_KEY = re.compile(r"^(?P<indent>[ \t]+)(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<rest>.*)$")
_SECTION = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*):")

# What a managed value is allowed to look like *before* we touch it. Anything
# else means a person has written prose there, and prose wins over this module.
_REPLACEABLE = re.compile(
    r"^\s*(null|~|true|false|-?\d[\d_]*\.?\d*(?:[eE][-+]?\d+)?|\"[^\"#]*\"|'[^'#]*')\s*$"
)


class SyncError(RuntimeError):
    """The file and this module disagree about what exists. Never guess."""


@dataclass(frozen=True)
class Measure:
    """One measured number and the two places it lives."""

    section: str
    key: str
    source: tuple[str, ...]  # path into metrics.json
    fmt: str = "{:d}"

    @property
    def where(self) -> str:
        return f"{self.section}.{self.key}"


def _dig(d: Any, path: tuple[str, ...]) -> Any:
    for step in path:
        if not isinstance(d, dict):
            return None
        d = d.get(step)
    return d


# ---------------------------------------------------------------------------
# What gets synced
# ---------------------------------------------------------------------------

MEASURES: tuple[Measure, ...] = (
    Measure("downlink", "ratio", ("__repr", "ratio"), "{:.1f}"),
    Measure("downlink", "raw_scene_bytes", ("__repr", "raw"), "{:d}"),
    Measure("downlink", "payload_bytes", ("__repr", "payload"), "{:d}"),
    Measure("downlink", "ratio_scene", ("__repr", "scene"), "{}"),
    Measure("downlink", "ratio_min", ("compression", "ratio_min"), "{:.1f}"),
    Measure("downlink", "ratio_max", ("compression", "ratio_max"), "{:.1f}"),
    Measure("model", "parameters", ("evaluation", "n_parameters"), "{:d}"),
    Measure("model", "onnx_int8_bytes", ("quantization", "size", "int8_bytes"), "{:d}"),
    Measure(
        "model",
        "latency_ms_per_tile",
        ("quantization", "latency", "single_thread", "int8", "median_ms_per_tile"),
        "{:.3f}",
    ),
    Measure("model", "int8_degradation", ("quantization", "accuracy", "miou_drop"), "{:.4f}"),
)


def representative_scene(metrics: dict) -> dict | None:
    """The scene whose compression ratio is the one we quote.

    The three downlink numbers must come from the *same* scene. Taking a median
    numerator and a median denominator independently gives a pair whose quotient
    is not the median ratio — and a careful reader will divide the two numbers
    we printed and get a third one we never stated. One scene, three
    numbers, and the division works.
    """
    comp = metrics.get("compression") or {}
    scenes = comp.get("per_scene") or []
    target = comp.get("ratio_median")
    if not scenes or not isinstance(target, int | float):
        return None
    best = min(scenes, key=lambda s: abs((s.get("ratio") or 0.0) - target))
    return {
        "scene": str(best.get("scene_id", "")),
        "raw": int((best.get("numerator") or {}).get("sar_channels_png") or 0),
        "payload": int((best.get("denominator") or {}).get("total") or 0),
        "ratio": float(best.get("ratio") or 0.0),
    }


# ---------------------------------------------------------------------------
# The line editor
# ---------------------------------------------------------------------------


def _render(value: Any, fmt: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return '"' + value.replace('"', "") + '"'
    if fmt.endswith("d}"):
        return fmt.format(int(round(value)))
    return fmt.format(value)


def set_value(text: str, section: str, key: str, value: Any, fmt: str = "{}") -> str:
    """Replace ``section.key``'s value, leaving every other byte of the file alone.

    The trailing comment survives, and so does the column it sits in when the new
    value is no wider than the old one.
    """
    lines = text.split("\n")
    inside, target = False, None
    for i, line in enumerate(lines):
        if _SECTION.match(line):
            inside = line.startswith(f"{section}:")
            continue
        if not inside:
            continue
        m = _KEY.match(line)
        if m and m.group("key") == key:
            target = (i, m)
            break

    if target is None:
        raise SyncError(
            f"{section}.{key} does not exist in {ASSUMPTIONS}. This module does not add "
            f"keys: add it by hand, with the comment that explains what it means."
        )

    i, m = target
    rest = m.group("rest")
    body, sep, comment = rest.partition("#")
    if not _REPLACEABLE.match(body):
        raise SyncError(
            f"{section}.{key} contains '{body.strip()}', which is not a measured value. "
            f"Someone wrote it by hand: it will not be overwritten."
        )

    new = _render(value, fmt)
    if sep:
        # Keep the comment where it was if the value still fits under it.
        pad = max(1, len(body) - len(new) - 1)
        lines[i] = f"{m.group('indent')}{key}: {new}{' ' * pad}#{comment}"
    else:
        lines[i] = f"{m.group('indent')}{key}: {new}"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@dataclass
class Result:
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def wrote(self) -> bool:
        return bool(self.changed)


def sync(metrics: dict, text: str) -> Result:
    """Fold every measured number into the file text. Pure: no I/O."""
    res = Result()
    repr_scene = representative_scene(metrics)
    source = dict(metrics)
    if repr_scene:
        source["__repr"] = repr_scene

    before = yaml.safe_load(text) or {}

    for meas in MEASURES:
        value = _dig(source, meas.source)
        if value is None or (isinstance(value, str) and not value):
            res.missing.append(meas.where)
            continue
        current = (before.get(meas.section) or {}).get(meas.key)
        text = set_value(text, meas.section, meas.key, value, meas.fmt)
        rendered = yaml.safe_load(text)[meas.section][meas.key]
        (res.unchanged if current == rendered else res.changed).append(meas.where)

    # Provisional is not a measurement, it is a consequence of one. A section
    # stops being provisional exactly when nothing in it is still missing.
    for section in ("downlink", "model"):
        wanted = [m for m in MEASURES if m.section == section]
        still_missing = any(m.where in res.missing for m in wanted)
        if section in (yaml.safe_load(text) or {}) and "provisional" in (
            yaml.safe_load(text).get(section) or {}
        ):
            was = yaml.safe_load(text)[section]["provisional"]
            text = set_value(text, section, "provisional", still_missing)
            if was != still_missing:
                res.changed.append(f"{section}.provisional")

    res.text = text
    return res


def run(repo: Path = Path("."), *, dry_run: bool = False) -> Result:
    metrics_path, target = repo / METRICS, repo / ASSUMPTIONS
    if not metrics_path.exists():
        raise SyncError(f"{METRICS} does not exist: run make report first.")
    if not target.exists():
        raise SyncError(
            f"{ASSUMPTIONS} does not exist. It is mandatory: every published figure "
            f"must trace back to it."
        )

    metrics = json.loads(metrics_path.read_text())
    original = target.read_text()
    res = sync(metrics, original)

    if res.missing:
        log.warning("Not measured, still provisional: %s", ", ".join(res.missing))
    if res.text != original and not dry_run:
        target.write_text(res.text)
    verb = "to update" if dry_run else "updated"
    for where in res.changed:
        log.info("  %s %s", where, verb)
    log.info(
        "%s: %d %s, %d already in sync, %d not measured",
        ASSUMPTIONS,
        len(res.changed),
        verb,
        len(res.unchanged),
        len(res.missing),
    )
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bring configs/assumptions.yaml in line with the measurements."
    )
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument(
        "--check", action="store_true", help="do not write; exit 1 if the file is out of sync"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    res = run(args.repo, dry_run=args.check)
    if args.check and res.wrote:
        log.error("configs/assumptions.yaml is behind the measurements: run make report")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
