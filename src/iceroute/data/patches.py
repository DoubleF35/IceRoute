"""Extract and cache 256x256 training patches.

The sampling rule is the part of this file that decides how good the model
gets, so it is worth stating plainly:

An Arctic scene is mostly open water or uniform ice. Sample it uniformly and
the model learns to predict one class everywhere, which scores well on pixel
accuracy and is useless for routing — routing decisions happen exactly at the
boundaries. So patches containing an **ice margin** (at least two SOD classes,
each covering at least 10% of the valid pixels) are kept preferentially, to a
configured target fraction.

Storage
-------
One memory-mapped ``.npy`` triplet per scene rather than one ``.npz`` per patch:
thousands of small files are slow to open, and a single archive cannot be
memory-mapped. ``np.load(..., mmap_mode='r')`` lets the OS page cache hold the
working set, so the cache can exceed RAM.

Augmentation is *not* baked into the cache. Flips and 90-degree rotations are
applied on the fly in ``PatchDataset``, which keeps the cache four times
smaller and gives a different draw each epoch.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from iceroute.data.loader import FILL_VALUE, Scene, load_config, load_scene

log = logging.getLogger("patches")

# Channel order in the cached array. Fixed here so the model and the cache
# cannot disagree; configs/train.yaml selects a prefix of it.
CHANNELS = ["hh", "hv", "incidence_angle", "distance_to_land"]


@dataclass
class PatchStats:
    """What came out of one scene."""

    scene_id: str
    region: str
    month: int
    n_candidates: int
    n_valid_enough: int
    n_margin: int
    n_uniform: int
    n_kept: int
    class_counts: dict[int, int]
    valid_fraction: float

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["class_counts"] = {str(k): v for k, v in self.class_counts.items()}
        return d


def is_margin_patch(
    labels: np.ndarray, valid: np.ndarray, *, min_classes: int, min_fraction: float
) -> bool:
    """True if the patch straddles an ice boundary.

    Criterion: at least ``min_classes`` distinct SOD classes present, each
    covering at least ``min_fraction`` of the *valid* pixels. Requiring a
    minimum share rather than mere presence is what excludes patches where a
    second class appears as a handful of speckled pixels.
    """
    n_valid = int(valid.sum())
    if n_valid == 0:
        return False
    vals, counts = np.unique(labels[valid], return_counts=True)
    keep = [(v, c) for v, c in zip(vals, counts, strict=True) if v != FILL_VALUE]
    if len(keep) < min_classes:
        return False
    significant = sum(1 for _, c in keep if c / n_valid >= min_fraction)
    return significant >= min_classes


def iter_windows(shape: tuple[int, int], size: int, stride: int):
    """Top-left corners of a tiling, with the last row/column flush to the edge."""
    h, w = shape
    if h < size or w < size:
        return
    rows = list(range(0, h - size + 1, stride))
    cols = list(range(0, w - size + 1, stride))
    if rows[-1] != h - size:
        rows.append(h - size)
    if cols[-1] != w - size:
        cols.append(w - size)
    for r in rows:
        for c in cols:
            yield r, c


@dataclass
class SceneScan:
    """Result of the cheap classification pass over one scene."""

    scene_id: str
    region: str
    month: int
    valid_fraction: float
    n_candidates: int
    margin: list[tuple[int, int]]
    uniform: list[tuple[int, int]]


def scan_scene(path: Path, cfg: dict) -> SceneScan:
    """Classify a scene's windows without loading the SAR channels.

    Deciding the margin/uniform balance needs only the ice chart and its
    validity mask — two of the six full-resolution arrays. Reading just those
    makes a global two-pass balance affordable: the alternative, balancing
    per-scene, breaks badly on scenes that contain no ice margin at all, which
    then contribute hundreds of uniform patches and swamp the target ratio.
    """
    import netCDF4

    from iceroute.data.loader import SCENE_RE, _invalid_mask, _read, parse_region

    pc = cfg["patches"]
    size, stride = int(pc["size"]), int(pc["stride"])
    max_invalid = float(pc["max_invalid_fraction"])
    crit = pc["margin_criterion"]
    tgt = cfg["variables"]["targets"]

    with netCDF4.Dataset(path, "r") as ds:
        sod_raw = _read(ds, tgt["sod"], "float64")
        scene_id = str(getattr(ds, "scene_id", path.name))
        original_id = str(getattr(ds, "original_id", ""))
        ice_service = str(getattr(ds, "ice_service", ""))

    valid = ~_invalid_mask(sod_raw)
    sod = np.where(valid, sod_raw, FILL_VALUE).astype("uint8")

    margin: list[tuple[int, int]] = []
    uniform: list[tuple[int, int]] = []
    n_candidates = 0
    for r, c in iter_windows(sod.shape, size, stride):
        n_candidates += 1
        vwin = valid[r : r + size, c : c + size]
        if 1.0 - vwin.mean() > max_invalid:
            continue
        ywin = sod[r : r + size, c : c + size]
        if is_margin_patch(
            ywin,
            vwin,
            min_classes=int(crit["min_classes"]),
            min_fraction=float(crit["min_class_fraction"]),
        ):
            margin.append((r, c))
        else:
            uniform.append((r, c))

    month = 0
    if m := SCENE_RE.match(scene_id):
        month = int(m.group(1)[4:6])

    return SceneScan(
        scene_id=scene_id,
        region=parse_region(original_id, ice_service),
        month=month,
        valid_fraction=float(valid.mean()),
        n_candidates=n_candidates,
        margin=margin,
        uniform=uniform,
    )


def allocate_uniform_quota(
    scans: list[SceneScan], target_margin: float, *, balance: str = "sampler"
) -> dict[str, int]:
    """How many uniform patches each scene may contribute.

    With ``balance="sampler"`` the answer is "all of them": the target margin
    fraction is achieved at training time by weighting the sampler instead of by
    throwing data away. See docs/decisions.md D-017 — the discard route kept 14.7% of
    the available patches and the model overfitted for lack of data while the
    ratio it was starved for was perfectly correct.

    With ``balance="discard"`` every margin patch is kept — they are the scarce,
    informative ones — and the global uniform budget that satisfies the ratio is
    split across scenes in proportion to how many uniform patches each offers,
    so a margin-free scene still contributes variety without dominating.
    """
    if balance == "sampler":
        return {s.scene_id: len(s.uniform) for s in scans}
    if balance != "discard":
        raise ValueError(f"balance must be 'sampler' or 'discard', not {balance!r}")

    total_margin = sum(len(s.margin) for s in scans)
    total_uniform = sum(len(s.uniform) for s in scans)

    if total_margin == 0 or target_margin <= 0:
        # Nothing to balance against; take everything and let the caller warn.
        return {s.scene_id: len(s.uniform) for s in scans}

    budget = min(total_uniform, int(total_margin * (1 - target_margin) / target_margin))

    quota: dict[str, int] = {}
    assigned = 0
    for s in scans:
        share = int(budget * len(s.uniform) / total_uniform) if total_uniform else 0
        quota[s.scene_id] = min(share, len(s.uniform))
        assigned += quota[s.scene_id]

    # Hand out the rounding remainder to scenes that still have spare patches.
    for s in sorted(scans, key=lambda s: len(s.uniform), reverse=True):
        if assigned >= budget:
            break
        spare = len(s.uniform) - quota[s.scene_id]
        take = min(spare, budget - assigned)
        quota[s.scene_id] += take
        assigned += take

    return quota


def extract_from_scene(
    scene: Scene,
    cfg: dict,
    rng: np.random.Generator,
    scan: SceneScan,
    uniform_quota: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PatchStats]:
    """Extract the pre-selected patches from one scene.

    Returns ``(x, y, mask, stats)`` where ``x`` is ``(N, C, S, S)`` float32,
    ``y`` is ``(N, S, S)`` uint8 and ``mask`` is ``(N, S, S)`` bool.
    """
    size = int(cfg["patches"]["size"])
    stack = scene.channels(CHANNELS)  # (C, H, W)

    margin_idx, uniform_idx = scan.margin, scan.uniform
    n_margin, n_uniform = len(margin_idx), len(uniform_idx)

    if n_uniform > uniform_quota:
        sel = rng.choice(n_uniform, size=uniform_quota, replace=False)
        uniform_keep = [uniform_idx[i] for i in sorted(sel)]
    else:
        uniform_keep = uniform_idx

    windows = margin_idx + uniform_keep
    n = len(windows)

    x = np.empty((n, len(CHANNELS), size, size), dtype="float32")
    y = np.empty((n, size, size), dtype="uint8")
    m = np.empty((n, size, size), dtype="bool")

    for i, (r, c) in enumerate(windows):
        x[i] = stack[:, r : r + size, c : c + size]
        y[i] = scene.sod[r : r + size, c : c + size]
        m[i] = scene.mask_valid[r : r + size, c : c + size]

    # Zero the SAR padding explicitly on invalid pixels. It is already 0.0 in
    # the source (D-006), but making it an invariant of the cache means a later
    # scene with a different padding convention cannot leak garbage in.
    x[~m[:, None, :, :].repeat(len(CHANNELS), axis=1)] = 0.0

    counts: dict[int, int] = {}
    if n:
        vals, cts = np.unique(y[m], return_counts=True)
        counts = {int(v): int(c) for v, c in zip(vals, cts, strict=True)}

    stats = PatchStats(
        scene_id=scene.meta.scene_id,
        region=scene.meta.region,
        month=scene.meta.acquired.month if scene.meta.acquired else 0,
        n_candidates=scan.n_candidates,
        n_valid_enough=n_margin + n_uniform,
        n_margin=n_margin,
        n_uniform=len(uniform_keep),
        n_kept=n,
        class_counts=counts,
        valid_fraction=scene.valid_fraction,
    )
    return x, y, m, stats


def cache_scene(
    scene: Scene,
    cfg: dict,
    out_dir: Path,
    rng: np.random.Generator,
    scan: SceneScan,
    uniform_quota: int,
) -> PatchStats:
    """Extract and persist one scene's patches as memory-mappable .npy files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = scene.meta.scene_id.replace(".nc", "")
    x, y, m, stats = extract_from_scene(scene, cfg, rng, scan, uniform_quota)

    if stats.n_kept == 0:
        log.warning(
            "%s: no usable patch (annotated %.1f%%, candidates %d)",
            stem,
            100 * stats.valid_fraction,
            stats.n_candidates,
        )
        return stats

    np.save(out_dir / f"{stem}_x.npy", x)
    np.save(out_dir / f"{stem}_y.npy", y)
    np.save(out_dir / f"{stem}_m.npy", m)

    # Which patches contain an ice margin. extract_from_scene() writes the
    # margin ones first, so this could be inferred from n_margin — but an
    # ordering convention is exactly the kind of thing a later refactor breaks
    # silently, and the sampler weights depend on it being right.
    flags = np.zeros(stats.n_kept, dtype="bool")
    flags[: stats.n_margin] = True
    np.save(out_dir / f"{stem}_f.npy", flags)
    return stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PatchDataset:
    """Memory-mapped patch dataset with on-the-fly augmentation.

    Implements the torch ``Dataset`` protocol without importing torch, so the
    cache can be inspected from a plain script. ``train.py`` wraps it directly.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        scene_ids: list[str],
        *,
        channels: list[str] | None = None,
        augment: bool = False,
        seed: int = 0,
        collapse_classes: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.channels = channels or CHANNELS[:3]
        self.channel_idx = [CHANNELS.index(c) for c in self.channels]
        self.augment = augment
        self._rng = np.random.default_rng(seed)

        # Applied on read rather than baked into the cache: the six-class labels
        # stay on disk, so switching schemes costs nothing and the 15 GB of
        # patches do not have to be extracted twice. See data/collapse.py.
        self.collapse_classes = collapse_classes
        self._collapse_lut = None
        if collapse_classes:
            from iceroute.data.collapse import lookup_table

            self._collapse_lut = lookup_table()

        self._x: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._m: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []  # (bundle, row)
        margin: list[np.ndarray] = []

        missing = []
        for sid in scene_ids:
            stem = sid.replace(".nc", "")
            xp = self.cache_dir / f"{stem}_x.npy"
            if not xp.exists():
                missing.append(stem)
                continue
            b = len(self._x)
            self._x.append(np.load(xp, mmap_mode="r"))
            self._y.append(np.load(self.cache_dir / f"{stem}_y.npy", mmap_mode="r"))
            self._m.append(np.load(self.cache_dir / f"{stem}_m.npy", mmap_mode="r"))
            self._index.extend((b, i) for i in range(len(self._x[b])))

            fp = self.cache_dir / f"{stem}_f.npy"
            n = len(self._x[b])
            # A cache written before flags existed reads as "no margins", which
            # makes the sampler fall back to uniform weights rather than
            # silently upweighting the wrong patches.
            margin.append(np.load(fp) if fp.exists() else np.zeros(n, dtype="bool"))

        if missing:
            log.warning("%d scenes without cache: %s", len(missing), ", ".join(missing[:5]))

        self.is_margin = np.concatenate(margin) if margin else np.zeros(0, dtype="bool")
        if not self._index:
            raise RuntimeError(
                f"No patches in {self.cache_dir} for the {len(scene_ids)} requested scenes. "
                f"Did you run 'make data'?"
            )

    def __len__(self) -> int:
        return len(self._index)

    def margin_fraction(self) -> float:
        """Fraction of cached patches that contain an ice margin."""
        return float(self.is_margin.mean()) if self.is_margin.size else 0.0

    def sampler_weights(self, target_margin: float = 0.40) -> np.ndarray | None:
        """Per-patch weights that make a draw hit ``target_margin`` in expectation.

        This is the half of the balancing that used to be done by deleting
        patches (D-017). With ``M`` margin and ``U`` uniform patches, weighting
        each margin patch ``target/M`` and each uniform one ``(1-target)/U``
        makes any single draw contain a margin with probability ``target``,
        while every patch stays reachable.

        Returns ``None`` when weighting is unnecessary or impossible — no flags,
        or one of the two groups empty — so the caller falls back to plain
        shuffling rather than dividing by zero.
        """
        if self.is_margin.size != len(self._index) or not len(self._index):
            return None
        n_margin = int(self.is_margin.sum())
        n_uniform = len(self._index) - n_margin
        if n_margin == 0 or n_uniform == 0 or not 0.0 < target_margin < 1.0:
            return None

        w = np.empty(len(self._index), dtype="float64")
        w[self.is_margin] = target_margin / n_margin
        w[~self.is_margin] = (1.0 - target_margin) / n_uniform
        return w

    def __getitem__(self, i: int):
        b, r = self._index[i]
        x = np.asarray(self._x[b][r][self.channel_idx], dtype="float32")
        y = np.asarray(self._y[b][r], dtype="uint8")
        if self._collapse_lut is not None:
            y = self._collapse_lut[y]
        y = y.astype("int64")
        m = np.asarray(self._m[b][r], dtype="bool")

        if self.augment:
            # Flips and 90-degree rotations only. An arbitrary rotation would
            # interpolate a categorical target; colour jitter would alter a
            # physical backscatter value. Both are excluded by design.
            if self._rng.random() < 0.5:
                x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
            if self._rng.random() < 0.5:
                x, y, m = x[:, ::-1, :], y[::-1, :], m[::-1, :]
            k = int(self._rng.integers(0, 4))
            if k:
                x = np.rot90(x, k, axes=(1, 2))
                y = np.rot90(y, k)
                m = np.rot90(m, k)
            x, y, m = np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(m)

        # Invalid pixels carry the loss's ignore_index, so they contribute no
        # gradient. Never let FILL_VALUE reach CrossEntropy as a class id.
        y = np.where(m, y, FILL_VALUE).astype("int64")
        return x, y, m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract and cache the training patches.")
    ap.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--split", type=Path, default=None, help="default: from configs/data.yaml")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/train"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--stats", type=Path, default=Path("artifacts/data_stats.json"))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N scenes")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    cfg = load_config(args.config)
    import yaml

    split_path = args.split or Path(cfg.get("split_file", "configs/split.yaml"))
    split = yaml.safe_load(Path(split_path).read_text())["split"]

    wanted = list(split["train"]) + list(split["val"])
    if args.limit:
        wanted = wanted[: args.limit]

    present = [s for s in wanted if (args.raw_dir / s).exists()]
    absent = len(wanted) - len(present)
    if absent:
        log.warning(
            "%d scenes not downloaded yet: skipping them (re-run after 'make download')", absent
        )
    if not present:
        log.error("No scene available in %s. Run 'make download' first.", args.raw_dir)
        return 1

    rng = np.random.default_rng(int(split.get("seed", 0)))
    target_margin = float(cfg["patches"]["margin_fraction_target"])

    # --- pass 1: classify -------------------------------------------------
    # Reads only the ice chart, so it is cheap enough to do over every scene
    # before committing to a selection.
    log.info("Pass 1/2 — classification (%d scenes)", len(present))
    scans: list[SceneScan] = []
    for i, sid in enumerate(present, 1):
        sc = scan_scene(args.raw_dir / sid, cfg)
        scans.append(sc)
        log.info(
            "  [%d/%d] %-32s annotated %5.1f%%  margin %4d  uniform %4d",
            i,
            len(present),
            sid,
            100 * sc.valid_fraction,
            len(sc.margin),
            len(sc.uniform),
        )

    balance = str(cfg["patches"].get("balance", "sampler"))
    quota = allocate_uniform_quota(scans, target_margin, balance=balance)
    tm, tu = sum(len(s.margin) for s in scans), sum(quota.values())
    if balance == "sampler":
        log.info(
            "Balance 'sampler': keeping all %d valid patches. The %.0f%% margin "
            "fraction is reached by weighting the sampler during training, not by "
            "throwing data away (D-017).",
            tm + tu,
            100 * target_margin,
        )
    log.info(
        "Global budget: %d with margin + %d uniform = %d patches (%.0f%% margin, target %.0f%%)",
        tm,
        tu,
        tm + tu,
        100 * tm / max(tm + tu, 1),
        100 * target_margin,
    )
    if tm == 0:
        log.warning(
            "NO patch with an ice margin in the whole subset. "
            "The criterion (>=%d classes, each >=%.0f%%) is too strict for these scenes, "
            "or the downloaded scenes are all uniform ice.",
            cfg["patches"]["margin_criterion"]["min_classes"],
            100 * cfg["patches"]["margin_criterion"]["min_class_fraction"],
        )

    # --- pass 2: extract --------------------------------------------------
    log.info("Pass 2/2 — extraction")
    all_stats: list[PatchStats] = []
    for i, (sid, sc) in enumerate(zip(present, scans, strict=True), 1):
        scene = load_scene(args.raw_dir / sid, cfg)
        st = cache_scene(scene, cfg, args.out, rng, sc, quota.get(sc.scene_id, 0))
        all_stats.append(st)
        log.info(
            "  [%d/%d] %-32s kept %4d  (margin %d, uniform %d)",
            i,
            len(present),
            sid,
            st.n_kept,
            st.n_margin,
            st.n_uniform,
        )

    # --- aggregate --------------------------------------------------------
    total_counts: dict[int, int] = {}
    for st in all_stats:
        for k, v in st.class_counts.items():
            total_counts[k] = total_counts.get(k, 0) + v
    grand = sum(total_counts.values())

    n_kept = sum(s.n_kept for s in all_stats)
    n_margin = sum(s.n_margin for s in all_stats)

    names = {int(k): v["name"] for k, v in cfg["sod_classes"].items()}
    summary = {
        "n_scenes": len(all_stats),
        "n_patches": n_kept,
        "n_margin_patches": n_margin,
        "margin_fraction": n_margin / n_kept if n_kept else 0.0,
        "margin_fraction_target": cfg["patches"]["margin_fraction_target"],
        "patch_size": cfg["patches"]["size"],
        "channels": CHANNELS,
        "class_pixel_counts": {str(k): v for k, v in sorted(total_counts.items())},
        "class_frequency": {str(k): v / grand for k, v in sorted(total_counts.items())}
        if grand
        else {},
        "class_names": {str(k): v for k, v in names.items()},
        "per_scene": [s.as_dict() for s in all_stats],
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(summary, indent=2))

    log.info("=" * 66)
    log.info("Total patches: %d from %d scenes", n_kept, len(all_stats))
    if n_kept:
        log.info(
            "With margin: %d (%.0f%%, target %.0f%%)",
            n_margin,
            100 * n_margin / n_kept,
            100 * cfg["patches"]["margin_fraction_target"],
        )
        log.info("Class distribution over valid pixels:")
        for k, v in sorted(total_counts.items()):
            bar = "█" * int(round(40 * v / grand))
            log.info("  %d %-12s %5.1f%%  %s", k, names.get(k, "?"), 100 * v / grand, bar)
    log.info("Statistics in %s", args.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
