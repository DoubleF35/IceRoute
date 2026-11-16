#!/usr/bin/env python3
"""Download a stratified subset of the AI4Arctic ready-to-train dataset.

SPEC §5.3. The full training article is 60 GB across 512 scenes; we take 60,
stratified by region and month so the model sees both winter and the melt
season, and all 20 test scenes with their reference labels.

Why stratify on region at all
-----------------------------
The filename is ``<YYYYMMDD>T<HHMMSS>_<service>_prep.nc`` and carries no region
token — ``service`` is only ``dmi`` or ``cis``, a coarse Greenland/Canada proxy.
The region lives in the ``original_id`` HDF5 attribute inside the file. Rather
than download 60 GB to find out where the scenes are, the region for all 552
scenes was extracted with 2 KB ranged GETs and is checked in at
``data/scene_regions.json``.

The selection is deterministic given the seed, and is frozen into
``configs/split.yaml``. Once frozen it is never regenerated: a validation split
that changes underneath you makes yesterday's mIoU incomparable with today's.

Usage
-----
    make download          # foreground, with progress
    make download-bg       # background, logs to artifacts/logs/download.log
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from iceroute.data.figshare import (  # noqa: E402
    DiskSpaceError,
    DownloadError,
    FigshareFile,
    download_file,
    free_bytes,
    list_article_files,
)

REPO = Path(__file__).resolve().parent.parent
SCENE_RE = re.compile(r"^(\d{8})T(\d{6})_(dmi|cis)_prep\.nc$")

log = logging.getLogger("download")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def scene_month(name: str) -> int | None:
    m = SCENE_RE.match(name)
    return int(m.group(1)[4:6]) if m else None


def scene_date(name: str) -> datetime | None:
    m = SCENE_RE.match(name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def load_file_index(article_id: int, cache: Path) -> list[FigshareFile]:
    """File listing for an article, from the checked-in cache or the API."""
    if cache.exists():
        payload = json.loads(cache.read_text())
        log.info("indice da cache: %s (%d file)", cache.name, len(payload))
    else:
        log.info("interrogo Figshare per l'articolo %d…", article_id)
        specs = list_article_files(article_id)
        payload = [
            {
                "id": s.id,
                "name": s.name,
                "size": s.size,
                "download_url": s.download_url,
                "supplied_md5": s.md5,
            }
            for s in specs
        ]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload, indent=2))
        log.info("indice salvato in %s (%d file)", cache.name, len(payload))
    return [FigshareFile.from_api(item) for item in payload]


def stratified_select(
    candidates: list[FigshareFile],
    regions: dict[str, dict],
    n_target: int,
    seed: int,
) -> list[FigshareFile]:
    """Pick ``n_target`` scenes spread across region and month.

    Largest-remainder allocation across regions (so every region present in the
    dataset contributes, proportionally to how many scenes it has), then within
    each region the scenes are ordered by month and sampled at even intervals so
    a region's quota is not all drawn from the same season.
    """
    import random

    rng = random.Random(seed)

    by_region: dict[str, list[FigshareFile]] = defaultdict(list)
    unknown = 0
    for spec in candidates:
        info = regions.get(spec.name)
        if not info:
            unknown += 1
            continue
        by_region[info["region"]].append(spec)
    if unknown:
        log.warning("%d scene senza regione nota: escluse dalla stratificazione", unknown)

    total = sum(len(v) for v in by_region.values())
    if total == 0:
        raise RuntimeError("Nessuna scena con regione nota: data/scene_regions.json è vuoto?")
    n_target = min(n_target, total)

    # Largest remainder: floor of the proportional share, then hand out the
    # leftovers to the regions with the biggest fractional part.
    exact = {r: len(v) * n_target / total for r, v in by_region.items()}
    quota = {r: int(v) for r, v in exact.items()}
    leftover = n_target - sum(quota.values())
    for region, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if leftover <= 0:
            break
        quota[region] += 1
        leftover -= 1

    selected: list[FigshareFile] = []
    for region, specs in sorted(by_region.items()):
        k = quota.get(region, 0)
        if k <= 0:
            continue
        # Order by month, then jitter deterministically inside the month so we
        # do not always take the chronologically first scene of each month.
        ordered = sorted(specs, key=lambda s: (scene_month(s.name) or 0, rng.random()))
        if k >= len(ordered):
            picked = ordered
        else:
            step = len(ordered) / k
            picked = [ordered[int(i * step)] for i in range(k)]
        selected.extend(picked)

    return sorted(selected, key=lambda s: s.name)


def report_coverage(specs: list[FigshareFile], regions: dict[str, dict]) -> None:
    months = Counter(scene_month(s.name) for s in specs)
    regs = Counter(regions.get(s.name, {}).get("region", "?") for s in specs)
    total_gb = sum(s.size for s in specs) / 1e9

    log.info("Selezionate %d scene, %.1f GB", len(specs), total_gb)
    log.info("  per mese:   %s", " ".join(f"{m:02d}:{months[m]}" for m in sorted(months) if m))
    winter = sum(months.get(m, 0) for m in (12, 1, 2, 3))
    melt = sum(months.get(m, 0) for m in (6, 7, 8, 9))
    log.info("  inverno (12-03): %d · scioglimento (06-09): %d", winter, melt)
    log.info("  regioni:    %d distinte", len(regs))
    for region, n in sorted(regs.items()):
        log.info("    %-28s %d", region, n)


SPLIT_HEADER = """# Split CONGELATO — generato da scripts/download_subset.py, NON modificare a mano.
#
# SPEC §5.3: split per scena, mai per patch. Patch della stessa scena in train e
# val causano leakage e producono metriche gonfie che crollano in demo.
#
# Una volta congelato non si rigenera: se cambiasse, il mIoU di oggi non sarebbe
# più confrontabile con quello di ieri.
#
# Le 5 scene demo sono un'estrazione casuale: vanno RIVISTE A MANO dopo il
# download, scegliendo quelle visivamente convincenti (margine di ghiaccio netto,
# canali navigabili plausibili, nessun artefatto grosso).
"""


def freeze_split(selected: list[FigshareFile], split_path: Path, seed: int) -> dict[str, list[str]]:
    """Write the train/val/demo split into configs/split.yaml, once.

    Kept in its own file rather than appended to configs/data.yaml because
    ``yaml.safe_dump`` cannot round-trip comments, and data.yaml's comments
    document the SOD-to-POLARIS mapping — the joint between model and routing.
    Generated content and hand-written content do not share a file.

    SPEC §5.3: split per scene, never per patch, and never regenerated.
    """
    if split_path.exists():
        existing = yaml.safe_load(split_path.read_text()) or {}
        if existing.get("split", {}).get("frozen"):
            log.info("Split già congelato in %s: lo riuso.", split_path.name)
            return {k: existing["split"].get(k, []) for k in ("train", "val", "demo")}

    import random

    names = sorted(s.name for s in selected)
    rng = random.Random(seed)
    rng.shuffle(names)

    n_val, n_demo = 10, 5
    val = sorted(names[:n_val])
    demo = sorted(names[n_val : n_val + n_demo])
    train = sorted(names[n_val + n_demo :])

    payload = {
        "split": {
            "frozen": True,
            "seed": seed,
            "train": train,
            "val": val,
            "demo": demo,
        }
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        SPLIT_HEADER + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    )
    log.info("Split congelato: %d train, %d val, %d demo", len(train), len(val), len(demo))
    return {"train": train, "val": val, "demo": demo}


def download_many(
    specs: list[FigshareFile], dest: Path, min_free: int, label: str
) -> tuple[int, int]:
    """Download a list of files. Returns (ok, failed)."""
    ok = failed = 0
    total = len(specs)
    for i, spec in enumerate(specs, 1):
        log.info("[%s %d/%d] %s (%.0f MB)", label, i, total, spec.name, spec.size / 1e6)
        try:
            download_file(spec, dest, min_free_bytes=min_free)
            ok += 1
        except DiskSpaceError as exc:
            log.error("SPAZIO ESAURITO: %s", exc)
            log.error("Libera spazio e rilancia: il download riprende da qui.")
            raise
        except DownloadError as exc:
            log.error("fallita: %s", exc)
            failed += 1
    return ok, failed


def extend_train(args, cfg, scenes, regions, split_path, seed, min_free, avail) -> int:
    """Download extra TRAINING scenes without regenerating the frozen split.

    SPEC §5.3 freezes the split and forbids regenerating it: if val or demo
    changed, today's mIoU would not be comparable with yesterday's. So the extra
    scenes go into a separate ``configs/split_extra.yaml`` and extend **train
    only**. Validation and demo stay bit-for-bit identical, which is exactly
    what makes the before/after comparison mean anything.

    Scenes already in any split are excluded, so a rerun adds new ones rather
    than redownloading.
    """
    frozen = (yaml.safe_load(split_path.read_text()) or {}).get("split", {})
    used = {name for key in ("train", "val", "demo") for name in frozen.get(key, [])}

    extra_path = REPO / "configs" / "split_extra.yaml"
    if extra_path.exists():
        prev = (yaml.safe_load(extra_path.read_text()) or {}).get("extra_train", [])
        used |= set(prev)
    else:
        prev = []

    pool = [f for f in scenes if f.name not in used]
    log.info("Scene gia' allocate: %d · disponibili per l'estensione: %d", len(used), len(pool))
    if not pool:
        log.error("Nessuna scena nuova disponibile.")
        return 1

    # A different seed from the frozen split, or stratified_select would return
    # the same regional ordering and cluster the extension in the same places.
    picked = stratified_select(pool, regions, min(args.extend_train, len(pool)), seed + 1)
    report_coverage(picked, regions)

    planned = sum(f.size for f in picked)
    log.info("Da scaricare: %d scene, %.1f GB", len(picked), planned / 1e9)
    if avail - planned < min_free:
        log.error(
            "Spazio insufficiente: servono %.1f GB, liberi %.1f GB, soglia %.0f GB. "
            "Riduci --extend-train.",
            planned / 1e9,
            avail / 1e9,
            min_free / 1e9,
        )
        return 1
    if args.dry_run:
        for f in picked:
            log.info("  %s  %s", f.name, regions[f.name]["region"])
        return 0

    names = sorted(prev + [f.name for f in picked])
    extra_path.write_text(
        "# Scene AGGIUNTIVE di training. Vedi DECISIONS.md D-023.\n"
        "#\n"
        "# Generato da: scripts/download_subset.py --extend-train\n"
        "# NON tocca configs/split.yaml, che resta congelato (SPEC §5.3): val e demo\n"
        "# sono identiche, quindi il mIoU prima/dopo resta confrontabile. Se cambiassero\n"
        "# anche quelle, il confronto non direbbe piu' niente.\n"
        + yaml.safe_dump({"extra_train": names}, sort_keys=False, width=100)
    )
    log.info("Elenco in %s (%d scene in totale)", extra_path, len(names))

    ok, failed = download_many(picked, REPO / "data" / "raw" / "train", min_free, "extra")
    log.info("=" * 60)
    log.info("Estensione: %d scaricate, %d fallite", ok, failed)
    log.info("Ora: make data && make train")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "data.yaml")
    ap.add_argument("--n-train", type=int, default=None, help="override del numero di scene")
    ap.add_argument("--dry-run", action="store_true", help="mostra la selezione senza scaricare")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument(
        "--extend-train",
        type=int,
        default=None,
        help="scarica N scene AGGIUNTIVE per il training, senza toccare lo split "
        "congelato. Le nuove finiscono in configs/split_extra.yaml.",
    )
    args = ap.parse_args(argv)

    setup_logging()
    cfg = yaml.safe_load(args.config.read_text())
    src, subset = cfg["source"], cfg["subset"]

    min_free = int(subset["min_free_gb"] * 1e9)
    avail = free_bytes(REPO / "data")
    log.info("Spazio libero: %.1f GB (soglia minima %.0f GB)", avail / 1e9, min_free / 1e9)

    regions = json.loads((REPO / subset["region_source"]).read_text())

    # --- train ------------------------------------------------------------
    train_art = src["articles"]["train"]
    train_files = load_file_index(
        train_art["id"], REPO / "data" / f"figshare_files_{train_art['id']}.json"
    )
    scenes = [f for f in train_files if SCENE_RE.match(f.name)]
    log.info("Articolo train: %d scene disponibili", len(scenes))

    split_path = REPO / cfg.get("split_file", "configs/split.yaml")
    seed = int(
        (yaml.safe_load(split_path.read_text()) or {}).get("split", {}).get("seed", 20260903)
        if split_path.exists()
        else 20260903
    )

    if args.extend_train:
        return extend_train(args, cfg, scenes, regions, split_path, seed, min_free, avail)

    n_train = args.n_train or subset["n_train_scenes"]
    selected = stratified_select(scenes, regions, n_train, seed)
    report_coverage(selected, regions)

    # --- test -------------------------------------------------------------
    test_art = src["articles"]["test"]
    test_files = load_file_index(
        test_art["id"], REPO / "data" / f"figshare_files_{test_art['id']}.json"
    )
    test_scenes = [f for f in test_files if f.name.endswith("_prep.nc")]
    test_refs = [f for f in test_files if f.name.endswith("_prep_reference.nc")]
    log.info(
        "Articolo test: %d scene + %d reference (%.2f GB)",
        len(test_scenes),
        len(test_refs),
        sum(f.size for f in test_scenes + test_refs) / 1e9,
    )

    planned = sum(s.size for s in selected)
    if not args.skip_test:
        planned += sum(f.size for f in test_scenes + test_refs)
    log.info("Totale da scaricare: %.1f GB", planned / 1e9)
    if avail - planned < min_free:
        log.error(
            "Spazio insufficiente: servono %.1f GB, liberi %.1f GB, soglia %.0f GB.",
            planned / 1e9,
            avail / 1e9,
            min_free / 1e9,
        )
        return 1

    if args.dry_run:
        log.info("--dry-run: nessun download. Scene selezionate:")
        for s in selected:
            log.info("  %s  %-28s %.0f MB", s.name, regions[s.name]["region"], s.size / 1e6)
        return 0

    freeze_split(selected, split_path, seed)

    train_dir = REPO / "data" / "raw" / "train"
    test_dir = REPO / "data" / "raw" / "test"

    ok_t, fail_t = download_many(selected, train_dir, min_free, "train")
    ok_e = fail_e = 0
    if not args.skip_test:
        # References first: 16 MB total, and useless to have scenes without them.
        ok_r, fail_r = download_many(test_refs, test_dir, min_free, "ref")
        ok_e, fail_e = download_many(test_scenes, test_dir, min_free, "test")
        ok_e += ok_r
        fail_e += fail_r

    log.info("=" * 60)
    log.info("Train: %d scaricate, %d fallite", ok_t, fail_t)
    log.info("Test:  %d scaricate, %d fallite", ok_e, fail_e)
    log.info("Spazio libero residuo: %.1f GB", free_bytes(REPO / "data") / 1e9)
    if fail_t or fail_e:
        log.warning("Rilancia 'make download': le già presenti vengono saltate.")
        return 1
    log.info("Completato. Prossimo passo: make inspect SCENE=...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
