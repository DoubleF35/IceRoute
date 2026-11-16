"""Everything that must be true before the laptop goes on stage. SPEC Gate 4.

This is the check to run the morning of 7 October, with the network off, after
a reboot — and every time an artefact is regenerated before then.

It is deliberately paranoid about one class of failure: a demo that *looks*
built because the directories exist. `data/demo/` can hold an index pointing at
PNGs that were never written, routes from a model that has since been replaced,
or a carousel scene the evaluation now says is the worst of the set. None of
those raise an exception. All of them are visible on stage.

What this cannot check is the part of Gate 4 that matters most: three people who
did not build it using it on their own. No script attests to that.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

log = logging.getLogger("preflight")

OK, WARN, FAIL = "  ok  ", " avviso ", " ERRORE "


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, what: str, detail: str = "") -> None:
        self.rows.append((status, what, detail))

    def ok(self, what, detail=""):
        self.add(OK, what, detail)

    def warn(self, what, detail=""):
        self.add(WARN, what, detail)

    def fail(self, what, detail=""):
        self.add(FAIL, what, detail)

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == WARN)


def check_demo(repo: Path, rep: Report) -> dict:
    index_path = repo / "data/demo/index.json"
    if not index_path.exists():
        rep.fail("data/demo/index.json", "assente: lancia make precompute")
        return {}
    index = json.loads(index_path.read_text())

    if not index.get("generated_from_predicted_mask"):
        rep.fail(
            "rotte dalla maschera predetta",
            "l'indice viene da una classifica su ice chart, non dal modello",
        )
    else:
        rep.ok("rotte dalla maschera predetta")

    scenes = index.get("scenes") or []
    carousel = [s for s in scenes if not s.get("worst_case")]
    worst = [s for s in scenes if s.get("worst_case")]
    if not carousel:
        rep.fail("scene nel carosello", "nessuna")
    else:
        rep.ok("scene nel carosello", f"{len(carousel)} + {len(worst)} caso peggiore")

    # Every file the UI will try to open, actually opened. The paths are
    # resolved exactly as app.py resolves them — bare names under images/ and
    # routes/ — because a check that looks somewhere else proves nothing.
    #
    # `files` also carries an "inference_stats" entry that is a JSON blob, not a
    # filename. It is skipped here, but it should not be in a dict called
    # `files` at all.
    missing = []
    for s in scenes:
        for key, rel in (s.get("files") or {}).items():
            if key == "inference_stats" or not isinstance(rel, str) or not rel.endswith(".png"):
                continue
            if not (repo / "data/demo/images" / rel).exists():
                missing.append(f"{s['scene_id'][:14]}/{key}")
        for ic, entry in (s.get("classes") or {}).items():
            gj = entry.get("geojson")
            if gj and not (repo / "data/demo/routes" / gj).exists():
                missing.append(f"{s['scene_id'][:14]}/{ic}.geojson")
            ovl = entry.get("overlay")
            if ovl and not (repo / "data/demo/overlays" / ovl).exists():
                missing.append(f"{s['scene_id'][:14]}/{ic}.overlay")
    if missing:
        rep.fail("file referenziati dall'indice", f"{len(missing)} mancanti: {missing[:3]}")
    else:
        rep.ok("file referenziati dall'indice", "tutti presenti")

    # Leftovers from earlier builds. Harmless to the UI, which only opens what
    # the index names, but they inflate the one artefact that has to be backed
    # up before the pitch, and anyone browsing the folder sees scenes that were
    # dropped. Reported, never deleted: this directory is not under git, so a
    # wrong guess here is unrecoverable.
    img_dir = repo / "data/demo/images"
    if img_dir.exists():
        referenced = {
            v
            for s in scenes
            for k, v in (s.get("files") or {}).items()
            if isinstance(v, str) and v.endswith(".png")
        }
        on_disk = list(img_dir.glob("*.png"))
        orphans = [p for p in on_disk if p.name not in referenced]
        if orphans:
            mb = sum(p.stat().st_size for p in orphans) / 1e6
            rep.warn(
                "immagini orfane in data/demo/images",
                f"{len(orphans)} su {len(on_disk)} non sono nell'indice ({mb:.0f} MB)",
            )
        else:
            rep.ok("immagini orfane in data/demo/images", "nessuna")

    differ = sum(1 for s in carousel if s.get("routes_differ"))
    if differ == 0:
        rep.fail(
            "momento 'cambio classe di nave'", "nessuna scena lo sostiene: il selettore è teatro"
        )
    elif differ < 2:
        rep.warn("momento 'cambio classe di nave'", f"solo {differ} scena su {len(carousel)}")
    else:
        rep.ok("momento 'cambio classe di nave'", f"{differ}/{len(carousel)} scene")
    return index


def check_model_is_current(repo: Path, index: dict, rep: Report) -> None:
    """The demo, the picked config and the evaluation must be the same model.

    They drift apart silently: retraining writes a new checkpoint, the picker
    changes its mind, and data/demo/ keeps the routes of whatever ran last.
    """
    picked = repo / "artifacts/best_config.txt"
    if not picked.exists():
        rep.warn("modello scelto", "artifacts/best_config.txt assente")
        return
    cfg_path = repo / picked.read_text().strip()
    if not cfg_path.exists():
        rep.fail("modello scelto", f"{cfg_path} non esiste")
        return
    cfg = yaml.safe_load(cfg_path.read_text())
    want = (cfg.get("output") or {}).get("best_checkpoint")
    got = index.get("checkpoint")
    if got != want:
        rep.fail(
            "demo allineata al modello scelto",
            f"la demo usa {got}, il modello scelto è {want}: rilancia precompute",
        )
    else:
        rep.ok("demo allineata al modello scelto", str(want))

    # And the *measurements* must come from that same model. They did not, once:
    # pitch/metrics.py defaulted to configs/train3.yaml — the baseline — so
    # `make report` quietly restated the headline compression ratio as 347:1
    # instead of 439:1, with no code in the measurement chain having changed.
    # Nothing caught it, because the check above only looks at data/demo/.
    comp = (
        json.loads((repo / "artifacts/metrics.json").read_text()).get("compression")
        if (repo / "artifacts/metrics.json").exists()
        else None
    ) or {}
    measured = comp.get("checkpoint")
    if not comp:
        return
    if measured is None:
        rep.warn(
            "compressione misurata sul modello scelto",
            "metrics.json non registra con quale modello: rilancia make report",
        )
    elif measured != want:
        rep.fail(
            "compressione misurata sul modello scelto",
            f"misurata su {measured}, schierato è {want}: rilancia make report",
        )
    else:
        rep.ok("compressione misurata sul modello scelto", f"{comp.get('ratio_median')}:1")


def check_worst_case_is_worst(repo: Path, rep: Report) -> None:
    demo_cfg = repo / "configs/demo.yaml"
    metrics = repo / "artifacts/metrics.json"
    if not (demo_cfg.exists() and metrics.exists()):
        return
    cfg = yaml.safe_load(demo_cfg.read_text())
    worst_id = (cfg.get("worst_case") or {}).get("id")
    per = ((json.loads(metrics.read_text()).get("evaluation") or {}).get("scenes") or {}).get(
        "per_scene"
    ) or []
    scored = {r["scene_id"]: (r.get("rio") or {}).get("worst_unsafe_fraction") for r in per}
    if not worst_id or not isinstance(scored.get(worst_id), float):
        rep.warn("caso peggiore dichiarato", "non valutato")
        return
    offenders = [
        e["id"]
        for e in cfg["scenes"]
        if isinstance(scored.get(e["id"]), float) and scored[e["id"]] >= scored[worst_id]
    ]
    if offenders:
        rep.fail(
            "il caso peggiore è davvero il peggiore", f"più permissive nel carosello: {offenders}"
        )
    else:
        rep.ok(
            "il caso peggiore è davvero il peggiore", f"{worst_id[:14]} a {scored[worst_id]:.1%}"
        )


def check_routes_are_drawable(repo: Path, index: dict, rep: Report) -> None:
    """Not "is there an overlay file" — does drawing it change any pixels.

    The failure being guarded against here already happened once and shipped:
    the drawing code looked up something nothing wrote, skipped every track and
    saved a perfectly clean image. Checking that the file exists would not have
    caught it. Checking that the picture changes does.
    """
    from PIL import Image

    from iceroute.demo import overlay as ov

    drawn, failures = 0, []
    for s in index.get("scenes") or []:
        png = repo / "data/demo/images" / ((s.get("files") or {}).get("mask_geo") or "")
        for ic, entry in (s.get("classes") or {}).items():
            if not entry.get("route_exists"):
                continue
            data = ov.load(repo / "data/demo/overlays" / (entry.get("overlay") or ""))
            tag = f"{s['scene_id'][:14]}/{ic}"
            if not png.is_file() or not data:
                failures.append(tag)
                continue
            try:
                img = Image.open(png)
                if img.convert("RGB").tobytes() == ov.draw(img, data, show="both").tobytes():
                    failures.append(f"{tag} (nessun pixel cambiato)")
                else:
                    drawn += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{tag} ({exc})")

    if failures:
        rep.fail("le rotte si disegnano davvero", f"{len(failures)} problemi: {failures[:3]}")
    elif drawn:
        rep.ok("le rotte si disegnano davvero", f"{drawn} rotte verificate a pixel")
    else:
        rep.warn("le rotte si disegnano davvero", "nessuna rotta da verificare")


def check_claims(repo: Path, rep: Report) -> None:
    claims = repo / "artifacts/claims.md"
    if not claims.exists():
        rep.warn("artifacts/claims.md", "assente: lancia make report")
        return
    text = claims.read_text()
    if "non misurat" in text:
        rep.warn("verdetto sulle affermazioni", "qualche riga non è ancora misurata")
    else:
        rep.ok("verdetto sulle affermazioni", "ogni riga ha una misura")


def check_standalone(repo: Path, rep: Report) -> None:
    """The offline copy must exist and must not predate the build it renders.

    Freshness, not existence, is the thing worth checking. A ``demo.html`` left
    over from before the last precompute would open perfectly, look right, and
    show the previous model's routes — and it is the file that gets used
    precisely when nobody is in a position to notice.
    """
    page = repo / "demo.html"
    index = repo / "data/demo/index.json"
    if not page.exists():
        rep.warn("copia statica demo.html", "assente: make demo-static")
        return
    mb = page.stat().st_size / 1e6
    if index.exists() and page.stat().st_mtime < index.stat().st_mtime:
        rep.fail(
            "copia statica demo.html",
            f"più vecchia di data/demo/: mostra una build superata. Rilancia "
            f"make demo-static ({mb:.1f} MB)",
        )
    else:
        rep.ok("copia statica demo.html", f"{mb:.1f} MB, aggiornata")


def check_assumptions(repo: Path, rep: Report) -> None:
    """The file the footer shows must not still say `null` for measured things.

    SPEC §10.1 makes ``configs/assumptions.yaml`` the source of every economic and
    performance claim, and the demo renders it verbatim in front of the audience.
    For weeks it read ``provisional: true`` beside quantities ``metrics.json`` had
    already measured — the pitch quoting a number while the file behind it said
    the number was not yet known.
    """
    from iceroute.report import assumptions as asm

    try:
        res = asm.run(repo, dry_run=True)
    except asm.SyncError as exc:
        rep.warn("assunzioni allineate alle misure", str(exc))
        return
    if res.changed:
        rep.fail(
            "assunzioni allineate alle misure",
            f"{len(res.changed)} indietro ({', '.join(res.changed[:3])}): make report",
        )
    elif res.missing:
        rep.warn(
            "assunzioni allineate alle misure",
            f"{len(res.missing)} non ancora misurate: {', '.join(res.missing[:3])}",
        )
    else:
        rep.ok("assunzioni allineate alle misure", f"{len(res.unchanged)} voci")


def check_backup_video(repo: Path, rep: Report) -> None:
    vids = (
        list((repo / "artifacts/video").glob("*.mp4"))
        if (repo / "artifacts/video").exists()
        else []
    )
    if not vids:
        rep.warn("video di riserva", "non registrato: make video")
    else:
        newest = max(vids, key=lambda p: p.stat().st_mtime)
        rep.ok("video di riserva", f"{newest.name} ({newest.stat().st_size / 1e6:.0f} MB)")


def check_imports(rep: Report) -> None:
    """The UI must import with the network off, before anyone clicks anything."""
    try:
        import iceroute.demo.app  # noqa: F401

        rep.ok("la UI si importa")
    except Exception as exc:  # noqa: BLE001
        rep.fail("la UI si importa", str(exc))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Controlli prima del palco. Gate 4.")
    ap.add_argument("--repo", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rep = Report()
    index = check_demo(args.repo, rep)
    if index:
        check_model_is_current(args.repo, index, rep)
        check_routes_are_drawable(args.repo, index, rep)
    check_worst_case_is_worst(args.repo, rep)
    check_claims(args.repo, rep)
    check_standalone(args.repo, rep)
    check_assumptions(args.repo, rep)
    check_backup_video(args.repo, rep)
    check_imports(rep)

    log.info("")
    for status, what, detail in rep.rows:
        log.info("[%s] %-38s %s", status, what, detail)
    log.info("")
    log.info("%d controlli · %d errori · %d avvisi", len(rep.rows), rep.failed, rep.warned)
    log.info("")
    log.info("Questo script NON può verificare il Gate 4 per intero: resta la prova con")
    log.info("la rete staccata, la macchina riavviata e tre persone esterne che la usano")
    log.info("da sole. Nessun file su disco lo attesta.")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
