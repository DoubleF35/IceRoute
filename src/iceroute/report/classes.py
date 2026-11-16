"""The ice-class table for the deck, generated from what the model measured.

A-11 of §98: the June decks print five ice classes, one of which — *Deformed
Ice / Ridges* — does not exist in any AI4Arctic label, and the model actually
deploys three. Replacing a wrong table by hand invites a second wrong table, so
this writes it from ``artifacts/metrics.json``, ``configs/assumptions.yaml`` and
``data/polaris_rv.csv``.

The reason it is worth generating rather than typing is not tidiness. Writing the
replacement by hand, I produced three rows that read as equally trustworthy —
and they are not. **Thin ice scores 0.136 IoU.** A table that hides that is the
same species of error as the one it replaces: a claim the measurement does not
support, printed confidently.

So the table carries what the model does when it is wrong, and the distinction
that matters is not accuracy but *direction*:

* thin ice is mostly called **thick** — an error toward caution, which costs a
  longer route and nothing else;
* thick ice called **open water** is the error that could put a ship somewhere
  it may not go, and it is the number to publish.

That second figure is the honest headline of this whole section, and it is
better said by us first than found by someone else.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger("classes")

METRICS = Path("artifacts/metrics.json")
ASSUMPTIONS = Path("configs/assumptions.yaml")
RV_TABLE = Path("data/polaris_rv.csv")
OUT = Path("pitch/class_table.md")

# What each deployed class means to a bridge, in the deck's own register. The
# wording is a judgement, so it lives here and not in a config: everything
# numeric on the row is measured, this column is what we choose to say.
MEANING = {
    "Acqua libera": ("Open water", "Navigate freely"),
    "Ghiaccio sottile": ("Thin ice", "POLARIS decides — depends on concentration"),
    "Ghiaccio spesso": ("Thick ice", "POLARIS decides — often forbidden at high concentration"),
}


@dataclass
class Row:
    name: str
    english: str
    meaning: str
    polaris_type: str
    iou: float | None
    support_px: int | None
    riv: dict[str, int]
    mistaken_for: str = ""
    mistaken_pct: float | None = None


def load_riv(path: Path, ice_classes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """RIV per ice type, for the vessel classes the demo shows."""
    rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    table: dict[str, dict[str, int]] = {}
    for rec in csv.DictReader(rows):
        vessel = rec.get("ice_class")
        if vessel not in ice_classes:
            continue
        for ice_type, value in rec.items():
            if ice_type in {"ice_class", "source"} or value is None:
                continue
            try:
                table.setdefault(ice_type, {})[vessel] = int(value)
            except ValueError:
                continue
    return table


def confusion_rows(agg: dict) -> tuple[dict[str, tuple[str, float]], float | None]:
    """For each class: what it is most often mistaken for, and how often.

    Also returns the one number that belongs on a safety slide — the share of
    thick ice called open water — because that is the mistake with a
    consequence. Everything else costs a detour.
    """
    cm = agg.get("confusion_matrix")
    names = list((agg.get("per_class") or {}).keys())
    if not cm or len(names) != len(cm):
        return {}, None
    m = np.asarray(cm, dtype=float)
    worst: dict[str, tuple[str, float]] = {}
    for i, name in enumerate(names):
        total = m[i].sum()
        if total <= 0:
            continue
        off = [(m[i][j] / total, names[j]) for j in range(len(names)) if j != i]
        pct, other = max(off)
        worst[name] = (other, pct * 100)

    permissive = None
    if "Ghiaccio spesso" in names and "Acqua libera" in names:
        i, j = names.index("Ghiaccio spesso"), names.index("Acqua libera")
        if m[i].sum() > 0:
            permissive = float(m[i][j] / m[i].sum() * 100)
    return worst, permissive


def collect(metrics: dict, assumptions: dict, rv_path: Path) -> dict:
    ev = (metrics.get("evaluation") or {}).get("scenes") or {}
    agg = ev.get("aggregate") or {}
    per_class = agg.get("per_class") or {}
    seg = (assumptions.get("segmentation") or {}).get("classes") or {}
    vessels = tuple((assumptions.get("vessel") or {}).get("classes_shown") or ("PC6",))
    riv = load_riv(rv_path, vessels) if rv_path.exists() else {}
    worst, permissive = confusion_rows(agg)

    rows = []
    for idx in sorted(seg):
        entry = seg[idx] or {}
        name = entry.get("name", "")
        scores = per_class.get(name) or {}
        english, meaning = MEANING.get(name, (name, ""))
        other, pct = worst.get(name, ("", None))
        rows.append(
            Row(
                name=name,
                english=english,
                meaning=meaning,
                polaris_type=entry.get("polaris_type", ""),
                iou=scores.get("iou"),
                support_px=scores.get("support_px"),
                riv=riv.get(entry.get("polaris_type", ""), {}),
                mistaken_for=other,
                mistaken_pct=pct,
            )
        )

    per_scene = [
        r
        for r in (ev.get("per_scene") or [])
        if isinstance((r.get("rio") or {}).get("worst_unsafe_fraction"), float)
    ]
    fr = [r["rio"]["worst_unsafe_fraction"] for r in per_scene]
    return {
        "rows": rows,
        "vessels": vessels,
        "miou": agg.get("miou"),
        "binary": agg.get("binary_accuracy"),
        "n_scenes": ev.get("n_scenes"),
        "permissive_thick_as_water": permissive,
        "unsafe_median": float(np.median(fr)) if fr else None,
        "unsafe_max": max(fr) if fr else None,
        "n_unsafe_scenes": len(fr),
        "scheme": (assumptions.get("segmentation") or {}).get("scheme", ""),
    }


def _n(v, spec="{:.3f}", missing="non misurato"):
    return spec.format(v) if isinstance(v, int | float) else missing


def render(c: dict) -> str:
    vessels = c["vessels"]
    head = " | ".join(f"RIV {v}" for v in vessels)
    sep = "|".join(["---"] * len(vessels))

    deck = [f"| Class | What it means to a bridge | {head} |", f"|---|---|{sep}|"]
    detail = [
        "| Classe | IoU misurata | pixel | quando sbaglia, la chiama | tipo POLARIS |",
        "|---|---:|---:|---|---|",
    ]
    for r in c["rows"]:
        rivs = " | ".join(
            f"{r.riv.get(v, '—'):+d}" if isinstance(r.riv.get(v), int) else "—" for v in vessels
        )
        deck.append(f"| **{r.english}** | {r.meaning} | {rivs} |")
        px = f"{r.support_px / 1e6:.0f} M" if isinstance(r.support_px, int) else "—"
        mis = (
            f"{r.mistaken_for.lower()} nel {r.mistaken_pct:.0f}% dei casi"
            if r.mistaken_pct is not None
            else "—"
        )
        detail.append(f"| {r.name} | {_n(r.iou)} | {px} | {mis} | `{r.polaris_type}` |")

    perm = c["permissive_thick_as_water"]
    return f"""# Le classi di ghiaccio, come stanno davvero

> **Generato da `make report`.** Non modificare a mano: la fonte è
> `artifacts/metrics.json`, `configs/assumptions.yaml` e `data/polaris_rv.csv`.
> Chiude **A-11** di §98, che nei deck di giugno dichiarava cinque classi di cui una
> — *Deformed Ice / Ridges* — non esiste in nessuna etichetta AI4Arctic.

Schema attivo: **{c["scheme"]}**. mIoU **{_n(c["miou"])}**, ghiaccio/acqua
**{_n(c["binary"])}**, su **{c["n_scenes"]}** scene del test set ufficiale.

## Tabella per la slide

{chr(10).join(deck)}

`RIO = C × RIV(tipo) + (10 − C) × RIV(acqua libera)` — IMO MSC.1/Circ.1519.

Il RIV è la ragione per cui la tabella non può fermarsi al tipo: la stessa classe è
navigazione normale a bassa concentrazione e vietata ad alta. È la lezione di D-031, ed è
anche il motivo per cui la conversazione conviene spostarla dal *numero di classi*
all'*implementare una circolare IMO*.

## Quello che la slide non deve nascondere

{chr(10).join(detail)}

**Le tre classi non sono ugualmente affidabili, e vale la pena dirlo per primi.**

Ma la domanda giusta non è quanto sbaglia: è **da che parte**. Il ghiaccio sottile viene
preso male, e quasi sempre viene chiamato **spesso** — cioè più pericoloso di quello che è.
Costa una rotta più lunga e nient'altro.

L'errore che conta è l'opposto: **ghiaccio spesso chiamato acqua libera, {_n(perm, "{:.1f}%")}
dei suoi pixel.** È l'unico che potrebbe mandare una nave dove non può andare, ed è il numero
da pubblicare.

Sull'intera scena, la frazione di mappa in cui siamo **più permissivi** della carta
dell'analista ha mediana **{_n(c["unsafe_median"], "{:.1%}")}** e caso peggiore
**{_n(c["unsafe_max"], "{:.1%}")}** su {c["n_unsafe_scenes"]} scene.

**E il caso peggiore è nella demo, marcato.** Non è un dato che nascondiamo in una nota: è
una delle scene che si possono scegliere sul palco. Mostrare da soli il proprio caso peggiore
è la cosa che rende credibile tutto il resto — e toglie a chiunque la possibilità di trovarlo
per primo.

## Se chiedono «perché tre e non sei?»

Perché sei è oltre il punto in cui il dato etichettato regge. Il modello a sei classi si è
fermato a **mIoU 0,272** sbagliando con sicurezza (registrato in D-018; l'artefatto di quella
valutazione non è più nel repo, la voce sì). A tre classi fa {_n(c["miou"])}, e sulla
distinzione che decide la sicurezza — ghiaccio sì o no — fa **{_n(c["binary"])}**.

Il ripiego a tre classi è previsto dalla SPEC §7.4 ed è documentato in D-020. **Attenzione:**
questi numeri non vanno confrontati con la soglia 0,60, che era definita sulle sei classi SOD.

## E i ridge?

Non esistono come etichetta. AI4Arctic annota concentrazione (SIC), stadio di sviluppo (SOD) e
dimensione dei banchi (FLOE): **nessuna delle tre codifica la deformazione**. Non è una classe
che il modello sbaglia, è una che non può imparare.

La risposta buona non è togliere la riga e tacere: è **dire che è il problema successivo**.
«Nessun dataset pubblico etichetta la deformazione — è il prossimo problema, non un risultato
nostro» mostra che sappiamo dov'è il limite, che è più convincente di una classe in più.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genera la tabella delle classi per il deck.")
    ap.add_argument("--metrics", type=Path, default=METRICS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics.exists():
        log.warning("%s assente: la tabella delle classi non viene scritta.", args.metrics)
        return 0
    metrics = json.loads(args.metrics.read_text())
    assumptions = yaml.safe_load(ASSUMPTIONS.read_text()) if ASSUMPTIONS.exists() else {}

    c = collect(metrics, assumptions, RV_TABLE)
    if not c["rows"]:
        log.warning("Nessuna classe in configs/assumptions.yaml: niente da scrivere.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(c), encoding="utf-8")
    log.info(
        "%s — %d classi, permissivo %s",
        args.out,
        len(c["rows"]),
        _n(c["permissive_thick_as_water"], "{:.1f}%"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
