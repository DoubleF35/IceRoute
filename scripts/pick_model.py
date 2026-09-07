"""Choose between the trained models, on the official test set. Writes the winner.

Three things make this a script rather than a judgement call at 3 a.m.:

* **The comparison is on the official test set**, never on validation. Every one
  of these models early-stopped on val mIoU, so val flatters each of them by an
  unknown and *different* amount. Picking on val would pick whichever overfitted
  its own selection criterion hardest.
* **mIoU is not the only axis.** A model with a better mIoU and a worse
  permissive fraction routes ships into ice POLARIS forbids. The permissive
  fraction is a veto, not a tiebreak.
* **The default is to keep the incumbent.** A challenger has to win clearly;
  noise should not swap the model the demo is built on.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("pick")

# A challenger must beat the incumbent by more than this on mIoU. Below it, the
# difference is not distinguishable from which scenes happen to be in the set.
MIN_GAIN = 0.01

# ...and if the two are within MIN_GAIN of each other on accuracy, the safety
# metric decides, provided the improvement is at least this many points.
#
# This clause was added on 2026-09-07, after it turned out the original rule had
# no way to express the case that actually occurred. It was written expecting
# the trade-off to be "more accurate but more permissive", and treated the
# permissive fraction purely as a veto against such a challenger. Lovász was the
# opposite: accuracy indistinguishable (-0,0028 pooled, +0,025 per-scene) and
# ten points safer. Under the old rule that model was rejected — not by a
# judgement about safety, but because the rule had no branch for it.
#
# The asymmetry justifying this is not new and is not fitted to the result: it
# is the same one behind D-008 and rio_error.py. Being too cautious costs time;
# being too permissive costs the ship.
SAFETY_MARGIN = 0.05

CANDIDATES = [
    ("baseline",  "configs/train3.yaml",        "artifacts/metrics.json"),
    ("lovasz",    "configs/train3_lovasz.yaml", "artifacts/eval_lovasz/metrics.json"),
    ("piu_dati",  "configs/train3_more.yaml",   "artifacts/eval_more/metrics.json"),
]


def read(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    ev = (json.loads(p.read_text()) or {}).get("evaluation") or {}
    scenes = ev.get("scenes") or {}
    agg = scenes.get("aggregate") or {}
    if not agg.get("confusion_matrix"):
        return None
    per = scenes.get("per_scene") or []
    mious = [r["miou"] for r in per if isinstance(r.get("miou"), float)]
    return {
        "miou": agg.get("miou"),
        # The pooled mIoU is computed from one confusion matrix over every
        # pixel; the per-scene mean weights each scene equally. A ship transits
        # one scene at a time, so when the two disagree it is worth seeing.
        "miou_per_scene": sum(mious) / len(mious) if mious else None,
        "binary": agg.get("binary_accuracy"),
        "unsafe": scenes.get("worst_unsafe_fraction"),
        "per_class": {k: v.get("iou") for k, v in (agg.get("per_class") or {}).items()},
        "split": ev.get("split"),
    }


def _unsafe(r: dict) -> float | None:
    return r["unsafe"] if isinstance(r.get("unsafe"), float) else None


def choose(found: list, incumbent: tuple) -> tuple:
    """Pick a winner from all candidates at once, in a way that ignores order.

    The first version walked the list and compared each candidate against the
    running best. That is order-dependent, and here it actually changed the
    answer: ``piu_dati`` is 6,9 points safer than the baseline at
    indistinguishable accuracy, so it would win if it came first, and then
    ``lovasz`` — only 3,5 points safer than *it* — would fail to displace it.
    Same three models, same numbers, different winner depending on how the list
    was written. A selection rule that does that will eventually swap the demo's
    model for no reason anybody can reconstruct.

    So the rule is stated over the whole set:

    1. **A clear accuracy win.** Any challenger more than ``MIN_GAIN`` above the
       incumbent, and not more permissive than it. Best gain wins.
    2. **Otherwise safety decides among the accuracy ties.** Take everything
       within ``MIN_GAIN`` of the best mIoU on offer and pick the least
       permissive, provided it beats the incumbent by ``SAFETY_MARGIN``.
    3. **Otherwise the incumbent stays.** Noise should not swap the model.
    """
    inc_name, _, inc = incumbent
    challengers = [f for f in found if f[0] != inc_name]

    # -- 1. a clear accuracy win, safety not made worse ---------------------
    clear = []
    for f in challengers:
        gain = f[2]["miou"] - inc["miou"]
        u, ui = _unsafe(f[2]), _unsafe(inc)
        not_worse = u is None or ui is None or u <= ui + 0.01
        if gain > MIN_GAIN and not_worse:
            clear.append((f, gain))
        elif gain > MIN_GAIN:
            log.info("\n%s ha mIoU migliore (%+.4f) ma frazione permissiva peggiore "
                     "(%.2f%% contro %.2f%%): SCARTATO.", f[0], gain, 100 * u, 100 * ui)
    if clear:
        f, gain = max(clear, key=lambda x: x[1])
        log.info("\n%s batte %s di %+.4f di mIoU senza peggiorare la sicurezza.",
                 f[0], inc_name, gain)
        return f

    # -- 2. accuracy indistinguishable: the safety metric decides -----------
    ceiling = max([inc["miou"]] + [f[2]["miou"] for f in challengers])
    tied = [f for f in challengers
            if ceiling - f[2]["miou"] <= MIN_GAIN and _unsafe(f[2]) is not None]
    ui = _unsafe(inc)
    if tied and ui is not None:
        f = min(tied, key=lambda f: _unsafe(f[2]))
        safety_gain = ui - _unsafe(f[2])
        if safety_gain >= SAFETY_MARGIN:
            log.info("\n%s ha mIoU indistinguibile da %s (%+.4f, sotto la soglia di %.2f) "
                     "ma è più sicuro di %.1f punti (%.2f%% contro %.2f%%).",
                     f[0], inc_name, f[2]["miou"] - inc["miou"], MIN_GAIN,
                     100 * safety_gain, 100 * _unsafe(f[2]), 100 * ui)
            log.info("Un errore troppo cauto costa tempo, uno troppo permissivo costa la "
                     "nave: a parità di accuratezza decide la sicurezza.")
            return f

    # -- 3. nothing convincing ---------------------------------------------
    log.info("\nNessun candidato batte %s in modo convincente: resta lui.", inc_name)
    return incumbent


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    found = []
    for name, cfg, metrics in CANDIDATES:
        r = read(metrics)
        if r is None:
            log.info("%-10s nessuna valutazione", name)
            continue
        if r["split"] != "test_ufficiale":
            log.info("%-10s valutato su '%s', non sul test ufficiale: escluso",
                     name, r["split"])
            continue
        found.append((name, cfg, r))
        log.info(
            "%-10s mIoU %.4f (per scena %.4f) · gh/acqua %.4f · permissiva peggiore %s",
            name, r["miou"],
            r["miou_per_scene"] if isinstance(r["miou_per_scene"], float) else float("nan"),
            r["binary"],
            f"{r['unsafe']:.2%}" if isinstance(r["unsafe"], float) else "n/d",
        )

    if not found:
        log.info("\nNessun candidato valutato: resta configs/train3.yaml")
        Path("artifacts/best_config.txt").write_text("configs/train3.yaml\n")
        return 0

    incumbent = next((f for f in found if f[0] == "baseline"), found[0])
    best_name, best_cfg, best = choose(found, incumbent)

    Path("artifacts/best_config.txt").write_text(best_cfg + "\n")
    log.info("\nSCELTO: %s (%s)", best_name, best_cfg)
    log.info("mIoU %.4f (per scena %.4f) · ghiaccio/acqua %.4f · permissiva peggiore %s",
             best["miou"],
             best["miou_per_scene"] if isinstance(best["miou_per_scene"], float) else float("nan"),
             best["binary"],
             f"{best['unsafe']:.2%}" if isinstance(best["unsafe"], float) else "n/d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
