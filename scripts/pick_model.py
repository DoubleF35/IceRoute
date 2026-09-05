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
    return {
        "miou": agg.get("miou"),
        "binary": agg.get("binary_accuracy"),
        "unsafe": scenes.get("worst_unsafe_fraction"),
        "split": ev.get("split"),
    }


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
            "%-10s mIoU %.4f · gh/acqua %.4f · permissiva peggiore %s",
            name, r["miou"], r["binary"],
            f"{r['unsafe']:.2%}" if isinstance(r["unsafe"], float) else "n/d",
        )

    if not found:
        log.info("\nNessun candidato valutato: resta configs/train3.yaml")
        Path("artifacts/best_config.txt").write_text("configs/train3.yaml\n")
        return 0

    incumbent = next((f for f in found if f[0] == "baseline"), found[0])
    best_name, best_cfg, best = incumbent

    for name, cfg, r in found:
        if name == best_name:
            continue
        gain = r["miou"] - best["miou"]
        # The veto: more accuracy is not worth a model that is more permissive
        # about ice a vessel may not enter.
        safer = (
            not isinstance(r["unsafe"], float)
            or not isinstance(best["unsafe"], float)
            or r["unsafe"] <= best["unsafe"] + 0.01
        )
        if gain > MIN_GAIN and safer:
            log.info("\n%s batte %s di %+.4f di mIoU senza peggiorare la sicurezza.",
                     name, best_name, gain)
            best_name, best_cfg, best = name, cfg, r
        elif gain > MIN_GAIN:
            log.info("\n%s ha mIoU migliore (%+.4f) ma frazione permissiva peggiore "
                     "(%.2f%% contro %.2f%%): SCARTATO.",
                     name, gain, 100 * r["unsafe"], 100 * best["unsafe"])

    Path("artifacts/best_config.txt").write_text(best_cfg + "\n")
    log.info("\nSCELTO: %s (%s)", best_name, best_cfg)
    log.info("mIoU %.4f · ghiaccio/acqua %.4f", best["miou"], best["binary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
