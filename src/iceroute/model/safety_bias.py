"""Break ties towards the more dangerous ice class. Answers the 26% finding.

Measured on the AI4Arctic official test set, the trained model places up to
**26.4% of pixels in a less restrictive POLARIS operation level than reality**
— it says the ice is safer than it is. That is the one error direction this
project cannot ship: being too cautious costs hours, being too permissive costs
a ship.

``argmax`` is the wrong decision rule for that. It minimises classification
error, which treats "called old ice thick first-year" and "called thick
first-year old ice" as the same mistake. POLARIS does not: the first can route a
vessel into ice it is not certified for, the second adds a detour.

So the rule here is: among the classes the model considers **plausible** — those
whose probability is at least ``tau`` times the top probability — take the one
with the lowest Risk Index Value for the vessel in question. With ``tau = 1.0``
this is exactly argmax. As ``tau`` falls the model becomes progressively more
conservative, and mIoU falls with it.

That trade is the point, and it has to be *measured*, not assumed: a rule that
made the map uniformly "old ice" would score zero unsafe pixels and be useless.
``sweep_tau`` produces the curve so the operating point is chosen on evidence.

This is a decision rule applied at inference, not a retraining. It costs one
comparison per pixel and changes no weights, so it is available on the FPGA
target too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from iceroute.model.metrics import ConfusionMatrix
from iceroute.model.rio_error import rio_error
from iceroute.routing.polaris import PolarisTable

log = logging.getLogger(__name__)


def danger_order(
    ice_class: str,
    sod_to_polaris: dict[int, str],
    *,
    table: PolarisTable | None = None,
    n_classes: int = 6,
) -> np.ndarray:
    """RIV per SOD class id. Lower means more dangerous for this vessel."""
    table = table or PolarisTable.load()
    return table.riv_lookup(ice_class, sod_to_polaris, n_classes=n_classes)


def safety_biased_labels(
    proba: np.ndarray,
    riv: np.ndarray,
    tau: float = 1.0,
) -> np.ndarray:
    """Argmax, except ties within ``tau`` go to the most dangerous class.

    Parameters
    ----------
    proba
        ``(n_classes, H, W)`` softmax probabilities.
    riv
        Risk index value per class, from ``danger_order``.
    tau
        Plausibility threshold in (0, 1]. A class competes if its probability is
        at least ``tau * max_probability`` at that pixel. ``tau = 1.0`` reduces
        to plain argmax.

    Notes
    -----
    Relative to the top probability rather than absolute: an absolute floor
    behaves completely differently in a confident region than in an ambiguous
    one, and the whole point is to act on *relative* uncertainty.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau deve stare in (0, 1], non {tau}")
    if proba.ndim != 3:
        raise ValueError(f"Attese (n_classes, H, W), ricevuto {proba.shape}")
    if len(riv) != proba.shape[0]:
        raise ValueError(f"{len(riv)} risk value per {proba.shape[0]} classi: non corrispondono.")

    if tau == 1.0:
        return proba.argmax(axis=0).astype("uint8")

    top = proba.max(axis=0, keepdims=True)
    plausible = proba >= tau * top

    # Among plausible classes pick the lowest RIV. Implemented by ranking:
    # order classes by danger, then take the first plausible one.
    order = np.argsort(riv, kind="stable")  # most dangerous first
    ranked = plausible[order]  # (n_classes, H, W)
    first = ranked.argmax(axis=0)  # index into `order`
    return order[first].astype("uint8")


@dataclass
class TradeOff:
    """One operating point on the safety/accuracy curve."""

    tau: float
    miou: float
    binary_accuracy: float
    unsafe_fraction: float
    overcautious_fraction: float
    rio_mae: float
    rio_bias: float

    def as_dict(self) -> dict:
        return {
            "tau": self.tau,
            "miou": round(self.miou, 4),
            "binary_accuracy": round(self.binary_accuracy, 4),
            "unsafe_fraction": round(self.unsafe_fraction, 5),
            "overcautious_fraction": round(self.overcautious_fraction, 5),
            "rio_mae": round(self.rio_mae, 3),
            "rio_bias": round(self.rio_bias, 3),
        }


def sweep_tau(
    proba: np.ndarray,
    truth: np.ndarray,
    ice_class: str,
    sod_to_polaris: dict[int, str],
    *,
    taus: list[float] | None = None,
    table: PolarisTable | None = None,
    sic: np.ndarray | None = None,
    mask_valid: np.ndarray | None = None,
    n_classes: int = 6,
) -> list[TradeOff]:
    """Measure the cost of caution across a range of ``tau``.

    Returns one row per threshold so the operating point can be argued from the
    curve. Without this the bias would be a knob someone turned until the number
    looked good.
    """
    table = table or PolarisTable.load()
    riv = danger_order(ice_class, sod_to_polaris, table=table, n_classes=n_classes)
    taus = taus or [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]

    out = []
    for tau in taus:
        pred = safety_biased_labels(proba, riv, tau)
        cm = ConfusionMatrix(n_classes)
        cm.update(truth, pred)
        try:
            err = rio_error(
                truth,
                pred,
                ice_class,
                sod_to_polaris,
                table=table,
                sic=sic,
                mask_valid=mask_valid,
                n_classes=n_classes,
            )
        except ValueError as exc:
            log.warning("tau=%.2f: errore RIO non calcolabile — %s", tau, exc)
            continue
        out.append(
            TradeOff(
                tau=tau,
                miou=cm.miou(),
                binary_accuracy=cm.binary_accuracy(),
                unsafe_fraction=err.unsafe_fraction,
                overcautious_fraction=err.overcautious_fraction,
                rio_mae=err.rio_mae,
                rio_bias=err.rio_bias,
            )
        )
    return out


def choose_tau(
    curve: list[TradeOff], *, max_unsafe: float = 0.05, min_miou_ratio: float = 0.80
) -> tuple[float, str]:
    """Pick the least conservative tau that meets the safety budget.

    Two constraints, both needed. Without the mIoU floor the answer is always
    the most conservative point — a map that calls everything old ice has zero
    unsafe pixels and no information. Without the unsafe budget there is no
    safety argument at all.

    Returns ``(tau, rationale)``. The mIoU floor is treated as a **hard**
    constraint and safety as the objective within it: a map conservative enough
    to have no unsafe pixels but too degraded to route on is not a safe map, it
    is a useless one, and shipping it would trade a visible risk for an
    invisible one. When the safety budget cannot be met inside the floor, the
    answer is plain argmax plus an explicit statement that it cannot be met.
    """
    if not curve:
        return 1.0, "Nessuna curva misurata: resta argmax."

    baseline = next((t for t in curve if t.tau == 1.0), curve[0])
    floor = baseline.miou * min_miou_ratio
    usable = [t for t in curve if t.miou >= floor]

    feasible = [t for t in usable if t.unsafe_fraction <= max_unsafe]
    if feasible:
        best = max(feasible, key=lambda t: t.tau)
        return best.tau, (
            f"tau={best.tau:g}: permissiva {best.unsafe_fraction:.2%} entro il budget "
            f"{max_unsafe:.0%}, mIoU {best.miou:.3f} ({best.miou / baseline.miou:.0%} "
            f"di argmax). Cauta {best.overcautious_fraction:.2%}."
        )

    if usable:
        safest = min(usable, key=lambda t: t.unsafe_fraction)
        return safest.tau, (
            f"NESSUN punto rispetta la permissiva ≤ {max_unsafe:.0%} restando sopra "
            f"mIoU {floor:.3f}. Il più cauto utilizzabile è tau={safest.tau:g}, "
            f"permissiva {safest.unsafe_fraction:.2%}. Il budget va rinegoziato "
            f"esplicitamente, non allargato in silenzio."
        )

    return 1.0, (
        f"Nessun punto della curva resta sopra mIoU {floor:.3f}: la regola di "
        f"decisione non può recuperare la sicurezza senza distruggere la mappa. "
        f"Resta argmax, e il problema è del modello, non della soglia."
    )


def format_curve(curve: list[TradeOff]) -> str:
    lines = [
        f"  {'tau':>5}{'mIoU':>9}{'gh/acqua':>11}{'PERMISSIVA':>13}{'cauta':>9}{'RIO MAE':>10}"
    ]
    lines.append("  " + "-" * 57)
    for t in curve:
        lines.append(
            f"  {t.tau:5.2f}{t.miou:9.3f}{t.binary_accuracy:11.3f}"
            f"{t.unsafe_fraction:12.2%}{t.overcautious_fraction:9.2%}{t.rio_mae:10.2f}"
        )
    return "\n".join(lines)
