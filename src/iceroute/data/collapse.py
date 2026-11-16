"""Reduce the six SOD classes to three. The SPEC §7.4 fallback.

Why this exists
---------------
Six-class SOD did not reach the mIoU threshold: 0.272 on the official test set,
against a required 0.60, with the model confidently wrong rather than uncertain
(D-018). SPEC §7.4 prescribes the response — *"ridurre a binario ghiaccio/acqua
+ una classe di ghiaccio spesso, e ridimensionare le affermazioni nel pitch.
Non aggiustare il numero."*

**This is a reduction of the problem, not an improvement of the model.** Any
number produced under this scheme describes an easier task and must be labelled
as such wherever it appears. ``describe()`` exists so that label travels with
the data instead of relying on someone remembering.

The grouping, and why it is safe
--------------------------------
    0  Open water      <- SOD 0
    1  Thin ice        <- SOD 1 New ice, 2 Young ice, 3 Thin FYI
    2  Thick ice       <- SOD 4 Thick FYI, 5 Old ice

Two facts make this defensible rather than convenient:

* **New ice and Young ice carry identical risk values** for every vessel class
  we ship, so merging them costs nothing at all in POLARIS terms (D-015).
* Each merged group takes the risk value of its **most dangerous** member —
  Thin FYI for the thin group, Old ice for the thick one. Verified against the
  circular in ``conservative_member``: the answer is the same for PC6, PC7 and
  1A, so one POLARIS ice type serves all three.

The second point matters more than the first. Because the merge rounds *towards
danger*, it cannot manufacture a permissive error: a pixel the six-class model
called Thick FYI when it was really Old ice now maps to the same collapsed
class, priced at Old ice. The reduction should therefore lower the permissive
fraction as a side effect — which is measured, not assumed.
"""

from __future__ import annotations

import logging

import numpy as np

from iceroute.model.metrics import IGNORE_INDEX
from iceroute.routing.polaris import PolarisTable

log = logging.getLogger(__name__)

# Collapsed class ids and the SOD classes each absorbs.
GROUPS: dict[int, list[int]] = {
    0: [0],  # open water
    1: [1, 2, 3],  # new ice, young ice, thin first-year
    2: [4, 5],  # thick first-year, old ice
}

NAMES: dict[int, str] = {
    0: "Acqua libera",
    1: "Ghiaccio sottile",
    2: "Ghiaccio spesso",
}

N_CLASSES = len(GROUPS)


def lookup_table(n_source: int = 6) -> np.ndarray:
    """Array mapping a SOD class id to its collapsed id, for vectorised remap.

    Index ``IGNORE_INDEX`` maps to itself so ignored pixels stay ignored; a
    table that silently turned 255 into class 0 would relabel every unannotated
    pixel as open water, which is both wrong and permissive.
    """
    table = np.full(256, IGNORE_INDEX, dtype="uint8")
    for new, olds in GROUPS.items():
        for old in olds:
            if old < n_source:
                table[old] = new
    return table


def collapse(labels: np.ndarray, n_source: int = 6) -> np.ndarray:
    """Remap a SOD label array to the three-class scheme."""
    arr = np.asarray(labels)
    if arr.dtype != np.uint8:
        # Anything outside 0-255 cannot be a class id; clip rather than wrap.
        arr = np.clip(arr, 0, 255).astype("uint8")
    return lookup_table(n_source)[arr]


def collapse_proba(proba: np.ndarray) -> np.ndarray:
    """Sum class probabilities within each group.

    Summing is the correct reduction: the collapsed class is the *union* of the
    source classes, so its probability is the sum of theirs. Taking the maximum
    instead — a tempting shortcut — would under-report a collapsed class whose
    mass is spread across its members, which is exactly the case that matters
    here.
    """
    if proba.ndim != 3:
        raise ValueError(f"Attese (n_classes, H, W), ricevuto {proba.shape}")
    out = np.zeros((N_CLASSES, *proba.shape[1:]), dtype=proba.dtype)
    for new, olds in GROUPS.items():
        for old in olds:
            if old < proba.shape[0]:
                out[new] += proba[old]
    return out


def conservative_member(
    group: list[int],
    sod_to_polaris: dict[int, str],
    ice_classes: list[str],
    *,
    table: PolarisTable | None = None,
) -> tuple[int, bool]:
    """The SOD class in ``group`` with the lowest risk value.

    Returns ``(class_id, agrees_across_vessels)``. The flag matters: if the most
    dangerous member differed between a PC6 and a 1A, one POLARIS ice type could
    not represent the group and the mapping would have to be per vessel. It does
    not differ for our three classes — but that is a property of the circular,
    not a guarantee, so it is checked rather than assumed.
    """
    table = table or PolarisTable.load()
    picks = {ic: min(group, key=lambda k: table.riv(ic, sod_to_polaris[k])) for ic in ice_classes}
    values = set(picks.values())
    return next(iter(values)) if len(values) == 1 else picks[ice_classes[0]], len(values) == 1


def polaris_mapping(
    sod_to_polaris: dict[int, str],
    ice_classes: list[str],
    *,
    table: PolarisTable | None = None,
) -> dict[int, str]:
    """POLARIS ice type per collapsed class, derived from the risk table.

    Derived rather than written down, so a change to ``data/polaris_rv.csv`` or
    to the SOD mapping propagates instead of leaving a stale constant behind.
    """
    table = table or PolarisTable.load()
    out = {}
    for new, olds in GROUPS.items():
        pick, agree = conservative_member(olds, sod_to_polaris, ice_classes, table=table)
        if not agree:
            raise ValueError(
                f"Il membro più cauto del gruppo {new} ({olds}) cambia a seconda della "
                f"classe di nave: un solo tipo di ghiaccio POLARIS non lo rappresenta. "
                f"Il raggruppamento va rifatto per classe di nave."
            )
        out[new] = sod_to_polaris[pick]
    return out


def describe(source_miou: float | None = None) -> str:
    """The disclaimer that must travel with every number from this scheme."""
    text = (
        "SCHEMA RIDOTTO A 3 CLASSI (ripiego SPEC §7.4): acqua libera / ghiaccio "
        "sottile / ghiaccio spesso. È un problema PIÙ FACILE di quello a 6 classi: "
        "questi numeri non sono confrontabili con la soglia mIoU 0,60, che si "
        "riferisce alle 6 classi SOD."
    )
    if source_miou is not None:
        text += f" Il modello a 6 classi si era fermato a mIoU {source_miou:.3f}."
    return text


def summary() -> str:
    lines = ["Raggruppamento SOD → 3 classi (SPEC §7.4):"]
    for new, olds in GROUPS.items():
        lines.append(f"  {new} {NAMES[new]:<18} ← SOD {olds}")
    return "\n".join(lines)
