"""Tests for the model selector.

The selector decides which network the demo is built on. Its failure mode is not
a crash: it is picking a defensible-looking model for a reason nobody can
reconstruct three weeks later.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pick_model import MIN_GAIN, SAFETY_MARGIN, choose  # noqa: E402


def _cand(name, miou, unsafe, per_scene=None):
    return (name, f"configs/{name}.yaml",
            {"miou": miou, "miou_per_scene": per_scene if per_scene is not None else miou,
             "binary": 0.9, "unsafe": unsafe, "split": "test_ufficiale", "per_class": {}})


def test_a_clear_accuracy_win_is_taken():
    inc = _cand("baseline", 0.50, 0.30)
    ch = _cand("sfidante", 0.55, 0.30)
    assert choose([inc, ch], inc)[0] == "sfidante"


def test_a_more_permissive_challenger_is_refused_however_accurate():
    """The veto. More accuracy does not buy a model that sends ships into ice
    POLARIS forbids."""
    inc = _cand("baseline", 0.50, 0.20)
    ch = _cand("sfidante", 0.70, 0.60)
    assert choose([inc, ch], inc)[0] == "baseline"


def test_safety_decides_when_accuracy_is_indistinguishable():
    """The case that actually occurred: Lovász, 0,0028 behind on pooled mIoU and
    ten points less permissive. The original rule had no branch for it."""
    inc = _cand("baseline", 0.5352, 0.5172)
    ch = _cand("lovasz", 0.5324, 0.4129)
    assert choose([inc, ch], inc)[0] == "lovasz"


def test_a_small_safety_gain_does_not_swap_the_model():
    """Below SAFETY_MARGIN the difference is not worth rebuilding the demo for."""
    inc = _cand("baseline", 0.50, 0.30)
    ch = _cand("sfidante", 0.50, 0.30 - SAFETY_MARGIN / 2)
    assert choose([inc, ch], inc)[0] == "baseline"


def test_the_incumbent_wins_ties():
    inc = _cand("baseline", 0.50, 0.30)
    ch = _cand("sfidante", 0.50, 0.30)
    assert choose([inc, ch], inc)[0] == "baseline"


def test_a_challenger_within_min_gain_alone_is_not_enough():
    """Accuracy noise must not swap the model when safety is equal."""
    inc = _cand("baseline", 0.50, 0.30)
    ch = _cand("sfidante", 0.50 + MIN_GAIN / 2, 0.30)
    assert choose([inc, ch], inc)[0] == "baseline"


def test_the_winner_does_not_depend_on_the_order_of_the_list():
    """The bug this function was rewritten to fix.

    Walking the list and comparing against the running best made the answer
    depend on how CANDIDATES happened to be written: `piu_dati` is 6,9 points
    safer than the baseline and would win if seen first, after which `lovasz` --
    only 3,5 points safer than *it*, below the margin -- could no longer
    displace it. Same numbers, different model, no way to reconstruct why.
    """
    inc = _cand("baseline", 0.5352, 0.5172)
    lov = _cand("lovasz", 0.5324, 0.4129)
    more = _cand("piu_dati", 0.5335, 0.4481)

    winners = {choose(list(p), inc)[0] for p in itertools.permutations([inc, lov, more])}
    assert winners == {"lovasz"}, f"il vincitore dipende dall'ordine: {winners}"


@pytest.mark.parametrize("missing", ["challenger", "incumbent"])
def test_an_unmeasured_safety_fraction_never_triggers_a_swap(missing):
    """Absent evidence is not evidence of safety."""
    inc = _cand("baseline", 0.50, None if missing == "incumbent" else 0.50)
    ch = _cand("sfidante", 0.50, None if missing == "challenger" else 0.10)
    assert choose([inc, ch], inc)[0] == "baseline"
