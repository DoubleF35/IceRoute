"""Tests for the live-computation path.

The claim this code makes on stage is "what you are watching is the real
pipeline, not pictures I made earlier". That claim is only worth anything if
two things hold: it runs the same function the product runs, and when it
disagrees with the precomputed result it *says so* rather than showing whichever
answer came last.
"""

from __future__ import annotations

import inspect

import pytest

from iceroute.demo import live


def _ov(cells):
    return {"grid_shape": [100, 100], "iceroute": {"path": cells}}


# ---------------------------------------------------------------------------
# Same code as the product
# ---------------------------------------------------------------------------


def test_the_live_path_calls_process_scene_rather_than_reimplementing_it():
    """A second implementation of the chain would make "watch it compute" a
    statement about code the product does not ship."""
    src = inspect.getsource(live.compute)
    assert "process_scene" in src
    for reimplemented in ("predict_scene", "find_route", "geocode_scene"):
        assert reimplemented not in src, (
            f"live.compute chiama {reimplemented} direttamente: "
            f"sta rifacendo la catena invece di eseguirla"
        )


def test_process_scene_accepts_the_progress_hook():
    """The only thing the live path adds. If the signature drifts, the stage
    display silently stops updating and the run looks frozen."""
    from iceroute.demo.precompute import process_scene

    assert "on_stage" in inspect.signature(process_scene).parameters


def test_process_scene_runs_without_a_hook():
    """precompute itself passes nothing; the hook must be optional."""
    from iceroute.demo.precompute import process_scene

    assert inspect.signature(process_scene).parameters["on_stage"].default is None


# ---------------------------------------------------------------------------
# Failure is returned, not raised
# ---------------------------------------------------------------------------


def test_a_broken_run_returns_an_error_instead_of_raising(tmp_path):
    """This sits behind a button on a stage. A traceback across the projector
    is a worse outcome than a red box."""
    run = live.compute(
        tmp_path / "non-esiste.nc",
        model=None,
        device="cpu",
        out_dir=tmp_path,
        train_config="configs/train3.yaml",
    )
    assert not run.ok
    assert run.error
    assert run.total_seconds >= 0


def test_a_missing_checkpoint_is_a_clear_error(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("output:\n  best_checkpoint: /non/esiste.pt\n")
    with pytest.raises(FileNotFoundError, match="Checkpoint assente"):
        live.load_model(cfg)


# ---------------------------------------------------------------------------
# Agreement with the precomputed result
# ---------------------------------------------------------------------------


def test_an_identical_route_is_reported_as_identical():
    cells = [[r, r] for r in range(20)]
    v = live.agreement(_ov(cells), _ov(cells))
    assert v["identical"] and v["comparable"] and v["n_cells"] == 20


def test_a_different_route_is_flagged_loudly_not_swallowed():
    """If the two ever diverge, that is a Tuesday problem — but only if the
    demo says so instead of quietly displaying the newer answer."""
    v = live.agreement(_ov([[r, r] for r in range(20)]), _ov([[r, r + 1] for r in range(20)]))
    assert v["comparable"] and not v["identical"]
    assert "DIVERSA" in v["note"]


def test_a_route_of_a_different_length_is_a_difference():
    v = live.agreement(_ov([[1, 1], [2, 2]]), _ov([[1, 1], [2, 2], [3, 3]]))
    assert not v["identical"]


@pytest.mark.parametrize("a,b", [({}, _ov([[1, 1]])), (_ov([[1, 1]]), {}), ({}, {})])
def test_a_missing_side_is_not_silently_an_agreement(a, b):
    v = live.agreement(a, b)
    assert not v["comparable"] and not v["identical"]


# ---------------------------------------------------------------------------
# Against the real build
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_live_run_reproduces_the_precomputed_route():
    """End to end, on a real scene: the whole point of the button.

    Marked slow — it loads a checkpoint and runs the full chain, about fifteen
    seconds on a GPU. Skipped when the raw scene or the model is absent.
    """
    import tempfile
    from pathlib import Path

    from iceroute.demo import overlay as ov

    picked = Path("artifacts/best_config.txt")
    raw = Path("data/raw/test/20200701T114012_cis_prep.nc")
    stored_dir = Path("data/demo/overlays")
    if not (picked.exists() and raw.exists() and stored_dir.exists()):
        pytest.skip("scena grezza, modello o precomputato assenti")

    cfg = Path(picked.read_text().strip())
    model, device = live.load_model(cfg)
    out = Path(tempfile.mkdtemp(prefix="iceroute_test_live_"))
    run = live.compute(raw, model=model, device=device, out_dir=out, train_config=cfg)

    assert run.ok, run.error
    assert [s.name for s in run.stages], "nessuna fase riportata"
    assert all(s.done for s in run.stages)

    stem = raw.name.replace(".nc", "")
    for ice_class in ("PC6", "IA"):
        verdict = live.agreement(
            ov.load(out / "overlays" / f"{stem}__{ice_class}.json"),
            ov.load(stored_dir / f"{stem}__{ice_class}.json"),
        )
        assert verdict["identical"], f"{ice_class}: {verdict['note']}"
