"""The CLI dispatches; it must never become a second implementation."""

from __future__ import annotations

import importlib

import pytest

from iceroute import cli


def test_help_lists_every_stage(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in [*cli.STAGES, "run", "demo"]:
        assert name in out


def test_unknown_command_fails_with_usage(capsys):
    assert cli.main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("name", sorted(cli.STAGES))
def test_every_stage_points_at_a_main(name):
    module = importlib.import_module(cli.STAGES[name][0])
    assert callable(getattr(module, "main", None)), name


def test_run_rejects_a_missing_scene(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["run", str(tmp_path / "nope.nc")])


def test_run_forwards_to_precompute(monkeypatch, tmp_path):
    scene = tmp_path / "20180707T113313_cis_prep.nc"
    scene.write_bytes(b"")
    seen = {}

    from iceroute.demo import precompute

    monkeypatch.setattr(precompute, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    assert cli.main(["run", str(scene), "--out", str(tmp_path / "o"), "--device", "cpu"]) == 0
    argv = seen["argv"]
    assert argv[argv.index("--scenes") + 1] == scene.name
    assert argv[argv.index("--raw-dir") + 1] == str(tmp_path)
    assert argv[argv.index("--device") + 1] == "cpu"
