"""Tests for the status page generator.

The page exists so a human can see project state at a glance, which makes a
wrong number on it worse than no page. Two of these tests pin bugs that shipped:
a download count that rendered negative, and Italian formatting that silently
lost thousands separators.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from iceroute.pitch.dashboard import (
    MISSING,
    _it,
    _md,
    bar_rows,
    build_page,
    collect,
    download_status,
    line_chart,
    parse_decisions,
    parse_table,
    read_runs,
)

TODAY = date(2026, 9, 5)


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (1_964_182, 0, "1.964.182"),
        (14_188, 0, "14.188"),
        (0.3649, 3, "0,365"),
        (2.053, 2, "2,05"),
        (999, 0, "999"),
        (0, 0, "0"),
        (-1234.5, 1, "-1.234,5"),
    ],
)
def test_italian_number_formatting(value, decimals, expected):
    """Regression: a placeholder-based swap once turned the separator into a
    null byte, and stripping it deleted every thousands separator."""
    assert _it(value, decimals) == expected


def test_thousands_separator_survives():
    """The specific failure: 1964182 must not render as 1964182 or 1964,182."""
    out = _it(1_964_182)
    assert out.count(".") == 2
    assert "," not in out


# ---------------------------------------------------------------------------
# Download counting
# ---------------------------------------------------------------------------


def test_download_count_is_never_negative(tmp_path):
    """Regression: scenes and reference labels were both counted, then
    subtracted, so a partly-downloaded test set rendered as -7/20."""
    test_dir = tmp_path / "data/raw/test"
    test_dir.mkdir(parents=True)
    for i in range(3):
        (test_dir / f"2020010{i}T000000_dmi_prep.nc").touch()
    for i in range(20):
        (test_dir / f"202001{i:02d}T000000_dmi_prep_reference.nc").touch()

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/split.yaml").write_text(
        "split:\n  train: [a.nc]\n  val: []\n  demo: []\n"
    )

    out = download_status(tmp_path)
    for key, v in out.items():
        assert v["have"] >= 0, f"{key} negativo: {v}"
    assert out["test ufficiale"]["have"] == 3
    assert out["etichette test"]["have"] == 20


def test_glob_does_not_count_references_as_scenes(tmp_path):
    test_dir = tmp_path / "data/raw/test"
    test_dir.mkdir(parents=True)
    (test_dir / "s_prep.nc").touch()
    (test_dir / "s_prep_reference.nc").touch()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/split.yaml").write_text("split:\n  train: []\n  val: []\n  demo: []\n")

    out = download_status(tmp_path)
    assert out["test ufficiale"]["have"] == 1


# ---------------------------------------------------------------------------
# Parsing DECISIONS.md
# ---------------------------------------------------------------------------


def test_parse_table_reads_claim_rows():
    text = (
        "| A-01 | qualcosa | **CHIUSA** misurata | usare il misurato |\n"
        "| A-02 | altro | aperta | da fare |\n"
        "| P-01 | provvisorio | Chiusa 03/09 | vedi D-006 |\n"
    )
    claims = parse_table(text, "A")
    assert [c["id"] for c in claims] == ["A-01", "A-02"]
    assert "CHIUSA" in claims[0]["detail"]
    assert [p["id"] for p in parse_table(text, "P")] == ["P-01"]


def test_parse_table_ignores_headers_and_separators():
    text = "| ID | Cosa | Stato | Chiude |\n|---|---|---|---|\n| P-01 | a | b | c |\n"
    assert len(parse_table(text, "P")) == 1


def test_parse_decisions_extracts_id_title_and_date():
    text = (
        "## D-001 — Prima decisione\n\n**Data:** 2026-09-02\n\ntesto\n\n"
        "## D-017 — Ultima decisione\n\n**Data:** 2026-09-05\n\naltro\n"
    )
    out = parse_decisions(text)
    assert [d["id"] for d in out] == ["D-001", "D-017"]
    assert out[1]["title"] == "Ultima decisione"
    assert out[1]["date"] == "2026-09-05"


def test_real_decisions_file_parses():
    """The page is only useful if it tracks the actual file."""
    path = Path("DECISIONS.md")
    if not path.exists():
        pytest.skip("DECISIONS.md assente")
    text = path.read_text()
    assert len(parse_decisions(text)) >= 10
    assert len(parse_table(text, "A")) >= 5
    assert len(parse_table(text, "P")) >= 5


# ---------------------------------------------------------------------------
# Training log
# ---------------------------------------------------------------------------


def test_runs_split_on_epoch_reset(tmp_path):
    """The CSV is append-only, so several runs share one file."""
    csv_path = tmp_path / "train_log.csv"
    csv_path.write_text(
        "epoch,val_miou\n1,0.1\n2,0.2\n1,0.3\n2,0.4\n3,0.5\n"
    )
    runs = read_runs(csv_path)
    assert len(runs) == 2
    assert len(runs[0]) == 2
    assert len(runs[1]) == 3


def test_missing_training_log_is_not_an_error(tmp_path):
    assert read_runs(tmp_path / "nope.csv") == []


def test_malformed_rows_are_skipped(tmp_path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("epoch,val_miou\n1,0.1\nboh,0.2\n2,0.3\n")
    assert sum(len(r) for r in read_runs(csv_path)) == 2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_line_chart_handles_no_data():
    assert "Nessun dato" in line_chart([{"label": "x", "colour": "#fff", "points": []}])


def test_line_chart_handles_a_flat_series():
    """A constant series must not divide by a zero range."""
    svg = line_chart([{"label": "x", "colour": "#fff", "points": [(1, 0.5), (2, 0.5)]}])
    assert "<svg" in svg
    assert "nan" not in svg.lower()


def test_line_chart_emits_one_path_per_series():
    svg = line_chart([
        {"label": "a", "colour": "#111", "points": [(1, 0.1), (2, 0.2)]},
        {"label": "b", "colour": "#222", "points": [(1, 0.3), (2, 0.4)]},
    ])
    assert svg.count("<path") == 2


def test_bar_rows_scale_to_the_largest_value():
    out = bar_rows([("a", 1.0, "100%"), ("b", 0.5, "50%")])
    assert "width:100.0%" in out
    assert "width:50.0%" in out


def test_bar_rows_survive_all_zeros():
    assert "<div" in bar_rows([("a", 0, "0"), ("b", 0, "0")])


def test_markdown_bold_and_code_are_rendered_not_escaped():
    out = _md("un **grassetto** e un `codice`")
    assert "<b" in out
    assert "mono" in out


def test_markdown_escapes_html():
    assert "&lt;script&gt;" in _md("<script>alert(1)</script>")


# ---------------------------------------------------------------------------
# The page as a whole
# ---------------------------------------------------------------------------


def test_page_builds_from_an_empty_repo(tmp_path):
    """With no artifacts at all the page must still render, saying so."""
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()

    page = build_page(collect(tmp_path, TODAY))
    assert "<title>" in page
    assert MISSING in page
    assert "nan" not in page.lower()


def test_page_has_no_external_resources():
    """SPEC §10.4: the demo laptop runs with the network off."""
    import re

    ctx = collect(Path("."), TODAY)
    page = build_page(ctx)
    assert not re.search(r'(?:src|href)=["\']https?://', page)


def test_page_reports_days_remaining():
    page = build_page(collect(Path("."), TODAY))
    assert "32" in page          # to the pitch, 7 Oct 2026
    assert "26" in page          # to the freeze, 1 Oct 2026


def test_page_never_shows_a_number_that_was_not_measured(tmp_path):
    """The project's central rule, applied to the page a human reads."""
    (tmp_path / "artifacts").mkdir()
    for d in ("tests", "src", "scripts"):
        (tmp_path / d).mkdir()

    page = build_page(collect(tmp_path, TODAY))
    # With no metrics.json there is nothing to quote, so the compression and
    # INT8 cards must say so rather than render 0.00.
    assert page.count(MISSING) >= 2
    assert "0,00 MB" not in page
