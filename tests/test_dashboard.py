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

from iceroute.report.dashboard import (
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
    (tmp_path / "configs/split.yaml").write_text("split:\n  train: [a.nc]\n  val: []\n  demo: []\n")

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
    csv_path.write_text("epoch,val_miou\n1,0.1\n2,0.2\n1,0.3\n2,0.4\n3,0.5\n")
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
    svg = line_chart(
        [
            {"label": "a", "colour": "#111", "points": [(1, 0.1), (2, 0.2)]},
            {"label": "b", "colour": "#222", "points": [(1, 0.3), (2, 0.4)]},
        ]
    )
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


CSV_HEADER = (
    "epoch,train_loss,train_ce,train_dice,val_loss,val_miou,"
    "val_accuracy,val_binary_accuracy,lr,seconds,peak_vram_gb\n"
)


def _repo(
    tmp_path,
    *,
    n_classes=3,
    log_csv="artifacts/logs/mine.csv",
    val_miou=0.62,
    test_miou=None,
    picked="configs/mine.yaml",
):
    """A repo skeleton with one training config and its own log."""
    for d in ("tests", "src", "scripts", "configs", "artifacts/logs"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / picked).write_text(
        f"model:\n  n_classes: {n_classes}\noutput:\n  log_csv: {log_csv}\n"
    )
    (tmp_path / "artifacts/best_config.txt").write_text(picked + "\n")
    (tmp_path / log_csv).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / log_csv).write_text(CSV_HEADER + f"1,1,1,1,1,{val_miou},0.9,0.9,1e-4,10,1\n")
    if test_miou is not None:
        import json

        (tmp_path / "artifacts/metrics.json").write_text(
            json.dumps(
                {
                    "evaluation": {
                        "split": "test_ufficiale",
                        "scenes": {"aggregate": {"miou": test_miou}},
                    }
                }
            )
        )
    return tmp_path


def test_dashboard_reads_the_log_of_the_deployed_model_not_the_last_trained(tmp_path):
    """The bug this page shipped with.

    Every config appends to its own CSV. Reading a fixed path meant that once
    the §7.4 fallback moved training to a new config, the page kept reporting a
    six-class run that had already been abandoned — for days, with no error.
    """
    repo = _repo(tmp_path, val_miou=0.62)
    # A stale log from an abandoned config, at the path the page used to read.
    (repo / "artifacts/logs/train_log.csv").write_text(
        CSV_HEADER + "1,1,1,1,1,0.375,0.9,0.9,1e-4,10,1\n"
    )

    ctx = collect(repo, TODAY)
    assert ctx["best_miou"] == pytest.approx(0.62)
    assert ctx["selected_config"] == "configs/mine.yaml"


def test_the_six_class_threshold_is_not_applied_to_a_three_class_score(tmp_path):
    """SPEC §7.4's 0,60 is about six classes; three is a strictly easier task.

    Tagging a three-class score "sotto 0,60" compares two different problems
    and reads as a verdict on the model.
    """
    page = build_page(collect(_repo(tmp_path, n_classes=3, test_miou=0.535), TODAY))
    assert "sotto 0,60" not in page
    assert "ripiego §7.4 a 3 classi" in page


def test_the_threshold_still_applies_when_the_scheme_is_six_class(tmp_path):
    """The tag is suppressed because the schemes differ, not because it is
    inconvenient. On six classes it must come back."""
    page = build_page(collect(_repo(tmp_path, n_classes=6, test_miou=0.535), TODAY))
    assert "sotto 0,60" in page


def test_headline_is_the_held_out_score_not_validation(tmp_path):
    """Validation picked the checkpoint, so it flatters it by an unknown
    amount. The number a human reads first has to be the one that did not."""
    ctx = collect(_repo(tmp_path, val_miou=0.62, test_miou=0.535), TODAY)
    page = build_page(ctx)
    assert "mIoU sul test ufficiale" in page
    assert _it(0.535, 3) in page


def test_headline_says_so_when_there_is_no_held_out_score(tmp_path):
    """Without an evaluation it falls back to validation — but must label it,
    not pass a validation number off as a held-out one."""
    page = build_page(collect(_repo(tmp_path, val_miou=0.62), TODAY))
    assert "mIoU di validazione" in page
    assert "nessun held-out" in page


def test_gate_4_is_never_green_from_artefacts_alone(tmp_path):
    """Its real criterion is three uninvolved people using it offline after a
    reboot. No file on disk can attest to that, so files must not turn it
    green."""
    import json

    repo = _repo(tmp_path, test_miou=0.535)
    (repo / "data/demo").mkdir(parents=True)
    (repo / "data/demo/index.json").write_text(
        json.dumps({"scenes": [{"scene_id": f"s{i}.nc"} for i in range(5)]})
    )
    (repo / "artifacts/figures").mkdir(parents=True)
    for i in range(4):
        (repo / f"artifacts/figures/f{i}.png").touch()

    gate4 = next(g for g in collect(repo, TODAY)["gates"] if g.name == "Gate 4")
    assert gate4.done is False
    assert "5 scene precalcolate" in gate4.detail
    assert "video di riserva mancante" in gate4.detail


def test_parameter_count_is_never_invented(tmp_path):
    """It used to fall back to a literal, which keeps being displayed after the
    architecture changes under it."""
    assert collect(_repo(tmp_path), TODAY)["n_params"] is None


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
    assert "32" in page  # to the pitch, 7 Oct 2026
    assert "26" in page  # to the freeze, 1 Oct 2026


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


# ---------------------------------------------------------------------------
# The route figure and its caption
# ---------------------------------------------------------------------------


def _repo_with_route(tmp_path, *, ice_class="PC6", blocked=True, lead_class=None):
    """A repo where the figure generator has run and left its lead behind."""
    import json

    repo = _repo(tmp_path)
    (repo / "artifacts/figures").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts/figures/route_lead.json").write_text(
        json.dumps(
            {
                "scene_id": "SCENA.nc",
                "ice_class": lead_class or ice_class,
                "figure": "02_useful.png",
            }
        )
    )
    (repo / "data/demo").mkdir(parents=True, exist_ok=True)
    (repo / "data/demo/index.json").write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": "SCENA.nc",
                        "region": "Kara",
                        "classes": {
                            ice_class: {
                                "route_exists": True,
                                "n_waypoints": 5,
                                "separation": {"pct_diagonal": 9.4, "max_km": 81.0},
                                "comparison": {
                                    "iceroute": {"time_hours": 23.7},
                                    "baseline": {
                                        "blocked_by_ice": blocked,
                                        "n_no_go_cells": 104 if blocked else 0,
                                    },
                                },
                            }
                        },
                    }
                ]
            }
        )
    )
    return repo


def test_the_route_caption_names_the_class_the_figure_was_drawn_for(tmp_path):
    """The drift this guards against was real and lived in the first version.

    The caption ranked over every vessel class and picked IA; the figure had
    been generated for PC6. The page then showed a PC6 route under a sentence
    about IA, with numbers to match — a wrong claim next to a correct picture.
    """
    repo = _repo_with_route(tmp_path, ice_class="PC6")
    (repo / "data/demo/index.json").write_text(
        (repo / "data/demo/index.json")
        .read_text()
        .replace(
            '"PC6": {',
            '"IA": {"route_exists": true, "n_waypoints": 9,'
            '"separation": {"pct_diagonal": 30.0, "max_km": 250.0},'
            '"comparison": {"iceroute": {"time_hours": 30.0},'
            '"baseline": {"blocked_by_ice": true, "n_no_go_cells": 999}}}, "PC6": {',
        )
    )

    story = collect(repo, TODAY)["route"]
    assert story["ice_class"] == "PC6", "la didascalia ha scelto una classe diversa dalla figura"
    assert story["n_no_go"] == 104
    assert "999" not in build_page(collect(repo, TODAY))


def test_no_recorded_lead_means_no_route_claim(tmp_path):
    """Absent evidence must not produce a caption."""
    repo = _repo_with_route(tmp_path)
    (repo / "artifacts/figures/route_lead.json").unlink()
    assert collect(repo, TODAY)["route"] == {}
    assert "celle vietate" not in build_page(collect(repo, TODAY))


def test_a_lead_pointing_at_a_missing_scene_makes_no_claim(tmp_path):
    import json

    repo = _repo_with_route(tmp_path)
    (repo / "artifacts/figures/route_lead.json").write_text(
        json.dumps({"scene_id": "SPARITA.nc", "ice_class": "PC6"})
    )
    assert collect(repo, TODAY)["route"] == {}


def test_a_passable_direct_route_is_labelled_a_weak_comparison(tmp_path):
    """If the figure fell back to a scene POLARIS allows, the two lines
    overlap and the picture argues nothing. The page must say so."""
    repo = _repo_with_route(tmp_path, blocked=False)
    page = build_page(collect(repo, TODAY))
    assert "confronto debole" in page
    assert "non può percorrere" not in page


def test_the_route_section_never_quotes_a_time_saving(tmp_path):
    """D-026: against an impossible voyage there is no time to save."""
    page = build_page(collect(_repo_with_route(tmp_path), TODAY))
    assert "risparmi" in page and "% in meno" not in page
    assert "ore risparmiate" not in page
