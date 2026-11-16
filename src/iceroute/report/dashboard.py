"""Generate index.html: the internal status page for IceRoute.

Every number on the page is read from an artifact on disk — ``metrics.json``,
``data_stats.json``, the training CSV, ``DECISIONS.md``, git. Nothing is typed
in by hand, so the page cannot drift from reality, and a value that has not been
measured yet renders as "non ancora misurato" rather than as a plausible
placeholder. That is the same rule the rest of the project runs on (SPEC §2.2),
applied to the one artefact a human actually looks at.

The page is **fully self-contained**: inline CSS, inline SVG, no fonts or
scripts from a CDN. The demo laptop has to work with the network card switched
off (SPEC §10.4), and a status page that goes blank offline would be the first
thing to fail on 7 October.

Run with ``make dashboard``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

log = logging.getLogger("dashboard")

FREEZE = date(2026, 10, 1)
PITCH = date(2026, 10, 7)

# ---------------------------------------------------------------------------
# Reading the artifacts
# ---------------------------------------------------------------------------


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_runs(csv_path: Path) -> list[list[dict]]:
    """Training history, split into runs.

    The CSV is append-only so it survives a crash, which means several runs sit
    end to end in one file. A run boundary is an epoch number that does not
    increase.
    """
    if not csv_path.exists():
        return []
    runs: list[list[dict]] = []
    current: list[dict] = []
    last = 0
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                epoch = int(row["epoch"])
            except (KeyError, ValueError):
                continue
            if epoch <= last and current:
                runs.append(current)
                current = []
            current.append({k: _num(v) for k, v in row.items()})
            last = epoch
    if current:
        runs.append(current)
    return runs


def _num(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def parse_table(text: str, prefix: str) -> list[dict]:
    """Rows of a DECISIONS.md markdown table whose first cell starts with ``prefix``."""
    out = []
    for line in text.splitlines():
        if not line.startswith(f"| {prefix}-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) >= 4:
            out.append(
                {
                    "id": cells[0],
                    "what": cells[1],
                    "detail": cells[2],
                    "action": cells[3],
                }
            )
    return out


def parse_decisions(text: str) -> list[dict]:
    """The D-xxx entries: id, title and date."""
    out = []
    for m in re.finditer(r"^## (D-\d+) — (.+?)$", text, re.MULTILINE):
        tail = text[m.end() : m.end() + 200]
        dm = re.search(r"\*\*Data:\*\*\s*([\d-]+)", tail)
        out.append({"id": m.group(1), "title": m.group(2), "date": dm.group(1) if dm else ""})
    return out


def git_info(repo: Path) -> dict:
    def run(*args) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    log_lines = run("log", "--pretty=%h|%ad|%s", "--date=short", "-12").splitlines()
    return {
        "n_commits": len(run("log", "--pretty=%h").splitlines()),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "recent": [
            dict(zip(("hash", "date", "subject"), line.split("|", 2), strict=False))
            for line in log_lines
            if "|" in line
        ],
    }


def count_tests(repo: Path) -> int:
    """Number of test functions, counted from source rather than by running them."""
    n = 0
    for p in (repo / "tests").glob("test_*.py"):
        n += len(re.findall(r"^def test_", p.read_text(), re.MULTILINE))
    return n


def count_lines(repo: Path, *dirs: str) -> int:
    n = 0
    for d in dirs:
        for p in (repo / d).rglob("*.py"):
            if "__pycache__" not in str(p):
                n += len(p.read_text().splitlines())
    return n


def download_status(repo: Path) -> dict:
    import yaml

    try:
        split = yaml.safe_load((repo / "configs/split.yaml").read_text())["split"]
    except (OSError, KeyError, yaml.YAMLError):
        return {}
    raw = repo / "data/raw/train"
    out = {}
    for k in ("train", "val", "demo"):
        ids = split.get(k, [])
        out[k] = {"have": sum(1 for i in ids if (raw / i).exists()), "total": len(ids)}
    # glob("*_prep.nc") already excludes "*_prep_reference.nc" — those end in
    # _reference.nc. Subtracting the references as well drove this negative.
    test_dir = repo / "data/raw/test"
    scenes = list(test_dir.glob("*_prep.nc")) if test_dir.exists() else []
    refs = list(test_dir.glob("*_prep_reference.nc")) if test_dir.exists() else []
    out["test ufficiale"] = {"have": len(scenes), "total": 20}
    out["etichette test"] = {"have": len(refs), "total": 20}
    return out


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


@dataclass
class Gate:
    name: str
    label: str
    done: bool
    detail: str
    blocking: str = ""


def gates(ctx: dict) -> list[Gate]:
    """Gate status derived from artifacts, not from a hand-maintained list."""
    stats = ctx["data_stats"]
    metrics = ctx["metrics"]
    dl = ctx["download"]
    best = ctx["best_miou"]
    routes = ctx["routes"]

    scenes_ok = all(
        dl.get(k, {}).get("have", 0) == dl.get(k, {}).get("total", 1)
        for k in ("train", "val", "demo")
    )
    quant = metrics.get("quantization", {})
    evaluation = metrics.get("evaluation", {})

    return [
        Gate(
            "Gate 0",
            "Ambiente e ispezione",
            True,
            "CUDA verificata con una matmul reale, nomi delle variabili letti dal dato",
        ),
        Gate(
            "Gate 1",
            "Pipeline dati",
            bool(stats.get("n_patches")),
            f"{_it(stats.get('n_patches', 0))} patch da {stats.get('n_scenes', 0)} scene"
            if stats
            else "cache non costruita",
            "" if scenes_ok else "download incompleto",
        ),
        Gate(
            "Gate 2",
            "Modello e quantizzazione",
            bool(quant.get("size")) and bool(evaluation),
            _gate2_detail(best, quant, evaluation),
            "manca la valutazione" if not evaluation else "",
        ),
        Gate(
            "Gate 3",
            "Routing POLARIS",
            bool(routes.get("scenes")),
            f"{sum(1 for s in routes.get('scenes', []) if s.get('usable'))} scene instradate"
            if routes
            else "non ancora eseguito",
        ),
        Gate(
            "Gate 4",
            "Demo e materiali",
            False,
            _gate4_detail(ctx["demo"]),
            # Deliberately never green from files alone. The gate is "three
            # people who did not build it can use it, offline, after a
            # reboot" — and no artefact on disk can attest to that. Marking it
            # done because precompute ran is exactly how a demo reaches a
            # stage untested.
            "resta la prova in condizioni reali: rete staccata, macchina "
            "riavviata, tre persone esterne che la usano da sole",
        ),
    ]


def _gate4_detail(demo: dict) -> str:
    bits = []
    if demo.get("n_scenes"):
        bits.append(
            f"{demo['n_scenes']} scene precalcolate"
            + (" + caso peggiore" if demo.get("has_worst_case") else "")
        )
    if demo.get("n_figures"):
        bits.append(f"{demo['n_figures']} figure")
    bits.append(
        "video di riserva registrato" if demo.get("has_video") else "video di riserva mancante"
    )
    return " · ".join(bits) if bits else "non iniziato"


def _gate2_detail(best, quant, evaluation) -> str:
    bits = []
    if best is not None:
        bits.append(f"mIoU {_it(best, 3)} val")
    if quant.get("size"):
        bits.append(f"INT8 {_it(quant['size']['int8_mb'], 2)} MB")
    if evaluation:
        bits.append("valutato")
    return " · ".join(bits) if bits else "training non concluso"


# ---------------------------------------------------------------------------
# SVG helpers — no chart library, the page must work offline
# ---------------------------------------------------------------------------


def line_chart(series: list[dict], width=680, height=240, pad=38) -> str:
    """Multi-series line chart as inline SVG.

    Each series is ``{"label", "colour", "points": [(x, y), ...]}``. Written by
    hand rather than pulled from a library because the page has to render with
    the network off.
    """
    pts = [p for s in series for p in s["points"]]
    if not pts:
        return '<p class="muted">Nessun dato di training ancora.</p>'

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1.0
    # A little headroom so the peak is not glued to the frame.
    span = y1 - y0
    y0, y1 = y0 - 0.05 * span, y1 + 0.08 * span

    def sx(x):
        return pad + (x - x0) / max(x1 - x0, 1) * (width - 2 * pad)

    def sy(y):
        return height - pad - (y - y0) / (y1 - y0) * (height - 2 * pad)

    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    # Horizontal guides with labels.
    for i in range(5):
        v = y0 + (y1 - y0) * i / 4
        y = sy(v)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" class="grid"/>')
        out.append(
            f'<text x="{pad - 7}" y="{y + 3.5:.1f}" class="axis" text-anchor="end">{v:.2f}</text>'
        )
    for i in range(5):
        x = x0 + (x1 - x0) * i / 4
        out.append(
            f'<text x="{sx(x):.1f}" y="{height - pad + 16}" class="axis" '
            f'text-anchor="middle">{x:.0f}</text>'
        )

    for s in series:
        if not s["points"]:
            continue
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
            for i, (x, y) in enumerate(s["points"])
        )
        out.append(f'<path d="{d}" fill="none" stroke="{s["colour"]}" stroke-width="2"/>')
        lx, ly = s["points"][-1]
        out.append(f'<circle cx="{sx(lx):.1f}" cy="{sy(ly):.1f}" r="3.5" fill="{s["colour"]}"/>')

    out.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:{s["colour"]}"></i>{html.escape(s["label"])}</span>'
        for s in series
    )
    return "".join(out) + f'<div class="legend">{legend}</div>'


def bar_rows(items: list[tuple[str, float, str]]) -> str:
    """Horizontal bars for a distribution: (label, fraction 0-1, display value)."""
    top = max((v for _, v, _ in items), default=1) or 1
    rows = []
    for label, value, shown in items:
        pct = 100 * value / top
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{html.escape(label)}</span>'
            f'<span class="bar-track"><i style="width:{pct:.1f}%"></i></span>'
            f'<span class="bar-value">{html.escape(shown)}</span></div>'
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

CSS = """
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:#0a0e14;color:#c9d4e3;
 font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:31px;margin:0;letter-spacing:-.5px;color:#eaf1fa;font-weight:650}
h2{font-size:13px;text-transform:uppercase;letter-spacing:1.3px;color:#6b7f99;
 margin:44px 0 14px;font-weight:650}
a{color:#5eb0ef}
.muted{color:#6b7f99}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}

header{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;
 flex-wrap:wrap;border-bottom:1px solid #1d2633;padding-bottom:22px}
.sub{color:#6b7f99;margin:7px 0 0;max-width:620px}
.count{text-align:right;flex-shrink:0}
.count b{display:block;font-size:38px;color:#eaf1fa;line-height:1;font-weight:650}
.count small{color:#6b7f99;font-size:12px;text-transform:uppercase;letter-spacing:1px}
.count .warn b{color:#f0b849}

.grid{display:grid;gap:13px}
.g4{grid-template-columns:repeat(auto-fit,minmax(215px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}

.card{background:#111823;border:1px solid #1d2633;border-radius:11px;padding:17px 19px}
.card h3{margin:0 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:1.1px;
 color:#6b7f99;font-weight:650}
.big{font-size:27px;color:#eaf1fa;font-weight:650;line-height:1.25;
 font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.note{font-size:12.5px;color:#6b7f99;margin-top:5px}

.tag{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.7px;
 padding:2.5px 8px;border-radius:5px;text-transform:uppercase;vertical-align:middle}
.ok{background:#0d3b2a;color:#4ade9b}
.no{background:#3b1c1c;color:#f08a8a}
.prov{background:#3d3212;color:#f0b849}
.wait{background:#1b2735;color:#7d92ad}

.gate{display:flex;gap:14px;align-items:flex-start;padding:13px 0;
 border-bottom:1px solid #1a2230}
.gate:last-child{border-bottom:0}
.dot{width:11px;height:11px;border-radius:50%;flex-shrink:0;margin-top:6px}
.dot.y{background:#4ade9b;box-shadow:0 0 9px #4ade9b66}
.dot.n{background:#33404f}
.gate b{color:#eaf1fa;font-weight:600}
.gate .d{color:#8397ae;font-size:13.5px}

table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:#6b7f99;font-weight:650;font-size:11px;text-transform:uppercase;
 letter-spacing:1px;padding:0 11px 9px;border-bottom:1px solid #1d2633}
td{padding:11px;border-bottom:1px solid #161e29;vertical-align:top;color:#9fb1c7}
td:first-child{white-space:nowrap;color:#eaf1fa;font-weight:600;
 font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
tr:last-child td{border-bottom:0}
.scroll{overflow-x:auto;background:#111823;border:1px solid #1d2633;border-radius:11px;
 padding:15px 6px}

.chart{width:100%;height:auto;display:block}
.grid line,line.grid{stroke:#1d2633;stroke-width:1}
text.axis{fill:#55677d;font-size:10.5px;font-family:ui-monospace,Menlo,monospace}
.legend{display:flex;gap:17px;flex-wrap:wrap;margin-top:9px;padding-left:38px}
.key{font-size:12px;color:#8397ae;display:flex;align-items:center;gap:6px}
.key i{width:11px;height:3px;border-radius:2px;display:inline-block}

.bar-row{display:flex;align-items:center;gap:11px;padding:5px 0;font-size:13px}
.bar-label{width:112px;flex-shrink:0;color:#9fb1c7}
.bar-track{flex:1;height:8px;background:#161e29;border-radius:4px;overflow:hidden}
.bar-track i{display:block;height:100%;background:linear-gradient(90deg,#2d6ea8,#5eb0ef);
 border-radius:4px}
.bar-value{width:78px;text-align:right;flex-shrink:0;color:#eaf1fa;
 font-family:ui-monospace,Menlo,monospace;font-size:12px}

details{background:#111823;border:1px solid #1d2633;border-radius:11px;padding:0}
summary{cursor:pointer;padding:15px 19px;color:#9fb1c7;font-size:13.5px;user-select:none}
summary:hover{color:#eaf1fa}
details[open] summary{border-bottom:1px solid #1d2633}
details .inner{padding:6px 13px 10px}

footer{margin-top:52px;padding-top:20px;border-top:1px solid #1d2633;
 color:#55677d;font-size:12.5px}
@media(max-width:620px){
 .count{text-align:left}h1{font-size:25px}.bar-label{width:84px}
 .wrap{padding:22px 15px 60px}
}
"""


MISSING = "non ancora misurato"


def _it(v, decimals: int = 0) -> str:
    """Italian number formatting: dot for thousands, comma for decimals.

    The page is in Italian and sits next to DECISIONS.md; ``14,188 patch`` reads
    as fourteen point one in that context.
    """
    # str.translate swaps both separators in one pass. Chained replace() needs a
    # placeholder character, and a placeholder is one more thing that can go
    # wrong — it already did once, silently deleting every thousands separator.
    return f"{v:,.{decimals}f}".translate(str.maketrans({",": ".", ".": ","}))


def _fmt(v, decimals: int = 3, missing: str = MISSING) -> str:
    return _it(v, decimals) if isinstance(v, int | float) else missing


def route_figure(repo: Path, max_px: int = 1000) -> str:
    """The route comparison, inlined as a data URI.

    Inlined rather than linked because this page has to survive being emailed
    or opened from a USB stick with the network off, which is also why it has
    no external resources at all. The deck version is 300 DPI and far too heavy
    to paste into an HTML file, so it is downscaled here — the page wants a
    thumbnail of the argument, not the printable asset.
    """
    import base64
    import io

    path = repo / "artifacts/figures/02_useful.png"
    if not path.exists():
        return (
            '<div class="note">Figura della rotta non generata: '
            '<span class="mono">make report</span>.</div>'
        )
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width > max_px:
        img = img.resize((max_px, round(img.height * max_px / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    uri = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<img src="data:image/jpeg;base64,{uri}" alt="Rotta IceRoute contro '
        f'rotta diretta" style="width:100%;border-radius:8px;display:block">'
    )


def _route_note(r: dict) -> str:
    """The caption. Every figure in it comes from the demo index.

    Deliberately makes no claim about time saved. On these scenes the direct
    route is forbidden, so there is no feasible voyage to be faster than, and
    a percentage would have to be invented against a baseline that does not
    exist (D-026). "That route is illegal and this one is not" is both the
    honest statement and the stronger one.
    """
    if not r:
        return (
            '<h3>Nessun confronto disponibile</h3><div class="note">Serve '
            '<span class="mono">make report</span>: la figura mostra la rotta '
            "diretta contro quella calcolata, e senza scene non c'è confronto.</div>"
        )

    head = f"<h3>{html.escape(r['scene_id'][:15])} · nave {html.escape(r['ice_class'])}</h3>"
    if r["blocked_by_ice"]:
        claim = f"""<div class="big">{_it(r["n_no_go"])} celle vietate</div>
<div class="note">La rotta diretta — la linea che seguirebbe una nave con la sola
carta nautica — attraversa {_it(r["n_no_go"])} celle oltre il limite POLARIS per una
nave {html.escape(r["ice_class"])}. Non è una rotta più lenta: è una rotta che quella
nave <b>non può percorrere</b>. IceRoute ne trova una in {_fmt(r["hours"], 1)} h,
riassunta in {r["n_waypoints"]} waypoint — gli unici dati che scendono a terra.</div>"""
    else:
        # The figure fell back to a scene whose direct route is legal. Saying
        # anything stronger here would be describing a picture that is not on
        # the page.
        claim = """<div class="big"><span class="tag prov">confronto debole</span></div>
<div class="note">Su questa scena la rotta diretta <b>è percorribile</b>: la figura
mostra due tracce che in larga parte coincidono, il che non dimostra la tesi.
Rigenerare con una scena in cui POLARIS vieta la linea retta.</div>"""

    return f"""{head}{claim}
<div class="note" style="margin-top:11px">Le due tracce si separano fino a
{_fmt(r["separation_km"], 0)} km, il {_fmt(r["separation_pct"], 1)}% della diagonale
dell'immagine. La scena è scelta su questo numero: un confronto in cui le due linee
si sovrappongono non dimostra niente, per quante celle siano vietate sotto.</div>
<div class="note" style="margin-top:11px">Nessuna percentuale di tempo risparmiato:
contro una rotta non percorribile non esiste un tempo di riferimento da cui
risparmiare (D-026).</div>"""


def build_page(ctx: dict) -> str:
    today = ctx["today"]
    to_freeze = (FREEZE - today).days
    to_pitch = (PITCH - today).days

    stats, metrics = ctx["data_stats"], ctx["metrics"]
    quant = metrics.get("quantization", {})
    size = quant.get("size", {})
    lat = (quant.get("latency") or {}).get("single_thread", {})
    best = ctx["best_miou"]
    dl = ctx["download"]
    g = ctx["git"]

    # -- headline cards ----------------------------------------------------
    # Two traps here, and the page fell into both.
    #
    # First, the headline must be the HELD-OUT number. Validation chose the
    # checkpoint, so it flatters it; the official test set did not.
    #
    # Second, SPEC §7.4's threshold of 0,60 is about the SIX-class problem.
    # Three classes is a strictly easier task, so tagging a three-class score
    # "sotto 0,60" compares two different things and reads as a verdict on the
    # model. The tag only appears when the numbers are commensurable.
    n_cls = ctx["n_classes"]
    headline = ctx["test_miou"] if ctx["test_miou"] is not None else best
    if headline is not None and n_cls == 6:
        miou_tag = (
            f'<span class="tag {"ok" if headline >= 0.60 else "prov"}">'
            f"{'sopra soglia' if headline >= 0.60 else 'sotto 0,60'}</span>"
        )
    elif n_cls:
        miou_tag = f'<span class="tag prov">ripiego §7.4 a {n_cls} classi</span>'
    else:
        miou_tag = ""

    if ctx["test_miou"] is not None:
        miou_note = f"su {ctx['eval_split'] or 'held-out'} · validazione {_fmt(best)} {miou_tag}"
        miou_label = "mIoU sul test ufficiale"
    else:
        miou_note = f"solo validazione, nessun held-out {miou_tag}"
        miou_label = "mIoU di validazione"

    int8 = size.get("int8_mb")
    ratio = ctx["compression"]

    cards = [
        (
            "Parametri del modello",
            _it(ctx["n_params"]) if ctx["n_params"] else "—",
            "budget SPEC 2.000.000"
            + (
                ' <span class="tag ok">entro</span>'
                if ctx["n_params"] and ctx["n_params"] < 2_000_000
                else ""
            ),
        ),
        (
            "Modello INT8 misurato",
            f"{_it(int8, 2)} MB" if int8 else MISSING,
            (
                f"{_it(size.get('bytes_per_parameter_int8', 0), 2)} byte/parametro · "
                "il deck dichiara 58 KB"
            )
            if int8
            else "serve <span class='mono'>make quantize</span>",
        ),
        (miou_label, _fmt(headline), miou_note),
        (
            "Latenza per tile (1 thread)",
            f"{_it(lat.get('int8', {}).get('median_ms_per_tile', 0))} ms"
            if lat.get("int8")
            else "non ancora misurata",
            "INT8 su CPU, mediana",
        ),
    ]
    cards_html = "".join(
        f'<div class="card"><h3>{h}</h3><div class="big">{v}</div><div class="note">{n}</div></div>'
        for h, v, n in cards
    )

    # -- gates -------------------------------------------------------------
    # Built with concatenation rather than nested f-strings: Python 3.11 rejects
    # a backslash inside an f-string expression, and HTML attributes need quotes.
    gate_parts = []
    for gt in ctx["gates"]:
        badge = '<span class="tag ok">verde</span>' if gt.done else ""
        blocked = (
            '<span class="tag wait">' + html.escape(gt.blocking) + "</span>" if gt.blocking else ""
        )
        gate_parts.append(
            '<div class="gate"><span class="dot ' + ("y" if gt.done else "n") + '"></span>'
            "<div><b>"
            + html.escape(gt.name)
            + " — "
            + html.escape(gt.label)
            + "</b> "
            + badge
            + " "
            + blocked
            + '<div class="d">'
            + html.escape(gt.detail)
            + "</div></div></div>"
        )
    gates_html = "".join(gate_parts)

    # -- training ----------------------------------------------------------
    runs = ctx["runs"]
    chart = '<p class="muted">Nessun training registrato.</p>'
    train_note = ""
    if runs:
        last = runs[-1]
        chart = line_chart(
            [
                {
                    "label": "mIoU validazione",
                    "colour": "#5eb0ef",
                    "points": [
                        (r["epoch"], r["val_miou"])
                        for r in last
                        if isinstance(r.get("val_miou"), float)
                    ],
                },
                {
                    "label": "accuratezza ghiaccio/acqua",
                    "colour": "#4ade9b",
                    "points": [
                        (r["epoch"], r["val_binary_accuracy"])
                        for r in last
                        if isinstance(r.get("val_binary_accuracy"), float)
                    ],
                },
            ]
        )
        prev = ""
        if len(runs) > 1:
            best_prev = max(
                (
                    r["val_miou"]
                    for run in runs[:-1]
                    for r in run
                    if isinstance(r.get("val_miou"), float)
                ),
                default=None,
            )
            if best_prev:
                prev = (
                    f" Il run precedente si era fermato a mIoU {_it(best_prev, 3)} "
                    f"con la cache ridotta (vedi D-017)."
                )
        train_note = (
            f'<p class="note">Run corrente: {len(last)} epoche registrate su '
            f"{len(runs)} run totali nel log.{prev}</p>"
        )

    # -- class distribution ------------------------------------------------
    freq = stats.get("class_frequency") or {}
    names = stats.get("class_names") or {}
    dist = (
        bar_rows(
            [
                (str(names.get(k, k)), v, f"{100 * v:.1f}%")
                for k, v in sorted(freq.items(), key=lambda kv: str(kv[0]))
            ]
        )
        if freq
        else '<p class="muted">Cache non costruita.</p>'
    )

    # -- download ----------------------------------------------------------
    dl_rows = (
        bar_rows(
            [
                (
                    k.replace("_", " "),
                    (v["have"] / v["total"] if v["total"] else 0),
                    f"{v['have']}/{v['total']}",
                )
                for k, v in dl.items()
            ]
        )
        if dl
        else '<p class="muted">Split non congelato.</p>'
    )

    # -- claims and provisional -------------------------------------------
    def rows(items, closed_marker=None):
        out = []
        for it in items:
            detail = it["detail"]
            closed = closed_marker and closed_marker in detail
            tag = (
                '<span class="tag ok">chiusa</span>'
                if closed
                else '<span class="tag prov">aperta</span>'
            )
            out.append(
                f"<tr><td>{html.escape(it['id'])} {tag}</td>"
                f"<td>{_md(it['what'])}</td><td>{_md(detail)}</td>"
                f"<td>{_md(it['action'])}</td></tr>"
            )
        return "".join(out)

    claims_html = rows(ctx["claims"], "CHIUSA")
    prov_html = rows(ctx["provisional"], "Chiusa")

    decisions_html = "".join(
        f"<tr><td>{html.escape(d['id'])}</td><td>{html.escape(d['title'])}</td>"
        f'<td class="mono">{html.escape(d["date"])}</td></tr>'
        for d in ctx["decisions"]
    )
    commits_html = "".join(
        f'<tr><td class="mono">{html.escape(c.get("hash", ""))}</td>'
        f'<td class="mono">{html.escape(c.get("date", ""))}</td>'
        f"<td>{html.escape(c.get('subject', ''))}</td></tr>"
        for c in g.get("recent", [])
    )

    route_img = route_figure(ctx["repo"])
    route_note = _route_note(ctx.get("route") or {})

    comp_note = (
        f"{_it(ratio['value'])}:1 <span class='tag prov'>limite superiore</span>"
        if ratio
        else MISSING
    )
    if size.get("compression_ratio"):
        model_comp = f"{_it(size['compression_ratio'], 2)}×"
        model_comp_note = (
            f"fp32 {_it(size['fp32_mb'], 2)} MB → INT8 {_it(size['int8_mb'], 2)} MB. "
            "Non è 4× perché grafo, scale QDQ e parametri GroupNorm restano in float32."
        )
    else:
        model_comp = MISSING
        model_comp_note = (
            "Serve <span class='mono'>make quantize</span>. Il rapporto atteso è sotto 4×: "
            "grafo, scale QDQ e GroupNorm restano in float32."
        )

    return f"""<title>IceRoute — stato del progetto</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>
<div class="wrap">
<header>
  <div>
    <h1>IceRoute</h1>
    <p class="sub">Segmentazione del ghiaccio marino a bordo e rotta ottimale
    vincolata da POLARIS. Ogni numero in questa pagina è letto da un artefatto
    su disco: quello che non è stato misurato lo dice.</p>
  </div>
  <div class="count">
    <b>{to_pitch}</b><small>giorni al pitch · {PITCH.strftime("%d/%m/%Y")}</small>
    <div style="margin-top:13px" class="{"warn" if to_freeze < 8 else ""}">
      <b style="font-size:23px">{to_freeze}</b>
      <small>al feature freeze · {FREEZE.strftime("%d/%m")}</small>
    </div>
  </div>
</header>

<h2>Numeri misurati</h2>
<div class="grid g4">{cards_html}</div>

<h2>La rotta</h2>
<div class="grid g2">
  <div class="card">{route_img}</div>
  <div class="card">{route_note}</div>
</div>

<h2>Gate</h2>
<div class="card">{gates_html}</div>

<h2>Training</h2>
<div class="grid g2">
  <div class="card">{chart}{train_note}</div>
  <div class="card">
    <h3>Distribuzione delle classi sui pixel annotati</h3>{dist}
    <div class="note" style="margin-top:11px">
      {_it(stats.get("n_patches", 0))} patch da {stats.get("n_scenes", 0)} scene ·
      margine in cache {_it(100 * stats.get("margin_fraction", 0))}%, portato al
      {_it(100 * stats.get("margin_fraction_target", 0.4))}% dal campionamento pesato
    </div>
  </div>
</div>

<h2>Dati scaricati</h2>
<div class="card">{dl_rows}
  <div class="note" style="margin-top:11px">Il set di test ufficiale AI4Arctic è
  l'unica stima non distorta: le sue etichette stanno in file
  <span class="mono">_prep_reference.nc</span> separati.</div>
</div>

<h2>Compressione del downlink</h2>
<div class="grid g2">
  <div class="card"><h3>Rapporto misurato</h3><div class="big">{comp_note}</div>
  <div class="note">Misurato sull'ice chart, che è disegnata a poligoni e si comprime
  molto meglio dell'uscita di una rete. Va rimisurato sulla maschera predetta: il
  codice si rifiuta di restituire un numero citabile finché non lo è (D-016).</div></div>
  <div class="card"><h3>Compressione del modello</h3>
  <div class="big">{model_comp}</div>
  <div class="note">{model_comp_note}</div></div>
</div>

<h2>Affermazioni pubbliche da riconciliare</h2>
<div class="scroll"><table>
<thead><tr><th>ID</th><th>Affermazione</th><th>Problema</th><th>Azione</th></tr></thead>
<tbody>{claims_html}</tbody></table></div>

<h2>Voci provvisorie</h2>
<div class="scroll"><table>
<thead><tr><th>ID</th><th>Cosa</th><th>Stato</th><th>Chiude quando</th></tr></thead>
<tbody>{prov_html}</tbody></table></div>

<h2>Stato del codice</h2>
<div class="grid g4">
  <div class="card"><h3>Commit</h3><div class="big">{g.get("n_commits", 0)}</div>
    <div class="note">branch <span class="mono">{html.escape(g.get("branch", "?"))}</span></div></div>
  <div class="card"><h3>Test</h3><div class="big">{ctx["n_tests"]}</div>
    <div class="note">{_it(ctx["test_lines"])} righe di test</div></div>
  <div class="card"><h3>Righe di codice</h3><div class="big">{_it(ctx["src_lines"])}</div>
    <div class="note">src + scripts</div></div>
  <div class="card"><h3>Decisioni registrate</h3><div class="big">{len(ctx["decisions"])}</div>
    <div class="note">DECISIONS.md</div></div>
</div>

<h2>Registro</h2>
<details><summary>Decisioni tecniche ({len(ctx["decisions"])})</summary>
  <div class="inner"><table><tbody>{decisions_html}</tbody></table></div></details>
<details style="margin-top:11px"><summary>Ultimi commit</summary>
  <div class="inner"><table><tbody>{commits_html}</tbody></table></div></details>

<footer>
  Generata da <span class="mono">make dashboard</span> il
  {today.strftime("%d/%m/%Y")}. Pagina autonoma: nessuna risorsa esterna, funziona
  con la rete spenta. Space Eagle · Wave by Vento, OGR Torino.
</footer>
</div>
"""


def _md(text: str) -> str:
    """Minimal inline markdown: bold, code, escaped."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b style='color:#eaf1fa'>\1</b>", out)
    out = re.sub(r"`(.+?)`", r"<span class='mono'>\1</span>", out)
    return out


def selected_training_config(repo: Path) -> tuple[str, dict]:
    """The config of the model that is actually deployed, not the last trained.

    ``scripts/pick_model.py`` writes the winner to ``artifacts/best_config.txt``.
    Reading a fixed path instead is how this page spent days reporting the val
    mIoU of a six-class run that had already been abandoned for the §7.4
    fallback — every training run appends to its own CSV, so "the last log" and
    "the deployed model" are different things the moment more than one config
    exists.
    """
    picked = repo / "artifacts/best_config.txt"
    name = picked.read_text().strip() if picked.exists() else "configs/train3.yaml"
    path = repo / name
    return name, (yaml.safe_load(path.read_text()) if path.exists() else {})


def route_story(repo: Path, demo_index: dict) -> dict:
    """The numbers for the scene the route figure is actually showing.

    The subject is *read* from ``artifacts/figures/route_lead.json``, which the
    figure generator writes, not re-derived here. Two places choosing the lead
    scene by reimplementing the same rule is two places that can disagree, and
    the first version of this function did: it captioned an IA route under a
    PC6 picture, because it ranked over every vessel class while the figure was
    generated for one. A caption naming a different subject than the image is
    worse than no caption — it is a wrong claim with a picture next to it.
    """
    lead = _json(repo / "artifacts/figures/route_lead.json")
    if not lead:
        return {}
    scene = next(
        (s for s in demo_index.get("scenes") or [] if s["scene_id"] == lead.get("scene_id")), None
    )
    entry = ((scene or {}).get("classes") or {}).get(lead.get("ice_class")) or {}
    if not entry.get("route_exists"):
        return {}
    cmp_ = entry.get("comparison") or {}
    base, sep = cmp_.get("baseline") or {}, entry.get("separation") or {}
    return {
        "scene_id": scene["scene_id"],
        "region": scene.get("region", ""),
        "ice_class": lead["ice_class"],
        "blocked_by_ice": bool(base.get("blocked_by_ice")),
        "n_no_go": base.get("n_no_go_cells"),
        "hours": (cmp_.get("iceroute") or {}).get("time_hours"),
        "n_waypoints": entry.get("n_waypoints"),
        "separation_pct": sep.get("pct_diagonal"),
        "separation_km": sep.get("max_km"),
    }


def collect(repo: Path, today: date) -> dict:
    decisions_text = (repo / "DECISIONS.md").read_text() if (repo / "DECISIONS.md").exists() else ""
    metrics = _json(repo / "artifacts/metrics.json")
    stats = _json(repo / "artifacts/data_stats.json")
    summary = _json(repo / "artifacts/train_summary.json")
    routes = _json(repo / "data/demo/routes_index.json")

    cfg_name, train_cfg = selected_training_config(repo)
    log_csv = (train_cfg.get("output") or {}).get("log_csv") or "artifacts/logs/train_log.csv"
    runs = read_runs(repo / log_csv)

    best = max(
        (r["val_miou"] for r in (runs[-1] if runs else []) if isinstance(r.get("val_miou"), float)),
        default=None,
    )

    # The headline number is the held-out one. Validation picked the checkpoint,
    # so it flatters it by an unknown amount; the official test set did not.
    evaluation = metrics.get("evaluation") or {}
    test_miou = ((evaluation.get("scenes") or {}).get("aggregate") or {}).get("miou")
    n_classes = (train_cfg.get("model") or {}).get("n_classes")

    demo_index = _json(repo / "data/demo/index.json")
    demo = {
        "n_scenes": len([s for s in demo_index.get("scenes", []) if not s.get("worst_case")]),
        "has_worst_case": any(s.get("worst_case") for s in demo_index.get("scenes", [])),
        "n_figures": len(list((repo / "artifacts/figures").glob("*.png"))),
        "has_video": any((repo / "artifacts/video").glob("*.mp4")),
    }

    route = route_story(repo, demo_index)

    comp = metrics.get("compression") or {}
    ratio = None
    if comp.get("ratio_median"):
        ratio = {"value": comp["ratio_median"], "provisional": comp.get("provisional", True)}

    ctx = {
        "today": today,
        "repo": repo,
        "metrics": metrics,
        "data_stats": stats,
        "routes": routes,
        "runs": runs,
        "best_miou": best,
        "test_miou": test_miou,
        "n_classes": n_classes,
        "selected_config": cfg_name,
        "eval_split": evaluation.get("split"),
        "demo": demo,
        "route": route,
        "compression": ratio,
        "download": download_status(repo),
        "git": git_info(repo),
        "n_tests": count_tests(repo),
        "test_lines": count_lines(repo, "tests"),
        "src_lines": count_lines(repo, "src", "scripts"),
        "claims": parse_table(decisions_text, "A"),
        "provisional": parse_table(decisions_text, "P"),
        "decisions": parse_decisions(decisions_text),
        # No literal fallback: a hardcoded parameter count is a number that
        # keeps being displayed after the architecture changes under it.
        "n_params": (
            metrics.get("export", {}).get("fp32", {}).get("n_parameters")
            or summary.get("n_parameters")
        ),
    }
    ctx["gates"] = gates(ctx)
    return ctx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genera index.html con lo stato del progetto.")
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("index.html"))
    ap.add_argument("--today", default=None, help="ISO date, per test riproducibili")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    today = date.fromisoformat(args.today) if args.today else date.today()

    ctx = collect(args.repo, today)
    args.out.write_text(build_page(ctx))

    done = sum(1 for g in ctx["gates"] if g.done)
    log.info(
        "%s scritta — %d/%d gate verdi, %d commit, %d test, %d giorni al pitch",
        args.out,
        done,
        len(ctx["gates"]),
        ctx["git"].get("n_commits", 0),
        ctx["n_tests"],
        (PITCH - today).days,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
