"""What goes down the link, and what the numbers become under an assumption.

Two panels of the UI live here, together because they are the same idea from
opposite ends: one shows a quantity we **measured**, the other a quantity we
**cannot** measure and therefore hand to the viewer as a dial.

The payload
-----------
The satellite sends a mask, a route and a few bytes of metadata — not the
scene. That is the product. :func:`payload` reports the byte counts, taking the
route from :func:`compression.downlink_route` so the panel and the compression
figure in the deck are computing the same thing from the same function.

Time and fuel
-------------
These have no source. ``open_water_speed_kn: 12.0`` is the one unsourced value
in ``configs/routing.yaml`` (P-05), fuel consumption is not in the project at
all (P-09), and SPEC §10.1 forbids putting a currency on either. Measurement
last night (D-026) put numbers on how bad it is: transit hours move 95% across
10–19.5 kn, and on four of five demo scenes the direct route is forbidden, so
there is no reference voyage to be faster than.

The answer is not to pick a plausible number and print it. It is to expose the
assumption as an input the viewer sets, label every figure downstream as theirs
rather than ours, and — the part that makes this honest rather than merely
deniable — **refuse to rescale when the rescale would be wrong**.

Which it often would be. Transit time is not simply distance over speed. The
POLARIS speed model has two bands: at RIO ≥ 0 the vessel runs at open-water
speed, which does scale with the dial; in the elevated band it is capped at a
**fixed** Table 1.2 limit — 3.0 kn for PC6 and PC7 — which does not scale with
anything. A route crossing that band has a component immune to the slider, and
scaling the total by v_nominal/v_new would quietly invent time.

So :func:`rescale_transit` checks that the route stayed at RIO ≥ 0 for its
whole length, and only then rescales — exactly, not approximately. Otherwise it
returns nothing and says why. On the current demo scenes the check passes (A\\*
routes them entirely through normal operation: the cost grids report 0.0% of
cells at reduced speed), but it is checked per route rather than assumed,
because the day a scene forces the vessel through the elevated band is the day
this silently starts lying.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

# Table 1.2 of IMO MSC.1/Circ.1519: the speed cap in the elevated risk band.
# Below PC7 that band is already no-go, so there is no cap to apply.
ELEVATED_CAP_KN = {"PC1": 11.0, "PC2": 8.0, "PC3": 5.0, "PC4": 5.0,
                   "PC5": 5.0, "PC6": 3.0, "PC7": 3.0}

HOURS_PER_DAY = 24.0


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


@dataclass
class Payload:
    """The bytes that actually leave the satellite, and what they replace."""

    route_bytes: int
    n_waypoints: int
    route_text: str
    mask_bytes: int | None = None
    metadata_bytes: int | None = None
    sar_bytes: int | None = None
    ratio: float | None = None
    measured: bool = False
    note: str = ""

    @property
    def total_bytes(self) -> int | None:
        if self.mask_bytes is None:
            return None
        return self.mask_bytes + self.route_bytes + (self.metadata_bytes or 0)


def payload(entry: dict, routes_dir: Path, metrics: dict, scene_id: str) -> Payload | None:
    """What this scene's downlink weighs.

    The route is read from disk and stripped to the transmitted feature, so the
    figure is true for any scene. The mask and SAR sizes come from the
    compression measurement, which only covers the scenes ``make report`` ran
    over — where it is missing, the panel shows the route alone and says the
    rest is unmeasured rather than filling it in.
    """
    from iceroute.pitch.compression import downlink_route

    gj_path = routes_dir / (entry.get("geojson") or "")
    if not gj_path.is_file():
        return None
    sent = downlink_route(json.loads(gj_path.read_text()))
    if not sent:
        return None

    # The exact string that is counted is the exact string that is shown.
    # Pretty-printing it for display would put sixty lines on screen under a
    # label saying "432 bytes" — and the whole point of the panel is that the
    # payload is small, which a wall of indented JSON argues against.
    text = json.dumps(sent, separators=(",", ":"))
    feat = sent["features"][0]
    out = Payload(
        route_bytes=len(text.encode()),
        n_waypoints=len(feat.get("geometry", {}).get("coordinates") or []),
        route_text=textwrap.fill(text, width=78),
    )

    per = (metrics.get("compression") or {}).get("per_scene") or []
    row = next((r for r in per if r.get("scene_id") == scene_id), None)
    if not row:
        out.note = ("Maschera e SAR non misurati per questa scena: "
                    "`make report` copre solo le prime scene dell'indice.")
        return out

    den, num = row.get("denominator") or {}, row.get("numerator") or {}
    out.mask_bytes = den.get("mask_png")
    out.metadata_bytes = den.get("metadata")
    out.sar_bytes = num.get("sar_channels_png")
    out.measured = True
    if out.sar_bytes and out.total_bytes:
        out.ratio = out.sar_bytes / out.total_bytes
    return out


# ---------------------------------------------------------------------------
# What the payload costs on the link
# ---------------------------------------------------------------------------


@dataclass
class LinkTime:
    """How long the payload takes to reach the bridge, and on what terminal."""

    seconds: float | None = None
    terminal: str = ""
    kbps: int | None = None
    sbd_messages: int | None = None
    reason: str = ""


def link_time(payload_bytes: int | None, route_bytes: int | None,
              assumptions: dict) -> LinkTime:
    """Seconds on the *smallest* terminal a ship is likely to carry.

    This exists because of A-10. Stating that the downlink is 148 KB invites
    exactly one objection — "can the ship's link carry that?" — and the honest
    answer is thirteen seconds, which is better than the question expects. A
    panel that gives the size and not the cost leaves the objection standing.

    The rates are third-party specifications, not our measurements, so they live
    in ``configs/assumptions.yaml`` with a source and this function refuses
    without them. The seconds are **derived from rated peak**: real throughput is
    lower, so the figure is an order of magnitude and the UI says so.

    The slowest terminal is chosen deliberately. Quoting FleetBroadband's three
    seconds would be quoting the best case at someone who may not have it.
    """
    link = (assumptions or {}).get("link") or {}
    rates = [(v, k) for k, v in link.items()
             if k.endswith("_kbps") and isinstance(v, int | float) and v > 0]
    if not rates or not payload_bytes:
        return LinkTime(reason="Velocità del collegamento non in configs/assumptions.yaml.")

    kbps, key = min(rates)
    msgs = None
    per_msg = link.get("iridium_sbd_message_bytes")
    if route_bytes and isinstance(per_msg, int) and per_msg > 0:
        msgs = -(-route_bytes // per_msg)
    return LinkTime(seconds=payload_bytes * 8 / (kbps * 1000),
                    terminal=key[:-5].replace("_", " ").title(),
                    kbps=int(kbps), sbd_messages=msgs)


# ---------------------------------------------------------------------------
# Time and fuel, under an assumption the viewer sets
# ---------------------------------------------------------------------------


@dataclass
class Transit:
    hours: float | None = None
    exact: bool = False
    reason: str = ""


def rescale_transit(
    hours: float | None,
    min_rio: float | None,
    *,
    nominal_kn: float,
    chosen_kn: float,
    ice_class: str,
) -> Transit:
    """The transit time at a different open-water speed, or nothing.

    Exact when the route never left normal operation, because then every
    segment ran at open-water speed and the total is strictly proportional to
    1/v. Refused otherwise: the elevated band is capped at a fixed number of
    knots that the dial does not move, so a proportional rescale would invent
    time on the part of the voyage the regulation, not the vessel, controls.
    """
    if hours is None or chosen_kn <= 0 or nominal_kn <= 0:
        return Transit(reason="Nessun tempo di transito da riscalare.")
    if min_rio is None:
        return Transit(reason="RIO minimo sulla rotta non noto: non si può "
                              "sapere se la rotta resta in operazione normale.")
    if min_rio < 0:
        cap = ELEVATED_CAP_KN.get(ice_class)
        detail = (f"limitata a {cap:.0f} kn dalla Tabella 1.2 IMO"
                  if cap else "già oltre il limite per questa classe")
        return Transit(reason=(
            f"La rotta entra in rischio elevato (RIO minimo {min_rio:+.0f}), dove la "
            f"velocità è {detail} e non dipende da questo cursore. Riscalare il totale "
            f"inventerebbe del tempo: il numero non viene mostrato."))
    return Transit(hours=hours * nominal_kn / chosen_kn, exact=True)


def fuel_tonnes(hours: float | None, tonnes_per_day: float | None) -> float | None:
    """Fuel burnt over a transit, if someone supplied a consumption.

    Nothing clever: hours times a rate. It is here rather than inline to keep
    the ``None`` in one place — no consumption figure means no fuel figure, and
    never a zero, which on a dashboard reads as a measurement.
    """
    if hours is None or not tonnes_per_day:
        return None
    return hours / HOURS_PER_DAY * tonnes_per_day
