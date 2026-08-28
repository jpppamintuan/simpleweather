import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ecmwf_client import (
    AVAILABLE_THRESHOLDS_MM,
    THRESHOLD_COLORS,
    MODEL_LABELS,
    PH_TZ,
    fetch_forecast_table,
)

st.set_page_config(page_title="Rainfall Exceedance Forecast", page_icon="🌧️", layout="wide")

# Load the actual webfont Streamlit's own UI uses (Source Sans) so the
# tables' font-family declaration has something real to point to, instead
# of silently falling back to each browser's own default font.
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Source+Sans+3:wght@400;600;700;800&display=swap\" "
    'rel="stylesheet">',
    unsafe_allow_html=True,
)

st.title("🌧️ Rainfall Exceedance Forecast")
st.caption("ECMWF ENS open data — probability of 24h rainfall exceeding each threshold")

LOCATIONS = {
    "Mandaluyong City, Metro Manila": (14.576975, 121.052521),
    "Guiguinto, Bulacan": (14.842279, 120.859681),
    "Makati City, Metro Manila": (14.555539, 121.002918),
    "Bambang, Nueva Vizcaya": (16.389440, 121.106919),
    "Bacoor, Cavite": (14.454261, 120.941266),
}

CACHE_TTL_SECONDS = 3 * 60 * 60  # ENS updates twice a day (00/12 UTC)

# Shared across ALL browser sessions on this running server process (unlike
# st.session_state, which is per-session). This is what makes repeat
# requests from *different* users/tabs instant instead of re-fetching.
_GLOBAL_CACHE: dict = {}

# Persisted to disk so cached forecasts survive an app sleep/restart, not
# just this process's uptime -- Streamlit Community Cloud puts free-tier
# apps to sleep after inactivity, which wipes any purely in-memory cache.
# Caveat: this is a best-effort convenience, not guaranteed persistence --
# a full redeploy (new git push) or the app moving to a different host
# would still reset it, since it's a plain file next to the script, not an
# external database.
_CACHE_FILE = Path(__file__).parent / "forecast_cache.json"


def _serialize_result(result: dict) -> dict:
    def dt(x):
        return x.isoformat() if x else None

    return {
        "run_time": dt(result["run_time"]),
        "grid_lat": result["grid_lat"],
        "grid_lon": result["grid_lon"],
        "windows": [
            {"label": w["label"], "end_step": w["end_step"],
             "start_utc": dt(w["start_utc"]), "end_utc": dt(w["end_utc"])}
            for w in result["windows"]
        ],
        "data": {str(k): v for k, v in result["data"].items()},
        "available_since": dt(result["available_since"]),
        "next_expected": dt(result["next_expected"]),
        "downloaded_bytes": result.get("downloaded_bytes"),
        "fetch_mode": result.get("fetch_mode"),
        "model": result.get("model"),
    }


def _deserialize_result(d: dict) -> dict:
    def pdt(s):
        return datetime.fromisoformat(s) if s else None

    return {
        "run_time": pdt(d["run_time"]),
        "grid_lat": d["grid_lat"],
        "grid_lon": d["grid_lon"],
        "windows": [
            {"label": w["label"], "end_step": w["end_step"],
             "start_utc": pdt(w["start_utc"]), "end_utc": pdt(w["end_utc"])}
            for w in d["windows"]
        ],
        "data": {int(k): v for k, v in d["data"].items()},
        "available_since": pdt(d["available_since"]),
        "next_expected": pdt(d["next_expected"]),
        "downloaded_bytes": d.get("downloaded_bytes"),
        "fetch_mode": d.get("fetch_mode"),
        "model": d.get("model"),
    }


def _cache_key_to_str(key: tuple) -> str:
    lat, lon, lead_days, model = key
    return f"{lat}|{lon}|{lead_days}|{model}"


def _cache_key_from_str(s: str) -> tuple:
    lat, lon, lead_days, model = s.split("|")
    return (float(lat), float(lon), int(lead_days), model)


def _load_disk_cache() -> None:
    if not _CACHE_FILE.exists():
        return
    try:
        raw = json.loads(_CACHE_FILE.read_text())
        now = time.time()
        for key_str, entry in raw.items():
            if now - entry["ts"] >= CACHE_TTL_SECONDS:
                continue  # skip stale entries rather than loading dead data
            _GLOBAL_CACHE[_cache_key_from_str(key_str)] = {
                "ts": entry["ts"],
                "result": _deserialize_result(entry["result"]),
            }
    except Exception:
        pass  # corrupt or unreadable cache file -- just start fresh


def _save_disk_cache() -> None:
    try:
        raw = {
            _cache_key_to_str(key): {"ts": entry["ts"], "result": _serialize_result(entry["result"])}
            for key, entry in _GLOBAL_CACHE.items()
        }
        _CACHE_FILE.write_text(json.dumps(raw))
    except Exception:
        pass  # best-effort -- a failed write shouldn't crash the app


_load_disk_cache()  # populate the in-memory cache from disk once, at startup


def _cache_lookup(key: tuple):
    now = time.time()
    session_cache = st.session_state.setdefault("_forecast_cache", {})
    cached = session_cache.get(key)
    if cached and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["result"]
    global_entry = _GLOBAL_CACHE.get(key)
    if global_entry and (now - global_entry["ts"] < CACHE_TTL_SECONDS):
        session_cache[key] = global_entry  # promote into this session too
        return global_entry["result"]
    return None


def _cache_store(key: tuple, result: dict) -> None:
    entry = {"ts": time.time(), "result": result}
    st.session_state.setdefault("_forecast_cache", {})[key] = entry
    _GLOBAL_CACHE[key] = entry


def get_forecast_with_progress(lat: float, lon: float, lead_days: int, load_aifs: bool):
    """Fetches ENS, and AIFS too if load_aifs is set. Three optimizations
    over a naive "just fetch both" approach:
      1. Each model is cached independently (by lat/lon/lead_days/model),
         so toggling "Load AIFS" after ENS is already cached only fetches
         the new AIFS data, not both again.
      2. When both models genuinely need fetching, they run concurrently
         in separate threads -- wall time is ~max(ens_time, aifs_time)
         instead of the sum, since these are I/O-bound network requests.
      3. Reuses the existing single-request-per-model optimization (5
         thresholds combined into 1 file) underneath, so worst case is 2
         requests total (1 per model), not 10.

    Returns {"ifs": (result, was_cached), "aifs-ens": (result_or_None, was_cached)}
    -- the aifs-ens entry is only present if load_aifs was True.
    """
    models = ["ifs"] + (["aifs-ens"] if load_aifs else [])
    keys = {m: (lat, lon, lead_days, m) for m in models}
    cached_results = {m: _cache_lookup(keys[m]) for m in models}
    to_fetch = [m for m in models if cached_results[m] is None]

    if not to_fetch:
        return {m: (cached_results[m], True) for m in models}

    progress = st.progress(0, text="Connecting to ECMWF Open Data...")
    progress_lock = threading.Lock()
    progress_state = {m: (0.0, "Waiting...") for m in to_fetch}

    fetched = {}
    errors = {}

    def worker(m: str):
        # Runs on a background thread. Must NOT call any Streamlit UI
        # function (st.*, or methods on an element like `progress`) --
        # those require a script-run context that background threads
        # don't have by default, which silently breaks every fetch (this
        # was the actual bug: even the single-model/ENS-only path went
        # through this same threaded code, so it broke everything, not
        # just AIFS). This callback only writes to a plain dict; the only
        # place that touches the real `progress` element is the polling
        # loop below, which runs on the main thread.
        def on_progress(frac: float, msg: str):
            with progress_lock:
                progress_state[m] = (frac, msg)

        try:
            fetched[m] = fetch_forecast_table(
                lat, lon, max_lead_days=lead_days, progress_callback=on_progress, model=m
            )
        except Exception as e:
            errors[m] = e

    threads = [threading.Thread(target=worker, args=(m,), daemon=True) for m in to_fetch]
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        with progress_lock:
            fracs = [v[0] for v in progress_state.values()]
            combined_frac = sum(fracs) / len(fracs) if fracs else 0.0
            msg = "  |  ".join(f"{MODEL_LABELS[m]}: {progress_state[m][1]}" for m in to_fetch)
        progress.progress(min(max(int(combined_frac * 100), 0), 100), text=msg)
        time.sleep(0.1)

    for t in threads:
        t.join()

    progress.progress(100, text="Done!")
    time.sleep(0.2)
    progress.empty()

    if "ifs" in to_fetch and "ifs" in errors:
        raise errors["ifs"]  # ENS is required -- surface the failure

    out = {}
    for m in models:
        if cached_results[m] is not None:
            out[m] = (cached_results[m], True)
        elif m in fetched:
            _cache_store(keys[m], fetched[m])
            out[m] = (fetched[m], False)
        else:
            out[m] = (None, False)  # AIFS failed -- app degrades to ENS-only, not a crash

    if fetched:
        _save_disk_cache()
    return out


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _relative_luminance(r: int, g: int, b: int) -> float:
    """Perceived brightness (0=black, 1=white). Used to decide whether a
    color needs light or dark text on top of it."""
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


# The tables are permanently styled as light mode, regardless of the app's
# actual theme. Previously these used "inherit"/"transparent" to follow
# Streamlit's live theme -- but since data-cell backgrounds can't do the
# same trick (they're deliberately colored, not just "whatever's behind
# them"), that led to a half-and-half look after a theme toggle: reactive
# text/borders next to frozen cell colors. Fixing everything to explicit
# light-mode values sidesteps that entirely.
BASE_TEXT = "#111111"
CARD_BG = "#ffffff"
BORDER = "rgba(0,0,0,0.12)"


# Streamlit's default body font is Source Sans (Adobe's open-source
# typeface). Just naming it in font-family isn't enough -- if the font
# file isn't actually loaded, the browser silently falls back to its own
# default (SF Pro on iOS, Arial on Windows), which was happening here. The
# webfont is loaded explicitly via Google Fonts in the script below.
FONT_STACK = "'Source Sans 3', 'Source Sans Pro', -apple-system, sans-serif"


def _neutral_base_rgb() -> tuple[int, int, int]:
    """The <5% tier's color: always white now, in both light and dark mode.
    This used to switch to black in dark mode, but that required Python to
    know the current theme -- and since toggling the theme doesn't trigger
    a script rerun, that color would freeze at whatever it was when the
    tables were last rendered until something else caused a rerun (which,
    with the refresh button removed, might be never). Every tier now uses
    the exact same white-tint rule regardless of theme, so there's nothing
    left that can go stale."""
    return (255, 255, 255)


def _tint_toward_white_rgb(r: int, g: int, b: int, alpha: float) -> tuple[int, int, int]:
    """Every non-neutral, non-base tier blends toward WHITE, regardless of
    theme -- keeps every color's tones identical between light and dark
    mode. (This started as an orange-specific fix -- darkening pure orange
    toward black read as "brown", with no separate name for "dark orange"
    the way there is for dark red/blue/purple -- but applying it to every
    color is simpler and gives one consistent rule instead of a
    per-threshold special case.)"""
    return (
        round(r * alpha + 255 * (1 - alpha)),
        round(g * alpha + 255 * (1 - alpha)),
        round(b * alpha + 255 * (1 - alpha)),
    )


# Discrete color tiers instead of a continuous gradient -- easier to read
# at a glance, like a legend, and sidesteps needing "good at every possible
# opacity" colors. Boundaries match ECMWF's own product visualization
# convention:
#   <5%: neutral (white/black, the only theme-dependent tier)
#   5-35% / 35-65% / 65-95%: tone3 / tone2 / tone1 (lightest -> strongest)
#   >=95%: pure base color
_TIER_FILL_ALPHA = {"tone3": 0.05, "tone2": 0.35, "tone1": 0.65}


def _tier_for_value(val: float) -> str:
    v = round(val)  # use the *displayed* value so tier and label never disagree
    if v < 5:
        return "neutral"
    elif v < 35:
        return "tone3"
    elif v < 65:
        return "tone2"
    elif v < 95:
        return "tone1"
    return "base"


def _cell_rgb_for_value(threshold_mm: int, val: float) -> tuple[int, int, int]:
    tier = _tier_for_value(val)
    r, g, b = _hex_to_rgb(THRESHOLD_COLORS[threshold_mm])

    if tier == "neutral":
        return _neutral_base_rgb()
    if tier == "base":
        return r, g, b
    return _tint_toward_white_rgb(r, g, b, _TIER_FILL_ALPHA[tier])


def _fmt_ph(dt: datetime) -> str:
    return dt.astimezone(PH_TZ).strftime("%a, %d %b %I%p")


def _fmt_window(w: dict) -> str:
    return f"{_fmt_ph(w['start_utc'])}<br>to<br>{_fmt_ph(w['end_utc'])}"


def _cell_text_color(blended_luminance: float) -> str:
    """Pick black or white text based on the *actual rendered* blended
    color's brightness -- correct at every probability level now that the
    background is a solid pre-blended color rather than true transparency."""
    return "#111111" if blended_luminance > 0.5 else "#ffffff"


def _estimate_col_width_px(windows: list[dict]) -> int:
    """table-layout:fixed makes every column equal width, but needs ONE
    width value to apply to all of them -- this computes that value from
    the actual longest line of text that will appear in any column
    (usually a date header line, e.g. "Sun, 06 Sep 08AM"), rather than a
    guessed flat number. That guess (90px) was too narrow for the header
    text specifically, which is what caused the overlap on mobile.

    Note: header text is multi-line (stacked via <br>), so it's the
    longest SINGLE LINE within that stack that matters, not the full
    concatenated string length.
    """
    candidates = ["Threshold", "≥100 mm", "100%"]  # widest label/value text
    if windows:
        sample_header = _fmt_window(windows[0])
        candidates.extend(sample_header.split("<br>"))

    longest_line = max(len(line) for line in candidates)

    char_width_px = 7.5  # generous estimate for ~12px Source Sans (proportional font)
    padding_px = 24  # 12px left + 12px right cell padding
    return int(longest_line * char_width_px) + padding_px


def render_ribbon_chart_html(result: dict) -> str:
    """Nested-area 'ribbon' chart: all 5 thresholds as overlapping filled
    areas from 0 up to that day's probability, drawn in order from 1mm
    (largest area, drawn first/underneath) to 100mm (smallest area, drawn
    last/on top). This works cleanly here specifically because the
    thresholds are monotonically nested -- P(>=1mm) is always >= P(>=5mm)
    >= ... >= P(>=100mm) on any given day -- so the bands never cross and
    naturally read as "shrinking severity, shrinking likelihood."

    Rendered via components.v1.html (a real iframe), not st.markdown --
    unlike the HTML tables elsewhere in this file, this needs actual
    <script> execution (Chart.js), which st.markdown's unsafe_allow_html
    silently strips. The iframe has its own isolated document, so the
    color fills' transparency composites against a background we set
    explicitly (white) rather than whatever Streamlit's live theme
    happens to be -- sidesteps the whole light/dark inconsistency problem
    the HTML tables had, without needing the solid-tier color system.
    """
    windows = result["windows"]
    data = result["data"]
    if not windows:
        return ""

    day_labels = [w["start_utc"].astimezone(PH_TZ).strftime("%a %d") for w in windows]

    datasets = []
    for threshold in AVAILABLE_THRESHOLDS_MM:
        values = [data[threshold].get(w["label"]) or 0 for w in windows]
        datasets.append({"label": f"{threshold}mm", "values": values, "color": THRESHOLD_COLORS[threshold]})

    labels_json = json.dumps(day_labels)
    datasets_json = json.dumps(datasets)

def _tooltip_row_html(threshold: int, val) -> str:
    """One row of the hover/click tooltip balloon, colored identically to
    the corresponding full-table cell (same _cell_rgb_for_value /
    _cell_text_color functions), so the balloon and the table never show
    conflicting colors for the same value."""
    if val is None:
        bg, text_color = "#ffffff", BASE_TEXT
        text = "No data"
    else:
        r, g, b = _cell_rgb_for_value(threshold, val)
        bg = f"rgb({r},{g},{b})"
        text_color = _cell_text_color(_relative_luminance(r, g, b))
        text = f"{val:.0f}% chance of rain (&ge;{threshold} mm)"
    return (
        f"<div style='padding:6px 10px;text-align:center;background-color:{bg};"
        f"color:{text_color};font-weight:600;font-size:12px;"
        f"border-bottom:1px solid rgba(0,0,0,0.12);'>{text}</div>"
    )


def _tooltip_html_for_window(w: dict, data: dict) -> str:
    """The full 6-row day-detail content: a header (the date range) plus
    one colored row per threshold, single-column, matching the requested
    layout exactly. Styling is minimal here (no shadow/radius/min-width)
    since this now renders flush inside dayDetailPanel's own bordered
    container, not as a floating element in its own right."""
    header = (
        f"<div style='padding:6px 10px;text-align:center;font-weight:700;font-size:12px;"
        f"color:{BASE_TEXT};background-color:#ffffff;"
        f"border-bottom:1px solid rgba(0,0,0,0.12);'>{_fmt_ph(w['start_utc'])} to {_fmt_ph(w['end_utc'])}</div>"
    )
    rows = "".join(_tooltip_row_html(t, data[t].get(w["label"])) for t in AVAILABLE_THRESHOLDS_MM)
    return f"<div style='font-family:{FONT_STACK};'>{header}{rows}</div>"


def render_ribbon_chart_html(result: dict) -> str:
    """Nested-area 'ribbon' chart: all 5 thresholds as bands between
    adjacent threshold curves (1mm's band fills to the 5mm line, 5mm's to
    the 20mm line, etc., down to 100mm filling to zero) -- well-defined
    regardless of paint order, since it works because the thresholds are
    monotonically nested: P(>=1mm) is always >= P(>=5mm) >= ... >=
    P(>=100mm) on any given day, so adjacent bands never cross.

    Colors use each threshold's tone2 (35%) shade for the FILL, but the
    BORDER stroke uses the pure base hex -- a same-color-as-fill border
    (the earlier version) let adjacent pastel bands visually bleed into
    each other with nothing but a thin same-toned line between them,
    which is most of why the graph's tone2 read as more washed-out /
    "transparent-looking" than the same tone2 RGB values look in the
    table (verified byte-identical between the two elsewhere) -- there,
    every cell sits inside a neutral grey grid border with bold dark text
    printed directly on it, both of which make the fill read as more
    saturated by contrast. A distinct, more saturated stroke around each
    band restores some of that "grounded" look without changing the fill
    color itself.

    Instead of a floating hover tooltip (removed -- on a phone screen it
    covered most of the chart), the same per-day detail is now a
    persistent panel below the graph that updates on hover/tap and
    defaults to showing day 1 as soon as the chart loads.

    Rendered via components.v1.html (a real iframe), not st.markdown --
    unlike the HTML tables elsewhere in this file, this needs actual
    <script> execution (Chart.js), which st.markdown's unsafe_allow_html
    silently strips. The iframe has its own isolated document with an
    explicit white background, so it isn't affected by Streamlit's theme,
    and needs its own font loaded separately from the main page's.
    """
    windows = result["windows"]
    data = result["data"]
    if not windows:
        return ""

    day_labels = [w["start_utc"].astimezone(PH_TZ).strftime("%a %d") for w in windows]

    datasets = []
    for threshold in AVAILABLE_THRESHOLDS_MM:
        values = [data[threshold].get(w["label"]) or 0 for w in windows]
        base_hex = THRESHOLD_COLORS[threshold]
        r, g, b = _tint_toward_white_rgb(*_hex_to_rgb(base_hex), _TIER_FILL_ALPHA["tone2"])
        datasets.append({
            "label": f"{threshold}mm",
            "values": values,
            "fillColor": f"rgb({r},{g},{b})",
            "borderColor": base_hex,
        })

    day_detail_html_by_day = [_tooltip_html_for_window(w, data) for w in windows]

    labels_json = json.dumps(day_labels)
    datasets_json = json.dumps(datasets)
    day_details_json = json.dumps(day_detail_html_by_day)

    return f"""
    <link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>body {{ font-family: {FONT_STACK}; margin: 0; }}</style>
    <div style="background-color:#ffffff;padding:8px;">
      <div style="position:relative;width:100%;height:300px;">
        <canvas id="ribbonChart"></canvas>
      </div>
      <div id="dayDetailPanel" style="margin-top:10px;border-radius:6px;overflow:hidden;
        border:1px solid rgba(0,0,0,0.12);"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    (function() {{
      const labels = {labels_json};
      const rawDatasets = {datasets_json};
      const dayDetailHtmlByDay = {day_details_json};
      const panelEl = document.getElementById("dayDetailPanel");

      const datasets = rawDatasets.map((d, i) => ({{
        label: d.label,
        data: d.values,
        borderColor: d.borderColor,
        backgroundColor: d.fillColor,
        pointBackgroundColor: d.borderColor,
        pointBorderColor: d.borderColor,
        // Each band fills the area between THIS curve and the NEXT one
        // (1mm fills to the 5mm line, 5mm fills to the 20mm line, etc.),
        // not "down to zero" for every dataset -- well-defined regardless
        // of paint order, since the data is nested and descending.
        fill: (i === rawDatasets.length - 1) ? "origin" : (i + 1),
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.3,
      }}));

      function showDayDetail(context) {{
        const {{ tooltip }} = context;
        if (tooltip.opacity === 0 || tooltip.dataPoints.length === 0) {{
          return;  // leave the panel showing whatever day it last showed
        }}
        const dayIndex = tooltip.dataPoints[0].dataIndex;
        panelEl.innerHTML = dayDetailHtmlByDay[dayIndex];
      }}

      const chart = new Chart(document.getElementById("ribbonChart"), {{
        type: "line",
        data: {{ labels: labels, datasets: datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{
            // Chart.js v4's built-in "colors" plugin auto-assigns its own
            // palette and can override explicit per-dataset colors in
            // some configs -- disabling it guarantees ours are used.
            colors: {{ enabled: false, forceOverride: false }},
            legend: {{
              display: true,
              position: "bottom",
              labels: {{ boxWidth: 12, font: {{ size: 11 }}, color: "#333333" }},
            }},
            // No floating tooltip -- the same content goes into the
            // persistent panel below the graph instead (see
            // showDayDetail), via this same hover/tap detection.
            tooltip: {{ enabled: false, external: showDayDetail }},
          }},
          scales: {{
            y: {{
              min: 0, max: 100,
              ticks: {{ callback: (v) => v + "%", color: "#666666" }},
              grid: {{ color: "#e5e5e5" }},
            }},
            x: {{
              ticks: {{ color: "#666666", font: {{ size: 10 }} }},
              grid: {{ display: false }},
            }},
          }},
        }},
      }});

      // Default to day 1 as soon as the chart is ready, before any hover.
      panelEl.innerHTML = dayDetailHtmlByDay[0];
    }})();
    </script>
    """


def render_table_html(result: dict) -> str:
    windows = result["windows"]
    data = result["data"]

    header_cells = "".join(
        f"<th style='padding:8px 12px;font-size:12px;white-space:nowrap;"
        f"color:{BASE_TEXT};border-bottom:2px solid {BORDER};'>{_fmt_window(w)}</th>"
        for w in windows
    )

    rows_html = ""
    for threshold in AVAILABLE_THRESHOLDS_MM:
        color_hex = THRESHOLD_COLORS[threshold]
        row_cells = ""
        for w in windows:
            val = data[threshold].get(w["label"])
            if val is None:
                row_cells += (
                    f"<td style='padding:8px 12px;text-align:center;color:{BASE_TEXT};"
                    f"border-bottom:1px solid {BORDER};'>—</td>"
                )
                continue
            br, bg_g, bb = _cell_rgb_for_value(threshold, val)
            bg = f"rgb({br},{bg_g},{bb})"
            text_color = _cell_text_color(_relative_luminance(br, bg_g, bb))
            row_cells += (
                f"<td style='padding:8px 12px;text-align:center;"
                f"background-color:{bg};color:{text_color};font-weight:600;"
                f"border-bottom:1px solid {BORDER};'>{val:.0f}%</td>"
            )
        rows_html += (
            f"<tr><td style='padding:8px 12px;font-weight:700;white-space:nowrap;"
            f"color:{BASE_TEXT};background-color:{color_hex}33;"
            f"border-bottom:1px solid {BORDER};'>≥{threshold} mm</td>{row_cells}</tr>"
        )

    # min-width scales with the number of columns AND the actual text
    # width each needs, so on mobile the wrapper's overflow-x:auto kicks
    # in for horizontal scroll instead of cramming/overlapping text, while
    # table-layout:fixed still keeps every column exactly equal.
    col_width_px = _estimate_col_width_px(windows)
    min_width_px = (len(windows) + 1) * col_width_px  # +1 for the "Threshold" label column


    # NOTE: no leading whitespace on any line below -- st.markdown treats
    # 4+ leading spaces as a Markdown code block, which silently breaks
    # raw-HTML rendering (that was the root cause of a bug reported earlier).
    return (
        f"<div style=\"overflow-x:auto;background-color:{CARD_BG};border-radius:8px;padding:4px;\">"
        f"<table style=\"border-collapse:collapse;width:100%;min-width:{min_width_px}px;"
        f"table-layout:fixed;font-family:{FONT_STACK};font-size:13px;\">"
        f"<thead><tr>"
        f"<th style='padding:8px 12px;text-align:left;color:{BASE_TEXT};border-bottom:2px solid {BORDER};'>Threshold</th>"
        f"{header_cells}"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
        f"</div>"
    )


def _pick_headline_threshold(data: dict, window_label: str):
    """Normally the headline figure is 1mm ('any rain'). Exception: if a
    higher threshold has reached >=95% (i.e. it's about as certain as the
    1mm figure would be), show that more severe threshold instead -- it's
    more informative than a near-guaranteed 'any rain' number. Among
    thresholds that qualify, the most severe (largest mm) one is used.

    Comparisons use the *rounded* (displayed) value, not the raw one --
    otherwise a probability that displays as "50%" but is actually 49.9
    under the hood would silently fail a ">=50" check, which looks like a
    bug to anyone just looking at the screen."""
    for threshold in (100, 50, 20, 5):
        val = data.get(threshold, {}).get(window_label)
        if val is not None and round(val) >= 95:
            return threshold, val
    val_1mm = data.get(1, {}).get(window_label)
    if val_1mm is None:
        return None
    return 1, val_1mm


def _pick_secondary_threshold(data: dict, window_label: str, headline_threshold: int):
    """The next more severe threshold above the headline that still clears
    65% -- e.g. if 5mm became the headline (>=95%), show the next threshold
    above 5mm (checked 100 -> 50 -> 20, most severe first) that's >=65%.
    65% matches the tone1 color-tier boundary (the same breakpoints used
    for the full table's cell coloring), so a threshold only shows up here
    once it's visually in the "strong" tier, not just "medium"."""
    for threshold in (100, 50, 20, 5, 1):
        if threshold <= headline_threshold:
            continue
        val = data.get(threshold, {}).get(window_label)
        if val is not None and round(val) >= 65:
            return threshold, val
    return None


def _cell_style(threshold_mm: int, val) -> tuple[str, str]:
    """Shared with the full table: same discrete-tier color by value,
    text color chosen for contrast against the actual rendered color."""
    if val is None:
        return "background-color:transparent;", BASE_TEXT
    br, bg_g, bb = _cell_rgb_for_value(threshold_mm, val)
    bg = f"background-color:rgb({br},{bg_g},{bb});"
    text_color = _cell_text_color(_relative_luminance(br, bg_g, bb))
    return bg, text_color


def render_three_day_table_html(result: dict, num_days: int = 3) -> str:
    windows = result["windows"][:num_days]
    data = result["data"]
    if not windows:
        return ""

    header_cells = "".join(
        f"<th style='padding:10px 16px;font-size:10pt;font-weight:600;text-align:center;"
        f"color:{BASE_TEXT};border-bottom:2px solid {BORDER};'>"
        f"{w['start_utc'].astimezone(PH_TZ).strftime('%a, %d %b')}</th>"
        for w in windows
    )

    headline_cells = ""
    secondary_cells = ""
    for w in windows:
        label = w["label"]
        headline = _pick_headline_threshold(data, label)

        if headline is None:
            headline_cells += (
                f"<td style='padding:14px 16px;text-align:center;color:{BASE_TEXT};"
                f"border-bottom:1px solid {BORDER};'>—</td>"
            )
            secondary_cells += (
                f"<td style='padding:10px 16px;text-align:center;color:{BASE_TEXT};"
                f"font-size:10pt;border-bottom:1px solid {BORDER};'>—</td>"
            )
            continue

        t_mm, t_val = headline
        bg, text_color = _cell_style(t_mm, t_val)
        headline_cells += (
            f"<td style='padding:14px 16px;text-align:center;{bg}"
            f"border-bottom:1px solid {BORDER};'>"
            f"<div style=\"font-size:20pt;font-weight:800;color:{text_color};line-height:1.15;\">{t_val:.0f}%</div>"
            f"<div style=\"font-size:10pt;font-weight:400;color:{text_color};margin-top:2px;\">chance of rain (&ge; {t_mm} mm)</div>"
            f"</td>"
        )

        secondary = _pick_secondary_threshold(data, label, t_mm)
        if secondary:
            s_mm, s_val = secondary
            s_bg, s_text_color = _cell_style(s_mm, s_val)
            secondary_cells += (
                f"<td style='padding:10px 16px;text-align:center;{s_bg}color:{s_text_color};"
                f"font-size:10pt;border-bottom:1px solid {BORDER};'>"
                f"{s_val:.0f}% chance of rain (&ge; {s_mm} mm)</td>"
            )
        else:
            secondary_cells += (
                f"<td style='padding:10px 16px;text-align:center;color:{BASE_TEXT};"
                f"font-size:10pt;border-bottom:1px solid {BORDER};'>—</td>"
            )

    return (
        f"<div style=\"overflow-x:auto;background-color:{CARD_BG};border-radius:8px;padding:4px;\">"
        f"<table style=\"border-collapse:collapse;width:100%;table-layout:fixed;font-family:{FONT_STACK};\">"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody><tr>{headline_cells}</tr><tr>{secondary_cells}</tr></tbody>"
        f"</table>"
        f"</div>"
    )


col1, col2 = st.columns([2, 1])
with col1:
    location_name = st.selectbox("Location", list(LOCATIONS.keys()))
    lat, lon = LOCATIONS[location_name]
with col2:
    lead_days = st.slider("Forecast range (days)", min_value=1, max_value=15, value=15)

load_aifs = st.checkbox(
    "Load AIFS",
    help="Also fetch ECMWF's AI-based ensemble (AIFS ENS) alongside the standard ENS. "
         "Roughly doubles the data fetched, though both run concurrently so it's not "
         "twice the wait.",
)

get_forecast_clicked = st.button("Get forecast", type="primary")

elapsed_placeholder = st.empty()

if get_forecast_clicked:
    request_started_at = time.time()
    try:
        fetch_out = get_forecast_with_progress(lat, lon, lead_days, load_aifs)
    except Exception as e:
        st.error(f"Failed to fetch forecast: {type(e).__name__}: {e}")
        with st.expander("Full error details"):
            st.exception(e)
        st.stop()
    elapsed = time.time() - request_started_at

    st.session_state["last_fetch_out"] = fetch_out
    st.session_state["last_load_aifs"] = load_aifs
    st.session_state["last_location_name"] = location_name
    st.session_state["last_lat"] = lat
    st.session_state["last_lon"] = lon
    st.session_state["last_elapsed"] = elapsed

if "last_fetch_out" in st.session_state:
    fetch_out = st.session_state["last_fetch_out"]
    load_aifs_active = st.session_state["last_load_aifs"]
    location_name = st.session_state["last_location_name"]
    lat = st.session_state["last_lat"]
    lon = st.session_state["last_lon"]
    elapsed = st.session_state["last_elapsed"]

    ens_result, ens_was_cached = fetch_out["ifs"]
    aifs_result, aifs_was_cached = fetch_out.get("aifs-ens", (None, False))

    any_fresh_fetch = not ens_was_cached or (load_aifs_active and not aifs_was_cached)
    elapsed_placeholder.caption(
        f"⏱️ Loaded in {elapsed:.1f}s" + ("" if any_fresh_fetch else " (from cache)")
    )

    # --- Model selector: only relevant once AIFS has actually loaded ---
    available_models = ["ifs"]
    if load_aifs_active:
        if aifs_result is not None:
            available_models.append("aifs-ens")
        else:
            st.warning("AIFS data couldn't be loaded for this request; showing ECMWF ENS only.")

    if len(available_models) > 1:
        show_aifs = st.toggle(f"Show {MODEL_LABELS['aifs-ens']}")
        selected_model = "aifs-ens" if show_aifs else "ifs"
    else:
        selected_model = "ifs"

    result = ens_result if selected_model == "ifs" else aifs_result
    was_cached = ens_was_cached if selected_model == "ifs" else aifs_was_cached
    model_label = MODEL_LABELS[selected_model]

    if not result["windows"]:
        st.warning("No aligned 00 UTC windows available for this range.")
        st.stop()

    if result.get("fetch_mode") == "separate" and not was_cached:
        st.caption(f"⚠️ Combined request wasn't available for {model_label}; fetched thresholds individually (slower).")

    if not result.get("aligned_to_utc_midnight", True):
        run_hour_str = f"{result['run_time'].hour:02d} UTC"
        st.caption(
            f"ℹ️ This {model_label} run started at {run_hour_str}, not 00/12 UTC, so windows below "
            f"are aligned to that run's own hour rather than the usual 00 UTC boundary."
        )

    if result.get("capped_to_day6"):
        st.caption(
            f"ℹ️ {model_label} is limited to 6 days here: its 06Z/18Z runs only publish that far out "
            f"(00Z/12Z runs go to 15 days, but this app can't tell which one it'll get in advance, "
            f"so it requests the range that's safe either way)."
        )

    # --- 3-day summary (essential info only) ---
    st.subheader(f"3-day summary for {location_name}")
    st.markdown(render_three_day_table_html(result, num_days=3), unsafe_allow_html=True)
    st.caption("All dates shown in UTC+8 (Philippine Time).")

    st.divider()

    # --- Full detailed table ---
    num_days_shown = len(result["windows"])
    st.subheader(f"Full {num_days_shown}-day forecast for {location_name}")
    show_graph = st.toggle("Show as graph")
    if show_graph:
        components.html(render_ribbon_chart_html(result), height=380)
    else:
        st.markdown(render_table_html(result), unsafe_allow_html=True)
    st.caption(f"All forecast windows shown in UTC+8 (Philippine Time). Source: ECMWF {model_label} Open Data (CC BY 4.0).")

    st.divider()

    # --- Run / location / grid info (moved to the bottom) ---
    run_time = result["run_time"]
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(f"**Model forecast run ({model_label}):** `{run_time.strftime('%Y-%m-%d %H UTC')}`")
        st.markdown(
            f"**Grid point used:** {result['grid_lat']:.3f}°N, {result['grid_lon']:.3f}°E "
            f"&nbsp;·&nbsp; **{location_name} (exact):** {lat:.6f}°N, {lon:.6f}°E"
        )
    with info_col2:
        now_utc = datetime.now(timezone.utc)
        available_ph = result["available_since"].astimezone(PH_TZ)
        next_ph = result["next_expected"].astimezone(PH_TZ)
        remaining = result["next_expected"] - now_utc
        if remaining.total_seconds() > 0:
            hrs = int(remaining.total_seconds() // 3600)
            mins = int((remaining.total_seconds() % 3600) // 60)
            remaining_str = f"in ~{hrs}h {mins}m"
        else:
            remaining_str = "due any time now"
        st.markdown(f"**Last updated (estimated):** {available_ph.strftime('%a, %d %b %Y %I:%M%p')} (UTC+8)")
        st.markdown(f"**Next update expected:** {next_ph.strftime('%a, %d %b %Y %I:%M%p')} (UTC+8) — {remaining_str}")

    schedule_note = (
        "\"Last updated\" and \"Next update\" are estimated from ECMWF's published IFS "
        "dissemination schedule, not a live timestamp from the server."
    )
    if selected_model == "aifs-ens":
        schedule_note += (
            " AIFS runs on a separate production pipeline with its own timing, so this "
            "estimate is rougher for AIFS than for ENS."
        )
    st.caption(schedule_note)
else:
    st.info("Choose a location and click **Get forecast**.")
