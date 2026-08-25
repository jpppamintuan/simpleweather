import colorsys
import time
from datetime import datetime, timezone

import streamlit as st

from ecmwf_client import (
    AVAILABLE_THRESHOLDS_MM,
    THRESHOLD_COLORS,
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
    "Guiguinto, Bulacan": (14.842279, 120.859681),
    "Mandaluyong City, Metro Manila": (14.576975, 121.052521),
    "Makati City, Metro Manila": (14.555539, 121.002918),
    "Bambang, Nueva Vizcaya": (16.389440, 121.106919),
    "Bacoor, Cavite": (14.454261, 120.941266),
}

CACHE_TTL_SECONDS = 3 * 60 * 60  # ENS updates twice a day (00/12 UTC)

# Shared across ALL browser sessions on this running server process (unlike
# st.session_state, which is per-session). This is what makes repeat
# requests from *different* users/tabs instant instead of re-fetching.
# Caveat: lives in memory only -- cleared on app reboot/redeploy/sleep, and
# wouldn't be shared across multiple server replicas if this app were ever
# scaled beyond a single instance (not a concern for the current setup).
_GLOBAL_CACHE: dict = {}


def get_forecast_with_progress(lat: float, lon: float, lead_days: int):
    """Three-tier lookup: this session's own cache (fastest) -> the
    cross-session global cache (any other user may have already fetched
    this) -> a genuine fetch, with real progress reported from inside
    fetch_forecast_table."""
    key = (lat, lon, lead_days)
    now = time.time()

    session_cache = st.session_state.setdefault("_forecast_cache", {})
    cached = session_cache.get(key)
    if cached and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["result"], True

    global_entry = _GLOBAL_CACHE.get(key)
    if global_entry and (now - global_entry["ts"] < CACHE_TTL_SECONDS):
        session_cache[key] = global_entry  # promote into this session too
        return global_entry["result"], True

    progress = st.progress(0, text="Connecting to ECMWF Open Data...")

    def on_progress(frac: float, msg: str):
        progress.progress(min(max(int(frac * 100), 0), 100), text=msg)

    try:
        result = fetch_forecast_table(lat, lon, max_lead_days=lead_days, progress_callback=on_progress)
    finally:
        progress.progress(100, text="Done!")
        time.sleep(0.2)
        progress.empty()

    entry = {"ts": now, "result": result}
    session_cache[key] = entry
    _GLOBAL_CACHE[key] = entry
    return result, False


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _relative_luminance(r: int, g: int, b: int) -> float:
    """Perceived brightness (0=black, 1=white). Used to decide whether a
    color needs light or dark text on top of it."""
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


# Default text/background for elements that AREN'T deliberately colored by
# a probability value (date headers, "--" placeholders, borders): just
# don't set an explicit color. "inherit"/"transparent" let these pick up
# whatever Streamlit is actually rendering around them, so they're correct
# in light mode, dark mode, or any custom theme, with no detection needed.
BASE_TEXT = "inherit"
CARD_BG = "transparent"
BORDER = "rgba(128,128,128,0.35)"

# Streamlit's default body font is Source Sans (Adobe's open-source
# typeface). Just naming it in font-family isn't enough -- if the font
# file isn't actually loaded, the browser silently falls back to its own
# default (SF Pro on iOS, Arial on Windows), which was happening here. The
# webfont is loaded explicitly via Google Fonts in the script below.
FONT_STACK = "'Source Sans 3', 'Source Sans Pro', -apple-system, sans-serif"


def _neutral_base_rgb() -> tuple[int, int, int]:
    """The <5% tier's color: white for light mode, black for dark mode.
    Rendered as a SOLID (opaque) color rather than true CSS opacity -- true
    opacity blends against whatever's actually behind it, so the same
    color looked like a clean pastel on a white page but a muddy smear on
    a dark one. Falls back to white if theme detection isn't available."""
    try:
        is_dark = st.context.theme.type == "dark"
    except Exception:
        is_dark = False
    return (0, 0, 0) if is_dark else (255, 255, 255)


def _blend_toward_neutral(r: int, g: int, b: int, alpha: float) -> tuple[int, int, int]:
    """Standard tier color: scale toward the neutral (white/black) base.
    This darkens/lightens while keeping saturation at 100%, which reads
    fine for red, blue, purple, yellow -- but not orange, see below."""
    nr, ng, nb = _neutral_base_rgb()
    return (
        round(r * alpha + nr * (1 - alpha)),
        round(g * alpha + ng * (1 - alpha)),
        round(b * alpha + nb * (1 - alpha)),
    )


def _desaturated_tier_rgb(r: int, g: int, b: int, target_saturation: float) -> tuple[int, int, int]:
    """Alternate tier color: keep the hue and lightness, only reduce
    saturation. Used for orange (20mm) instead of _blend_toward_neutral --
    darkening pure orange toward black makes it read as "brown" (there's
    no separate name for "dark orange" in common usage the way there is
    for dark red/blue/purple). Desaturating toward a muted/pastel version
    at roughly the same lightness avoids that entirely, and -- as a bonus
    -- looks reasonable on both light and dark pages without needing to
    know which one is active, since it isn't heading toward black or
    white."""
    h, l, _s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    nr, ng, nb = colorsys.hls_to_rgb(h, l, target_saturation)
    return round(nr * 255), round(ng * 255), round(nb * 255)


# Discrete color tiers instead of a continuous gradient -- easier to read
# at a glance, like a legend, and sidesteps needing "good at every possible
# opacity" colors. Boundaries match the requested scheme:
#   <5%: neutral (white/black)   >=95%: pure base color
#   5-35% / 35-65% / 65-95%: tone3 / tone2 / tone1 (lightest -> strongest)
_TIER_FILL_ALPHA = {"tone3": 0.20, "tone2": 0.50, "tone1": 0.80}
_TIER_SATURATION = {"tone3": 0.25, "tone2": 0.55, "tone1": 0.80}

# Thresholds that use saturation-based tiers instead of the standard
# lightness-based ones. Currently just 20mm/orange.
_DESATURATE_THRESHOLDS = {20}


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
    if tier == "neutral":
        return _neutral_base_rgb()

    r, g, b = _hex_to_rgb(THRESHOLD_COLORS[threshold_mm])
    if tier == "base":
        return r, g, b

    if threshold_mm in _DESATURATE_THRESHOLDS:
        return _desaturated_tier_rgb(r, g, b, _TIER_SATURATION[tier])

    return _blend_toward_neutral(r, g, b, _TIER_FILL_ALPHA[tier])


def _fmt_ph(dt: datetime) -> str:
    return dt.astimezone(PH_TZ).strftime("%a, %d %b %I%p")


def _fmt_window(w: dict) -> str:
    return f"{_fmt_ph(w['start_utc'])}<br>to<br>{_fmt_ph(w['end_utc'])}"


def _cell_text_color(blended_luminance: float) -> str:
    """Pick black or white text based on the *actual rendered* blended
    color's brightness -- correct at every probability level now that the
    background is a solid pre-blended color rather than true transparency."""
    return "#111111" if blended_luminance > 0.5 else "#ffffff"


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

    # NOTE: no leading whitespace on any line below -- st.markdown treats
    # 4+ leading spaces as a Markdown code block, which silently breaks
    # raw-HTML rendering (that was the root cause of a bug reported earlier).
    return (
        f"<div style=\"overflow-x:auto;background-color:{CARD_BG};border-radius:8px;padding:4px;\">"
        f"<table style=\"border-collapse:collapse;width:100%;font-family:{FONT_STACK};font-size:13px;\">"
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
    50% -- e.g. if 5mm became the headline (>=95%), show the next threshold
    above 5mm (checked 100 -> 50 -> 20, most severe first) that's >=50%."""
    for threshold in (100, 50, 20, 5, 1):
        if threshold <= headline_threshold:
            continue
        val = data.get(threshold, {}).get(window_label)
        if val is not None and round(val) >= 50:
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
        f"<div style=\"overflow-x:auto;border-radius:8px;padding:4px;\">"
        f"<table style=\"border-collapse:collapse;width:100%;font-family:{FONT_STACK};\">"
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

button_col, refresh_col = st.columns([1, 1])
with button_col:
    get_forecast_clicked = st.button("Get forecast", type="primary")
with refresh_col:
    # Toggling Streamlit's light/dark mode doesn't trigger a Python rerun,
    # so table colors (computed in Python, baked into static HTML) can't
    # react to it live. This button doesn't do anything special itself --
    # merely being clicked causes a rerun, and since rendering below always
    # re-runs from session_state (not just on "Get forecast"), that's
    # enough to recompute colors against whatever theme is active now.
    refresh_clicked = st.button("🎨 Refresh colors for current theme")

elapsed_placeholder = st.empty()

if get_forecast_clicked:
    request_started_at = time.time()
    try:
        result, was_cached = get_forecast_with_progress(lat, lon, lead_days)
    except Exception as e:
        st.error(f"Failed to fetch forecast: {e}")
        st.stop()
    elapsed = time.time() - request_started_at

    st.session_state["last_result"] = result
    st.session_state["last_location_name"] = location_name
    st.session_state["last_lat"] = lat
    st.session_state["last_lon"] = lon
    st.session_state["last_was_cached"] = was_cached
    st.session_state["last_elapsed"] = elapsed

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    location_name = st.session_state["last_location_name"]
    lat = st.session_state["last_lat"]
    lon = st.session_state["last_lon"]
    was_cached = st.session_state["last_was_cached"]
    elapsed = st.session_state["last_elapsed"]

    elapsed_placeholder.caption(
        f"⏱️ Loaded in {elapsed:.1f}s" + (" (from cache)" if was_cached else "")
    )

    if not result["windows"]:
        st.warning("No aligned 00 UTC windows available for this range.")
        st.stop()

    if result.get("fetch_mode") == "separate" and not was_cached:
        st.caption("⚠️ Combined request wasn't available; fetched thresholds individually (slower).")

    # --- 3-day summary (essential info only) ---
    st.subheader(f"3-day summary for {location_name}")
    st.markdown(render_three_day_table_html(result, num_days=3), unsafe_allow_html=True)
    st.caption("All dates shown in UTC+8 (Philippine Time).")

    st.divider()

    # --- Full detailed table ---
    num_days_shown = len(result["windows"])
    st.subheader(f"Full {num_days_shown}-day forecast for {location_name}")
    st.markdown(render_table_html(result), unsafe_allow_html=True)
    st.caption("All forecast windows shown in UTC+8 (Philippine Time). Source: ECMWF ENS Open Data (CC BY 4.0).")

    st.divider()

    # --- Run / location / grid info (moved to the bottom) ---
    run_time = result["run_time"]
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(f"**Model forecast run:** `{run_time.strftime('%Y-%m-%d %H UTC')}`")
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

    st.caption(
        "\"Last updated\" and \"Next update\" are estimated from ECMWF's published "
        "dissemination schedule, not a live timestamp from the server."
    )
else:
    st.info("Choose a location and click **Get forecast**.")
