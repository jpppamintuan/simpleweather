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


def get_forecast_with_progress(lat: float, lon: float, lead_days: int):
    """Session-level cache (per browser session) so repeat views of the same
    location/range are instant, while a genuinely new fetch shows real
    progress reported directly from inside fetch_forecast_table."""
    cache = st.session_state.setdefault("_forecast_cache", {})
    key = (lat, lon, lead_days)
    now = time.time()

    cached = cache.get(key)
    if cached and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["result"], True  # cache hit

    progress = st.progress(0, text="Connecting to ECMWF Open Data...")

    def on_progress(frac: float, msg: str):
        progress.progress(min(max(int(frac * 100), 0), 100), text=msg)

    try:
        result = fetch_forecast_table(lat, lon, max_lead_days=lead_days, progress_callback=on_progress)
    finally:
        progress.progress(100, text="Done!")
        time.sleep(0.2)
        progress.empty()

    cache[key] = {"ts": now, "result": result}
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

# Neutral gray that low-probability cells blend toward. A cell's color is
# pre-blended toward this and rendered as a SOLID (opaque) color, rather
# than using real CSS opacity -- true opacity blends against whatever's
# actually behind it, so the same "40% red" looks like a clean pastel on a
# white page but a muddy, hard-to-read smear on a dark one. Pre-blending
# ourselves means the result looks identical regardless of page theme.
_NEUTRAL_BASE_RGB = (170, 170, 170)

# Streamlit's own default UI font (its "sans serif" theme option resolves
# to this), so tables visually match the rest of the app instead of
# falling back to the browser's generic sans-serif (usually Arial).
FONT_STACK = "'Source Sans Pro', 'Source Sans 3', -apple-system, sans-serif"


def _blend_toward_neutral(r: int, g: int, b: int, alpha: float) -> tuple[int, int, int]:
    nr, ng, nb = _NEUTRAL_BASE_RGB
    return (
        round(r * alpha + nr * (1 - alpha)),
        round(g * alpha + ng * (1 - alpha)),
        round(b * alpha + nb * (1 - alpha)),
    )


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
        r, g, b = _hex_to_rgb(color_hex)
        row_cells = ""
        for w in windows:
            val = data[threshold].get(w["label"])
            if val is None:
                row_cells += (
                    f"<td style='padding:8px 12px;text-align:center;color:{BASE_TEXT};"
                    f"border-bottom:1px solid {BORDER};'>—</td>"
                )
                continue
            alpha = max(0.0, min(1.0, val / 100))
            br, bg_g, bb = _blend_toward_neutral(r, g, b, alpha)
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
    """Shared with the full table: solid pre-blended color by value,
    text color chosen for contrast against the actual rendered color."""
    if val is None:
        return "background-color:transparent;", BASE_TEXT
    r, g, b = _hex_to_rgb(THRESHOLD_COLORS[threshold_mm])
    alpha = max(0.0, min(1.0, val / 100))
    br, bg_g, bb = _blend_toward_neutral(r, g, b, alpha)
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

get_forecast_clicked = st.button("Get forecast", type="primary")
elapsed_placeholder = st.empty()

if get_forecast_clicked:
    request_started_at = time.time()
    try:
        result, was_cached = get_forecast_with_progress(lat, lon, lead_days)
    except Exception as e:
        st.error(f"Failed to fetch forecast: {e}")
        st.stop()

    elapsed = time.time() - request_started_at
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
