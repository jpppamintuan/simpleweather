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
    "Mandaluyong": (14.576975, 121.052521),
    "Makati": (14.555539, 121.002918),
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
    threshold's color needs light or dark text on top of it."""
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


# Streamlit exposes its live theme colors as CSS custom properties on the
# page. Using them directly (instead of detecting the theme in Python once
# per script run) means the table updates immediately when the person
# toggles light/dark mode, with no stale state.
CARD_BG = "var(--background-color, #ffffff)"
BASE_TEXT = "var(--text-color, #111111)"
BORDER = "rgba(128,128,128,0.35)"


def _fmt_ph(dt: datetime) -> str:
    return dt.astimezone(PH_TZ).strftime("%a, %d %b %Y %I%p")


def _fmt_window(w: dict) -> str:
    return f"{_fmt_ph(w['start_utc'])}<br>to<br>{_fmt_ph(w['end_utc'])}"


def _cell_text_color(threshold_luminance: float, alpha: float) -> str:
    """Below a certain fill strength the tint is faint enough that the
    page's own text color still reads fine on it. Above that, pick black
    or white based on how bright the *threshold's* color is -- e.g. yellow
    (5mm) always needs dark text even at 100% fill, while maroon (50mm)
    always needs white text, regardless of the probability value."""
    if alpha < 0.35:
        return BASE_TEXT
    return "#111111" if threshold_luminance > 0.5 else "#ffffff"


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
        luminance = _relative_luminance(r, g, b)
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
            bg = f"rgba({r},{g},{b},{alpha:.2f})"
            text_color = _cell_text_color(luminance, alpha)
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

    return f"""
    <div style="overflow-x:auto;background-color:{CARD_BG};border-radius:8px;padding:4px;">
    <table style="border-collapse:collapse;width:100%;font-family:sans-serif;font-size:13px;">
      <thead><tr>
        <th style='padding:8px 12px;text-align:left;color:{BASE_TEXT};
        border-bottom:2px solid {BORDER};'>Threshold</th>
        {header_cells}
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """


col1, col2 = st.columns([2, 1])
with col1:
    location_name = st.selectbox("Location", list(LOCATIONS.keys()))
    lat, lon = LOCATIONS[location_name]
with col2:
    lead_days = st.slider("Forecast range (days)", min_value=1, max_value=15, value=15)

if st.button("Get forecast", type="primary"):
    request_started_at = time.time()
    try:
        result, was_cached = get_forecast_with_progress(lat, lon, lead_days)
    except Exception as e:
        st.error(f"Failed to fetch forecast: {e}")
        st.stop()

    if not result["windows"]:
        st.warning("No aligned 00 UTC windows available for this range.")
        st.stop()

    if was_cached:
        st.caption("⚡ Served from cache (fetched earlier this session).")
    elif result.get("fetch_mode") == "separate":
        st.caption("⚠️ Combined request wasn't available; fetched thresholds individually (slower).")

    # --- Run / location / grid info ---
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
        "dissemination schedule, not a live timestamp from the server — see note below."
    )

    st.subheader("Exceedance probability by threshold and 24h window (00 UTC – 00 UTC)")
    st.markdown(render_table_html(result), unsafe_allow_html=True)
    st.caption("All forecast windows shown in UTC+8 (Philippine Time). Source: ECMWF ENS Open Data (CC BY 4.0).")

    elapsed = time.time() - request_started_at
    st.caption(f"⏱️ Loaded in {elapsed:.1f}s" + (" (from cache)" if was_cached else ""))
else:
    st.info("Choose a location and click **Get forecast**.")
