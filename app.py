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


def _theme_colors():
    """Streamlit's official runtime theme-detection API (st.context.theme,
    added 2025). This reflects the *actual* active theme per session and
    updates on rerun -- unlike CSS custom properties, which are only
    injected for real Streamlit components, not plain st.markdown HTML
    (that was the bug in the previous version: the table always fell back
    to its hardcoded light-mode default because those variables simply
    don't exist in this context)."""
    try:
        is_dark = st.context.theme.type == "dark"
    except Exception:
        is_dark = False

    if is_dark:
        return {"card_bg": "#1e1e1e", "text": "#f5f5f5", "border": "rgba(255,255,255,0.18)"}
    return {"card_bg": "#ffffff", "text": "#111111", "border": "rgba(0,0,0,0.12)"}


def _fmt_ph(dt: datetime) -> str:
    return dt.astimezone(PH_TZ).strftime("%a, %d %b %Y %I%p")


def _fmt_window(w: dict) -> str:
    return f"{_fmt_ph(w['start_utc'])}<br>to<br>{_fmt_ph(w['end_utc'])}"


def _cell_text_color(threshold_luminance: float, alpha: float, base_text: str) -> str:
    """Below a certain fill strength the tint is faint enough that the
    page's own text color still reads fine on it. Above that, pick black
    or white based on how bright the *threshold's* color is -- e.g. yellow
    (5mm) always needs dark text even at 100% fill, while maroon (50mm)
    always needs white text, regardless of the probability value."""
    if alpha < 0.35:
        return base_text
    return "#111111" if threshold_luminance > 0.5 else "#ffffff"


def render_table_html(result: dict) -> str:
    windows = result["windows"]
    data = result["data"]
    theme = _theme_colors()
    card_bg, base_text, border = theme["card_bg"], theme["text"], theme["border"]

    header_cells = "".join(
        f"<th style='padding:8px 12px;font-size:12px;white-space:nowrap;"
        f"color:{base_text};border-bottom:2px solid {border};'>{_fmt_window(w)}</th>"
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
                    f"<td style='padding:8px 12px;text-align:center;color:{base_text};"
                    f"border-bottom:1px solid {border};'>—</td>"
                )
                continue
            alpha = max(0.0, min(1.0, val / 100))
            bg = f"rgba({r},{g},{b},{alpha:.2f})"
            text_color = _cell_text_color(luminance, alpha, base_text)
            row_cells += (
                f"<td style='padding:8px 12px;text-align:center;"
                f"background-color:{bg};color:{text_color};font-weight:600;"
                f"border-bottom:1px solid {border};'>{val:.0f}%</td>"
            )
        rows_html += (
            f"<tr><td style='padding:8px 12px;font-weight:700;white-space:nowrap;"
            f"color:{base_text};background-color:{color_hex}33;"
            f"border-bottom:1px solid {border};'>≥{threshold} mm</td>{row_cells}</tr>"
        )

    return f"""
    <div style="overflow-x:auto;background-color:{card_bg};border-radius:8px;padding:4px;">
    <table style="border-collapse:collapse;width:100%;font-family:sans-serif;font-size:13px;">
      <thead><tr>
        <th style='padding:8px 12px;text-align:left;color:{base_text};
        border-bottom:2px solid {border};'>Threshold</th>
        {header_cells}
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """


def _pick_headline_threshold(data: dict, window_label: str):
    """Among the non-1mm thresholds, find the *largest* threshold (mm) whose
    probability still exceeds 50% -- e.g. if 5mm=98% and 20mm=55%, prefer
    showing 20mm (the more severe level that's still fairly likely) rather
    than 5mm (which is largely redundant with the 1mm 'any rain' figure)."""
    for threshold in (100, 50, 20, 5):
        val = data.get(threshold, {}).get(window_label)
        if val is not None and val > 50:
            return threshold, val
    return None


def render_summary_card_html(result: dict) -> str:
    windows = result["windows"]
    data = result["data"]
    if not windows:
        return ""

    theme = _theme_colors()
    card_bg, base_text, border = theme["card_bg"], theme["text"], theme["border"]

    w = windows[0]
    date_label = w["start_utc"].astimezone(PH_TZ).strftime("%a, %d %b %Y")
    val_1mm = data.get(1, {}).get(w["label"])
    val_1mm_str = f"{val_1mm:.0f}%" if val_1mm is not None else "—"

    headline = _pick_headline_threshold(data, w["label"])
    third_html = ""
    if headline:
        t_mm, t_val = headline
        third_html = f"""
        <div style="height:1px;background:{border};margin:10px 0;"></div>
        <div style="font-size:10pt;color:{base_text};">
          {t_val:.0f}% chance of rain (&ge; {t_mm} mm)
        </div>
        """

    return f"""
    <div style="max-width:280px;background-color:{card_bg};border:1px solid {border};
    border-radius:12px;padding:16px 20px;text-align:center;font-family:sans-serif;">
      <div style="font-size:10pt;font-weight:600;color:{base_text};">{date_label}</div>
      <div style="height:1px;background:{border};margin:10px 0;"></div>
      <div style="font-size:16pt;font-weight:800;color:{base_text};line-height:1.15;">{val_1mm_str}</div>
      <div style="font-size:10pt;color:{base_text};margin-top:2px;">chance of rain (&ge; 1 mm)</div>
      {third_html}
    </div>
    """


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

    st.markdown(render_summary_card_html(result), unsafe_allow_html=True)

    st.divider()

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
        "dissemination schedule, not a live timestamp from the server."
    )

    st.subheader("Exceedance probability by threshold and 24h window (00 UTC – 00 UTC)")
    st.markdown(render_table_html(result), unsafe_allow_html=True)
    st.caption("All forecast windows shown in UTC+8 (Philippine Time). Source: ECMWF ENS Open Data (CC BY 4.0).")
else:
    st.info("Choose a location and click **Get forecast**.")
