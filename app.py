import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from ecmwf_client import (
    AVAILABLE_THRESHOLDS_MM,
    THRESHOLD_COLORS,
    RESOLUTION_LABELS,
    PRODUCTS,
    estimated_release_time,
    estimated_next_update,
    PH_TZ,
    DEFAULT_PERCENTILES,
    AVAILABLE_PERIOD_HOURS,
    PERCENTILE_MAX_LEAD_HOURS,
    THRESHOLD_LEAD_DAYS,
    fetch_forecast_table,
    aggregate_percentile_bins_anchored,
    read_threshold_result_from_store,
    read_percentile_raw_from_store,
    read_temperature_percentiles_from_store,
    PH_BBOX,
)
import github_data_source
st.set_page_config(page_title="Probabilistic Rainfall Forecast", page_icon="🌧️", layout="wide")
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Source+Sans+3:wght@400;600;700;800&display=swap\" "
    'rel="stylesheet">',
    unsafe_allow_html=True,
)
st.title("🌧️ Probabilistic Rainfall Forecast")
st.caption("ECMWF ENS open data — rainfall amounts and chances of exceeding each threshold")
LOCATIONS = {
    "Mandaluyong City, Metro Manila": {"lat": 14.576975, "lon": 121.052521},
    "Guiguinto, Bulacan": {"lat": 14.842279, "lon": 120.859681},
    "Makati City, Metro Manila": {"lat": 14.555539, "lon": 121.002918},
    "Bambang, Nueva Vizcaya": {
        "lat": 16.389440, "lon": 121.106919,
        "query_overrides": {"0p25": (16.5, 121.25)},
    },
    "Bacoor, Cavite": {
        "lat": 14.454261, "lon": 120.941266,
        "query_overrides": {"0p1": (14.4, 120.9)},
    },
}

def resolve_query_point(lat: float, lon: float, resolution: str) -> tuple[float, float]:
    for loc in LOCATIONS.values():
        if loc["lat"] == lat and loc["lon"] == lon:
            return loc.get("query_overrides", {}).get(resolution, (lat, lon))
    return lat, lon

def get_true_point(location_name: str) -> tuple[float, float]:
    loc = LOCATIONS[location_name]
    return loc["lat"], loc["lon"]

_CUSTOM_COORDS_OPTION = "Custom coordinates"
_SEARCH_PLACE_OPTION = "Search Philippine city/municipality"
_PH_LOCATIONS_FILE = Path(__file__).parent / "ph_locations.json"
_ph_locations_cache: list[dict] | None = None

def _load_ph_locations() -> list[dict]:
    global _ph_locations_cache
    if _ph_locations_cache is None:
        with open(_PH_LOCATIONS_FILE, encoding="utf-8") as f:
            _ph_locations_cache = json.load(f)
    return _ph_locations_cache

def location_picker(key_prefix: str) -> tuple[str, float, float, float, float]:
    options = list(LOCATIONS.keys()) + [_SEARCH_PLACE_OPTION, _CUSTOM_COORDS_OPTION]
    selected = st.selectbox("Location", options, key=f"{key_prefix}_location")
    if selected not in (_SEARCH_PLACE_OPTION, _CUSTOM_COORDS_OPTION):
        true_lat, true_lon = get_true_point(selected)
        return selected, true_lat, true_lon, true_lat, true_lon
    if selected == _SEARCH_PLACE_OPTION:
        places = _load_ph_locations()
        provinces = sorted({p["province"] for p in places})
        search_col1, search_col2 = st.columns(2)
        with search_col1:
            chosen_province = st.selectbox("Province", provinces, key=f"{key_prefix}_province_search")
        places_in_province = [p for p in places if p["province"] == chosen_province]
        with search_col2:
            chosen_name = st.selectbox(
                "City/Municipality",
                [p["name"] for p in places_in_province],
                key=f"{key_prefix}_place_search_{chosen_province}",
            )
        place = next(p for p in places_in_province if p["name"] == chosen_name)
        st.map(pd.DataFrame({"lat": [place["lat"]], "lon": [place["lon"]]}), zoom=9)
        return place["display_name"], place["lat"], place["lon"], place["lat"], place["lon"]
    coord_col1, coord_col2 = st.columns(2)
    with coord_col1:
        custom_lat = st.number_input(
            "Latitude", min_value=PH_BBOX["lat_min"], max_value=PH_BBOX["lat_max"],
            value=14.6, step=0.01, format="%.4f", key=f"{key_prefix}_custom_lat",
        )
    with coord_col2:
        custom_lon = st.number_input(
            "Longitude", min_value=PH_BBOX["lon_min"], max_value=PH_BBOX["lon_max"],
            value=121.0, step=0.01, format="%.4f", key=f"{key_prefix}_custom_lon",
        )
    st.map(pd.DataFrame({"lat": [custom_lat], "lon": [custom_lon]}), zoom=9)
    location_name = f"Custom ({custom_lat:.4f}°N, {custom_lon:.4f}°E)"
    return location_name, custom_lat, custom_lon, custom_lat, custom_lon

CACHE_TTL_SECONDS = 15 * 60
_GLOBAL_CACHE: dict = {}
_CACHE_FILE = Path(__file__).parent / "forecast_cache_v2.json"

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
        "resolution": result.get("resolution"),
        "available_is_measured": result.get("available_is_measured", False),
        "next_description": result.get("next_description"),
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
        "resolution": d.get("resolution"),
        "available_is_measured": d.get("available_is_measured", False),
        "next_description": d.get("next_description"),
    }

def _cache_key_to_str(key: tuple) -> str:
    lat, lon, lead_days = key
    return f"{lat}|{lon}|{lead_days}"

def _cache_key_from_str(s: str) -> tuple:
    lat, lon, lead_days = s.split("|")
    return (float(lat), float(lon), int(lead_days))

def _load_disk_cache() -> None:
    if not _CACHE_FILE.exists():
        return
    try:
        raw = json.loads(_CACHE_FILE.read_text())
        now = time.time()
        for key_str, entry in raw.items():
            if now - entry["ts"] >= CACHE_TTL_SECONDS:
                continue
            _GLOBAL_CACHE[_cache_key_from_str(key_str)] = {
                "ts": entry["ts"],
                "result": _deserialize_result(entry["result"]),
            }
    except Exception:
        pass

def _save_disk_cache() -> None:
    try:
        raw = {
            _cache_key_to_str(key): {"ts": entry["ts"], "result": _serialize_result(entry["result"])}
            for key, entry in _GLOBAL_CACHE.items()
        }
        _CACHE_FILE.write_text(json.dumps(raw))
    except Exception:
        pass

_load_disk_cache()
_PCT_GLOBAL_CACHE: dict = {}
_PCT_CACHE_FILE = Path(__file__).parent / "percentile_cache_v2.json"

def _serialize_pct_result(result: dict) -> dict:
    def dt(x):
        return x.isoformat() if x else None
    return {
        "run_time": dt(result["run_time"]),
        "grid_lat": result["grid_lat"],
        "grid_lon": result["grid_lon"],
        "bins": [
            {
                "start_utc": dt(b["start_utc"]),
                "end_utc": dt(b["end_utc"]),
                "member_values_mm": b["member_values_mm"],
            }
            for b in result["bins"]
        ],
        "bin_hours": result["bin_hours"],
        "downloaded_bytes": result.get("downloaded_bytes"),
        "resolution": result.get("resolution"),
    }

def _deserialize_pct_result(d: dict) -> dict:
    def pdt(s):
        return datetime.fromisoformat(s) if s else None
    return {
        "run_time": pdt(d["run_time"]),
        "grid_lat": d["grid_lat"],
        "grid_lon": d["grid_lon"],
        "bins": [
            {
                "start_utc": pdt(b["start_utc"]),
                "end_utc": pdt(b["end_utc"]),
                "member_values_mm": b["member_values_mm"],
            }
            for b in d["bins"]
        ],
        "bin_hours": d["bin_hours"],
        "downloaded_bytes": d.get("downloaded_bytes"),
        "resolution": d.get("resolution"),
    }

def _pct_cache_key_to_str(key: tuple) -> str:
    lat, lon, lead_hours = key
    return f"{lat}|{lon}|{lead_hours}"

def _pct_cache_key_from_str(s: str) -> tuple:
    lat, lon, lead_hours = s.split("|")
    return (float(lat), float(lon), int(lead_hours))

def _load_pct_disk_cache() -> None:
    if not _PCT_CACHE_FILE.exists():
        return
    try:
        raw = json.loads(_PCT_CACHE_FILE.read_text())
        now = time.time()
        for key_str, entry in raw.items():
            if now - entry["ts"] >= CACHE_TTL_SECONDS:
                continue
            _PCT_GLOBAL_CACHE[_pct_cache_key_from_str(key_str)] = {
                "ts": entry["ts"],
                "result": _deserialize_pct_result(entry["result"]),
            }
    except Exception:
        pass

def _save_pct_disk_cache() -> None:
    try:
        raw = {
            _pct_cache_key_to_str(key): {"ts": entry["ts"], "result": _serialize_pct_result(entry["result"])}
            for key, entry in _PCT_GLOBAL_CACHE.items()
        }
        _PCT_CACHE_FILE.write_text(json.dumps(raw))
    except Exception:
        pass

_load_pct_disk_cache()

def _find_pct_superset_entry(cache_dict: dict, lat: float, lon: float, requested_lead_hours: int, now: float):
    candidates = []
    for key, entry in cache_dict.items():
        k_lat, k_lon, k_lead_hours = key
        if k_lat == lat and k_lon == lon and k_lead_hours >= requested_lead_hours:
            if now - entry["ts"] < CACHE_TTL_SECONDS:
                candidates.append((k_lead_hours, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def _truncate_pct_result(result: dict, lead_hours: int) -> dict:
    bin_hours = result["bin_hours"]
    num_bins = lead_hours // bin_hours
    truncated = dict(result)
    truncated["bins"] = result["bins"][:num_bins]
    return truncated

_STORE_DS_CACHE_TTL_SECONDS = 60
_store_ds_cache: dict = {
    "precip": {"ts": 0.0, "ds": None},
    "temperature": {"ts": 0.0, "ds": None},
}

def _load_store_cached(kind: str):
    entry = _store_ds_cache[kind]
    now = time.time()
    if entry["ds"] is not None and (now - entry["ts"] < _STORE_DS_CACHE_TTL_SECONDS):
        return entry["ds"]
    loader = github_data_source.load_percentile_grid if kind == "precip" else github_data_source.load_temperature_grid
    entry["ds"] = loader()
    entry["ts"] = now
    return entry["ds"]

def get_percentile_raw_with_progress(lat: float, lon: float, lead_hours: int):
    key = (lat, lon, lead_hours)
    now = time.time()
    session_cache = st.session_state.setdefault("_pct_forecast_cache", {})
    cached = session_cache.get(key)
    if cached and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["result"], True
    global_entry = _PCT_GLOBAL_CACHE.get(key)
    if global_entry and (now - global_entry["ts"] < CACHE_TTL_SECONDS):
        session_cache[key] = global_entry
        return global_entry["result"], True
    superset = (
        _find_pct_superset_entry(session_cache, lat, lon, lead_hours, now)
        or _find_pct_superset_entry(_PCT_GLOBAL_CACHE, lat, lon, lead_hours, now)
    )
    if superset:
        trimmed = _truncate_pct_result(superset["result"], lead_hours)
        entry = {"ts": now, "result": trimmed}
        session_cache[key] = entry
        _PCT_GLOBAL_CACHE[key] = entry
        return trimmed, True
    store_ds = _load_store_cached("precip")
    q_lat, q_lon = resolve_query_point(lat, lon, str(store_ds.attrs.get("resolution", "0p25")))
    store_result = read_percentile_raw_from_store(store_ds, q_lat, q_lon, lead_hours)
    if store_result is None:
        return None, False
    entry = {"ts": now, "result": store_result}
    session_cache[key] = entry
    _PCT_GLOBAL_CACHE[key] = entry
    _save_pct_disk_cache()
    return store_result, True

_TEMP_GLOBAL_CACHE: dict = {}

def get_temperature_percentiles_with_progress(lat: float, lon: float, lead_hours: int):
    key = (lat, lon, lead_hours)
    now = time.time()
    session_cache = st.session_state.setdefault("_temp_forecast_cache", {})
    cached = session_cache.get(key)
    if cached and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["result"], True
    global_entry = _TEMP_GLOBAL_CACHE.get(key)
    if global_entry and (now - global_entry["ts"] < CACHE_TTL_SECONDS):
        session_cache[key] = global_entry
        return global_entry["result"], True
    store_ds = _load_store_cached("temperature")
    q_lat, q_lon = resolve_query_point(lat, lon, str(store_ds.attrs.get("resolution", "0p25")))
    result = read_temperature_percentiles_from_store(store_ds, q_lat, q_lon, lead_hours)
    entry = {"ts": now, "result": result}
    session_cache[key] = entry
    _TEMP_GLOBAL_CACHE[key] = entry
    return result, True

def _find_superset_entry(cache_dict: dict, lat: float, lon: float, requested_lead_days: int, now: float):
    candidates = []
    for key, entry in cache_dict.items():
        k_lat, k_lon, k_lead_days = key
        if k_lat == lat and k_lon == lon and k_lead_days >= requested_lead_days:
            if now - entry["ts"] < CACHE_TTL_SECONDS:
                candidates.append((k_lead_days, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def _truncate_threshold_result(result: dict, lead_days: int) -> dict:
    truncated = dict(result)
    truncated["windows"] = result["windows"][:lead_days]
    return truncated

def _cache_lookup(key: tuple):
    lat, lon, lead_days = key
    now = time.time()
    session_cache = st.session_state.setdefault("_forecast_cache", {})
    cached = session_cache.get(key)
    if cached and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["result"]
    global_entry = _GLOBAL_CACHE.get(key)
    if global_entry and (now - global_entry["ts"] < CACHE_TTL_SECONDS):
        session_cache[key] = global_entry
        return global_entry["result"]
    superset = (
        _find_superset_entry(session_cache, lat, lon, lead_days, now)
        or _find_superset_entry(_GLOBAL_CACHE, lat, lon, lead_days, now)
    )
    if superset:
        trimmed = _truncate_threshold_result(superset["result"], lead_days)
        _cache_store(key, trimmed)
        return trimmed
    return None

def _cache_store(key: tuple, result: dict) -> None:
    entry = {"ts": time.time(), "result": result}
    st.session_state.setdefault("_forecast_cache", {})[key] = entry
    _GLOBAL_CACHE[key] = entry

def get_forecast_with_progress(lat: float, lon: float, lead_days: int):
    key = (lat, lon, lead_days)
    cached = _cache_lookup(key)
    if cached is not None:
        return cached, True
    store_ds = github_data_source.load_threshold_grid()
    if store_ds is not None:
        q_lat, q_lon = resolve_query_point(lat, lon, str(store_ds.attrs.get("resolution", "0p25")))
        store_result = read_threshold_result_from_store(store_ds, q_lat, q_lon, lead_days)
        _cache_store(key, store_result)
        return store_result, True
    progress = st.progress(0, text="Connecting to ECMWF Open Data...")
    def on_progress(frac: float, msg: str):
        progress.progress(min(max(int(frac * 100), 0), 100), text=msg)
    try:
        q_lat, q_lon = resolve_query_point(lat, lon, "0p25")
        result = fetch_forecast_table(q_lat, q_lon, max_lead_days=lead_days, progress_callback=on_progress)
    finally:
        progress.empty()
    _cache_store(key, result)
    _save_disk_cache()
    return result, False

def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def _relative_luminance(r: int, g: int, b: int) -> float:
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255

BASE_TEXT = "#111111"
CARD_BG = "#ffffff"
BORDER = "rgba(0,0,0,0.12)"
FONT_STACK = "'Source Sans 3', 'Source Sans Pro', -apple-system, sans-serif"

def _neutral_base_rgb() -> tuple[int, int, int]:
    return (255, 255, 255)

def _tint_toward_white_rgb(r: int, g: int, b: int, alpha: float) -> tuple[int, int, int]:
    return (
        round(r * alpha + 255 * (1 - alpha)),
        round(g * alpha + 255 * (1 - alpha)),
        round(b * alpha + 255 * (1 - alpha)),
    )

_TIER_FILL_ALPHA = {"tone3": 0.05, "tone2": 0.35, "tone1": 0.65}

def _tier_for_value(val: float) -> str:
    v = round(val)
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
    local = dt.astimezone(PH_TZ)
    return f"{local:%a, %d %b} {_hour_label(local.hour)}"

def _fmt_window(w: dict) -> str:
    return f"{_fmt_ph(w['start_utc'])}<br>to<br>{_fmt_ph(w['end_utc'])}"

def _cell_text_color(blended_luminance: float) -> str:
    return "#111111" if blended_luminance > 0.5 else "#ffffff"

def _estimate_col_width_px(windows: list[dict]) -> int:
    candidates = ["Threshold", "≥100 mm", "100%"]
    if windows:
        sample_header = _fmt_window(windows[0])
        candidates.extend(sample_header.split("<br>"))
    longest_line = max(len(line) for line in candidates)
    char_width_px = 7.5
    padding_px = 24
    return int(longest_line * char_width_px) + padding_px

def _tooltip_row_html(threshold: int, val) -> str:
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
    header = (
        f"<div style='padding:6px 10px;text-align:center;font-weight:700;font-size:12px;"
        f"color:{BASE_TEXT};background-color:#ffffff;"
        f"border-bottom:1px solid rgba(0,0,0,0.12);'>{_fmt_ph(w['start_utc'])} to {_fmt_ph(w['end_utc'])}</div>"
    )
    rows = "".join(_tooltip_row_html(t, data[t].get(w["label"])) for t in AVAILABLE_THRESHOLDS_MM)
    return f"<div style='font-family:{FONT_STACK};'>{header}{rows}</div>"

def render_ribbon_chart_html(result: dict) -> str:
    """Nested-area exceedance ribbon with persistent selected-day details."""
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
        fill: (i === rawDatasets.length - 1) ? "origin" : (i + 1),
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.3,
      }}));
      function showDayDetail(context) {{
        const {{ tooltip }} = context;
        if (tooltip.opacity === 0 || tooltip.dataPoints.length === 0) {{
          return;
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
            colors: {{ enabled: false, forceOverride: false }},
            legend: {{
              display: true,
              position: "bottom",
              labels: {{ boxWidth: 12, font: {{ size: 11 }}, color: "#333333" }},
            }},
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
      panelEl.innerHTML = dayDetailHtmlByDay[0];
    }})();
    </script>
    """

RAIN_BAR_CAP_MM = {3: 10, 6: 20}
ROW_ANCHOR_PH_HOUR = 2
_RAIN_BAR_GAP_PX = 6
_RAIN_BAR_MAX_PX = {8: 16, 4: 30}
_RAIN_SLOTS_MAX_PX = 170
_RAIN_MED = "#00BFFF"
_RAIN_P90 = "#CAD7E1"
_RAIN_TRACK = "#DFE3E9"

def _fmt_mm(v: float) -> str:
    return "0" if v < 0.05 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")

def _hour_label(hour: int) -> str:
    return f"{hour % 12 or 12}{'am' if hour % 24 < 12 else 'pm'}"

_tip_hour_label = _hour_label

def _hour_label_html(hour: int) -> str:
    return f"{hour % 12 or 12}<small class='rr-ap'>{'am' if hour % 24 < 12 else 'pm'}</small>"

def _chance_chip(threshold_mm: int, val, show_label: bool) -> str:
    if val is None:
        return "<div class='rr-chip rr-chip-na'>—</div>"
    bg, text_color = _cell_style(threshold_mm, val)
    label = f"<small>&ge;{threshold_mm} mm</small>" if show_label else ""
    return (
        f"<div class='rr-chip' style='{bg}color:{text_color};'>"
        f"<b>{val:.0f}<span class='rr-percent-sign'>%</span></b>{label}</div>"
    )


def render_rain_rows_html(pct_result: dict, threshold_result: dict | None = None, max_days: int = 6) -> str:
    """A compact selectable six-day rain forecast card.

    The intraday bar heights, scale, median/P90 colors, and individual bar
    tooltips are unchanged. Native CSS radio controls select one daily row
    and reveal its detail panel without a Streamlit rerun or a JS iframe.
    """
    from html import escape

    periods = pct_result["days"]
    step = pct_result["period_hours"]
    cap = RAIN_BAR_CAP_MM[step]
    n_slots = 24 // step
    if not periods:
        return ""

    rows: dict = {}
    for p in periods:
        start = p["start_utc"].astimezone(PH_TZ)
        rows.setdefault(start.date(), {})[(start.hour - ROW_ANCHOR_PH_HOUR) // step] = p

    bins_per_day = 24 // pct_result.get("bin_hours", 3)
    day_members: dict = {}
    for b in pct_result.get("raw_bins") or []:
        day_members.setdefault(b["start_utc"].astimezone(PH_TZ).date(), []).append(b["member_values_mm"])
    for day, row in rows.items():
        day_members.setdefault(day, [p["member_values_mm"] for p in row.values()])
    days = sorted(set(rows) | set(day_members))[:max_days]

    # 8am-8am threshold probabilities are intentionally distinct from the
    # 2am-2am day totals; preserve both labels in the expanded details.
    chance_data, chance_label = None, {}
    if threshold_result and threshold_result.get("windows"):
        chance_data = threshold_result["data"]
        chance_label = {
            w["start_utc"].astimezone(PH_TZ).date(): w["label"]
            for w in threshold_result["windows"]
        }
    has_chance = chance_data is not None
    row_cls = "rr-row rr-c" if has_chance else "rr-row"
    slot_hours = [ROW_ANCHOR_PH_HOUR + k * step for k in range(n_slots)]
    grid = f"grid-template-columns:repeat({n_slots},minmax(0,1fr));column-gap:{_RAIN_BAR_GAP_PX}px;"

    def tip_side(slot: int) -> str:
        if n_slots == 1:
            return ""
        pos = slot / (n_slots - 1)
        return " rr-tip-l" if pos < 0.25 else (" rr-tip-r" if pos > 0.75 else "")

    def bar(p: dict, slot: int, below: bool) -> str:
        st_ = p["stats"]
        med, p75, p90, p25, p10 = (
            st_["median"], st_["p75"], st_["p90"], st_["p25"], st_["p10"]
        )
        pct_of_cap = lambda v: min(max(v / cap, 0.0), 1.0) * 100
        start, end = p["start_utc"].astimezone(PH_TZ), p["end_utc"].astimezone(PH_TZ)
        when = f"{start:%a %d %b} · {_tip_hour_label(start.hour)}–{_tip_hour_label(end.hour)}"
        over_note = f" (P90 above the {cap} mm scale)" if p90 > cap else ""
        lines = [
            f"P90 {_fmt_mm(p90)} mm", f"P75 {_fmt_mm(p75)} mm",
            f"Median {_fmt_mm(med)} mm", f"P25 {_fmt_mm(p25)} mm",
            f"P10 {_fmt_mm(p10)} mm",
        ]
        cap_mark = "<span class='rr-over'></span>" if p90 > cap else ""
        tip_cls = "rr-tip" + tip_side(slot) + (" rr-tip-b" if below else "")
        return (
            f"<div class='rr-slot'><div class='rr-bar' tabindex='0'>"
            f"<div class='rr-fill'>"
            f"<span style='height:{pct_of_cap(p90):.1f}%;background:{_RAIN_P90}'></span>"
            f"<span style='height:{pct_of_cap(med):.1f}%;background:{_RAIN_MED}'></span>{cap_mark}</div>"
            f"<div class='{tip_cls}'><b>{when}</b><br>" + "<br>".join(lines) + f"{over_note}</div>"
            f"</div></div>"
        )

    empty_bar = "<div class='rr-slot'><div class='rr-bar rr-empty'><div class='rr-fill'></div></div></div>"
    chance_head = "<div class='rr-h rr-h-span'>Chance of rain</div>" if has_chance else ""
    head = (
        f"<div class='{row_cls} rr-head'><div></div>"
        f"<div class='rr-slots' style='{grid}'>"
        + "".join(f"<div class='rr-slot'><span>{_hour_label_html(h)}</span></div>" for h in slot_hours)
        + f"</div><div class='rr-h rr-h-r'>Day total</div>{chance_head}</div>"
    )

    body = ""
    details = ""
    for row_index, day in enumerate(days):
        row = rows.get(day, {})
        members = np.sum(day_members[day], axis=0)
        # Keep the original aggregation: sum each ensemble member first,
        # then compute percentiles across members.
        total_p10, total_p25, total_med, total_p75, total_p90 = [
            float(v) for v in np.percentile(members, [10, 25, 50, 75, 90])
        ]
        range_txt = f"{_fmt_mm(total_p10)}–{_fmt_mm(total_p90)}"
        is_partial = len(day_members[day]) < bins_per_day
        partial = "<span class='rr-part'>partial</span>" if is_partial else ""
        slots = "".join(bar(row[k], k, row_index < 2) if k in row else empty_bar for k in range(n_slots))

        chips = ""
        chance_detail = ""
        if has_chance:
            label = chance_label.get(day)
            v1 = chance_data[1].get(label) if label else None
            dyn = _pick_dynamic_threshold(chance_data, label) if label else None
            chips = _chance_chip(1, v1, True) + (
                _chance_chip(dyn[0], dyn[1], True) if dyn else _chance_chip(5, None, True)
            )
            fmt_prob = lambda v: '—' if v is None else f'{v:.0f}%'
            dyn_text = f"≥{dyn[0]} mm" if dyn else "≥5 mm"
            chance_detail = (
                "<div class='rr-detail-prob'>"
                "<div><small>Chance of ≥1 mm</small><b>" + fmt_prob(v1) + "</b></div>"
                "<div><small>Chance of " + dyn_text + "</small><b>" + fmt_prob(dyn[1] if dyn else None) + "</b></div>"
                "</div><div class='rr-detail-note'>Chance probabilities: 8am–8am (PHT), a different period from the day total.</div>"
            )

        # Hidden but keyboard-accessible native radio + whole-row label.
        # :has() switches the detail section locally, preserving bar tooltips.
        radio_id = f"rr-pick-{row_index}"
        checked = " checked" if row_index == 0 else ""
        body += (
            f"<input type='radio' class='rr-radio' name='rr-selected-day' id='{radio_id}'{checked}>"
            f"<label for='{radio_id}' class='{row_cls} rr-data rr-selectable' "
            f"aria-label='Show details for {day:%A %d %B}'>"
            f"<div class='rr-day'><b>{day:%a}</b><span>{day.day} {day:%b}</span></div>"
            f"<div class='rr-slots' style='{grid}'>{slots}</div>"
            f"<div class='rr-tot'><div class='rr-total-main'><b>{_fmt_mm(total_med)}</b>"
            f"<small>mm</small></div><span class='rr-range'>{range_txt}</span>{partial}</div>{chips}</label>"
        )
        items = [
            ("P10", total_p10), ("P25", total_p25),
            ("Median", total_med), ("P75", total_p75), ("P90", total_p90),
        ]
        quantiles_html = "".join(
            f"<div class='rr-quantile{' rr-qmedian' if name == 'Median' else ''}'>"
            f"<small>{name}</small><b>{_fmt_mm(amount)} <em>mm</em></b></div>"
            for name, amount in items
        )
        partial_note = " · Partial coverage" if is_partial else ""
        details += (
            f"<section class='rr-detail rr-detail-{row_index}' aria-label='Details for {day:%A %d %B}'>"
            f"<div class='rr-detail-top'><div><small>SELECTED DAY</small>"
            f"<strong>{day:%A, %d %B}</strong>"
            f"<span>Day total: 2am–2am (PHT){partial_note}</span></div>"
            f"<div class='rr-detail-main'><b>{_fmt_mm(total_med)}</b><small>mm median</small></div></div>"
            f"<div class='rr-quantiles'>{quantiles_html}</div>"
            f"{chance_detail}</section>"
        )

    legend_chance = (
        "<span>Chance columns: 24h, 8am–8am; ≥x mm = highest threshold at 65%+ (else 5 mm)</span>"
        if has_chance else ""
    )
    legend = (
        "<div class='rr-legend'>"
        f"<span><i style='background:{_RAIN_MED}'></i>Median</span>"
        f"<span><i style='background:{_RAIN_P90}'></i>P90</span>"
        "<span><i style='background:#0B3FBF;height:3px;vertical-align:3px'></i>above scale</span>"
        f"<span>Full bar = {cap} mm per {step}h</span>"
        "<span>Day total: median and P10–P90 of the 50 members</span>"
        f"{legend_chance}</div>"
    )

    # Keep the original bar-specific styling and colors intact; only the
    # total/chance typography, row selection and detail panel are new.
    css = (
        "<style>"
        f".rr-card{{background:#fff;color:#111;border-radius:10px;padding:10px 12px;font-family:{FONT_STACK};}}"
        f".rr-row{{display:grid;grid-template-columns:2.7rem minmax(0,{_RAIN_SLOTS_MAX_PX}px) 3.6rem;column-gap:10px;justify-content:space-evenly;align-items:center;}}"
        f".rr-row.rr-c{{grid-template-columns:2.3rem minmax(0,{_RAIN_SLOTS_MAX_PX}px) 3rem 2.7rem 2.7rem;column-gap:10px;}}"
        ".rr-data{padding:5px 0;border-top:1px solid #EEF0F3;}"
        ".rr-head{padding-bottom:2px;}"
        ".rr-head .rr-slot span{font-size:.58rem;color:#6b7280;white-space:nowrap;}.rr-ap{font-size:.46rem;}"
        ".rr-h{font-size:.58rem;color:#6b7280;text-align:center;line-height:1.1;white-space:nowrap;}"
        ".rr-h-r{text-align:center;}.rr-h-span{grid-column:span 2;}"
        ".rr-day{display:flex;flex-direction:column;line-height:1.15;}"
        ".rr-day b{font-size:.82rem;font-weight:700;}.rr-day span{font-size:.62rem;color:#6b7280;}"
        ".rr-slots{display:grid;}"
        ".rr-slot{display:flex;justify-content:center;}"
        f".rr-bar{{position:relative;width:100%;max-width:{_RAIN_BAR_MAX_PX[n_slots]}px;height:60px;outline:none;cursor:pointer;}}"
        f".rr-fill{{position:absolute;inset:0;background:{_RAIN_TRACK};border-radius:7px;overflow:hidden;}}"
        ".rr-fill span{position:absolute;left:0;right:0;bottom:0;display:block;}"
        ".rr-fill .rr-over{top:0;bottom:auto;height:3px;background:#0B3FBF;}"
        ".rr-empty{cursor:default;}.rr-empty .rr-fill{opacity:.25;}"
        ".rr-tot{text-align:center;white-space:nowrap;display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1.12;}"
        ".rr-total-main{display:flex;align-items:baseline;justify-content:center;gap:2px;}"
        ".rr-tot .rr-total-main b{font-size:.95rem;font-weight:750;letter-spacing:-.025em;font-variant-numeric:tabular-nums;}"
        ".rr-tot .rr-total-main small{font-size:.57rem;font-weight:400;color:#737D88;}"
        ".rr-tot .rr-range{display:block;font-size:.59rem;color:#858D98;margin-top:2px;}"
        ".rr-tot .rr-part{font-size:.57rem;font-style:italic;color:#6b7280;}"
        ".rr-chip{box-sizing:border-box;width:100%;height:2.5rem;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;line-height:1.1;box-shadow:inset 0 0 0 1px rgba(0,0,0,.08);}"
        ".rr-chip b{display:block;font-size:.90rem;font-weight:750;letter-spacing:-.015em;font-variant-numeric:tabular-nums;}"
        ".rr-chip .rr-percent-sign{font-size:.73em;font-weight:600;}"
        ".rr-chip small{display:block;font-size:.54rem;margin-top:1px;white-space:nowrap;opacity:.82;}"
        ".rr-chip-na{color:#9ca3af;font-size:.75rem;}"
        ".rr-tip{display:none;position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);"
        "background:#111;color:#fff;font-size:.68rem;line-height:1.4;padding:6px 9px;border-radius:7px;"
        "white-space:nowrap;z-index:50;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.25);}"
        ".rr-tip-l{left:0;transform:none;}.rr-tip-r{left:auto;right:0;transform:none;}"
        ".rr-tip-b{bottom:auto;top:calc(100% + 6px);}"
        ".rr-bar:hover .rr-tip,.rr-bar:focus .rr-tip{display:block;}"
        ".rr-legend{display:flex;flex-wrap:wrap;gap:3px 12px;font-size:.62rem;color:#6b7280;margin-top:8px;}"
        ".rr-legend i{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:4px;vertical-align:-1px;}"
        ".rr-radio{position:absolute;width:1px;height:1px;opacity:0;overflow:hidden;}"
        ".rr-selectable{cursor:pointer;border-radius:6px;transition:background-color .12s ease;}"
        ".rr-selectable:hover{background-color:#F8FAFC;}"
        ".rr-radio:checked+.rr-selectable{background-color:#F3F8FB;box-shadow:inset 2px 0 #00BFFF;}"
        ".rr-radio:focus-visible+.rr-selectable{outline:2px solid #159BD6;outline-offset:1px;}"
        ".rr-detail{display:none;margin-top:10px;padding:12px 13px;border-radius:9px;background:#F7FAFC;border:1px solid #E7EDF2;}"
        ".rr-detail-top{display:flex;justify-content:space-between;align-items:center;gap:10px;}"
        ".rr-detail-top>div:first-child{display:flex;flex-direction:column;gap:2px;}"
        ".rr-detail-top small{font-size:.55rem;font-weight:700;color:#82909E;letter-spacing:.08em;}"
        ".rr-detail-top strong{font-size:.84rem;font-weight:700;color:#1D2832;}"
        ".rr-detail-top span{font-size:.62rem;color:#798694;}"
        ".rr-detail-main{display:flex;align-items:baseline;gap:4px;white-space:nowrap;}"
        ".rr-detail-main b{font-size:1.22rem;letter-spacing:-.035em;color:#172532;}"
        ".rr-detail-main small{font-size:.62rem;color:#7D8995;letter-spacing:0;}"
        ".rr-quantiles{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:4px;margin-top:12px;}"
        ".rr-quantile{min-width:0;text-align:center;padding:7px 1px;border-radius:6px;background:#EFF3F6;}"
        ".rr-quantile small{display:block;font-size:.59rem;color:#788491;}"
        ".rr-quantile b{display:block;font-size:.77rem;font-weight:700;color:#3F4C58;white-space:nowrap;}"
        ".rr-quantile em{font-style:normal;font-size:.58rem;font-weight:400;color:#81909A;}"
        ".rr-qmedian{background:#E2F5FD;}"
        ".rr-qmedian b{color:#126F96;font-size:.85rem;}"
        ".rr-detail-prob{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:11px;}"
        ".rr-detail-prob>div{display:flex;align-items:baseline;justify-content:space-between;gap:5px;padding:6px 9px;border-radius:6px;background:#EDF3F8;}"
        ".rr-detail-prob small{font-size:.62rem;color:#657484;}"
        ".rr-detail-prob b{font-size:.83rem;color:#1E394D;white-space:nowrap;}"
        ".rr-detail-note{font-size:.57rem;color:#83909B;margin-top:5px;}"
        "@media(max-width:460px){.rr-detail{padding:10px 7px;}.rr-quantiles{gap:2px;}"
        ".rr-quantile b{font-size:.68rem;}.rr-qmedian b{font-size:.75rem;}"
        ".rr-detail-prob{grid-template-columns:1fr;gap:4px;}}"
        + "".join(
            f".rr-card:has(#rr-pick-{i}:checked) .rr-detail-{i}{{display:block;}}"
            for i in range(len(days))
        )
        + "</style>"
    )
    return f"{css}<div class='rr-card'>{head}{body}{legend}{details}</div>"


HEAT_INDEX_CATEGORIES = [
    ("caution", "Caution", "Caution (27°C+)", 27, "#F4C430"),
    ("extreme_caution", "Extreme Caution", "Extreme Caution (33°C+)", 33, "#FF9F1C"),
    ("danger", "Danger", "Danger (42°C+)", 42, "#E8450C"),
    ("extreme_danger", "Extreme Danger", "Extreme Danger (52°C+)", 52, "#C0122F"),
]

def render_temperature_chart_html(
    temp_result: dict,
    show_temp: bool = True,
    show_heat_index: bool = True,
    show_ranges: bool = True,
    show_categories: dict | None = None,
) -> str:
    steps = temp_result["steps"]
    if not steps or not (show_temp or show_heat_index):
        return ""
    show_categories = show_categories or {}
    time_labels = [s["time_utc"].astimezone(PH_TZ).strftime("%a %d, %I%p") for s in steps]
    labels_json = json.dumps(time_labels)
    temp_fill = "rgba(0,123,255,0.15)"
    hi_fill = "rgba(220,53,69,0.15)"
    datasets = []
    data_values = []
    if show_temp:
        if show_ranges:
            temp_p90 = [s["temperature_stats"].get("p90", s["temperature_stats"]["median"]) for s in steps]
            temp_p10 = [s["temperature_stats"].get("p10", s["temperature_stats"]["median"]) for s in steps]
            data_values.extend(temp_p90)
            data_values.extend(temp_p10)
            datasets.append({
                "label": "Temp P90", "data": temp_p90, "borderWidth": 0,
                "backgroundColor": temp_fill, "fill": "+1", "pointRadius": 0, "tension": 0.3,
            })
            datasets.append({
                "label": "Temp P10", "data": temp_p10, "borderWidth": 0,
                "backgroundColor": temp_fill, "fill": False, "pointRadius": 0, "tension": 0.3,
            })
        temp_median = [s["temperature_stats"]["median"] for s in steps]
        data_values.extend(temp_median)
        datasets.append({
            "label": "Temperature (median)", "data": temp_median,
            "borderColor": "#0056b3", "backgroundColor": "#0056b3", "fill": False,
            "pointRadius": 0, "tension": 0.3, "borderWidth": 3,
        })
    if show_heat_index:
        if show_ranges:
            hi_p90 = [s["heat_index_stats"].get("p90", s["heat_index_stats"]["median"]) for s in steps]
            hi_p10 = [s["heat_index_stats"].get("p10", s["heat_index_stats"]["median"]) for s in steps]
            data_values.extend(hi_p90)
            data_values.extend(hi_p10)
            datasets.append({
                "label": "Heat Index P90", "data": hi_p90, "borderWidth": 0,
                "backgroundColor": hi_fill, "fill": "+1", "pointRadius": 0, "tension": 0.3,
            })
            datasets.append({
                "label": "Heat Index P10", "data": hi_p10, "borderWidth": 0,
                "backgroundColor": hi_fill, "fill": False, "pointRadius": 0, "tension": 0.3,
            })
        hi_median = [s["heat_index_stats"]["median"] for s in steps]
        data_values.extend(hi_median)
        datasets.append({
            "label": "Heat Index (median)", "data": hi_median,
            "borderColor": "#c82333", "backgroundColor": "#c82333", "fill": False,
            "pointRadius": 0, "tension": 0.3, "borderWidth": 3,
            "borderDash": [6, 4],
        })
    data_min, data_max = min(data_values), max(data_values)
    span = data_max - data_min
    if span <= 10:
        y_step = 2
    elif span <= 25:
        y_step = 5
    else:
        y_step = 10
    y_min = math.floor(data_min / y_step) * y_step
    y_max = math.ceil(data_max / y_step) * y_step
    if y_max == y_min:
        y_max += y_step
    category_datasets = []
    category_lines_for_labels = []
    category_full_labels = []
    for key, short_name, full_label, threshold_c, color in HEAT_INDEX_CATEGORIES:
        if show_categories.get(key):
            category_datasets.append({
                "label": full_label, "data": [threshold_c] * len(steps),
                "borderColor": color, "backgroundColor": color, "fill": False,
                "pointRadius": 0, "tension": 0, "borderWidth": 1.5, "borderDash": [3, 3],
            })
            category_lines_for_labels.append({"name": short_name, "value": threshold_c, "color": color})
            category_full_labels.append(full_label)
    datasets = category_datasets + datasets
    datasets_json = json.dumps(datasets)
    category_lines_json = json.dumps(category_lines_for_labels)
    category_full_labels_json = json.dumps(category_full_labels)
    return f"""
    <link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>body {{ font-family: {FONT_STACK}; margin: 0; }}</style>
    <div style="background-color:#ffffff;padding:8px;">
      <div style="position:relative;width:100%;height:320px;">
        <canvas id="tempChart"></canvas>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    (function() {{
      const labels = {labels_json};
      const datasets = {datasets_json};
      const categoryLines = {category_lines_json};
      const categoryFullLabels = new Set({category_full_labels_json});
      const categoryLabelPlugin = {{
        id: "categoryLabelPlugin",
        afterDatasetsDraw(chart) {{
          const {{ ctx, chartArea, scales }} = chart;
          const yScale = scales.y;
          ctx.save();
          ctx.font = "600 10px {FONT_STACK}";
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          categoryLines.forEach((cat) => {{
            if (cat.value < yScale.min || cat.value > yScale.max) return;
            const y = yScale.getPixelForValue(cat.value);
            ctx.fillStyle = cat.color;
            ctx.fillText(cat.name, chartArea.left + 4, y - 2);
          }});
          ctx.restore();
        }},
      }};
      new Chart(document.getElementById("tempChart"), {{
        type: "line",
        data: {{ labels: labels, datasets: datasets }},
        plugins: [categoryLabelPlugin],
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{
            colors: {{ enabled: false, forceOverride: false }},
            legend: {{
              display: true, position: "bottom",
              labels: {{
                boxWidth: 12, font: {{ size: 10 }}, color: "#333333",
                filter: (item) => !item.text.includes("P10") && !item.text.includes("P90")
                  && !categoryFullLabels.has(item.text),
              }},
            }},
            tooltip: {{
              filter: (item) => !categoryFullLabels.has(item.dataset.label),
              itemSort: (a, b) => {{
                const seriesRank = (label) => label.toLowerCase().includes("heat index") ? 1 : 0;
                const statRank = (label) => {{
                  if (label.includes("P90")) return 0;
                  if (label.includes("median")) return 1;
                  if (label.includes("P10")) return 2;
                  return 3;
                }};
                const sr = seriesRank(a.dataset.label) - seriesRank(b.dataset.label);
                if (sr !== 0) return sr;
                return statRank(a.dataset.label) - statRank(b.dataset.label);
              }},
              callbacks: {{ label: (ctx) => ctx.dataset.label + ": " + ctx.parsed.y.toFixed(1) + "\\u00b0C" }},
            }},
          }},
          scales: {{
            y: {{
              min: {y_min},
              max: {y_max},
              ticks: {{ stepSize: {y_step}, callback: (v) => v + "\\u00b0C", color: "#666666" }},
              grid: {{ color: "#e5e5e5" }},
            }},
            x: {{ ticks: {{ color: "#666666", font: {{ size: 10 }}, maxTicksLimit: 10 }}, grid: {{ display: false }} }},
          }},
        }},
      }});
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
    col_width_px = _estimate_col_width_px(windows)
    min_width_px = (len(windows) + 1) * col_width_px
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

def _pick_dynamic_threshold(data: dict, window_label: str):
    for threshold in (100, 50, 20, 5):
        val = data.get(threshold, {}).get(window_label)
        if val is not None and round(val) >= 65:
            return threshold, val
    val_5mm = data.get(5, {}).get(window_label)
    return (5, val_5mm) if val_5mm is not None else None

def _cell_style(threshold_mm: int, val) -> tuple[str, str]:
    if val is None:
        return "background-color:transparent;", BASE_TEXT
    br, bg_g, bb = _cell_rgb_for_value(threshold_mm, val)
    bg = f"background-color:rgb({br},{bg_g},{bb});"
    text_color = _cell_text_color(_relative_luminance(br, bg_g, bb))
    return bg, text_color

def _coords_caption(location_name: str, true_lat: float, true_lon: float, result: dict) -> str:
    resolution = result.get("resolution") or "0p25"
    res_label = RESOLUTION_LABELS.get(resolution, resolution)
    parts = [
        f"**{location_name}, exact:** {true_lat:.6f}°N, {true_lon:.6f}°E",
        f"**{res_label} grid point used:** {result['grid_lat']:.3f}°N, {result['grid_lon']:.3f}°E",
    ]
    query = resolve_query_point(true_lat, true_lon, resolution)
    if query != (true_lat, true_lon):
        parts.append(f"(query point overridden to {query[0]}°N, {query[1]}°E for {res_label} data)")
    return " · ".join(parts)

_PRODUCT_LABELS = {
    "threshold": "Threshold forecast",
    "percentile": "Percentile rainfall",
    "temperature": "Temperature / heat index",
}

def _fmt_ph_minutes(dt: datetime) -> str:
    return dt.astimezone(PH_TZ).strftime("%a %d %b %I:%M%p")

def _relative(delta_seconds: float) -> str:
    if delta_seconds <= 0:
        return "due now"
    hours, minutes = int(delta_seconds // 3600), int((delta_seconds % 3600) // 60)
    return f"in {hours}h {minutes:02d}m" if hours else f"in {minutes}m"

def build_status_footer_html(manifest: dict | None, now: datetime) -> str:
    cell = "padding:2px 10px 2px 0;white-space:nowrap;vertical-align:top;"
    head = "".join(
        f"<th style='{cell}text-align:left;font-weight:600;'>{h}</th>"
        for h in ("Product", "Cycle (UTC)", "Grid", "Released (UTC+8)", "Next update, est. (UTC+8)")
    )
    if manifest is None:
        return ("<div style='font-size:0.72rem;opacity:0.8;'>Data status unavailable "
                "(couldn't reach the data store).</div>")
    run_times = manifest.get("run_times", {})
    resolutions = manifest.get("resolutions", {})
    published = manifest.get("published_at", {})
    failing = set(manifest.get("failures", []))
    rows = ""
    for name in PRODUCTS:
        label = _PRODUCT_LABELS[name] + (" ⚠" if name in failing else "")
        if name not in run_times:
            rows += (f"<tr><td style='{cell}'>{label}</td>"
                     f"<td colspan='4' style='{cell}'>no data ingested yet</td></tr>")
            continue
        run_time = datetime.fromisoformat(run_times[name])
        if run_time.tzinfo is None:
            run_time = run_time.replace(tzinfo=timezone.utc)
        resolution = resolutions.get(name, "0p25")
        if published.get(name):
            released = _fmt_ph_minutes(datetime.fromisoformat(published[name])) + " ✓"
        else:
            released = "~" + _fmt_ph_minutes(estimated_release_time(name, run_time, resolution))
        next_when, next_desc = estimated_next_update(name, run_time, resolution)
        next_text = (f"~{_fmt_ph_minutes(next_when)} ({_relative((next_when - now).total_seconds())}) "
                     f"– {next_desc}")
        rows += (
            f"<tr><td style='{cell}'>{label}</td>"
            f"<td style='{cell}'>{run_time:%HZ %d %b}</td>"
            f"<td style='{cell}'>{RESOLUTION_LABELS.get(resolution, resolution)}</td>"
            f"<td style='{cell}'>{released}</td>"
            f"<td style='{cell}'>{next_text}</td></tr>"
        )
    checked = ""
    generated_at = manifest.get("generated_at")
    if generated_at:
        age_min = (now - datetime.fromisoformat(generated_at)).total_seconds() / 60
        checked = f" Store last checked {int(age_min)} min ago."
    note = (
        "✓ = publish time recorded from ECMWF; ~ = estimated from ECMWF's dissemination schedule. "
        "Each 0.25° cycle is replaced by its 0.1° version 2h later." + checked
    )
    return (
        "<div style='font-size:0.72rem;line-height:1.35;opacity:0.85;overflow-x:auto;'>"
        f"<table style='border-collapse:collapse;'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
        f"<div style='margin-top:4px;opacity:0.8;'>{note}</div></div>"
    )

def render_status_footer() -> None:
    st.divider()
    st.markdown(
        build_status_footer_html(github_data_source.get_manifest(), datetime.now(timezone.utc)),
        unsafe_allow_html=True,
    )

location_name, lat, lon, true_lat, true_lon = location_picker("shared")
st.divider()
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "Rainfall Forecast"
_selected_mode = st.segmented_control(
    "View",
    ["Rainfall Forecast", "Temperature Forecast"],
    default=st.session_state["view_mode"],
)
if _selected_mode is not None:
    st.session_state["view_mode"] = _selected_mode
view_mode = st.session_state["view_mode"]
main = st.container()
footer_slot = st.container()
with footer_slot:
    render_status_footer()
with main:
    if view_mode == "Rainfall Forecast":
        st.caption(
            "Rainfall amounts from all 50 ECMWF ensemble members (median and P10–P90 range), "
            "alongside ECMWF's precomputed chances of exceeding fixed rainfall thresholds. "
            "Reads pre-computed data from the scheduled ingestion job; only the threshold "
            "probabilities fall back to a live 0.25° fetch if the store isn't fresh."
        )
        st.markdown(
            "<style>"
            "@media (orientation: portrait){"
            ".st-key-rain_split [data-testid='stHorizontalBlock']{flex-direction:column !important;}"
            ".st-key-rain_split [data-testid='stColumn'],.st-key-rain_split [data-testid='column']"
            "{width:100% !important;flex:1 1 auto !important;min-width:0 !important;}"
            "}"
            "@media (orientation: landscape){.st-key-thr_divider{display:none;}}"
            "</style>",
            unsafe_allow_html=True,
        )
        lead_days = THRESHOLD_LEAD_DAYS
        pct_lead_hours = PERCENTILE_MAX_LEAD_HOURS
        rain_clicked = st.button("Get rainfall forecast", type="primary", key="rain_button")
        rain_elapsed_placeholder = st.empty()
        if rain_clicked:
            rain_started_at = time.time()
            thr_out = None
            pct_out = None
            try:
                thr_out = get_forecast_with_progress(lat, lon, lead_days)
            except Exception as e:
                st.error(f"Failed to fetch threshold probabilities: {type(e).__name__}: {e}")
                with st.expander("Full error details (threshold)"):
                    st.exception(e)
            try:
                pct_out = get_percentile_raw_with_progress(lat, lon, pct_lead_hours)
            except Exception as e:
                st.error(f"Failed to read rainfall percentiles: {type(e).__name__}: {e}")
                with st.expander("Full error details (percentile)"):
                    st.exception(e)
            pct_raw_new, pct_cached_new = pct_out if pct_out else (None, False)
            if thr_out is None and pct_raw_new is None:
                st.error("Rainfall data isn't available right now -- please try again shortly.")
                st.stop()
            st.session_state["last_rain"] = {
                "location_name": location_name,
                "true": (true_lat, true_lon),
                "thr": thr_out[0] if thr_out else None,
                "thr_cached": thr_out[1] if thr_out else True,
                "pct_raw": pct_raw_new,
                "pct_cached": pct_cached_new,
                "elapsed": time.time() - rain_started_at,
            }
        if "last_rain" in st.session_state:
            rain = st.session_state["last_rain"]
            rain_name = rain["location_name"]
            rain_true = rain["true"]
            thr_result = rain["thr"]
            pct_raw_result = rain["pct_raw"]
            all_cached = (thr_result is None or rain["thr_cached"]) and (pct_raw_result is None or rain["pct_cached"])
            rain_elapsed_placeholder.caption(
                f"⏱️ Loaded in {rain['elapsed']:.1f}s" + (" (from cache)" if all_cached else "")
            )
            card_thr = thr_result if (thr_result is not None and thr_result["windows"]) else None
            with st.container(key="rain_split"):
                left_col, right_col = st.columns([5, 6], gap="large")
            with left_col:
                st.subheader(f"6-day rainfall forecast for {rain_name}")
                pct_result = None
                if pct_raw_result is None:
                    st.warning(
                        "Rainfall amounts (percentile data) aren't available from the latest ingestion "
                        "right now (store unreachable or not yet published) -- please try again shortly."
                    )
                else:
                    period_labels = {h: f"{h}h" for h in AVAILABLE_PERIOD_HOURS}
                    if st.session_state.get("pct_period_hours") not in AVAILABLE_PERIOD_HOURS:
                        st.session_state["pct_period_hours"] = 3
                    selected_period_label = st.segmented_control(
                        "Time step",
                        list(period_labels.values()),
                        default=period_labels[st.session_state["pct_period_hours"]],
                        key="pct_period_control",
                    )
                    if selected_period_label is not None:
                        st.session_state["pct_period_hours"] = next(
                            h for h, label in period_labels.items() if label == selected_period_label
                        )
                    pct_period_hours = st.session_state["pct_period_hours"]
                    pct_result = aggregate_percentile_bins_anchored(pct_raw_result, period_hours=pct_period_hours)
                    pct_days = pct_result["days"]
                    if not pct_days:
                        st.warning(
                            f"Not enough data for a complete {pct_period_hours}-hour period at this forecast "
                            f"range -- try a shorter time step."
                        )
                    else:
                        pct_percentiles = pct_result["percentiles"]
                        st.caption(
                            f"Each bar is a {pct_period_hours}-hour accumulation (UTC+8). Slices sit on the same 2am-based grid "
                            f"at every time step, so a day's total is identical however you slice it; a slice the forecast only "
                            f"partly covers is left empty."
                            + (" The chance columns are ECMWF's 24h threshold probabilities, which run 8am–8am, "
                               "so each is matched to the day it starts on." if card_thr else "")
                        )
                        show_pct_graph = st.toggle("Show as graph", value=True, key="pct_show_graph")
                        if show_pct_graph:
                            st.markdown(render_rain_rows_html(pct_result, card_thr, max_days=6), unsafe_allow_html=True)
                        else:
                            stat_rows = ["mean", "median"] + [f"p{p}" for p in pct_percentiles]
                            stat_display_labels = {"mean": "Mean", "median": "Median", **{f"p{p}": f"P{p}" for p in pct_percentiles}}
                            day_col_labels = [f"{d['start_utc'].astimezone(PH_TZ):%a %d %b}, {_hour_label(d['start_utc'].astimezone(PH_TZ).hour)}" for d in pct_days]
                            table_data = {
                                day_col_labels[i]: [round(d["stats"][row], 1) for row in stat_rows]
                                for i, d in enumerate(pct_days)
                            }
                            pct_df = pd.DataFrame(table_data, index=[stat_display_labels[r] for r in stat_rows])
                            st.dataframe(pct_df, use_container_width=True)
            with right_col:
                if thr_result is not None:
                    resolution = thr_result.get("resolution") or "0p25"
                    model_label = f"ECMWF ENS {RESOLUTION_LABELS.get(resolution, resolution)}"
                    with st.container(key="thr_divider"):
                        st.divider()
                    if not thr_result["windows"]:
                        st.warning("No aligned 00 UTC windows available for this range.")
                    else:
                        if thr_result.get("fetch_mode") == "separate" and not rain["thr_cached"]:
                            st.caption(f"⚠️ Combined request wasn't available for {model_label}; fetched thresholds individually (slower).")
                        st.caption(
                            "ℹ️ ECMWF publishes these threshold probabilities for the 00Z and 12Z ensemble runs only "
                            "(there is no 06Z or 18Z probability product), so they update twice a day -- first at 0.25°, "
                            "then at 0.1° about two hours later. The rainfall amounts use all four daily runs."
                        )
                        num_days_shown = len(thr_result["windows"])
                        st.subheader(f"Full {num_days_shown}-day threshold forecast for {rain_name}")
                        show_graph = st.toggle("Show as graph", key="thr_show_graph")
                        if show_graph:
                            components.html(render_ribbon_chart_html(thr_result), height=650)
                        else:
                            st.markdown(render_table_html(thr_result), unsafe_allow_html=True)
                        st.caption(f"All forecast windows shown in UTC+8 (Philippine Time). Source: ECMWF {model_label} Open Data (CC BY 4.0).")
            if pct_result is not None:
                st.caption(_coords_caption(rain_name, *rain_true, pct_result))
            if thr_result is not None:
                st.caption(_coords_caption(rain_name, *rain_true, thr_result))
        else:
            st.info("Choose a location and click **Get rainfall forecast**.")

    elif view_mode == "Temperature Forecast":
        st.caption(
            'Temperature and heat index ("feels like") percentiles across all 50 ECMWF ensemble '
            'members, at native 3-hourly resolution. Always reads pre-computed data from the '
            'scheduled ingestion job, same as the views above -- no live fetch.'
        )
        temp_location_name, temp_lat, temp_lon = location_name, lat, lon
        temp_lead_hours = PERCENTILE_MAX_LEAD_HOURS
        temp_clicked = st.button("Get temperature forecast", key="temp_button")
        temp_elapsed_placeholder = st.empty()
        if temp_clicked:
            temp_request_started_at = time.time()
            try:
                temp_result, temp_was_cached = get_temperature_percentiles_with_progress(
                    temp_lat, temp_lon, temp_lead_hours
                )
            except Exception as e:
                st.error(f"Failed to read temperature forecast: {type(e).__name__}: {e}")
                with st.expander("Full error details"):
                    st.exception(e)
                st.stop()
            temp_elapsed = time.time() - temp_request_started_at
            st.session_state["last_temp_result"] = temp_result
            st.session_state["last_temp_location_name"] = temp_location_name
            st.session_state["last_temp_true"] = (true_lat, true_lon)
            st.session_state["last_temp_was_cached"] = temp_was_cached
            st.session_state["last_temp_elapsed"] = temp_elapsed
        if "last_temp_result" in st.session_state:
            temp_result = st.session_state["last_temp_result"]
            temp_location_name = st.session_state["last_temp_location_name"]
            temp_was_cached = st.session_state["last_temp_was_cached"]
            temp_elapsed = st.session_state["last_temp_elapsed"]
            temp_elapsed_placeholder.caption(
                f"⏱️ Loaded in {temp_elapsed:.1f}s" + (" (from cache)" if temp_was_cached else "")
            )
            temp_steps = temp_result["steps"]
            if not temp_steps:
                st.warning("No data available for this forecast range.")
                st.stop()
            temp_percentiles = temp_result["percentiles"]
            st.subheader(f"Temperature & heat index for {temp_location_name}")
            temp_show_graph = st.toggle("Show as graph", value=True, key="temp_show_graph")
            if temp_show_graph:
                cb1, cb2, cb3 = st.columns(3)
                with cb1:
                    show_temp_series = st.checkbox("Temperature", value=False, key="temp_show_temp_series")
                with cb2:
                    show_hi_series = st.checkbox("Heat Index", value=True, key="temp_show_hi_series")
                with cb3:
                    show_ranges = st.checkbox("Show ranges", value=True, key="temp_show_ranges")
                show_all_categories = st.checkbox(
                    "Show Heat Index reference lines", value=True, key="temp_show_categories"
                )
                show_categories = {
                    key: show_all_categories for key, _short, _label, _threshold_c, _color in HEAT_INDEX_CATEGORIES
                }
                if not show_temp_series and not show_hi_series:
                    st.info("Select at least one of Temperature or Heat Index to show the graph.")
                else:
                    components.html(
                        render_temperature_chart_html(
                            temp_result,
                            show_temp=show_temp_series,
                            show_heat_index=show_hi_series,
                            show_ranges=show_ranges,
                            show_categories=show_categories,
                        ),
                        height=380,
                    )
            else:
                temp_stat_rows = ["mean", "median"] + [f"p{p}" for p in temp_percentiles]
                temp_stat_display_labels = {
                    "mean": "Mean", "median": "Median", **{f"p{p}": f"P{p}" for p in temp_percentiles}
                }
                temp_col_labels = [s["time_utc"].astimezone(PH_TZ).strftime("%a %d %b, %I%p") for s in temp_steps]
                temp_table_data = {
                    temp_col_labels[i]: [round(s["temperature_stats"][row], 1) for row in temp_stat_rows]
                    for i, s in enumerate(temp_steps)
                }
                hi_table_data = {
                    temp_col_labels[i]: [round(s["heat_index_stats"][row], 1) for row in temp_stat_rows]
                    for i, s in enumerate(temp_steps)
                }
                st.caption("Temperature (°C)")
                st.dataframe(
                    pd.DataFrame(temp_table_data, index=[temp_stat_display_labels[r] for r in temp_stat_rows]),
                    use_container_width=True,
                )
                st.caption('Heat Index / "feels like" (°C)')
                st.dataframe(
                    pd.DataFrame(hi_table_data, index=[temp_stat_display_labels[r] for r in temp_stat_rows]),
                    use_container_width=True,
                )
            st.caption(_coords_caption(temp_location_name, *st.session_state.get("last_temp_true", (true_lat, true_lon)), temp_result))
        else:
            st.info(
                "Choose a location and click **Get temperature forecast** to see temperature "
                "and heat index percentiles."
            )
