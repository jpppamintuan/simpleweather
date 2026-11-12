import json
from html import escape
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
from taiwan_support import taiwan_enabled, region_for_point
from regions import TAIWAN_BBOX
st.set_page_config(page_title="Simpleweather", page_icon="🌧️", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700;800&display=swap');
      :root {
        --simpleweather-font: 'Source Sans 3', 'Source Sans Pro', sans-serif;
      }
      /* Streamlit's main page and its native headers and form controls. */
      html, body, .stApp, [data-testid="stAppViewContainer"],
      [data-testid="stAppViewContainer"] :is(h1, h2, h3, h4, h5, h6, p, label,
          button, input, select, textarea),
      [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
      [data-testid="stSidebar"] :is(h1, h2, h3, h4, h5, h6, p, label,
          button, input, select, textarea) {
        font-family: var(--simpleweather-font) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Simpleweather")
st.caption(
    "Probabilistic rainfall and heat index forecasts across the Philippines, with Simple mode."
)
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
_TAIWAN_PLACE_OPTION = "Taiwan trip locations"
_TAIWAN_COORDS_OPTION = "Custom coordinates (Taiwan)"
_TW_LOCATIONS_FILE = Path(__file__).parent / "tw_locations.json"
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
    if taiwan_enabled():
        options += [_TAIWAN_PLACE_OPTION, _TAIWAN_COORDS_OPTION]
    selected = st.selectbox("Location", options, key=f"{key_prefix}_location")
    if selected == _TAIWAN_PLACE_OPTION and taiwan_enabled():
        with _TW_LOCATIONS_FILE.open(encoding="utf-8") as f:
            places = json.load(f)
        selected_place = st.selectbox(
            "Taiwan location", [p["name"] for p in places],
            key=f"{key_prefix}_taiwan_place",
        )
        place = next(p for p in places if p["name"] == selected_place)
        if st.toggle("Show map", value=False, key=f"{key_prefix}_taiwan_map"):
            st.map(pd.DataFrame({"lat": [place["lat"]], "lon": [place["lon"]]}), zoom=10)
        return place["name"] + ", Taiwan", place["lat"], place["lon"], place["lat"], place["lon"]
    if selected == _TAIWAN_COORDS_OPTION and taiwan_enabled():
        col1, col2 = st.columns(2)
        with col1:
            custom_lat = st.number_input(
                "Latitude", min_value=float(TAIWAN_BBOX["lat_min"]),
                max_value=float(TAIWAN_BBOX["lat_max"]), value=25.03,
                step=0.01, format="%.4f", key=f"{key_prefix}_taiwan_lat",
            )
        with col2:
            custom_lon = st.number_input(
                "Longitude", min_value=float(TAIWAN_BBOX["lon_min"]),
                max_value=float(TAIWAN_BBOX["lon_max"]), value=121.56,
                step=0.01, format="%.4f", key=f"{key_prefix}_taiwan_lon",
            )
        if st.toggle("Show map", value=False, key=f"{key_prefix}_taiwan_custom_map"):
            st.map(pd.DataFrame({"lat": [custom_lat], "lon": [custom_lon]}), zoom=10)
        display = f"Taiwan custom ({custom_lat:.4f}°N, {custom_lon:.4f}°E)"
        return display, custom_lat, custom_lon, custom_lat, custom_lon
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
        if st.toggle("Show map", value=False, key=f"{key_prefix}_show_map"):
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
    if st.toggle("Show map", value=False, key=f"{key_prefix}_show_map"):
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
    (region, kind): {"ts": 0.0, "ds": None}
    for region in ("ph", "taiwan")
    for kind in ("precip", "temperature")
}

def _load_store_cached(kind: str, region: str = "ph"):
    entry = _store_ds_cache[(region, kind)]
    now = time.time()
    if entry["ds"] is not None and (now - entry["ts"] < _STORE_DS_CACHE_TTL_SECONDS):
        return entry["ds"]
    loader = github_data_source.load_percentile_grid if kind == "precip" else github_data_source.load_temperature_grid
    entry["ds"] = loader(region=region)
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
    store_ds = _load_store_cached("precip", region_for_point(lat, lon))
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
    store_ds = _load_store_cached("temperature", region_for_point(lat, lon))
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
    store_ds = github_data_source.load_threshold_grid(region=region_for_point(lat, lon))
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

BASE_TEXT = "#283746"
CARD_BG = "#ffffff"
BORDER = "rgba(0,0,0,0.12)"
SHARED_RAIN_TYPOGRAPHY_CSS = r'''
/* Shared Simpleweather rainfall typography system (both independent cards).
   Sizes respond to the embedded card's own iframe viewport, not the outer app.
   Roles are matched between cards; special dense graph/chip labels remain small. */
:root {
  --rain-ink: #283746;
  --rain-muted: #687585;
  --rain-subtle: #8994A1;
  --rain-date: #737D88;
  --rain-section: #516678;
  --rain-selected: #1C3044;
  --rain-today: #1786B6;
  --rain-title-size: 16px;
  --rain-tabs-size: 12px;
  --rain-day-size: 12px;
  --rain-date-size: 11px;
  --rain-today-size: 10px;
  --rain-header-size: 11px;
  --rain-value-size: 14px;
  --rain-detail-title-size: 16px;
  --rain-section-size: 13px;
  --rain-detail-value-size: 15px;
  --rain-secondary-size: 11px;
  --rain-micro-size: 10px;
  --rain-descriptor-size: 12px;
  --rain-total-detail-size: 20px;
  --rain-quantile-size: 14px;
}
@media(min-width:430px){
  :root{
    --rain-title-size:18px;
    --rain-tabs-size:13px;
    --rain-day-size:14px;
    --rain-value-size:16px;
    --rain-descriptor-size:13px;
    --rain-quantile-size:15px;
    --rain-total-detail-size:22px;
  }
}
/* Card headings: identical weight, color, responsive sizing. */
.rr-card-title,.shell-title h2{
  font-size:var(--rain-title-size);font-weight:720;color:var(--rain-ink);
  line-height:1.3;letter-spacing:-.025em;
}
/* Tab buttons and the Simple mode control. */
.rr-card-tabs button,.view-tab{
  font-size:var(--rain-tabs-size);font-weight:700;color:#677586;
  line-height:1.2;
}
.rr-card-tabs button[aria-pressed="true"],.view-tab[aria-selected="true"]{
  color:var(--rain-selected);font-weight:700;
}
.rr-simple-toggle,.simple-toggle{font-size:13px;font-weight:650;color:#506070}
/* Daily date, TODAY label, column headers, and forecast values. */
.rr-day-weekday,.day-label .day-short strong{
  font-size:var(--rain-day-size);font-weight:700;color:var(--rain-ink);
  line-height:1.12;
}
.day-label .day-short{font-size:var(--rain-day-size);font-weight:400;color:var(--rain-ink);line-height:1.12}
.rr-day-calendar,.day-label .date,.day-label .date-inline{
  font-size:var(--rain-date-size);font-weight:400;color:var(--rain-date);line-height:1.15;
}
.rr-today-tag,.day-label .today-mini,.panel-day .today-mini,.matrix-today{
  font-size:var(--rain-today-size);font-weight:400;color:var(--rain-today);
  letter-spacing:.025em;line-height:1.15;
}
/* A small unboxed TODAY marker above each day's date in all cards. */
.rr-day .rr-today-tag,.temp-day-date .rr-today-tag,
.day-label .today-mini{margin:0 0 2px;white-space:nowrap;}
.matrix-today{display:inline-block;margin-left:6px;vertical-align:baseline;white-space:nowrap;}
.rr-h,.bar-header{
  font-size:var(--rain-header-size);font-weight:700;color:#7A8796;
}
.rr-h-r,.rr-h-chance{white-space:normal;overflow-wrap:normal;line-height:1.12}
.rr-chip b,.rr-chip .rr-prob-head b,.rr-tot .rr-total-main b,.probability{
  font-size:var(--rain-value-size);font-weight:750;
}
.rr-tot .rr-total-main b,.probability{color:var(--rain-ink)}
.rr-chip .rr-percent-sign,.rr-chip .rr-chip-threshold,
.rr-tot .rr-total-main small,.rr-secondary{
  font-size:var(--rain-secondary-size);font-weight:400;
}
.rr-tot .rr-total-main small,.rr-secondary{color:var(--rain-muted)}
/* Simple mode values (the six-day JS may shrink the WHOLE label together). */
.rr-simple-chance,.rr-total-plain,.probability.simple{
  font-size:var(--rain-descriptor-size);font-weight:680;line-height:1.13;
}
.rr-simple-threshold{font-size:var(--rain-secondary-size);font-weight:400}
.rr-total-range{font-size:var(--rain-descriptor-size)}
/* Expanded day headings, section headings, and headline details. */
.rr-detail-top strong,.panel-head h3{
  font-size:var(--rain-detail-title-size);font-weight:700;color:var(--rain-ink);
  line-height:1.24;
}
.rr-section-heading,.smallheading{
  font-size:var(--rain-section-size);font-weight:750;color:var(--rain-section);
}
.rr-detail-main b{font-size:var(--rain-total-detail-size);font-weight:750;color:var(--rain-ink)}
.rr-detail-main small,.rr-detail-main small.rr-detail-intensity,.panel-head .window{
  font-size:var(--rain-secondary-size);font-weight:400;color:var(--rain-muted);
}
.rr-quantile small,.amount-small .low{
  font-size:var(--rain-secondary-size);font-weight:400;color:#657383;
}
.rr-quantile b,.amount-small .big{
  font-size:var(--rain-quantile-size);font-weight:750;color:var(--rain-ink);
}
.rr-qmedian b{font-size:calc(var(--rain-quantile-size) + 1px);color:#126F96}
.rr-detail-chance small.rr-detail-chance-threshold,.chance-left strong{
  font-size:var(--rain-secondary-size);font-weight:700;
}
.rr-detail-chance small.rr-detail-chance-meaning,.chance-left span{
  font-size:var(--rain-micro-size);font-weight:400;
}
.rr-detail-chance b,.chance-right strong{
  font-size:var(--rain-detail-value-size);font-weight:750;
}
.rr-detail-chance .rr-detail-chance-desc,.chance-right span{
  font-size:var(--rain-micro-size);font-weight:400;
}
/* Explanations, legends, supporting notes, matrices, and chart labels. */
.rr-intro,.rr-reading-guide,.footnote{
  font-size:12px;font-weight:400;color:#697789;line-height:1.55;
}
.rr-reading-guide summary,.footnote summary{font-size:12px;font-weight:700;color:#576b80}
.rr-legend,.legend-pill,.context-line,.kicker,.rr-detail-note{
  font-size:var(--rain-secondary-size);font-weight:400;color:var(--rain-muted);
}
.rr-section-heading small,.rr-time-range,.rr-detail-bar-hours .rr-slot span,
.rr-head .rr-slot span,.rr-quantile em,.rr-quantile .rr-quantile-desc,
.panel-head .tap-hint,.finger-tip{
  font-size:var(--rain-micro-size);font-weight:400;color:var(--rain-muted);
}
.rr-quantile-help{font-size:9px!important;font-weight:400;color:var(--rain-subtle)}
.section-label{font-size:var(--rain-section-size);font-weight:750;color:var(--rain-section)}
.question{font-size:var(--rain-secondary-size);font-weight:650;color:var(--rain-muted)}
.threshold-chip{font-size:var(--rain-descriptor-size);font-weight:700}
.threshold-chip .chip-meaning{font-size:var(--rain-micro-size);font-weight:450}
.matrix-section th{font-size:var(--rain-secondary-size);font-weight:700;color:var(--rain-muted)}
.matrix-section td{font-size:12px}
.subview-title{font-size:12px;font-weight:600;color:var(--rain-muted)}
.subview-tab{font-size:12px;font-weight:650;color:var(--rain-muted)}
/* Use full threshold names on wider screens, concise names only on narrow phones. */
.chip-meaning-short{display:none}
/* The 14-day threshold chips display 5 long phrases in narrow columns. */
.outlook-simple .threshold-chip .chip-meaning{font-size:var(--rain-micro-size);font-weight:700}
@media(max-width:355px){
  /* Compact controls still need to fit on one line next to Simple mode. */
  .view-tab{font-size:10.5px;padding:7px 5px}
  .rr-card-tabs button{font-size:10.5px;padding:7px 9px}
  .outlook-simple .threshold-chip .chip-meaning{font-size:9px}
}
'''

# Keep this final style layer identical across the three self-contained forecast
# iframes. Each original component can retain its chart-specific data colors,
# but its chrome, controls, labels and interactions inherit these tokens.
SHARED_FORECAST_DESIGN_CSS = r'''
/* Simpleweather UI system v1: one source of truth for forecast-card chrome.
   Data colors (rainfall tiers and Heat Index severity) deliberately stay local. */
:root{
  --sw-font:'Source Sans 3','Source Sans Pro',sans-serif;
  --sw-ink:#283746;
  --sw-muted:#687585;
  --sw-subtle:#8994A1;
  --sw-date:#737D88;
  --sw-section:#516678;
  --sw-accent:#1786B6;
  --sw-focus:#159BD6;
  --sw-selection-accent:#00BFFF;
  --sw-surface:#FFFFFF;
  --sw-panel:#F7FAFC;
  --sw-hover:#F6F8FB;
  --sw-selected:#F3F8FB;
  --sw-line:#E7EBF0;
  --sw-segment-bg:#EEF1F5;
  --sw-segment-ink:#677586;
  --sw-segment-selected:#1C3044;
  --sw-switch-off:#D5DBE3;
  --sw-switch-on:#25A8DB;
  --sw-card-radius:14px;
  --sw-card-padding:15px;
  --sw-inner-radius:12px;
  --sw-day-radius:8px;
  --sw-tile-radius:8px;
  --sw-control-radius:10px;
  --sw-segment-radius:13px;
  --sw-fast:150ms;
  --sw-view:200ms;
  --sw-expand:260ms;
  /* Backwards-compatible aliases for the rainfall & temperature renderers. */
  --rain-ink:var(--sw-ink);
  --rain-muted:var(--sw-muted);
  --rain-subtle:var(--sw-subtle);
  --rain-date:var(--sw-date);
  --rain-section:var(--sw-section);
  --rain-selected:var(--sw-segment-selected);
  --rain-today:var(--sw-accent);
}
html,body{font-family:var(--sw-font);-webkit-font-smoothing:antialiased}
button,input,summary{font-family:inherit}
/* Frame-level chrome is the same at both sides of the Streamlit columns. */
.rr-root,.shell,.temp-root{
  box-sizing:border-box;
  color:var(--sw-ink);
  background:var(--sw-surface);
  border:1px solid var(--sw-line);
  border-radius:var(--sw-card-radius);
  padding:var(--sw-card-padding);
  box-shadow:none;
}
.rr-card-title,.shell-title h2{
  color:var(--sw-ink);font-size:var(--rain-title-size);
  font-weight:700;line-height:1.3;letter-spacing:-.025em;
}
.rr-section-heading,.smallheading,.section-label{
  color:var(--sw-section);font-size:var(--rain-section-size);font-weight:700;
}
/* Segmented tabs: same radius, typography, fill, and change of state. */
.rr-card-tabs,.view-tabs{
  display:flex;align-items:center;gap:3px;padding:4px;
  border-radius:var(--sw-segment-radius);background:var(--sw-segment-bg);
}
.rr-card-tabs button,.view-tab{
  border:0;border-radius:var(--sw-control-radius);background:transparent;
  color:var(--sw-segment-ink);padding:8px 14px;
  font-size:var(--rain-tabs-size);font-weight:700;line-height:1.2;
  transition:background-color var(--sw-fast) ease,color var(--sw-fast) ease;
}
.rr-card-tabs button[aria-pressed="true"],
.rr-card-tabs button[aria-selected="true"],.view-tab[aria-selected="true"]{
  color:var(--sw-segment-selected);background:var(--sw-surface);
  box-shadow:0 2px 8px #1e3b5020;
}
/* Compact graph/bar/matrix icon controls use the same tab treatment. */
.subview-tabs{
  background:var(--sw-segment-bg);border-radius:var(--sw-segment-radius);
  padding:3px;gap:3px;
}
.subview-tab{
  border-radius:var(--sw-control-radius);color:var(--sw-segment-ink);
  transition:background-color var(--sw-fast) ease,color var(--sw-fast) ease;
}
.subview-tab[aria-pressed="true"]{
  color:var(--sw-segment-selected);background:var(--sw-surface);
  box-shadow:0 2px 8px #1e3b5020;
}
/* The same Simple mode switch, regardless of which iframe owns it. */
.rr-simple-toggle,.simple-toggle{
  display:flex;align-items:center;gap:8px;color:#506070;
  font-size:13px;font-weight:650;cursor:pointer;user-select:none;
}
.rr-simple-toggle input,.simple-toggle input{
  appearance:none;width:37px;height:22px;margin:0;position:relative;
  flex:0 0 auto;border:0;border-radius:99px;background:var(--sw-switch-off);
  cursor:pointer;touch-action:manipulation;
  transition:background-color var(--sw-fast) ease;
}
.rr-simple-toggle input:after,.simple-toggle input:after{
  content:'';position:absolute;left:3px;top:3px;width:16px;height:16px;
  background:white;border-radius:50%;box-shadow:0 1px 3px #0002;
  transition:left var(--sw-fast) ease;
}
.rr-simple-toggle input:checked,.simple-toggle input:checked{background:var(--sw-switch-on)}
.rr-simple-toggle input:checked:after,.simple-toggle input:checked:after{left:18px}
/* Day-row treatment: identical pale selected fill and 2px cyan accent. */
.rr-selectable,.day-bar,.fingerprint,.temp-day-button{
  border-radius:var(--sw-day-radius);
  transition:background-color var(--sw-fast) ease;
}
.rr-selectable:hover,.day-bar:hover,.fingerprint:hover,.temp-day-button:hover{
  background-color:var(--sw-hover);
}
.rr-selectable.is-selected,.rr-selectable.is-collapsing,
.day-bar[aria-pressed="true"],.fingerprint[aria-pressed="true"],
.day-bar.is-collapsing,.fingerprint.is-collapsing,
.temp-day-button[aria-expanded="true"]{
  background:var(--sw-selected);
  box-shadow:inset 2px 0 0 var(--sw-selection-accent);
  border-radius:var(--sw-day-radius) var(--sw-day-radius) 0 0;
}
.temp-day-wrap{border-radius:var(--sw-day-radius)}
/* Same compact, unboxed TODAY marker; matrix dates remain inline. */
.rr-today-tag,.day-label .today-mini,.panel-day .today-mini,.matrix-today{
  color:var(--sw-accent);font-size:var(--rain-today-size);
  font-weight:400;letter-spacing:.025em;line-height:1.15;
}
.rr-day-weekday,.day-label .day-short strong{
  color:var(--sw-ink);font-weight:700;font-size:var(--rain-day-size);
}
.rr-day-calendar,.day-label .date,.day-label .date-inline{
  color:var(--sw-date);font-weight:400;font-size:var(--rain-date-size);
}
/* Consistent surfaces and corners, without touching colored forecast tiles. */
.part,.threshold-section,.temp-chart-shell{border-radius:var(--sw-inner-radius);border-color:var(--sw-line)}
.rr-quantile,.amount-small,.chance-tile,.temp-summary-tile,.temp-hour{
  border-radius:var(--sw-tile-radius);
}
.rr-detail,.panel,.temp-detail-body{background:var(--sw-panel)}
.rr-detail-content,.panel-slide-content,.temp-detail-body{
  background:var(--sw-panel);
  border-color:var(--sw-line);
  border-radius:0 0 var(--sw-day-radius) var(--sw-day-radius);
}
.threshold-chip{border-radius:var(--sw-control-radius)}
/* Keyboard users see the same focus style in all embedded cards. */
.rr-root button:focus-visible,.rr-root input:focus-visible,.rr-root summary:focus-visible,
.shell button:focus-visible,.shell input:focus-visible,.shell summary:focus-visible,
.temp-root button:focus-visible,.temp-root input:focus-visible,.temp-root summary:focus-visible{
  outline:2px solid var(--sw-focus);outline-offset:2px;
}
/* Shared motion vocabulary; never animate iframe height continuously. */
@keyframes rrFadeIn{from{opacity:0}to{opacity:1}}
@keyframes rrFadeOut{from{opacity:1}to{opacity:0}}
.rr-variant:not([hidden]){animation-duration:var(--sw-view)}
.rr-variant.is-closing{animation-duration:var(--sw-view)}
.panel{transition-duration:var(--sw-expand)}
.rr-detail{transition-duration:var(--sw-expand)}
.temp-detail.is-open{animation-duration:var(--sw-expand)}
.temp-chevron{transition-duration:var(--sw-expand)}
@media(max-width:720px){
  :root{--sw-card-radius:10px;--sw-card-padding:12px 9px}
  .rr-card-tabs button{padding:7px 10px}
  .view-tab{padding:7px 7px}
}
/* The Simple threshold selector uses the same five columns and dot colors
   as numerical mode at every viewport width. Stack the dot above text on
   very narrow screens so its column width is not lost to the dot itself. */
@media(max-width:430px){
  .outlook-simple .chips{grid-template-columns:repeat(5,minmax(0,1fr));gap:3px}
  .outlook-simple .threshold-chip{height:88px;min-height:88px;max-height:88px}
  .outlook-simple .threshold-chip .chip-meaning{
    flex-direction:column;align-items:center;justify-content:center;gap:4px;
  }
  .outlook-simple .threshold-chip .simple-swatch{
    display:inline-block;align-self:center;flex:none;width:7px;height:7px;
  }
  .outlook-simple .threshold-chip .chip-meaning,
  .outlook-simple .threshold-chip .chip-meaning-level,
  .outlook-simple .threshold-chip .chip-meaning-rest{
    font-size:clamp(9px,2.85vw,12px);
    line-height:1.12;letter-spacing:-.015em;
    white-space:normal;overflow-wrap:normal;
  }
  .outlook-simple .threshold-chip .chip-meaning-text>span{white-space:normal}
  .outlook-simple .chip-meaning-full{display:none}
  .outlook-simple .chip-meaning-short{display:flex}
}
@media(max-width:355px){
  .rr-card-tabs button{font-size:11px;padding:7px 7px}
  .view-tab{font-size:11px;padding:7px 5px}
  .rr-simple-toggle,.simple-toggle{font-size:11px;gap:4px}
  .rr-simple-toggle input,.simple-toggle input{width:33px;height:20px}
  .rr-simple-toggle input:after,.simple-toggle input:after{width:14px;height:14px}
  .rr-simple-toggle input:checked:after,.simple-toggle input:checked:after{left:16px}
}
@media(prefers-reduced-motion:reduce){
  .rr-root *, .shell *, .temp-root *{
    animation:none!important;transition:none!important;scroll-behavior:auto!important;
  }
}
'''

FONT_STACK = "'Source Sans 3', 'Source Sans Pro', sans-serif"

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
    return "#283746" if blended_luminance > 0.5 else "#ffffff"

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

def render_14day_outlook_html(result: dict, location_name: str) -> str:
    """Interactive 14-day outlook: linked probability ribbon, daily bars and rainfall fingerprints.

    All views use the SAME ECMWF 24-hour 08:00-to-08:00 PHT windows.
    Missing values stay missing; rainfall-range fingerprints require five valid,
    monotonically nonincreasing exceedance probabilities.
    """
    today_ph = datetime.now(PH_TZ).date()
    windows = [
        w for w in result.get("windows", [])
        if w["start_utc"].astimezone(PH_TZ).date() >= today_ph
    ]
    if not windows:
        # Show the latest available 14-day period if the stored run is stale;
        # the existing status footer and ECMWF run metadata give its age.
        windows = result.get("windows", [])
    windows = windows[:14]
    records = []
    for w in windows:
        values = []
        for threshold in AVAILABLE_THRESHOLDS_MM:
            candidate = result.get("data", {}).get(threshold, {}).get(w["label"])
            try:
                candidate = float(candidate) if candidate is not None else None
            except (ValueError, TypeError):
                candidate = None
            values.append(round(candidate, 4) if candidate is not None and math.isfinite(candidate) else None)
        records.append({
            "date": w["start_utc"].astimezone(PH_TZ).date().isoformat(),
            "start": w["start_utc"].isoformat(),
            "end": w["end_utc"].isoformat(),
            "values": values,
        })
    if not records:
        return ""
    # Escape angle brackets in JSON so data cannot prematurely close <script>.
    encoded = json.dumps(records, ensure_ascii=True).replace("<", "\\u003c")
    palette = [THRESHOLD_COLORS[mm] for mm in AVAILABLE_THRESHOLDS_MM]
    return (
        _FOURTEEN_DAY_TEMPLATE
        .replace("__RECORDS__", encoded)
        .replace("__LOCATION__", escape(location_name))
        .replace("__THRESHOLD_PALETTE__", json.dumps(palette))
        .replace("__SHARED_RAIN_TYPOGRAPHY__", SHARED_RAIN_TYPOGRAPHY_CSS + "\n" + SHARED_FORECAST_DESIGN_CSS)
    )

_FOURTEEN_DAY_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Simpleweather | Rainfall Outlook</title>

<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
:root{--ink:#283746;--muted:#687585;--muted2:#8994a1;--line:#e7ebf0;--surface:#fff;--bg:#f5f7fa;--blue:#00bfff;--shadow:0 10px 34px rgba(23,40,63,.035);--radius:20px}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:'Source Sans 3','Source Sans Pro',sans-serif;font-size:15px;-webkit-font-smoothing:antialiased}button,input{font:inherit}button{cursor:pointer}button:focus-visible,summary:focus-visible,svg:focus-visible{outline:3px solid rgba(0,156,220,.46);outline-offset:2px}button[aria-pressed="true"]{outline-offset:0}main{max-width:1150px;margin:auto;padding:30px 26px 70px}.eyebrow{font-size:12px;letter-spacing:.11em;color:#657991;font-weight:800;text-transform:uppercase}.page-heading{display:flex;justify-content:space-between;align-items:flex-end;gap:15px;margin-bottom:22px}.page-heading h1{font-weight:740;line-height:1.08;font-size:clamp(27px,3.5vw,38px);letter-spacing:-.04em;margin:6px 0 8px}.subtle{color:var(--muted);margin:0}.demo-pill{font-size:12px;letter-spacing:.012em;color:#5b6776;background:#e8edf3;border:1px solid #dde3ea;border-radius:50px;padding:7px 12px;white-space:nowrap}.shell{background:var(--surface);border:1px solid #e9edf2;border-radius:14px;padding:24px;box-shadow:var(--shadow)}.shell-title{display:flex;align-items:start;justify-content:space-between;gap:16px;margin-bottom:16px}.shell-title h2{font-size:22px;margin:2px 0 3px;font-weight:750;letter-spacing:-.03em}.shell-title p{margin:0;color:var(--muted);font-size:14px}.rain-icon{width:40px;height:40px;background:#e9f6fb;color:#149fce;border-radius:14px;display:grid;place-items:center;flex:none}.top-controls{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin:17px 0 18px}.view-tabs{display:flex;align-items:center;padding:4px;border-radius:13px;background:#eef1f5;gap:3px;width:max-content;max-width:100%}.view-tab{background:transparent;color:#677586;border:none;border-radius:10px;padding:8px 16px;font-size:13px;font-weight:700;white-space:nowrap;transition:background .15s}.view-tab[aria-selected="true"]{background:#fff;color:#1c3044;box-shadow:0 2px 8px #1e3b5020}.simple-toggle{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none;color:#506070;font-size:13px;font-weight:650}.simple-toggle input{appearance:none;position:relative;width:37px;height:22px;border-radius:99px;background:#d5dbe3;margin:0;transition:background .15s}.simple-toggle input:after{content:"";position:absolute;width:16px;height:16px;top:3px;left:3px;border-radius:50%;background:white;box-shadow:0 1px 3px #0002;transition:left .16s}.simple-toggle input:checked{background:#25a8db}.simple-toggle input:checked:after{left:18px}.simple-toggle input:focus-visible{outline:3px solid #7bd2ed;outline-offset:3px}.section-label{font-weight:750;font-size:14px;letter-spacing:-.01em}.kicker{font-size:12px;color:var(--muted);line-height:1.45}.part{border:1px solid var(--line);border-radius:17px;padding:17px 18px;margin-top:16px}.part-heading{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 12px}.threshold-section{background:#fbfcfe;border:1px solid #eef1f6;padding:15px 16px;border-radius:15px;margin-bottom:16px}.question{font-size:13px;color:#596a7b;font-weight:680;margin-bottom:11px}.chips{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}.threshold-chip{border:1px solid #e2e6eb;color:#526375;background:#fff;border-radius:10px;min-height:40px;display:flex;justify-content:center;align-items:center;gap:6px;padding:4px 5px;font-size:13px;font-weight:750;white-space:nowrap;transition:background .15s,border .15s}.threshold-chip .swatch{height:8px;width:8px;border-radius:50%;background:var(--chip);flex:none}.threshold-chip[aria-pressed="true"]{border:1.5px solid var(--chip-stroke);background:var(--chip-bg);color:#27394b;box-shadow:0 2px 5px #5263750b}.context-line{font-size:13px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:var(--muted);margin-top:11px}.context-line strong{color:#36495b}.graph-wrap{padding:4px 0 0}.graph-wrap svg{display:block;max-width:100%;width:100%;height:auto;touch-action:pan-y;cursor:crosshair}.legend{display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:8px 18px;margin-top:8px}.legend-pill{font-size:11.5px;color:#586679;display:flex;align-items:center;gap:5px}.legend-mark{height:9px;width:17px;border-radius:3px;display:inline-block}.selected-point-caption{display:none;color:#4b5968;margin-top:5px}.bars-section .part-heading{margin-bottom:10px}.bar-header{display:grid;grid-template-columns:102px minmax(0,1fr) 95px;align-items:end;gap:14px;color:#7a8796;font-size:11px;font-weight:700;margin:6px 8px}.bar-day-list{display:flex;flex-direction:column;gap:1px}.day-bar{width:100%;border:0;background:transparent;display:grid;grid-template-columns:102px minmax(0,1fr) 95px;gap:14px;text-align:left;align-items:center;min-height:40px;border-radius:10px;padding:5px 8px;color:var(--ink);transition:background .15s}.day-bar:hover{background:#f6f8fb}.day-bar[aria-pressed="true"]{background:#edf6fb;box-shadow:inset 3px 0 0 #31ace0}.day-label{display:flex;align-items:baseline;gap:5px;min-width:0}.day-label .weekday{font-size:13px;font-weight:650}.day-label .date{font-size:12px;color:var(--muted)}.day-label .today-mini{font-size:9px;letter-spacing:.045em;color:#228db9;font-weight:800}.track{width:100%;height:12px;border-radius:99px;overflow:hidden;background:#e9edf2;position:relative}.progress{display:block;height:100%;width:0;background:var(--bar);border-radius:99px;transition:width .24s ease,background .22s ease}.probability{font-size:14px;font-weight:750;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.probability.simple{font-size:11.5px;font-weight:680;white-space:normal;line-height:1.1}.panel{border:1px solid #e1e9f0;border-radius:16px;background:#f9fbfd;padding:16px 17px;margin-top:16px}.panel[hidden]{display:none}.panel-head{display:flex;gap:20px;justify-content:space-between;align-items:start;margin-bottom:9px}.panel-head h3{font-size:17px;margin:0;letter-spacing:-.02em}.panel-head .window{font-size:12px;color:var(--muted);margin-top:3px}.panel-head .tap-hint{font-size:11px;white-space:nowrap;color:#7c8995}.smallheading{font-size:12px;color:#516678;margin:14px 0 9px;font-weight:750}.five-chances{display:grid;grid-template-columns:repeat(5,1fr);gap:7px}.chance-tile{border-radius:10px;overflow:hidden;border:1px solid #e7eaf0;min-width:0;background:white}.chance-tile .amount{background:var(--soft);padding:6px 8px;font-size:11px;font-weight:750;color:#384657}.chance-tile .value{background:var(--tier);min-height:45px;display:flex;flex-direction:column;align-items:start;justify-content:center;padding:6px 8px;color:var(--tier-ink);font-size:16px;font-weight:780}.chance-tile .descriptor{font-size:10px;font-weight:500;color:inherit;opacity:.77}.amount-distribution{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.amount-small{background:#fff;border:1px solid #e7eaf0;border-radius:10px;padding:9px 10px}.amount-small .amount-swatch{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}.amount-small .low{font-size:11px;color:#657383}.amount-small .big{font-size:17px;font-weight:750;margin-top:2px}.footnote{padding:4px 1px 0;font-size:12px;color:#697789;line-height:1.6}.footnote details{margin-top:9px}.footnote summary{cursor:pointer;font-weight:700;color:#576b80}.exact{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}.exact details>summary{cursor:pointer;font-size:13px;color:#50667c;font-weight:750}.exact .table-scroll{overflow:auto;margin-top:12px}.exact table{width:100%;border-collapse:separate;border-spacing:3px;text-align:center;font-size:12px;min-width:630px}.exact th{font-size:11px;text-align:center;color:#637285;white-space:nowrap;padding:6px}.exact th:first-child,.exact td:first-child{text-align:left;white-space:nowrap}.exact td{border-radius:6px;padding:7px 5px;font-weight:700}.fingerprint-list{display:flex;flex-direction:column;gap:2px}.fingerprint{display:grid;grid-template-columns:102px minmax(0,1fr) 28px;gap:14px;border:none;border-radius:10px;padding:9px 8px;background:transparent;align-items:center;text-align:left;color:var(--ink);min-height:41px}.fingerprint:hover{background:#f6f8fb}.fingerprint[aria-pressed="true"]{background:#edf6fb;box-shadow:inset 3px 0 0 #31ace0}.finger-track{height:18px;border-radius:7px;overflow:hidden;display:flex;background:#e9edf2}.finger-segment{height:100%;transition:flex .22s}.legend-grid{display:flex;flex-wrap:wrap;gap:10px 13px;margin-top:15px}.finger-tip{font-size:12px;color:#718194}.card-divider{border-top:1px solid var(--line);margin:20px 0 14px}.empty{font-size:12px;color:#718194;font-style:italic}.hidden{display:none!important}
@media (max-width:720px){main{padding:17px 10px 45px}.page-heading{align-items:flex-start}.page-heading h1{font-size:29px}.shell{padding:17px 12px;border-radius:18px}.part{padding:13px 10px;border-radius:13px}.top-controls{align-items:stretch}.view-tabs{width:100%}.view-tab{flex:1;font-size:12px;padding:9px 8px}.simple-toggle{margin-left:auto}.threshold-section{padding:12px 10px}.chips{gap:5px}.threshold-chip{font-size:12px;gap:4px;padding:4px 2px;min-height:37px}.threshold-chip .swatch{width:6px;height:6px}.context-line{font-size:11.5px}.bar-header{grid-template-columns:70px minmax(0,1fr) 72px;gap:8px;margin:5px 4px}.day-bar{grid-template-columns:70px minmax(0,1fr) 72px;gap:8px;padding:5px 4px;min-height:39px}.day-label{flex-wrap:wrap;column-gap:4px;row-gap:0}.day-label .weekday{font-size:12px}.day-label .date{font-size:11px}.probability{font-size:13px}.probability.simple{font-size:10px}.track{height:11px}.fingerprint{grid-template-columns:70px minmax(0,1fr) 14px;gap:8px;padding:6px 4px}.finger-track{height:17px}.five-chances{grid-template-columns:repeat(5,minmax(0,1fr));gap:4px}.chance-tile .amount{font-size:10px;padding:5px}.chance-tile .value{font-size:13px;padding:7px 5px}.chance-tile .descriptor{font-size:9px}.amount-distribution{grid-template-columns:repeat(2,minmax(0,1fr))}.panel{padding:12px 10px}.panel-head h3{font-size:15px}.panel-head .tap-hint{display:none}.demo-pill{display:none}.shell-title h2{font-size:20px}.graph-wrap svg{min-height:0}}
@media (prefers-reduced-motion:reduce){*,*:before,*:after{transition:none!important;scroll-behavior:auto!important}}

html,body{margin:0;padding:0;min-height:0;height:auto;overflow-x:hidden!important;background:#fff}
main{padding:0!important;max-width:100%;margin:0}
.shell{border:1px solid #e8edf2;border-radius:15px;padding:14px 12px;box-shadow:none;background:#fff}
.shell-title{margin-bottom:8px}.shell-title h2{font-size:18px}.shell-title p{font-size:12px}
.part{padding:13px 12px;margin-top:12px;border-radius:13px}.threshold-section{margin-bottom:10px}
.top-controls{margin:10px 0 15px}.panel{margin-top:12px}
.graph-scroll{overflow-x:auto;overscroll-behavior-x:contain}
.graph-scroll svg{display:block;width:100%;min-width:545px;min-height:0;height:auto;cursor:crosshair;touch-action:pan-y}
.day-label .today-mini{display:none}.day-label.is-today .today-mini{display:inline}
.day-bar .probability.simple{font-size:11px;white-space:nowrap}
.chance-rows{display:flex;flex-direction:column;gap:5px}
.chance-row{display:flex;overflow:hidden;border-radius:9px;border:1px solid #e7ebf0;min-height:42px}
.chance-left{width:35%;display:flex;align-items:center;gap:6px;padding:7px 9px;min-width:0}
.chance-left strong{font-size:12px;white-space:nowrap}.chance-left span{font-size:10.5px;line-height:1.2;opacity:.75}
.chance-right{flex:1;display:flex;justify-content:flex-end;align-items:center;gap:12px;padding:7px 10px;min-width:0}
.chance-right strong{font-size:15px;font-variant-numeric:tabular-nums}
.chance-right span{min-width:60px;text-align:right;font-size:11px;opacity:.9}
.panel-head .window{line-height:1.45}.panel-head .tap-hint{white-space:normal;text-align:right}
.fingerprint.missing .finger-track{background:#f0f1f4;display:grid;place-items:center;color:#8793a0;font-size:11px}
.panel[hidden], [hidden]{display:none!important}
@media(max-width:720px){
 .shell{padding:12px 9px;border-radius:13px}.shell-title{gap:7px}.shell-title h2{font-size:16px}
 .shell-title p{font-size:11px}.rain-icon{width:32px;height:32px;border-radius:10px}
 .part{padding:11px 8px}.graph-scroll svg{width:545px}
 .chance-left{width:40%;gap:4px;padding:6px}.chance-left strong{font-size:11px}.chance-left span{font-size:10px}
 .chance-right{gap:5px;padding:6px}.chance-right strong{font-size:14px}.chance-right span{min-width:52px;font-size:10px}
 .probability.simple{font-size:10px!important}.bar-header,.day-bar{grid-template-columns:70px minmax(0,1fr) 82px}
 .part-heading{align-items:start}.part-heading .kicker{font-size:10px}
 .panel-head .tap-hint{display:none}
}
@media(max-width:355px){.chance-left span{font-size:9px}.bar-header,.day-bar{grid-template-columns:66px minmax(0,1fr) 82px;gap:5px}}
/* Match the calm type, spacing, and selection treatment of the six-day card. */
.shell{border-radius:14px;padding:15px 15px 17px;box-shadow:none}
.shell-title h2{font-size:18px;font-weight:720}
.shell-title p{font-size:12px}
.rain-icon{background:#eef6f9;color:#4a99ba}
.part{border-radius:11px;padding:13px 13px;margin-top:10px}
.threshold-section{border-radius:10px;margin-bottom:11px}
.day-label{display:flex;flex-direction:column;align-items:flex-start;justify-content:center;gap:1px}
.day-label .weekday{line-height:1.13;font-weight:700}
.day-label .date{font-size:11px;color:#737d88;line-height:1.13}
.day-label .today-mini{display:none;color:#1786b6;font-size:10px;font-weight:400;letter-spacing:.025em;line-height:1.15}
.day-label.is-today .today-mini{display:block}
.panel-day .today-mini{display:none;color:#1786b6;font-size:10px;font-weight:400;letter-spacing:.025em}
.panel-day .today-mini.show-today{display:block}
.panel-day h3{line-height:1.23}
.panel-head{align-items:center}
.day-bar,.fingerprint{border-radius:7px}
.day-bar[aria-pressed="true"],.fingerprint[aria-pressed="true"],.day-bar.is-collapsing,.fingerprint.is-collapsing{background:#f3f8fb;box-shadow:inset 2px 0 #00bfff;border-radius:7px 7px 0 0}
.subview-switch{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:13px 0 3px;padding:0 2px}
.subview-title{font-size:12px;color:#6d7b89;font-weight:600}
.subview-tabs{display:flex;gap:3px;background:#eef1f5;padding:3px;border-radius:9px}
.subview-tab{border:0;border-radius:7px;background:transparent;color:#637285;padding:6px 15px;font-size:12px;font-weight:650;white-space:nowrap}
.subview-tab[aria-pressed="true"]{background:#fff;color:#1d2530;box-shadow:0 1px 4px #1e3b5020}
.part[hidden],#outlook-exact[hidden]{display:none!important}
@media(max-width:720px){
 .shell{padding:12px 9px;border-radius:10px}
 .shell-title h2{font-size:16px}
 .subview-switch{gap:7px}
 .subview-tab{padding:6px 11px}
 .bar-header,.day-bar{grid-template-columns:69px minmax(0,1fr) 82px}
 .day-bar,.fingerprint{min-height:47px}
}
.threshold-chip{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px;min-height:51px;padding:5px 2px;line-height:1.12}
.threshold-chip .chip-top{display:flex;align-items:center;justify-content:center;gap:5px;white-space:nowrap}
.threshold-chip .chip-meaning{font-size:10px;font-weight:450;color:#697585;white-space:nowrap;letter-spacing:-.02em}
.threshold-chip[aria-pressed="true"] .chip-meaning{color:#425265}
@media(max-width:720px){.threshold-chip{min-height:49px;padding:4px 1px}.threshold-chip .chip-top{gap:3px}.threshold-chip .chip-meaning{font-size:9px}}
@media(max-width:355px){.threshold-chip .chip-meaning{font-size:8px}.threshold-chip .chip-top{gap:2px}}

/* Shared six-day visual language for the 14-day interface. */
#trend-svg:focus:not(:focus-visible){outline:none}
.day-label{gap:1px}.day-label .day-short{font-size:12px;font-weight:400;white-space:nowrap;line-height:1.15;color:#25303a}.day-label .day-short strong{font-weight:700}
.day-label .today-mini{color:#1786b6;font-size:10px;font-weight:400;letter-spacing:.025em}
.panel-head h3{font-size:16px;font-weight:700;letter-spacing:0;color:#1d2832}
.panel-head .window{font-size:11px;color:#737d88}
.chance-rows{gap:5px}.chance-row{border:0;border-radius:6px;min-height:0}
.chance-left{width:35%;gap:6px;padding:7px 7px}.chance-left strong{font-size:12px}.chance-left span{font-size:10px;opacity:.8}
.chance-right{gap:12px;padding:7px 10px}.chance-right strong{font-size:15px;font-weight:750}.chance-right span{font-size:10px;min-width:58px;opacity:.9}
.matrix-section .table-scroll{width:100%;overflow-x:auto}.matrix-section table{width:100%;min-width:600px;border-collapse:separate;border-spacing:3px;text-align:center;font-size:12px}
.matrix-section th{font-size:11px;color:#637285;padding:7px}.matrix-section td{padding:7px 5px;font-weight:700;border-radius:6px}
.matrix-section th:first-child,.matrix-section td:first-child{text-align:left;white-space:nowrap}
.threshold-chip{touch-action:manipulation}.threshold-chip .chip-meaning{white-space:normal;text-align:center;line-height:1.1}
.outlook-simple .threshold-chip{min-height:62px}.outlook-simple .threshold-chip .chip-top{display:none}
.outlook-simple .threshold-chip .chip-meaning{font-size:12px;font-weight:700;color:#344459;letter-spacing:-.015em}
.legend-button{border:0;background:transparent;padding:4px 5px;border-radius:5px;cursor:pointer;font-size:11.5px}
.legend-button[aria-pressed="true"]{background:#edf6fb}
@media(max-width:720px){
 .day-label .day-short{font-size:10.5px;letter-spacing:-.04em}
 .bar-header,.day-bar{grid-template-columns:85px minmax(0,1fr) 82px;gap:5px}
 .fingerprint{grid-template-columns:85px minmax(0,1fr) 16px;gap:5px}
 .chance-left{width:40%;gap:3px;padding:7px 4px}
 .chance-right{padding:7px 8px;gap:6px}
 .outlook-simple .threshold-chip{min-height:67px}
 .outlook-simple .threshold-chip .chip-meaning{font-size:10.5px}
}
@media(max-width:355px){.day-label .day-short{font-size:10px}.bar-header,.day-bar{grid-template-columns:81px minmax(0,1fr) 80px}.fingerprint{grid-template-columns:81px minmax(0,1fr) 12px}.outlook-simple .threshold-chip .chip-meaning{font-size:9.5px}}
/* Identical chip geometry in numerical and Simple modes, with a colored dot in both. */
.threshold-chip,.outlook-simple .threshold-chip{height:76px;min-height:76px;max-height:76px}
.threshold-chip .chip-meaning{display:flex;align-items:center;justify-content:center;gap:4px}
.threshold-chip .simple-swatch{display:none}
.outlook-simple .threshold-chip .chip-top{display:none}
.outlook-simple .threshold-chip .chip-meaning{font-weight:700;font-size:11px;line-height:1.15;white-space:normal}
.outlook-simple .threshold-chip .simple-swatch{display:inline-block;flex:none;width:7px;height:7px}
.matrix-section td:first-child{font-weight:400}
.matrix-section td:first-child strong{font-weight:700}
@media(max-width:720px){
.threshold-chip,.outlook-simple .threshold-chip{height:88px;min-height:88px;max-height:88px}
.outlook-simple .threshold-chip .chip-meaning{font-size:9.5px;gap:3px}
.outlook-simple .threshold-chip .simple-swatch{width:6px;height:6px}
}
@media(max-width:355px){
.threshold-chip,.outlook-simple .threshold-chip{height:96px;min-height:96px;max-height:96px}
.outlook-simple .threshold-chip .chip-meaning{font-size:9px}
}
/* Minimal, monochrome icons; keep text only as accessible labels/tooltips. */
.subview-tab{display:inline-flex;align-items:center;justify-content:center;width:40px;height:34px;padding:0}
.subview-tab svg{display:block;width:19px;height:19px;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;fill:none}
@media(max-width:720px){.subview-tab{width:38px;height:34px;padding:0}}
/* All forecast appearance changes fade in AND out over 200 ms. */
@keyframes forecastFadeIn{from{opacity:0}to{opacity:1}}
@keyframes forecastFadeOut{from{opacity:1}to{opacity:0}}
.part:not([hidden]),[role="tabpanel"]:not([hidden]),.threshold-section:not([hidden]){animation:forecastFadeIn .2s ease-out both}
.is-fading-out{animation:forecastFadeOut .2s ease-in both!important;pointer-events:none}
@media(prefers-reduced-motion:reduce){.panel,.part,[role="tabpanel"],.threshold-section,.is-fading-out{animation:none!important}}
/* Day details open directly from the selected day row; animate height, not opacity. */
.panel{display:grid;grid-template-rows:0fr;border:0;border-radius:0;
  background:#f7fafd;padding:0;margin:0;overflow:hidden;
  transition:grid-template-rows .26s ease}
.panel.is-expanded{grid-template-rows:1fr}
.panel-slide-wrap{min-height:0;overflow:hidden}
.panel-slide-content{padding:16px 17px;border:1px solid #e1e9f0;border-radius:12px;background:#f7fafd}
.inline-day-panel{width:100%;margin:0;border-radius:0 0 10px 10px}
.inline-day-panel .panel-slide-content{border-radius:0 0 10px 10px;border-top:0}
@media(max-width:720px){.panel-slide-content{padding:12px 10px}}
@media(prefers-reduced-motion:reduce){.panel{transition:none!important}}
/* Final consistency adjustments with the six-day card. */
.chips{grid-template-columns:repeat(5,minmax(0,1fr))}
.threshold-chip{min-width:0;position:relative}
.outlook-simple .threshold-chip .chip-meaning{display:flex;flex-direction:row;align-items:center;justify-content:center;gap:5px}
.outlook-simple .threshold-chip .simple-swatch{position:static;inset:auto;align-self:center;flex:0 0 7px;margin:0}
html,body{background:transparent!important}
main{border-radius:14px;overflow:hidden}
.shell{border-radius:14px;overflow:hidden}
.day-label .day-short{font-weight:400}
.day-label .day-short strong{font-weight:700}
.outlook-simple .threshold-chip .chip-meaning{font-size:12.5px;line-height:1.15;gap:5px}
.chip-meaning-text{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;min-width:0}
.chip-meaning-text>span{display:block;white-space:nowrap}
@media(max-width:720px){
 main,.shell{border-radius:10px}
 .outlook-simple .threshold-chip .chip-meaning{font-size:10px;letter-spacing:-.035em;gap:0}
 .chip-meaning-text{gap:2px}
}
@media(max-width:355px){.threshold-section{padding-left:7px;padding-right:7px}.chips{gap:3px}.outlook-simple .threshold-chip .chip-meaning{font-size:9.3px;letter-spacing:-.04em}}
/* Streamlit handles the two-column layout. Remove the legacy standalone
   14-day page gutters so both embedded card borders start at the same Y. */
html,body{margin:0!important;padding:0!important;overflow-x:hidden}
main{max-width:none;margin:0;padding:0!important;overflow:visible;border-radius:0}
.shell{margin:0;padding-top:13px}
@media(max-width:720px){.shell{padding-top:10px}}
/* Keep view selection compact and side-by-side with Simple mode on phones. */
@media(max-width:720px){
 .top-controls{align-items:center;flex-wrap:nowrap;gap:6px}
 .view-tabs{width:max-content;max-width:calc(100% - 104px);flex:0 1 auto;padding:3px;gap:2px}
 .view-tab{flex:0 0 auto;padding:7px 7px;font-size:12px;line-height:1.2}
 .simple-toggle{flex:0 0 auto;margin-left:auto;gap:5px;font-size:13px;white-space:nowrap}
}
@media(max-width:355px){
 .top-controls{gap:4px}
 .view-tabs{max-width:calc(100% - 98px)}
 .view-tab{padding:7px 5px;font-size:10.5px}
 .simple-toggle{gap:4px;font-size:10.5px}
 .simple-toggle input{width:33px;height:20px}
 .simple-toggle input:after{width:14px;height:14px}
 .simple-toggle input:checked:after{left:16px}
}
__SHARED_RAIN_TYPOGRAPHY__
/* In Simple mode, threshold text uses the same font size and color as the mm option. */
.outlook-simple .threshold-chip .chip-meaning,
.outlook-simple .threshold-chip .chip-meaning-level,
.outlook-simple .threshold-chip .chip-meaning-rest {
  font-size:var(--rain-descriptor-size);
  color:inherit;
  font-weight:700;
  letter-spacing:normal;
  line-height:1.12;
}
.outlook-simple .threshold-chip .chip-meaning {gap:3px}
.outlook-simple .threshold-chip .chip-meaning-text {gap:0}
@media(max-width:720px){
  .outlook-simple .threshold-chip .chip-meaning,
  .outlook-simple .threshold-chip .chip-meaning-level,
  .outlook-simple .threshold-chip .chip-meaning-rest {
    font-size:12px;
    letter-spacing:-.02em;
  }
  .outlook-simple .threshold-chip .simple-swatch{display:inline-block}
}
@media(max-width:355px){
  .outlook-simple .threshold-chip .chip-meaning,
  .outlook-simple .threshold-chip .chip-meaning-level,
  .outlook-simple .threshold-chip .chip-meaning-rest {
    font-size:var(--rain-descriptor-size);
    letter-spacing:-.035em;
  }
}
</style>
<meta name="color-scheme" content="light"></head>
<body>
<main>
<div class="shell">
    <div class="shell-title"><div><h2>14-day rainfall outlook for __LOCATION__</h2></div></div>
    <div class="top-controls">
      <div class="view-tabs" role="tablist" aria-label="14-day forecast view">
        <button class="view-tab" id="tab-chances" role="tab" aria-controls="view-chances" aria-selected="true" data-view="chances">Chance of rain</button>
        <button class="view-tab" id="tab-amounts" role="tab" aria-controls="view-amounts" aria-selected="false" data-view="amounts">Possible totals</button>
      </div>
      <label class="simple-toggle" for="outlook-simple"><input type="checkbox" id="outlook-simple"> Simple mode</label>
    </div>
    <div id="view-chances" role="tabpanel" aria-labelledby="tab-chances">
      <div class="subview-switch" role="group" aria-label="Explore chance of rain">
        <span class="subview-title">Explore chance of rain</span>
        <div class="subview-tabs">
          <button type="button" class="subview-tab" data-subview="graph" aria-pressed="true" aria-label="Graph" title="Graph"><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M3 20V4M3 20h18M5 16l5-6 4 3 6-8"/></svg></button>
          <button type="button" class="subview-tab" data-subview="bars" aria-pressed="false" aria-label="Bars" title="Bars"><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="3" y="3" width="18" height="4" rx="1"/><rect x="3" y="10" width="13" height="4" rx="1"/><rect x="3" y="17" width="8" height="4" rx="1"/></svg></button>
          <button type="button" class="subview-tab" data-subview="matrix" aria-pressed="false" aria-label="Matrix" title="Matrix"><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg></button>
        </div>
      </div>
      <div class="threshold-section" id="outlook-thresholds">
        <div class="question">Choose the amount of rain you want to check</div>
        <div class="chips" role="group" aria-label="Select rainfall threshold" id="threshold-chips"></div>
      </div>
      <div class="part graph-section" id="outlook-graph">
        <div class="part-heading"><div class="section-label">Rainfall chances over time</div><span class="kicker" id="graph-selected-label">Tap a day for details</span></div>
        <div class="graph-scroll"><svg id="trend-svg" viewBox="0 0 900 305" role="img" tabindex="0" aria-label="Interactive daily graph showing chances of different rainfall thresholds. Tap or click a date to select it."></svg></div>
        <div class="legend" id="graph-legend"></div>
      </div>
      <div class="part bars-section" id="outlook-bars" hidden>
        <div class="part-heading"><div class="section-label">Day-by-day chances</div><span id="bar-count" class="kicker">Daily outlook</span></div>
        <div class="bar-header"><span>Day</span><span>Chance of at least the chosen amount</span><span style="text-align:right">Chance</span></div>
        <div class="bar-day-list" id="bar-list"></div>
      </div>
      <div class="part matrix-section" id="outlook-matrix" hidden>
        <div class="part-heading"><div class="section-label">Exact rainfall probabilities</div></div>
        <div class="table-scroll" id="exact-table"></div>
      </div>
    </div>
    <div id="view-amounts" role="tabpanel" aria-labelledby="tab-amounts" hidden>
      <div class="part" style="margin-top:4px">
        <div class="part-heading"><div class="section-label">Possible rainfall totals for the day</div></div>
        <div class="fingerprint-list" id="fingerprint-list"></div>
        <div class="legend-grid" id="fingerprint-legend"></div>
        <div class="finger-tip" style="margin-top:9px">Longer segments mean more likely rainfall amounts.</div>
      </div>
    </div>
    <div id="day-panel-anchor"></div>
    <div class="panel" id="day-panel" aria-live="polite" hidden aria-hidden="true" inert></div>
    <div class="footnote"><span>Each day covers 8 AM–8 AM Philippine time.</span>
      <details><summary>How to read this forecast</summary><div style="padding-top:6px">Graph and Bars show the chance of reaching at least the chosen rainfall amount. Possible totals shows the chance of each separate rainfall range. Tap a day to see the full breakdown. Missing data is not the same as 0%. Rainfall chances do not directly measure flood risk.</div></details>
    </div>
  </div>
</main>
<script>
(() => {
  'use strict';
  const records = __RECORDS__;
  let simple = false; // Independent of the six-day card; off by default.
  const thresholds = [1,5,20,50,100];
  const sourceColors = __THRESHOLD_PALETTE__;
  // Identical base colors in the options, graph lines, legend and bars.
  const colors = sourceColors;
  const amountWords = ['very light','light','moderate','heavy','intense'];
  const shortAmountWords = ['v. light','light','mod.','heavy','intense'];
  const amountButtonLines = [
    ['very light','rain or more'], ['light rain','or more'],
    ['moderate','rain or more'], ['heavy rain','or more'],
    ['intense rain','or more']
  ];
  const ranges = ['<1 mm','1 to <5 mm','5 to <20 mm','20 to <50 mm','50 to <100 mm','100 mm or more'];
  const rangeColors = ['#CAD7E1', ...sourceColors.map(c=>tint(c,.65))];
  const byId = id => document.getElementById(id);
  // Keep this reference even when a row list is rebuilt with innerHTML.
  const dayPanel = byId('day-panel');
  const fadeMs = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 200;
  const fadeTimers = new WeakMap();
  const panelCloseMs = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 260;
  let panelCloseTimer = null;
  let closingRow = null;
  function clearClosingRow(){if(closingRow){closingRow.classList.remove('is-collapsing');closingRow=null;}}
  function slideDayPanel(show, reanimate=false){
    if(panelCloseTimer!==null){clearTimeout(panelCloseTimer);panelCloseTimer=null;}
    clearClosingRow();
    if(show){
      if(reanimate)dayPanel.classList.remove('is-expanded');
      dayPanel.hidden=false;
      dayPanel.inert=false;
      dayPanel.setAttribute('aria-hidden','false');
      if(reanimate && panelCloseMs){void dayPanel.offsetHeight;requestAnimationFrame(()=>dayPanel.classList.add('is-expanded'));}
      else dayPanel.classList.add('is-expanded');
    } else if(!dayPanel.hidden){
      dayPanel.inert=true;
      dayPanel.setAttribute('aria-hidden','true');
      const preceding=dayPanel.previousElementSibling;
      if(preceding?.matches('.day-bar,.fingerprint')){
        closingRow=preceding;closingRow.classList.add('is-collapsing');
      }
      dayPanel.classList.remove('is-expanded');
      if(!panelCloseMs){dayPanel.hidden=true;clearClosingRow();}
      else panelCloseTimer=setTimeout(()=>{
        dayPanel.hidden=true;panelCloseTimer=null;clearClosingRow();resizeFrame();
      },panelCloseMs);
    }
    resizeFrame();
  }
  function setVisible(el, visible){
    const pending = fadeTimers.get(el);
    if(pending !== undefined){window.clearTimeout(pending);fadeTimers.delete(el);}
    el.classList.remove('is-fading-out');
    if(visible){
      if(el.hidden) el.hidden = false;
    } else if(!el.hidden){
      if(!fadeMs){el.hidden = true;return;}
      el.classList.add('is-fading-out');
      const timer = window.setTimeout(()=>{
        el.hidden = true;
        el.classList.remove('is-fading-out');
        fadeTimers.delete(el);
        resizeFrame();
      }, fadeMs);
      fadeTimers.set(el, timer);
    }
  }
  const fmt = new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Manila',weekday:'short',day:'numeric',month:'short'});
  const fmtFull = new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Manila',weekday:'long',day:'numeric',month:'long'});
  const fmtWeek = new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Manila',weekday:'short'});
  const fmtWeekFull = new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Manila',weekday:'long'});
  const fmtDayMonth = new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Manila',day:'numeric',month:'short'});
  const fmtDayMonthFull = new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Manila',day:'numeric',month:'long'});
  const fmtHour = new Intl.DateTimeFormat('en-PH',{timeZone:'Asia/Manila',hour:'numeric',hour12:true});
  const state={view:'chances',subview:'graph',threshold:null,graphThreshold:null,barThreshold:0,day:null};
  const numeric = x => x!==null && Number.isFinite(x);
  const chanceWords = p=>!numeric(p)?'unavailable':p<5?'very low':p<35?'low':p<65?'medium':p<95?'high':'very high';
  const pct = p=>numeric(p)?`${Math.round(p)}%`:'—';
  function phDateKey(now){const p=new Intl.DateTimeFormat('en-US',{timeZone:'Asia/Manila',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(now); const find=t=>p.find(x=>x.type===t).value;return `${find('year')}-${find('month')}-${find('day')}`;}
  const dates = records.map((r,i)=>({...r,i,startDate:new Date(r.start),endDate:new Date(r.end)}));
  function dayName(d){return `${fmtWeekFull.format(d.startDate)}, ${fmtDayMonthFull.format(d.startDate)}`;}
  function windowName(d){return `${fmtHour.format(d.startDate).replace(' ','').toLowerCase()} ${fmt.format(d.startDate)} – ${fmtHour.format(d.endDate).replace(' ','').toLowerCase()} ${fmt.format(d.endDate)} (PHT)`;}
  function dayLabel(d){const today=d.date===phDateKey(new Date());return `<span class="day-label ${today?'is-today':''}"><span class="today-mini">TODAY</span><span class="day-short"><strong>${fmtWeek.format(d.startDate)}</strong> <span class="date-inline">${fmtDayMonth.format(d.startDate)}</span></span></span>`;}
  function rgb(hex){return [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));}
  function tint(hex,strength){return `rgb(${rgb(hex).map(v=>Math.round(255*(1-strength)+v*strength)).join(',')})`;}
  function tier(p){return p<5?0:p<35?.05:p<65?.35:p<95?.65:1;}
  function shade(hex,p){return !numeric(p)?'#f2f4f6':p<5?'#fff':tint(hex,tier(p));}
  function ink(hex,p){if(!numeric(p)||p<95)return '#273747';const [r,g,b]=rgb(hex);return (.299*r+.587*g+.114*b)/255>.5?'#273747':'#fff';}
  function bins(values){
    if(values.some(v=>!numeric(v)))return null;
    if(values.some((v,i)=>v<0||v>100||(i>0&&v>values[i-1]+.01)))return null;
    return [100-values[0],values[0]-values[1],values[1]-values[2],values[2]-values[3],values[3]-values[4],values[4]].map(v=>Math.max(0,v));
  }
  function setDay(i){state.day=state.day===i?null:i;redrawSelection();}
  function controls(){
    const choices=byId('threshold-chips');
    if(!choices.children.length)choices.innerHTML=thresholds.map((t,i)=>`<button type="button" class="threshold-chip" aria-pressed="false" data-threshold="${i}" style="--chip:${sourceColors[i]};--chip-stroke:${colors[i]};--chip-bg:${tint(sourceColors[i],.16)}" title="${amountWords[i]} rain or more" aria-label="Select chance of at least ${t} mm (${amountWords[i]} rain or more)"><span class="chip-top"><span class="swatch"></span><span class="chip-mm">${t} mm</span></span><span class="chip-meaning"><span class="swatch simple-swatch" aria-hidden="true"></span><span class="chip-meaning-text chip-meaning-full"><span class="chip-meaning-level">${amountButtonLines[i][0]}</span><span class="chip-meaning-rest">${amountButtonLines[i][1]}</span></span><span class="chip-meaning-text chip-meaning-short" aria-hidden="true"><span class="chip-meaning-level">${shortAmountWords[i]}</span><span class="chip-meaning-rest">or more</span></span></span></button>`).join('');
    choices.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.threshold)===state.threshold)));
    document.body.classList.toggle('outlook-simple',simple);
    document.querySelectorAll('.view-tab').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.view===state.view)));
    setVisible(byId('view-chances'),state.view==='chances');
    setVisible(byId('view-amounts'),state.view==='amounts');
    setVisible(byId('outlook-graph'),state.subview==='graph');
    setVisible(byId('outlook-bars'),state.subview==='bars');
    setVisible(byId('outlook-matrix'),state.subview==='matrix');
    setVisible(byId('outlook-thresholds'),state.subview!=='matrix');
    document.querySelectorAll('.subview-tab').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.subview===state.subview)));
    byId('bar-count').textContent=`${dates.length} days`;
    resizeFrame();
  }
  const X=i=>48+(dates.length<=1?0:i/(dates.length-1)*606), Y=p=>217-p*1.71;
  // Standard Catmull-Rom-to-Bezier curves, with moderate smoothing.
  // Each segment starts/ends at the real daily values. Missing dates break
  // the path rather than being interpolated as zero.
  const coord = n => Number(n).toFixed(2);
  function smoothCommands(points){
    if(points.length<2)return '';
    const curve=0.85; // 0 = straight, 1 = conventional Catmull-Rom
    let out='';
    for(let i=0;i<points.length-1;i++){
      const a=points[Math.max(0,i-1)];
      const b=points[i];
      const c=points[i+1];
      const d=points[Math.min(points.length-1,i+2)];
      const c1x=b[0]+(c[0]-a[0])*curve/6;
      const c1y=b[1]+(c[1]-a[1])*curve/6;
      const c2x=c[0]-(d[0]-b[0])*curve/6;
      const c2y=c[1]-(d[1]-b[1])*curve/6;
      out+=` C${coord(c1x)},${coord(c1y)} ${coord(c2x)},${coord(c2y)} ${coord(c[0])},${coord(c[1])}`;
    }
    return out;
  }
  function smoothPath(points){
    if(!points.length)return '';
    return `M${coord(points[0][0])},${coord(points[0][1])}`+smoothCommands(points);
  }
  function pathLine(series){
    const segments=[];let points=[];
    series.forEach((v,i)=>{
      if(numeric(v))points.push([X(i),Y(v)]);
      else if(points.length){segments.push(points);points=[];}
    });
    if(points.length)segments.push(points);
    return segments.map(smoothPath).join(' ');
  }
  function areaBands(upper,lower){
    const paths=[];let stack=[];
    function flush(){
      if(stack.length>=2){
        const top=stack.map(p=>p[0]);
        const bottom=stack.map(p=>p[1]).reverse();
        paths.push(smoothPath(top)+` L${coord(bottom[0][0])},${coord(bottom[0][1])}`+
                   smoothCommands(bottom)+' Z');
      }
      stack=[];
    }
    upper.forEach((v,i)=>{
      if(numeric(v)&&numeric(lower[i]))stack.push([[X(i),Y(v)],[X(i),Y(lower[i])]]);
      else flush();
    });
    flush();return paths;
  }
  function graph(){
    let out=''; const v=thresholds.map((_,t)=>dates.map(d=>d.values[t]));
    for(const p of [0,25,50,75,100]){out+=`<line x1="48" x2="654" y1="${Y(p)}" y2="${Y(p)}" stroke="#e7ebf0"/><text x="40" y="${Y(p)+4}" text-anchor="end" fill="#7d8b9a" font-size="12">${p}%</text>`;}
    dates.forEach((d,i)=>{if(i%2===0||i===dates.length-1){out+=`<text x="${X(i)}" y="239" text-anchor="middle" fill="#66788a" font-size="11">${fmtWeek.format(d.startDate)} ${fmtDayMonth.format(d.startDate)}</text>`;}});
    if(state.threshold===null){
      // Full five-threshold ribbon, each band bounded by the next threshold.
      for(let t=0;t<5;t++){
        const low=t===4?dates.map(()=>0):v[t+1];
        areaBands(v[t],low).forEach(d=>{out+=`<path d="${d}" fill="${sourceColors[t]}" fill-opacity=".19" stroke="none"/>`;});
      }
    } else {
      // One threshold: fill down to 0%, not merely to the next ribbon.
      const t=state.threshold;
      areaBands(v[t],dates.map(()=>0)).forEach(d=>{out+=`<path d="${d}" fill="${sourceColors[t]}" fill-opacity=".20" stroke="none"/>`;});
    }
    v.forEach((serie,t)=>{const focused=state.threshold===null||state.threshold===t;out+=`<path d="${pathLine(serie)}" fill="none" stroke="${focused?colors[t]:'#aab4bf'}" stroke-width="${state.threshold===null?2:focused?3.2:1.2}" stroke-opacity="${focused?1:.43}" stroke-linecap="round" stroke-linejoin="round"/>`;});
    if(state.day!==null){
      const i=state.day;
      out+=`<line x1="${X(i)}" x2="${X(i)}" y1="42" y2="217" stroke="#92aabb" stroke-dasharray="4 4"/>`;
      const shown=state.threshold===null?[0,1,2,3,4]:[state.threshold];
      shown.forEach(t=>{const p=v[t][i];if(numeric(p))out+=`<circle cx="${X(i)}" cy="${Y(p)}" r="${state.threshold===null?3.5:5.5}" fill="#fff" stroke="${colors[t]}" stroke-width="2.4"/>`;});
    }
    out+='<rect x="48" y="39" width="606" height="180" fill="transparent"/>';
    byId('trend-svg').innerHTML=out;
    const legend=byId('graph-legend');
    if(!legend.children.length)legend.innerHTML=thresholds.map((t,i)=>`<button type="button" class="legend-pill legend-button" data-legend-threshold="${i}" aria-pressed="false"><span class="legend-mark"></span>${t} mm</button>`).join('');
    legend.querySelectorAll('button').forEach(b=>{const i=Number(b.dataset.legendThreshold);b.setAttribute('aria-pressed',String(state.threshold===i));b.querySelector('.legend-mark').style.background=state.threshold===null||state.threshold===i?sourceColors[i]:'#abb5bf';});
    byId('graph-selected-label').textContent=state.day===null?'Tap a day for details':dayName(dates[state.day]);
  }
  function bars(){
    const list=byId('bar-list'),active=state.threshold??state.barThreshold;
    const signature=active+'|'+simple;
    if(list.dataset.renderSignature!==signature){
      if(dayPanel.parentElement===list)dayPanel.remove();
      list.innerHTML=dates.map((d,i)=>{const p=d.values[active];const w=numeric(p)?`width:${p}%`:'width:0';return `<button class="day-bar" type="button" data-day="${i}" aria-pressed="false">${dayLabel(d)}<span class="track"><span class="progress" style="${w};--bar:${sourceColors[active]}"></span></span><span class="probability ${simple?'simple':''}">${numeric(p)?(simple?chanceWords(p)+' chance':pct(p)):'No data'}</span></button>`;}).join('');
      list.dataset.renderSignature=signature;
    }
    list.querySelectorAll('button[data-day]').forEach(b=>b.setAttribute('aria-pressed',String(state.day===Number(b.dataset.day))));
  }
  function fingerprints(){
    const list=byId('fingerprint-list');
    if(list.dataset.rendered!=='true'){
      if(dayPanel.parentElement===list)dayPanel.remove();
      list.innerHTML=dates.map((d,i)=>{const b=bins(d.values);return `<button class="fingerprint ${b?'':'missing'}" type="button" data-day="${i}" aria-pressed="false" aria-label="${dayName(d)}. Open rainfall possibilities.">${dayLabel(d)}<span class="finger-track">${b?b.map((p,j)=>p>0?`<span class="finger-segment" style="flex:${p};background:${rangeColors[j]}" title="${ranges[j]}: ${p.toFixed(0)}%"></span>`:'').join(''):'Data unavailable'}</span><span aria-hidden="true" style="text-align:right;color:#8997a4">›</span></button>`;}).join('');
      list.dataset.rendered='true';
    }
    list.querySelectorAll('button[data-day]').forEach(b=>b.setAttribute('aria-pressed',String(state.day===Number(b.dataset.day))));
    byId('fingerprint-legend').innerHTML=ranges.map((r,i)=>`<span class="legend-pill"><span class="legend-mark" style="background:${rangeColors[i]}"></span>${r}</span>`).join('');
  }
  function details(){
    const panel=dayPanel;
    // Graph details stay at the bottom; Bars and Possible totals expand
    // directly below the selected day, preserving the same detail content.
    if(state.day===null){
      slideDayPanel(false);
      return;
    }
    const d=dates[state.day];const today=d.date===phDateKey(new Date());
    let h=`<div class="panel-head"><div><div class="panel-day"><span class="today-mini ${today?'show-today':''}">TODAY</span><h3>${dayName(d)}</h3></div><div class="window">${windowName(d)}</div></div><div class="tap-hint">Tap the day again to close</div></div>`;
    if(state.view==='amounts'){
      h+='<div class="smallheading">Possible rainfall totals for the day</div><div class="amount-distribution">';
      const b=bins(d.values);
      h+=b?b.map((p,i)=>`<div class="amount-small"><div class="low"><i class="amount-swatch" style="background:${rangeColors[i]}"></i>${ranges[i]}</div><div class="big">${Math.round(p)}%</div></div>`).join(''):'<div class="empty">The complete probability distribution is unavailable for this day.</div>';
      h+='</div>';
    } else {
      h+='<div class="smallheading">Chances of rain</div><div class="chance-rows">';
      thresholds.forEach((t,i)=>{const p=d.values[i];const s=shade(sourceColors[i],p),lc=tint(sourceColors[i],.20),tc=ink(sourceColors[i],p);h+=`<div class="chance-row"><div class="chance-left" style="background:${lc}"><strong>≥${t} mm</strong><span>${amountWords[i]} or more</span></div><div class="chance-right" style="background:${s};color:${tc}"><strong>${pct(p)}</strong><span>${numeric(p)?chanceWords(p):'no data'}</span></div></div>`;});
      h+='</div>';
    }
    const wasExpanded=panel.classList.contains('is-expanded') && !panel.hidden;
    const previousSibling=panel.previousElementSibling;
    panel.innerHTML=`<div class="panel-slide-wrap"><div class="panel-slide-content">${h}</div></div>`;
    let target=null;
    if(state.view==='amounts'){
      target=byId('fingerprint-list').querySelector(`.fingerprint[data-day="${state.day}"]`);
    }else if(state.subview==='bars'){
      target=byId('bar-list').querySelector(`.day-bar[data-day="${state.day}"]`);
    }
    panel.classList.toggle('inline-day-panel',Boolean(target));
    if(target)target.after(panel);
    else byId('day-panel-anchor').after(panel);
    slideDayPanel(true, !wasExpanded || (target && previousSibling!==target));
  }
  function exact(){
    const th=thresholds.map((t,i)=>`<th style="background:${tint(sourceColors[i],.20)}">≥${t} mm</th>`).join('');
    const todayKey=phDateKey(new Date());
    const rows=dates.map(d=>`<tr><td><strong>${fmtWeekFull.format(d.startDate)}</strong> ${fmtDayMonthFull.format(d.startDate)}${d.date===todayKey?' <span class="matrix-today">TODAY</span>':''}</td>${d.values.map((p,i)=>`<td style="background:${shade(sourceColors[i],p)};color:${ink(sourceColors[i],p)}">${pct(p)}</td>`).join('')}</tr>`).join('');
    byId('exact-table').innerHTML=`<table><thead><tr><th>Day</th>${th}</tr></thead><tbody>${rows}</tbody></table>`;
  }
  function resizeFrame(){
    requestAnimationFrame(()=>{
      const card=document.querySelector('.shell');if(!card)return;
      const h=Math.max(1,Math.ceil(card.getBoundingClientRect().bottom+3));
      try{const frame=window.frameElement;if(frame&&frame.style){frame.style.setProperty('height',`${h}px`,'important');frame.style.setProperty('min-height','0px','important');}}catch(e){}
      window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:h},'*');
    });
  }
  function redrawSelection(){graph();bars();fingerprints();details();resizeFrame();}
  function redraw(){controls();redrawSelection();}
  function focusThreshold(i){
    if(state.threshold===i){
      // The user's first repeat tap closes any open day panel, without
      // changing the active threshold or requiring a double tap.
      if(state.day!==null){state.day=null;redrawSelection();return;}
      // An additional tap in Graph restores the full-color five-ribbon view.
      if(state.subview==='graph'){showAllThresholds();return;}
      return;
    }
    state.threshold=i;
    if(state.subview==='bars')state.barThreshold=i;
    else state.graphThreshold=i;
    redraw();
  }
  function showAllThresholds(){if(state.subview!=='graph')return;state.threshold=null;state.graphThreshold=null;redraw();}
  byId('threshold-chips').addEventListener('click',e=>{const b=e.target.closest('button[data-threshold]');if(b)focusThreshold(Number(b.dataset.threshold));});
  byId('threshold-chips').addEventListener('dblclick',e=>{if(e.target.closest('button[data-threshold]'))showAllThresholds();});
  byId('graph-legend').addEventListener('click',e=>{const b=e.target.closest('button[data-legend-threshold]');if(b)focusThreshold(Number(b.dataset.legendThreshold));});
  byId('graph-legend').addEventListener('dblclick',e=>{if(e.target.closest('button[data-legend-threshold]'))showAllThresholds();});
  byId('bar-list').addEventListener('click',e=>{const b=e.target.closest('button[data-day]');if(b)setDay(Number(b.dataset.day));});
  byId('fingerprint-list').addEventListener('click',e=>{const b=e.target.closest('button[data-day]');if(b)setDay(Number(b.dataset.day));});
  document.querySelectorAll('.view-tab').forEach(b=>b.addEventListener('click',()=>{state.view=b.dataset.view;state.day=null;redraw();}));
  document.querySelectorAll('.subview-tab').forEach(b=>b.addEventListener('click',()=>{state.subview=b.dataset.subview;if(state.subview==='bars')state.threshold=state.barThreshold;else if(state.subview==='graph')state.threshold=state.graphThreshold;redraw();}));
  byId('outlook-simple').addEventListener('change',e=>{simple=e.target.checked;redraw();});
  const svg=byId('trend-svg');svg.setAttribute('viewBox','0 0 700 256');
  svg.addEventListener('click',e=>{const m=svg.getScreenCTM();if(!m)return;const point=svg.createSVGPoint();point.x=e.clientX;point.y=e.clientY;const p=point.matrixTransform(m.inverse());if(p.x<44||p.x>660||p.y<35||p.y>224)return;const n=Math.max(1,dates.length-1),i=Math.max(0,Math.min(dates.length-1,Math.round((p.x-48)/606*n)));setDay(i);});
  svg.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();state.day=Math.max(0,Math.min(dates.length-1,(state.day===null?0:state.day)+(e.key==='ArrowRight'?1:-1)));redrawSelection();}else if(e.key==='Enter'||e.key===' '){e.preventDefault();state.day=state.day===null?0:null;redrawSelection();}});
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(resizeFrame).observe(document.querySelector('.shell'));
  window.addEventListener('resize',resizeFrame);
  if(document.fonts?.ready)document.fonts.ready.then(resizeFrame);
  window.addEventListener('load',resizeFrame);
  exact();redraw();resizeFrame();
})();
</script></body></html>'''

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
_RAIN_MED = THRESHOLD_COLORS[1]
_RAIN_P90 = "#CAD7E1"
_RAIN_TRACK = "#DFE3E9"

def _fmt_mm(v: float) -> str:
    return "0" if v < 0.05 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")

def _hour_label(hour: int) -> str:
    return f"{hour % 12 or 12}{'am' if hour % 24 < 12 else 'pm'}"

_tip_hour_label = _hour_label

def _hour_label_html(hour: int) -> str:
    return f"{hour % 12 or 12}<small class='rr-ap'>{'am' if hour % 24 < 12 else 'pm'}</small>"

def _chance_descriptor(probability: float) -> str:
    """Simple lowercase categories for the chance of exceeding a threshold."""
    if probability < 5:
        return "very low"
    if probability < 35:
        return "low"
    if probability < 65:
        return "medium"
    if probability < 95:
        return "high"
    return "very high"


def _rain_amount_descriptor(rain_mm: float) -> str:
    """Unabbreviated description for one total rainfall amount."""
    if rain_mm < 5:
        return "very light"
    if rain_mm < 20:
        return "light"
    if rain_mm < 50:
        return "moderate"
    if rain_mm < 100:
        return "heavy"
    return "intense"


def _rain_range_descriptor(median_mm: float, p90_mm: float) -> str:
    """Use full words for single categories, short words only in a range."""
    low = _rain_amount_descriptor(median_mm)
    high = _rain_amount_descriptor(p90_mm)
    if low == high:
        return low
    abbreviations = {"very light": "v. light", "moderate": "mod."}
    return f"{abbreviations.get(low, low)} to {abbreviations.get(high, high)}"


def _simple_rain_range(median_mm: float, p90_mm: float) -> str:
    """Plain-language daily amount using the full median-to-P90 descriptors."""
    low = _rain_amount_descriptor(median_mm)
    high = _rain_amount_descriptor(p90_mm)
    return low if low == high else f"{low} to {high}"


def _simple_chance_chip(
    val_1mm, secondary: tuple[int, float] | None = None, *, partial: bool = False
) -> str:
    """The same probability and threshold as standard mode, without numbers."""
    threshold, probability = (1, val_1mm) if secondary is None else secondary
    if probability is None:
        return "<div class='rr-chip rr-chip-na' aria-label='Rain chance unavailable'>—</div>"
    bg, text_color = _cell_style(threshold, probability)
    chance_phrase = _chance_descriptor(float(probability)).capitalize() + " chance"
    amount_level = _rain_amount_descriptor(float(threshold))
    amount_short = {"very light": "v. light", "moderate": "mod."}.get(amount_level, amount_level)
    amount_phrase = amount_short.capitalize() + " or more"
    return (
        f"<div class='rr-chip rr-chip-simple' style='{bg}color:{text_color};' "
        f"aria-label='{probability:.0f}% chance of at least {threshold} mm: {chance_phrase}, {amount_level} rain or more'>"
        f"<span class='rr-simple-chance'>{chance_phrase}</span>"
        f"<span class='rr-simple-threshold'>{amount_phrase}</span>"
        "</div>"
    )


def _chance_chip(
    val_1mm, secondary: tuple[int, float] | None = None, *, partial: bool = False
) -> str:
    """Compact, color-coded card for the highest threshold with >=65% chance.

    Fall back to the >=1 mm probability when no higher level qualifies.
    Descriptors are reserved for the detailed view, where their meaning is clear.
    """
    threshold, probability = (1, val_1mm) if secondary is None else secondary
    if probability is None:
        return "<div class='rr-chip rr-chip-na' aria-label='Rain chance unavailable'>—</div>"
    bg, text_color = _cell_style(threshold, probability)
    return (
        f"<div class='rr-chip' style='{bg}color:{text_color};' "
        f"aria-label='{probability:.0f}% chance of at least {threshold} mm rainfall'>"
        "<span class='rr-prob-head'>"
        f"<b>{probability:.0f}</b><small class='rr-percent-sign'>%</small>"
        "</span>"
        f"<small class='rr-chip-threshold'>&ge;{threshold} mm</small>"
        "</div>"
    )


def render_rain_rows_html(pct_result: dict, threshold_result: dict | None = None, max_days: int = 6, *, simple_mode: bool = False, parts: bool = False) -> str:
    """Selectable six-day rainfall outlook with an accessible, detailed day panel.

    The overview bars retain the existing median/P90 colors and absolute scale.
    Only the detailed bars expose percentile tooltips. The card is rendered
    inside an isolated Streamlit iframe with explicit, toggleable day state.
    Clicking the active day closes its details; selecting another opens it
    immediately without a Streamlit rerun.
    """
    periods = pct_result["days"]
    step = pct_result["period_hours"]
    cap = RAIN_BAR_CAP_MM[step]
    n_slots = 24 // step
    if not periods:
        return ""

    rows: dict = {}
    for period in periods:
        start = period["start_utc"].astimezone(PH_TZ)
        rows.setdefault(start.date(), {})[(start.hour - ROW_ANCHOR_PH_HOUR) // step] = period

    bins_per_day = 24 // pct_result.get("bin_hours", 3)
    day_members: dict = {}
    for raw_bin in pct_result.get("raw_bins") or []:
        key = raw_bin["start_utc"].astimezone(PH_TZ).date()
        day_members.setdefault(key, []).append(raw_bin["member_values_mm"])
    for day, row in rows.items():
        day_members.setdefault(day, [p["member_values_mm"] for p in row.values()])
    # Drop past 2am-to-2am daily rows based on Philippine time, not the
    # ECMWF initialization day. The iframe checks the device clock too.
    today_ph = datetime.now(PH_TZ).date()
    days = [d for d in sorted(set(rows) | set(day_members)) if d >= today_ph][:max_days]
    if not days:
        return f"<div style='padding:14px;font-family:{FONT_STACK}'>No upcoming daily rainfall forecast is available. Try refreshing the data.</div>"

    # Threshold forecasts cover 8am–8am; rainfall totals cover 2am–2am.
    chance_data, chance_label = None, {}
    if threshold_result and threshold_result.get("windows"):
        chance_data = threshold_result["data"]
        chance_label = {
            w["start_utc"].astimezone(PH_TZ).date(): w["label"]
            for w in threshold_result["windows"]
        }
    has_chance = chance_data is not None
    row_cls = "rr-row rr-c" if has_chance else "rr-row"
    if simple_mode:
        row_cls += " rr-simple"
    slot_hours = [ROW_ANCHOR_PH_HOUR + i * step for i in range(n_slots)]
    grid = f"grid-template-columns:repeat({n_slots},minmax(0,1fr));column-gap:{_RAIN_BAR_GAP_PX}px;"
    # Widen only the detailed bars. The overview bars and their spacing stay intact.
    detail_grid = f"grid-template-columns:repeat({n_slots},minmax(0,1fr));column-gap:2px;"
    detail_bar_max_px = 2 * _RAIN_BAR_MAX_PX[n_slots]

    def tip_side(slot: int) -> str:
        if n_slots == 1:
            return ""
        pos = slot / (n_slots - 1)
        return " rr-tip-l" if pos < 0.25 else (" rr-tip-r" if pos > 0.75 else "")

    def bar(period: dict, slot: int, *, tooltip: bool = False, below: bool = False) -> str:
        start_ms = round(period["start_utc"].timestamp() * 1000)
        end_ms = round(period["end_utc"].timestamp() * 1000)
        slot_attrs = f"data-start-ms='{start_ms}' data-end-ms='{end_ms}'"
        stats = period["stats"]
        med, p75, p90, p25, p10 = (
            stats["median"], stats["p75"], stats["p90"], stats["p25"], stats["p10"]
        )
        pct_of_cap = lambda v: min(max(v / cap, 0.0), 1.0) * 100
        cap_mark = "<span class='rr-over'></span>" if med > cap else ""
        # Intentionally identical bar colors, heights, and scale to the previous card.
        bars = (
            "<div class='rr-fill'>"
            f"<span style='height:{pct_of_cap(p90):.1f}%;background:{_RAIN_P90}'></span>"
            f"<span style='height:{pct_of_cap(med):.1f}%;background:{_RAIN_MED}'></span>"
            f"{cap_mark}</div>"
        )
        if not tooltip:
            # No tooltip markup and no tabindex: the entire overview row receives
            # the click instead of a nested focusable bar intercepting it.
            return f"<div class='rr-slot' {slot_attrs}><div class='rr-bar'>{bars}</div></div>"
        start = period["start_utc"].astimezone(PH_TZ)
        end = period["end_utc"].astimezone(PH_TZ)
        when = f"{start:%a %d %b} · {_tip_hour_label(start.hour)}–{_tip_hour_label(end.hour)}"
        values = [
            f"P90 {_fmt_mm(p90)} mm", f"P75 {_fmt_mm(p75)} mm",
            f"Median {_fmt_mm(med)} mm", f"P25 {_fmt_mm(p25)} mm",
            f"P10 {_fmt_mm(p10)} mm",
        ]
        over_note = f" (median above the {cap} mm scale)" if med > cap else ""
        tip_cls = "rr-tip" + tip_side(slot) + (" rr-tip-b" if below else "")
        return (
            f"<div class='rr-slot' {slot_attrs}><button type='button' class='rr-bar' aria-label='Rainfall {when}; tap for percentiles'>"
            f"{bars}<span class='{tip_cls}'><b>{when}</b><br>"
            + "<br>".join(values) + f"{over_note}</span></button></div>"
        )

    empty_bar = "<div class='rr-slot'><div class='rr-bar rr-empty'><div class='rr-fill'></div></div></div>"
    slot_headers = "".join(
        f"<div class='rr-slot'><span>{_hour_label_html(h)}</span></div>"
        for h in slot_hours
    )
    # Full ranges above the expanded bars, with a line break on narrow screens.
    detail_slot_headers = "".join(
        "<div class='rr-slot'><span class='rr-time-range'>"
        f"<span class='rr-now-tag'>NOW:</span><span>{_tip_hour_label(h)}–</span><span>{_tip_hour_label(h + step)}</span>"
        "</span></div>"
        for h in slot_hours
    )
    chance_head = "<div class='rr-h rr-h-chance'>Chance of rain</div>" if has_chance else ""
    head = (
        f"<div class='{row_cls} rr-head'><div></div>{chance_head}"
        f"<div class='rr-slots' style='{grid}'>{slot_headers}</div>"
        "<div class='rr-h rr-h-r'>Total for the day</div></div>"
    )

    body = ""
    details = ""
    median_above_scale = False
    rendered_days = 0
    for index, day in enumerate(days):
        row = rows.get(day, {})
        if not day_members.get(day):
            continue
        index = rendered_days
        rendered_days += 1
        # Compute total percentiles after summing each member across the day.
        members = np.sum(day_members[day], axis=0)
        total_p10, total_p25, total_med, total_p75, total_p90 = [
            float(v) for v in np.percentile(members, [10, 25, 50, 75, 90])
        ]
        is_partial = len(day_members[day]) < bins_per_day
        partial = "<span class='rr-part'>(partial data)</span>" if is_partial else ""
        overview_bars = "".join(
            bar(row[i], i) if i in row else empty_bar for i in range(n_slots)
        )
        detailed_bars = "".join(
            bar(row[i], i, tooltip=True, below=False) if i in row else empty_bar
            for i in range(n_slots)
        )
        median_above_scale |= any(float(p["stats"]["median"]) > cap for p in row.values())

        chance_chip = ""
        chance_detail = ""
        if has_chance:
            label = chance_label.get(day)
            v1 = chance_data.get(1, {}).get(label) if label else None
            secondary = _pick_dynamic_threshold(chance_data, label) if label else None
            chance_chip = (
                _simple_chance_chip(v1, secondary, partial=is_partial)
                if simple_mode else _chance_chip(v1, secondary, partial=is_partial)
            )
            cards = []
            for threshold in AVAILABLE_THRESHOLDS_MM:
                val = chance_data.get(threshold, {}).get(label) if label else None
                value_bg, value_text = _cell_style(threshold, val)
                # The left segment always uses the full table's fixed 20% threshold tint.
                # The right segment uses that threshold's probability-dependent tier.
                # Their shared edge needs no border or separating line.
                threshold_rgb = _tint_toward_white_rgb(
                    *_hex_to_rgb(THRESHOLD_COLORS[threshold]), 0.20,
                )
                label_bg = f"background-color:rgb({','.join(map(str, threshold_rgb))});"
                label_text = _cell_text_color(_relative_luminance(*threshold_rgb))
                probability = "—" if val is None else f"{val:.0f}%"
                descriptor = "unavailable" if val is None else _chance_descriptor(float(val))
                rainfall_label = {
                    1: "very light", 5: "light", 20: "moderate",
                    50: "heavy", 100: "intense",
                }[threshold]
                cards.append(
                    f"<div class='rr-detail-chance' "
                    f"aria-label='Chance of at least {threshold} mm: {probability}, {descriptor}; {rainfall_label} rain or more'>"
                    f"<div class='rr-detail-chance-label' style='{label_bg}color:{label_text};'>"
                    f"<small class='rr-detail-chance-threshold'>&ge;{threshold} mm</small>"
                    f"<small class='rr-detail-chance-meaning'>{rainfall_label} or more</small></div>"
                    f"<div class='rr-detail-chance-value' style='{value_bg}color:{value_text};'>"
                    f"<b>{probability}</b><small class='rr-detail-chance-desc'>{descriptor}</small>"
                    "</div></div>"
                )
            chance_detail = (
                "<div class='rr-section-heading'>Chances of rain"
                "<small>Chance of reaching each rainfall amount</small></div>"
                "<div class='rr-detail-chances'>" + "".join(cards) + "</div>"
            )

        # A real button, not a radio/label pair: keyboard accessible, explicit
        # click state, and repeat clicks can collapse the expanded details.
        # Keep all day details closed on the initial render.
        initial_class = ""
        expanded = "false"
        simple_total_descriptor = _simple_rain_range(total_med, total_p90)
        total_is_range = " to " in simple_total_descriptor
        if total_is_range:
            first_amount, second_amount = simple_total_descriptor.split(" to ", 1)
            simple_total_markup = (
                f"<span class='rr-total-line'>{first_amount.capitalize()}</span>"
                f"<span class='rr-total-line'>to {second_amount}</span>"
            )
        else:
            simple_total_markup = f"<span class='rr-total-line'>{simple_total_descriptor.capitalize()}</span>"
        body += (
            f"<button type='button' class='{row_cls} rr-data rr-selectable{initial_class}' "
            f"data-rr-day='{index}' data-rr-date='{day:%Y-%m-%d}' aria-expanded='{expanded}' "
            f"aria-controls='rr-detail-{index}' "
            f"aria-label='Toggle rainfall details for {day:%A %d %B}'>"
            f"<div class='rr-day'><span class='rr-today-tag'>TODAY</span>"
            f"<strong class='rr-day-weekday'>{day:%a}</strong>"
            f"<span class='rr-day-calendar'>{day.day} {day:%b}</span></div>"
            f"{chance_chip}<div class='rr-slots' style='{grid}'>{overview_bars}</div>"
            f"<div class='rr-tot{' rr-tot-partial' if is_partial else ''}{' rr-total-simple' if simple_mode else ''}'>"
            + (
                f"<div class='rr-total-plain{' rr-total-range' if total_is_range else ''}'>{simple_total_markup}</div>"
                if simple_mode else
                f"<div class='rr-total-main'><b>{_fmt_mm(total_med)}</b>"
                "<small class='rr-secondary'>mm</small></div>"
                f"<small class='rr-secondary rr-range' title='Upper-end estimate: 90th-percentile ensemble total'>"
                f"High: {_fmt_mm(total_p90)} mm</small>"
            )
            + f"{partial}</div></button>"
        )
        quantile_values = [
            ("P10", total_p10), ("P25", total_p25),
            ("Median", total_med), ("P75", total_p75), ("P90", total_p90),
        ]
        quantile_meanings = {
            "P10": "lower estimate", "P25": "below typical",
            "Median": "most likely", "P75": "above typical",
            "P90": "higher estimate",
        }
        quantiles = "".join(
            f"<div class='rr-quantile{' rr-qmedian' if name == 'Median' else ''}'>"
            "<div class='rr-quantile-caption'>"
            f"<small class='rr-quantile-name'>{name}</small>"
            f"<small class='rr-quantile-help'><i>({quantile_meanings[name]})</i></small></div>"
            f"<b>{_fmt_mm(amount)} <em>mm</em></b>"
            f"<small class='rr-quantile-desc'>{_rain_amount_descriptor(amount)}</small></div>"
            for name, amount in quantile_values
        )
        details += (
            f"<section id='rr-detail-{index}' data-rr-date='{day:%Y-%m-%d}' class='rr-detail rr-detail-{index}' "
            f"aria-label='Details for {day:%A %d %B}' aria-hidden='true' inert>"
            "<div class='rr-detail-collapse'><div class='rr-detail-content'>"
            "<div class='rr-detail-top'>"
            f"<div class='rr-detail-date'><span class='rr-today-tag'>TODAY</span><strong>{day:%A}, {day.day} {day:%B}</strong></div>"
            "<div class='rr-detail-main'><div class='rr-detail-total-top'>"
            f"<small>Total for the day:</small><b>{_fmt_mm(total_med)}</b><small>mm</small></div>"
            f"<small class='rr-detail-intensity'>{_rain_amount_descriptor(total_med)}</small>"
            "</div></div>"
            "<div class='rr-section-heading'>Daily rainfall range"
            "<small>From lower to higher estimates</small></div>"
            f"<div class='rr-quantiles'>{quantiles}</div>"
            f"{chance_detail}"
            "<div class='rr-detail-bars'>"
            f"<div class='rr-section-heading'>{step}-hourly forecast"
            "<small>Tap a bar to see more details</small></div>"
            f"<div class='rr-detail-bar-hours rr-slots' style='{detail_grid}'>{detail_slot_headers}</div>"
            f"<div class='rr-detail-bar-values rr-slots' style='{detail_grid}'>{detailed_bars}</div>"
            "</div></div></div></section>"
        )

    legend = (
        "<div class='rr-legend'>"
        f"<span><i style='background:{_RAIN_MED}'></i>Most likely amount</span>"
        f"<span><i style='background:{_RAIN_P90}'></i>Higher estimate</span>"
        + ("<span><i style='background:#0B3FBF;height:3px;vertical-align:3px'></i>Above chart scale</span>" if median_above_scale else "")
        + "</div>"
    )

    # Compact baseline sizes match the previous phone view. Enlarged desktop
    # sizing applies only when the card itself is wide enough (not merely when
    # the desktop browser window is wide).
    css = (
        "<style>"
        f"html,body{{font-family:{FONT_STACK};font-size:16px;}}"
        f".rr-card{{--ink:#283746;--muted:#687585;--muted2:#8994a1;--today:#1786B6;container-type:inline-size;background:#fff;color:var(--ink);border-radius:10px;padding:10px 12px;font-family:{FONT_STACK};}}"
        f".rr-card button,.rr-card .rr-tip{{font-family:{FONT_STACK};}}"
        f".rr-row{{display:grid;grid-template-columns:2.7rem minmax(0,{_RAIN_SLOTS_MAX_PX}px) 3.6rem;column-gap:10px;justify-content:space-evenly;align-items:center;}}"
        f".rr-row.rr-c{{grid-template-columns:3.65rem 5.3rem minmax(0,{_RAIN_SLOTS_MAX_PX}px) 4.85rem;column-gap:6px;}}"
        ".rr-data{padding:5px 0;border-top:1px solid #EEF0F3;}"
        ".rr-head{padding-bottom:2px;}"
        ".rr-head .rr-slot span,.rr-detail-bar-hours .rr-slot span{font-size:.58rem;color:var(--muted2);white-space:nowrap;}"
        ".rr-ap{font-size:.46rem;}"
        ".rr-h{font-size:.58rem;color:#7A8796;font-weight:700;text-align:center;line-height:1.1;white-space:nowrap;}"
        ".rr-h-r{text-align:center;}"
        ".rr-day{display:flex;flex-direction:column;line-height:1.15;}"
        ".rr-day b{font-size:.82rem;font-weight:700;color:var(--ink);}"
        ".rr-day-weekday{font-size:.82rem;font-weight:700;line-height:1.1;color:var(--ink);}"
        ".rr-day-calendar{font-size:.60rem;letter-spacing:-.025em;white-space:nowrap;font-weight:400;line-height:1.15;color:#737D88;}"
        ".rr-secondary{font-size:.56rem;font-weight:400;line-height:1.2;letter-spacing:-.01em;color:var(--muted);white-space:nowrap;}"
        ".rr-slots{display:grid;}"
        ".rr-slot{display:flex;justify-content:center;}"
        ".rr-bar{position:relative;width:100%;max-width:var(--rr-overview-bar-max);height:60px;outline:none;}"
        f".rr-fill{{position:absolute;inset:0;background:{_RAIN_TRACK};border-radius:7px;overflow:hidden;}}"
        ".rr-fill span{position:absolute;left:0;right:0;bottom:0;display:block;}"
        ".rr-fill .rr-over{top:0;bottom:auto;height:3px;background:#0B3FBF;}"
        ".rr-empty .rr-fill{opacity:.25;}"
        ".rr-tot{text-align:center;white-space:nowrap;display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1.12;min-width:0;height:2.8rem;border-radius:6px;background:#F2F5F7;box-shadow:inset 0 0 0 1px rgba(0,0,0,.035);}"
        ".rr-tot-partial{height:3.3rem;}"
        ".rr-total-main{display:flex;align-items:baseline;justify-content:center;gap:2px;}"
        ".rr-tot .rr-total-main b{font-size:.95rem;font-weight:750;color:var(--ink);letter-spacing:-.025em;font-variant-numeric:tabular-nums;}"
        ".rr-tot .rr-total-main small{font-size:.56rem;font-weight:400;color:var(--muted);}"
        ".rr-tot .rr-range{display:block;margin-top:3px;color:var(--muted);font-variant-numeric:tabular-nums;}"
        ".rr-tot .rr-part{font-size:.57rem;font-style:italic;color:var(--muted);}"
        ".rr-chip{box-sizing:border-box;width:100%;height:2.8rem;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;line-height:1.1;box-shadow:inset 0 0 0 1px rgba(0,0,0,.08);}"
        ".rr-chip b{font-size:.98rem;font-weight:750;letter-spacing:-.015em;font-variant-numeric:tabular-nums;}"
        ".rr-prob-head{display:flex;align-items:baseline;justify-content:center;gap:2px;max-width:100%;white-space:nowrap;}"
        ".rr-chip .rr-prob-head b{font-size:.98rem;}"
        ".rr-chip .rr-percent-sign{font-size:.56rem;font-weight:400;opacity:.85;}"
        ".rr-chip .rr-chip-threshold{font-size:.56rem;font-weight:400;margin-top:3px;color:inherit;opacity:.85;}"
        ".rr-chip-na{color:var(--muted2);font-size:.75rem;}"
        ".rr-tip{display:none;position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);"
        "background:#36495B;color:#fff;font-size:.68rem;line-height:1.4;padding:6px 9px;border-radius:7px;"
        "white-space:nowrap;z-index:50;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.25);}"
        ".rr-tip-l{left:0;transform:none;}.rr-tip-r{left:auto;right:0;transform:none;}"
        ".rr-tip-b{bottom:auto;top:calc(100% + 6px);}"
        ".rr-detail .rr-bar{cursor:pointer;appearance:none;-webkit-appearance:none;border:0;padding:0;background:transparent;font:inherit;touch-action:manipulation;}"
        ".rr-detail .rr-bar:focus-visible{outline:2px solid #159BD6;outline-offset:2px;border-radius:7px;}"
        ".rr-detail .rr-bar:hover .rr-tip,.rr-detail .rr-bar:focus .rr-tip{display:block;}"
        ".rr-legend{display:flex;flex-wrap:wrap;gap:3px 12px;font-size:.62rem;color:var(--muted);margin-top:8px;}"
        ".rr-legend i{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:4px;vertical-align:-1px;}"
        ".rr-selectable{position:relative;display:grid;width:100%;box-sizing:border-box;appearance:none;-webkit-appearance:none;"
        "cursor:pointer;color:inherit;font:inherit;text-align:left;background:transparent;border:0;"
        "border-top:1px solid #EEF0F3;border-radius:6px;transition:background-color .12s ease;}"
        ".rr-selectable:hover{background-color:#F8FAFC;}"
        ".rr-selectable.is-selected,.rr-selectable.is-collapsing{background-color:#F3F8FB;box-shadow:inset 2px 0 #00BFFF;border-radius:8px 8px 0 0;}"
        ".rr-selectable:focus-visible{outline:2px solid #159BD6;outline-offset:1px;}"
        ".rr-detail{display:grid;grid-template-rows:0fr;min-height:0;margin:0;padding:0;"
        "border:0;border-radius:0 0 9px 9px;background:#F7FAFC;"
        "transition:grid-template-rows .26s ease;pointer-events:none;}"
        ".rr-detail.is-open{grid-template-rows:1fr;pointer-events:auto;}"
        ".rr-detail-collapse{min-height:0;overflow:hidden;}"
        ".rr-detail-content{padding:12px 13px;border-left:1px solid #E7EDF2;"
        "border-right:1px solid #E7EDF2;border-bottom:1px solid #E7EDF2;"
        "border-radius:0 0 9px 9px;background:#F7FAFC;}"
        "@media(prefers-reduced-motion:reduce){.rr-detail{transition:none!important;}}"
        ".rr-detail-top{display:flex;justify-content:space-between;align-items:center;gap:10px;}"
        ".rr-detail-top strong{font-size:.84rem;font-weight:700;color:var(--ink);}"
        ".rr-detail-main{display:flex;flex-direction:column;align-items:flex-end;justify-content:center;gap:2px;white-space:nowrap;text-align:right;}"
        ".rr-detail-total-top{display:flex;align-items:baseline;justify-content:flex-end;gap:4px;}"
        ".rr-detail-intensity{display:block;font-size:.62rem;color:var(--muted);line-height:1.15;font-weight:400;}"
        ".rr-detail-main b{font-size:1.22rem;font-weight:750;letter-spacing:-.035em;color:var(--ink);}"
        ".rr-detail-main small{font-size:.62rem;color:var(--muted);letter-spacing:0;}"
        ".rr-section-heading{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-top:15px;margin-bottom:7px;font-size:.77rem;font-weight:750;color:#516678;}"
        ".rr-section-heading small{font-size:.58rem;font-weight:400;color:var(--muted);text-align:right;}"
        ".rr-quantiles{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:4px;margin-top:0;}"
        ".rr-quantile{min-width:0;min-height:86px;display:flex;flex-direction:column;justify-content:center;gap:2px;text-align:center;padding:7px 1px;border-radius:6px;background:#EFF3F6;}"
        ".rr-quantile small{display:block;font-size:.59rem;color:#657383;font-weight:400;}"
        ".rr-quantile b{display:block;font-size:.77rem;font-weight:750;color:var(--ink);white-space:nowrap;}"
        ".rr-quantile em{font-style:normal;font-size:.58rem;font-weight:400;color:var(--muted2);}"
        ".rr-quantile .rr-quantile-desc{margin-top:0;font-size:.56rem;white-space:nowrap;}"
        ".rr-qmedian{background:#E2F5FD;}.rr-qmedian b{color:#126F96;font-size:.85rem;}"
        ".rr-detail-chances{display:flex;flex-direction:column;gap:5px;}"
        ".rr-detail-chance{min-width:0;display:grid;grid-template-columns:35% minmax(0,1fr);align-items:stretch;gap:0;overflow:hidden;border-radius:6px;background:transparent;}"
        ".rr-detail-chance-label{display:flex;flex-direction:row;align-items:center;justify-content:flex-start;gap:6px;padding:7px 7px;text-align:left;min-width:0;}"
        ".rr-detail-chance-value{min-width:0;display:flex;align-items:center;justify-content:flex-end;gap:12px;padding:7px 10px;}"
        ".rr-detail-chance small{font-size:.63rem;line-height:1.15;}"
        ".rr-detail-chance-threshold{font-weight:700;white-space:nowrap;flex-shrink:0;}"
        ".rr-detail-chance-meaning{font-weight:400;opacity:.8;white-space:normal;min-width:0;}"
        ".rr-detail-chance b{font-size:.92rem;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap;}"
        ".rr-detail-chance .rr-detail-chance-desc{font-size:.62rem;font-weight:400;min-width:58px;text-align:right;opacity:.9;white-space:nowrap;}"
        ".rr-detail-bars{margin-top:14px;padding-top:0;border-top:1px solid #E0E8EE;}"
        ".rr-detail-bar-hours{margin-bottom:7px;}"
        ".rr-time-range{display:flex;justify-content:center;white-space:nowrap;font-size:.55rem;letter-spacing:-.035em;color:var(--muted);}"
        ".rr-detail-bar-values .rr-bar{height:70px;max-width:var(--rr-detail-bar-max);}"
        ".rr-detail-note{font-size:.57rem;color:var(--muted2);margin-top:9px;line-height:1.35;}"
        "@container (min-width:430px){"
        ".rr-row.rr-c{grid-template-columns:3.8rem 6.4rem minmax(0,170px) 6.4rem;column-gap:8px;}"
        ".rr-row:not(.rr-c){grid-template-columns:3.1rem minmax(0,170px) 4.4rem;}"
        ".rr-day-weekday{font-size:.98rem;}.rr-day-calendar{font-size:.70rem;}"
        ".rr-secondary,.rr-tot .rr-total-main small,.rr-chip .rr-secondary{font-size:.68rem;}"
        ".rr-head .rr-slot span,.rr-detail-bar-hours .rr-slot span{font-size:.69rem;}"
        ".rr-ap{font-size:.53rem;}.rr-h{font-size:.69rem;}"
        ".rr-tot .rr-total-main b{font-size:1.12rem;}"
        ".rr-chip{height:3.2rem;}.rr-chip b,.rr-chip .rr-prob-head b{font-size:1.12rem;}"
        ".rr-chip .rr-percent-sign,.rr-chip .rr-chip-threshold{font-size:.68rem;}"
        ".rr-tot{height:3.2rem;}.rr-tot-partial{height:3.65rem;}"
        ".rr-legend,.rr-detail-note{font-size:.71rem;}"
        ".rr-detail-top strong{font-size:1rem;}.rr-detail-main b{font-size:1.4rem;}"
        ".rr-detail-main small{font-size:.75rem;}"
        ".rr-detail-main .rr-detail-intensity{font-size:.70rem;}"
        ".rr-quantile small,.rr-detail-chance small{font-size:.73rem;}"
        ".rr-section-heading{font-size:.94rem;}.rr-section-heading small{font-size:.73rem;}"
        ".rr-time-range{font-size:.69rem;}"
        ".rr-quantile b{font-size:.95rem;}.rr-qmedian b{font-size:1.05rem;}"
        ".rr-quantile em{font-size:.7rem;}"
        ".rr-quantile .rr-quantile-desc,.rr-detail-chance .rr-detail-chance-desc{font-size:.71rem;}"
        ".rr-detail-chance b{font-size:1rem;}"
        "}"
        "@media(max-width:460px){"
        ".rr-h-chance{white-space:normal;}"
        ".rr-row.rr-c{grid-template-columns:3.5rem 4.9rem minmax(0,170px) 4.5rem;column-gap:4px;}"
        ".rr-day-weekday{font-size:.78rem;}.rr-day-calendar{font-size:.59rem;}"
        ".rr-selectable.is-selected .rr-day{padding-left:6px;}"
        ".rr-detail-content{padding:10px 7px;}"
        ".rr-quantiles{gap:2px;}"
        ".rr-detail-chances{gap:4px;}"
        ".rr-detail-chance{grid-template-columns:40% minmax(0,1fr);}"
        ".rr-detail-chance-label{padding:7px 4px;gap:3px;}"
        ".rr-detail-chance-value{padding:7px 8px;gap:6px;}"
        ".rr-detail-chance-meaning{font-size:.56rem;}"
        ".rr-detail-chance .rr-detail-chance-desc{min-width:47px;}"
        ".rr-time-range{display:flex;flex-direction:column;align-items:center;font-size:.57rem;line-height:1.05;}"
        ".rr-section-heading{font-size:.78rem;}.rr-section-heading small{font-size:.59rem;}"
        ".rr-quantile b{font-size:.88rem;}.rr-qmedian b{font-size:.98rem;}"
        ".rr-quantile em{font-size:.58rem;}"
        ".rr-detail-chance b{font-size:.86rem;}"
        "}"
        ".rr-today-tag{display:none;font-size:.56rem;font-weight:400;color:var(--today);line-height:1.15;letter-spacing:.025em;}"
        ".rr-selectable.rr-today .rr-today-tag,.rr-detail.rr-today .rr-today-tag{display:block;}"
        ".rr-detail-date{display:flex;flex-direction:column;gap:2px;}"
        ".rr-now-tag{display:none;color:#087DB4;font-weight:800;font-size:.57rem;letter-spacing:.02em;margin-inline-end:4px;}"
        ".rr-detail-bar-hours .rr-slot.rr-now .rr-now-tag{display:block;}"
        ".rr-detail-bar-hours .rr-slot.rr-now .rr-time-range{color:#087DB4;font-weight:700;}"
        ".rr-detail-bar-values .rr-slot.rr-now .rr-bar,.rr-selectable .rr-slot.rr-now .rr-bar{outline:1.5px solid #168ECC;outline-offset:2px;border-radius:6px;}"
        ".rr-quantile-caption{min-height:2rem;display:flex;flex-direction:column;justify-content:center;gap:1px;margin-bottom:0;}"
        ".rr-quantile-help{display:block;min-height:0;font-size:.48rem!important;line-height:1.1;color:var(--muted2);font-weight:400;overflow-wrap:normal;}"
        ".rr-quantile-help i{font-style:italic;}"
        ".rr-chip-simple{gap:2px;padding:2px 3px;}"
        ".rr-simple-chance,.rr-simple-threshold{display:block;white-space:nowrap;letter-spacing:-.02em;}"
        ".rr-simple-chance{font-size:.78rem;font-weight:680;}"
        ".rr-simple-threshold{font-size:.68rem;font-weight:400;}"
        ".rr-total-simple{padding:2px;}"
        ".rr-total-plain{font-weight:680;color:var(--ink);font-size:.84rem;line-height:1.13;text-align:center;white-space:normal;overflow-wrap:normal;width:100%;}"
        ".rr-total-line{display:block;white-space:nowrap;}"
        ".rr-total-range{font-size:.77rem;color:var(--muted);}"
        "/* Both modes share the same .rr-row.rr-c grid; do not override it for Simple mode. */"
        "@container (min-width:430px){"
        "/* Desktop keeps the cards centered in exactly the same columns. */"
        ".rr-simple-chance{font-size:.88rem;}"
        ".rr-simple-threshold{font-size:.76rem;}"
        ".rr-total-plain{font-size:.94rem;}"
        ".rr-total-range{font-size:.86rem;}"
        ".rr-today-tag{font-size:.68rem;}"
        ".rr-quantile-help{font-size:.61rem!important;}"
        "}"
        + "</style>"
    )
    # Keep all interactive DOM inside the iframe. Streamlit's markdown element
    # can be re-rendered by React, which makes radio/label state unreliable.
    # Explicit click handling also permits closing an already expanded day.
    javascript = """<script>
    (() => {
      const card = document.querySelector('.rr-card');
      if (!card) return;
      const rows = Array.from(card.querySelectorAll('.rr-selectable[data-rr-day]'));
      const panels = Array.from(card.querySelectorAll('.rr-detail'));
      let selectedDay = -1;
      const slideMs = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 260;
      const closeTimers = new WeakMap();
      const formatPHTDate = (at) => {
        const parts = new Intl.DateTimeFormat('en-US', {
          timeZone:'Asia/Manila',year:'numeric',month:'2-digit',day:'2-digit'
        }).formatToParts(at);
        const part = name => parts.find(p => p.type === name).value;
        return `${part('year')}-${part('month')}-${part('day')}`;
      };

      function updateFrameHeight() {
        window.requestAnimationFrame(() => {
          const bounds = card.getBoundingClientRect();
          const height = Math.max(1, Math.ceil(bounds.bottom + 2));
          // Streamlit's modern st.iframe observes content height, but older
          // components.html versions can retain the original fixed height.
          // Trusted inline HTML is same-origin, so also resize its iframe
          // element directly when sandbox permissions allow this.
          try {
            const frame = window.frameElement;
            if (frame && frame.style) {
              frame.style.setProperty('height', `${height}px`, 'important');
              frame.style.setProperty('min-height', '0px', 'important');
            }
          } catch (err) { /* Cross-origin embedding: use the message below. */ }
          window.parent.postMessage({
            isStreamlitMessage: true,
            type: 'streamlit:setFrameHeight',
            height
          }, '*');
        });
      }

      function fitSimpleText() {
        if (card.hidden) return; // A hidden variant has no usable width.
        // Fit both chance lines together so the chance description always
        // retains a slightly larger type size than the rainfall description.
        card.querySelectorAll('.rr-chip-simple').forEach(chip => {
          const lines = [chip.querySelector('.rr-simple-chance'), chip.querySelector('.rr-simple-threshold')].filter(Boolean);
          if (lines.length !== 2) return;
          lines.forEach(el => {el.style.fontSize = '';});
          const base = lines.map(el => parseFloat(getComputedStyle(el).fontSize));
          const computed = getComputedStyle(chip);
          const room = chip.clientWidth - parseFloat(computed.paddingLeft) - parseFloat(computed.paddingRight) - 2;
          // These spans size to their text; compare with the card's inner
          // width rather than their own clientWidth (which hides overflow).
          // Apply a single scale factor to BOTH lines, retaining their
          // relative sizes even when a narrow card requires text fitting.
          let scale = 1;
          for (let attempt = 0; attempt < 18; attempt++) {
            const widest = Math.max(...lines.map(el => el.scrollWidth));
            if (widest <= room + 1) break;
            scale = Math.max(.58, scale * (room - 1) / widest);
            lines.forEach((el, i) => { el.style.fontSize = `${(base[i] * scale).toFixed(2)}px`; });
            if (scale <= .58) break;
          }
        });
        // Scale the ENTIRE daily total phrase together. Its two visual lines
        // must retain the same font size (e.g., "Light" / "to moderate").
        card.querySelectorAll('.rr-total-plain').forEach(group => {
          group.style.fontSize = '';
          const lines = Array.from(group.querySelectorAll('.rr-total-line'));
          if (!lines.length) return;
          let px = parseFloat(getComputedStyle(group).fontSize);
          const room = group.clientWidth;
          let attempts = 0;
          while (lines.some(line => line.scrollWidth > room + 1) && px > 7.5 && attempts++ < 35) {
            px = Math.max(7.5, px - .3);
            group.style.fontSize = `${px.toFixed(2)}px`;
          }
        });
      }

      function selectDay(nextIndex) {
        selectedDay = nextIndex;
        rows.forEach((row, index) => {
          const active = index === selectedDay;
          const pending = closeTimers.get(row);
          if (pending !== undefined) {clearTimeout(pending); closeTimers.delete(row);}
          row.classList.remove('is-collapsing');
          if (!active && row.classList.contains('is-selected') && slideMs) {
            row.classList.add('is-collapsing');
            const timer = setTimeout(() => {
              row.classList.remove('is-collapsing');closeTimers.delete(row);
            }, slideMs);
            closeTimers.set(row, timer);
          }
          row.classList.toggle('is-selected', active);
          row.setAttribute('aria-expanded', String(active));
        });
        panels.forEach((panel, index) => {
          const active = index === selectedDay;
          const wasOpen = panel.classList.contains('is-open');
          panel.inert = !active;
          panel.setAttribute('aria-hidden', String(!active));
          if (active && rows[index]) {
            // Reparent first, establish the collapsed grid, then animate open.
            // Otherwise the first expansion would appear instantly.
            rows[index].insertAdjacentElement('afterend', panel);
            if (!wasOpen) {
              panel.classList.remove('is-open');
              void panel.offsetHeight;
              requestAnimationFrame(() => {
                if (selectedDay === index) panel.classList.add('is-open');
              });
            }
          } else {
            panel.classList.remove('is-open');
          }
        });
        updateFrameHeight();
      }

      function refreshTime() {
        // The browser's system clock supplies the instant; display dates are
        // Philippine time because the forecast windows are labelled in PHT.
        const now = Date.now();
        const today = formatPHTDate(new Date(now));
        rows.forEach((row, index) => {
          row.classList.toggle('rr-today', row.dataset.rrDate === today);
          const panel = panels[index];
          if (panel) panel.classList.toggle('rr-today', panel.dataset.rrDate === today);
          for (const root of [row, panel]) {
            if (!root) continue;
            root.querySelectorAll('.rr-slot.rr-now').forEach(el => el.classList.remove('rr-now'));
          }
          const matchingSlots = Array.from(row.querySelectorAll('.rr-slots .rr-slot[data-start-ms]'));
          matchingSlots.forEach((slot, slotIndex) => {
            const from = Number(slot.dataset.startMs);
            const until = Number(slot.dataset.endMs);
            if (Number.isFinite(from) && from <= now && now < until) {
              slot.classList.add('rr-now');
              if (panel) {
                const detailSlot = panel.querySelectorAll('.rr-detail-bar-values .rr-slot')[slotIndex];
                const timeLabel = panel.querySelectorAll('.rr-detail-bar-hours .rr-slot')[slotIndex];
                if (detailSlot) detailSlot.classList.add('rr-now');
                if (timeLabel) timeLabel.classList.add('rr-now');
              }
            }
          });
        });
        updateFrameHeight();
      }

      rows.forEach((row, index) => {
        row.addEventListener('click', () => {
          selectDay(selectedDay === index ? -1 : index);
        });
      });
      if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(() => { updateFrameHeight(); }).observe(card);
      }
      if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(() => {fitSimpleText(); updateFrameHeight();});
      }
      window.addEventListener('resize', () => {fitSimpleText(); updateFrameHeight();});
      document.addEventListener('visibilitychange', () => {
        if (!document.hidden) refreshTime();
      });
      card.addEventListener('rr:activated', () => {
        fitSimpleText();refreshTime();updateFrameHeight();
      });
      window.addEventListener('load', () => {fitSimpleText();refreshTime();updateFrameHeight();});
      fitSimpleText();
      selectDay(selectedDay); // Start collapsed; the first tap opens the chosen day.
      refreshTime();
      window.setInterval(refreshTime, 60 * 1000);
    })();
    </script>"""
    viewport = (
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700;800&display=swap'>"
        "<style>html,body{margin:0;padding:0;min-height:0;height:auto;overflow-x:hidden;}"
        "body{min-width:0;}*{box-sizing:border-box;}</style>"
    )
    content = (f"<div class='rr-card' style='--rr-overview-bar-max:{_RAIN_BAR_MAX_PX[n_slots]}px;--rr-detail-bar-max:{detail_bar_max_px}px'>"
               f"<div class='rr-overview'>{head}{body}{legend}</div>"
               f"<div class='rr-details'>{details}</div></div>")
    if parts:
        return viewport, css, content, javascript
    return viewport + css + content + javascript



def render_six_day_forecast_html(period_results: dict[int, dict], threshold_result: dict | None, location_name: str) -> str:
    """Render a single embedded six-day card with native HTML controls.

    All 3/6-hourly and Standard/Simple variants share the same outer card and
    are rendered from the existing forecast calculations. Switching views is
    entirely client-side: it does not re-run Streamlit or request ECMWF data.
    """
    variant_parts = []
    variants = []
    for hours in (3, 6):
        if hours not in period_results or not period_results[hours].get("days"):
            continue
        for simple in (False, True):
            viewport, css, markup, script = render_rain_rows_html(
                period_results[hours], threshold_result, max_days=6,
                simple_mode=simple, parts=True,
            )
            variant_id = f"{hours}-{'simple' if simple else 'standard'}"
            markup = markup.replace(
                "<div class='rr-card'", f"<div class='rr-card rr-variant' data-rr-variant='{variant_id}'",
                1,
            )
            if variant_id != '3-standard' and variants:
                markup = markup.replace("data-rr-variant=", "hidden data-rr-variant=", 1)
            # A single control panel owns the frame size; inactive views cannot
            # override the height using their former private card bounds.
            script = script.replace(
                "document.querySelector('.rr-card')",
                f"document.querySelector('.rr-variant[data-rr-variant=\"{variant_id}\"]')",
            ).replace(
                "const bounds = card.getBoundingClientRect();",
                "const bounds = document.querySelector('.rr-root').getBoundingClientRect();",
            )
            variant_parts.append(markup)
            variants.append((variant_id, script))
    if not variants:
        return "<div>No rainfall forecast is available for these timing options.</div>"
    initial_id = variants[0][0]
    initial_hours = initial_id.split('-')[0]
    style = css + """<style>
    .rr-root{border:1px solid #e9edf2;border-radius:14px;padding:15px;background:#fff;color:#283746}
    .rr-root .rr-variant{padding:0;border:0;border-radius:0;background:transparent}
    .rr-variant[hidden]{display:none!important}
    .rr-variant:not([hidden]){animation:rrFadeIn .2s ease-out both}
    .rr-variant.is-closing{animation:rrFadeOut .2s ease-in both!important;pointer-events:none}
    @media(prefers-reduced-motion:reduce){.rr-variant:not([hidden]),.rr-variant.is-closing{animation:none!important}}
    .rr-card-controls{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin:14px 0 12px}
    .rr-card-title{font-size:18px;font-weight:720;line-height:1.3;letter-spacing:-.025em;color:#283746;margin:0}
    .rr-card-tabs{display:flex;align-items:center;padding:4px;border-radius:13px;background:#eef1f5;gap:3px}
    .rr-card-tabs button{font-family:inherit;border:0;border-radius:10px;background:transparent;color:#677586;padding:8px 16px;font-size:13px;font-weight:700;cursor:pointer;white-space:nowrap}
    .rr-card-tabs button[aria-pressed='true']{background:#fff;color:#1c3044;box-shadow:0 2px 8px #1e3b5020}
    .rr-simple-toggle{display:flex;align-items:center;gap:8px;color:#506070;font-size:13px;font-weight:650;cursor:pointer;user-select:none}
    .rr-simple-toggle input{appearance:none;position:relative;width:37px;height:22px;border-radius:99px;background:#d5dbe3;margin:0;cursor:pointer}
    .rr-simple-toggle input:after{content:'';position:absolute;width:16px;height:16px;top:3px;left:3px;background:#fff;border-radius:50%;box-shadow:0 1px 3px #0002;transition:left .16s}
    .rr-simple-toggle input:checked{background:#25a8db}.rr-simple-toggle input:checked:after{left:18px}
    .rr-root button:focus-visible,.rr-root input:focus-visible,.rr-root summary:focus-visible{outline:2px solid #159bd6;outline-offset:2px}
    .rr-intro{color:#687585;font-size:12px;margin-bottom:9px}
    .rr-reading-guide{margin-top:12px;color:#697789;font-size:12px;line-height:1.55}
    .rr-reading-guide summary{cursor:pointer;color:#576b80;font-weight:700}
    .rr-reading-guide .rr-guide-body{padding-top:7px}
    @media(max-width:720px){.rr-root{border-radius:10px;padding:12px 9px}.rr-card-title{font-size:16px}.rr-card-controls{gap:8px}.rr-card-tabs button{padding:7px 12px;font-size:12px}}
    @media(max-width:355px){.rr-card-tabs button{padding:7px 9px}}
    </style>""" + "<style>" + SHARED_RAIN_TYPOGRAPHY_CSS + "\n" + SHARED_FORECAST_DESIGN_CSS + "</style>"
    title = escape(location_name)
    controls = (
        f"<div class='rr-root rr-card' data-rain-period='{initial_hours}' data-rain-simple='false'>"
        f"<h2 class='rr-card-title'>Detailed rainfall forecast for {title}</h2>"
        "<div class='rr-card-controls'><div class='rr-card-tabs' role='group' aria-label='Rainfall timing'>"
        + "".join(
            f"<button type='button' data-rr-period='{hours}' aria-pressed='{str(hours==int(initial_hours)).lower()}'>{hours}-hourly</button>"
            for hours in (3, 6) if any(key.startswith(f'{hours}-') for key, _ in variants)
        )
        + "</div><label class='rr-simple-toggle'><input id='rr-mode-toggle' type='checkbox'>Simple mode</label></div>"
        "<div class='rr-intro'>Tap a day for details.</div>"
    )
    guide = (
        "<details class='rr-reading-guide'><summary>How to read this forecast</summary>"
        "<div class='rr-guide-body'>Bright blue shows the most likely rainfall amount; "
        "pale blue-gray shows a higher estimate. Chance cards show the likelihood "
        "of reaching their listed rainfall amounts. Tap a day for the full breakdown.</div></details>"
    )
    active_content = controls + "".join(variant_parts) + guide + "</div>"
    scripts = "".join(script for _, script in variants)
    controller = """<script>
    (()=>{
      const root=document.querySelector('.rr-root');
      const available=Array.from(root.querySelectorAll('.rr-variant'));
      const mode=root.querySelector('#rr-mode-toggle');
      const buttons=Array.from(root.querySelectorAll('[data-rr-period]'));
      const fadeMs=window.matchMedia('(prefers-reduced-motion: reduce)').matches?0:200;
      let transitionTimer=null;
      let transitionGeneration=0;
      function resize(){requestAnimationFrame(()=>{
        const h=Math.ceil(root.getBoundingClientRect().bottom+2);
        try {const frame=window.frameElement;if(frame){frame.style.setProperty('height',h+'px','important');frame.style.setProperty('min-height','0px','important');}} catch(e){}
        window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:h},'*');
      });}
      function show(){
        const generation=++transitionGeneration;
        if(transitionTimer!==null){window.clearTimeout(transitionTimer);transitionTimer=null;}
        const period=root.dataset.rainPeriod;
        const simple=mode.checked?'simple':'standard';
        let chosen=available.find(v=>v.dataset.rrVariant===period+'-'+simple);
        if(!chosen)chosen=available.find(v=>v.dataset.rrVariant.endsWith('-'+simple))||available[0];
        // Only one variant can participate in normal document flow. Previously
        // the outgoing and incoming variants were both rendered for 200 ms,
        // making the outgoing mode flash *below* the new card.
        const visible=available.find(v=>!v.hidden);
        available.forEach(v=>v.classList.remove('is-closing'));
        buttons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.rrPeriod===chosen.dataset.rrVariant.split('-')[0])));
        root.dataset.rainSimple=String(mode.checked);
        function activate(){
          if(generation!==transitionGeneration)return;
          available.forEach(v=>{
            v.hidden=v!==chosen;
            v.classList.remove('is-closing');
          });
          transitionTimer=null;
          requestAnimationFrame(()=>chosen.dispatchEvent(new Event('rr:activated')));
          resize();
        }
        if(visible&&visible!==chosen&&fadeMs){
          // Sequential 200-ms fade out / fade in; never show two full-height
          // variants at once. Rapid toggles cancel the pending activation.
          available.forEach(v=>{if(v!==visible)v.hidden=true;});
          void visible.offsetWidth;
          visible.classList.add('is-closing');
          transitionTimer=window.setTimeout(activate,fadeMs);
        }else activate();
        resize();
      }
      buttons.forEach(b=>b.addEventListener('click',()=>{root.dataset.rainPeriod=b.dataset.rrPeriod;show();}));
      mode.addEventListener('change',show);
      if(typeof ResizeObserver!=='undefined')new ResizeObserver(resize).observe(root);
      window.addEventListener('resize',resize);
      if(document.fonts?.ready)document.fonts.ready.then(resize);
      show();
    })();
    </script>"""
    return viewport + style + active_content + scripts + controller


HEAT_INDEX_CATEGORIES = [
    ("caution", "Caution", "Caution (27°C+)", 27, "#F4C430"),
    ("extreme_caution", "Extreme Caution", "Extreme Caution (33°C+)", 33, "#FF9F1C"),
    ("danger", "Danger", "Danger (42°C+)", 42, "#E8450C"),
    ("extreme_danger", "Extreme Danger", "Extreme Danger (52°C+)", 52, "#C0122F"),
]

def _temperature_stat_number(value):
    """Return a finite Celsius estimate or None (missing forecast)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 2) if math.isfinite(number) else None


def _temperature_severity(heat_index):
    if heat_index is None:
        return {"label": "Unavailable", "key": "unknown", "color": "#8994A1"}
    for key, short, _long, start, color in reversed(HEAT_INDEX_CATEGORIES):
        if heat_index >= start:
            return {"label": short, "key": key, "color": color}
    return {"label": "Below caution", "key": "below_caution", "color": "#7B9BA4"}


def _air_temperature_descriptor(value):
    """Informal reading aid for peak air temperature, not a hazard scale."""
    if value is None:
        return "Unavailable"
    if value < 24:
        return "Cool"
    if value < 30:
        return "Warm"
    if value < 35:
        return "Hot"
    return "Very hot"


def render_temperature_forecast_html(temp_result: dict, location_name: str) -> str:
    """Interactive six-day Heat Index-first card based on stored 3-hourly ENS data.

    Peaks are the maximum of available *3-hourly median* values in a local day.
    P10 and P90 at the selected peak time are not percentiles of daily maxima.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for step in temp_result.get("steps", []):
        utc_time = step.get("time_utc")
        if not isinstance(utc_time, datetime):
            continue
        local_time = utc_time.astimezone(PH_TZ)
        values = {}
        for kind, source in (("hi", "heat_index_stats"), ("temp", "temperature_stats")):
            stats = step.get(source) or {}
            values[kind] = {
                key: _temperature_stat_number(stats.get(key))
                for key in ("p10", "median", "p90")
            }
        grouped[local_time.date()].append({
            "time": local_time.isoformat(),
            "label": f"{local_time:%I %p}".lstrip("0"),
            "hour": local_time.hour,
            **values,
        })
    if not grouped:
        return "<div>Heat Index data isn't available for this location.</div>"

    today = datetime.now(PH_TZ).date()
    upcoming = sorted(date for date in grouped if date >= today)
    # If the stored run is stale, retain a useful (clearly dated) sample.
    days_to_show = (upcoming if upcoming else sorted(grouped))[:6]
    records = []
    for day in days_to_show:
        steps = sorted(grouped[day], key=lambda item: item["time"])
        def peak(kind):
            candidates = [(i, entry[kind]["median"]) for i, entry in enumerate(steps)
                          if entry[kind]["median"] is not None]
            if not candidates:
                return None
            ix, value = max(candidates, key=lambda item: item[1])
            return {"index": ix, "value": value, "time": steps[ix]["label"],
                    "p10": steps[ix][kind]["p10"], "p90": steps[ix][kind]["p90"]}

        hi_peak, air_peak = peak("hi"), peak("temp")
        air_low_entries = [entry["temp"]["median"] for entry in steps
                           if entry["temp"]["median"] is not None]
        air_low = min(air_low_entries) if air_low_entries else None
        records.append({
            "date": day.isoformat(),
            "weekday": day.strftime("%A"),
            "shortWeekday": day.strftime("%a"),
            "day": f"{day.day} {day:%b}",
            "full": f"{day:%A}, {day.day} {day:%B}",
            "today": day == today,
            "steps": steps,
            "hiPeak": hi_peak,
            "airPeak": air_peak,
            "airLow": air_low,
            "severity": _temperature_severity(hi_peak["value"] if hi_peak else None),
            "airDescriptor": _air_temperature_descriptor(air_peak["value"] if air_peak else None),
        })
    serialized = json.dumps(records, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    return (_TEMPERATURE_CARD_TEMPLATE
            .replace("__TEMPERATURE_DATA__", serialized)
            .replace("__LOCATION__", escape(location_name))
            .replace("__SHARED_RAIN_TYPOGRAPHY__", SHARED_RAIN_TYPOGRAPHY_CSS + "\n" + SHARED_FORECAST_DESIGN_CSS))


_TEMPERATURE_CARD_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Simpleweather | Heat Index</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700;800&display=swap');
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:transparent;overflow-x:hidden}
body,button,summary{font-family:'Source Sans 3','Source Sans Pro',sans-serif}
body{color:#283746;font-size:13px}
button{cursor:pointer}
[hidden]{display:none!important}
.temp-root{width:100%;max-width:1050px;margin:0 auto;border:1px solid #e9edf2;border-radius:14px;padding:15px;background:#fff;--temp-ink:#283746;--temp-muted:#687585;--temp-accent:#238FC2;--temp-hi:#DD8B47}
.temp-root button:focus-visible,.temp-root input:focus-visible,.temp-root summary:focus-visible{outline:2px solid #159bd6;outline-offset:3px}
.rr-card-title{font-size:var(--rain-title-size);font-weight:720;line-height:1.3;margin:0;color:var(--rain-ink);letter-spacing:-.025em}
.temp-top-subtitle{margin:3px 0 0;color:var(--rain-muted);font-size:var(--rain-secondary-size);font-weight:400}
.temp-card-controls{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:15px 0 11px;flex-wrap:wrap;position:relative;z-index:2;touch-action:manipulation}
.temp-card-controls .rr-card-tabs{max-width:100%}
.temp-card-controls .rr-simple-toggle{margin-left:auto}
.rr-card-tabs{display:flex;align-items:center;gap:3px;border-radius:13px;padding:4px;background:#eef1f5;min-width:0;flex:0 1 auto}
.rr-card-tabs button{color:#677586;border:0;border-radius:10px;background:transparent;padding:8px 14px;white-space:nowrap;line-height:1.2;font-size:var(--rain-tabs-size);font-weight:700;touch-action:manipulation;-webkit-tap-highlight-color:transparent}
.rr-card-tabs button[aria-selected="true"]{background:#fff;color:#1c3044;box-shadow:0 2px 8px #1e3b5020}
.rr-simple-toggle{display:flex;align-items:center;gap:8px;font-size:13px;color:#506070;font-weight:650;cursor:pointer;white-space:nowrap;flex:0 0 auto;user-select:none}
.rr-simple-toggle input{appearance:none;margin:0;width:37px;height:22px;position:relative;border-radius:50px;background:#d5dbe3;cursor:pointer;touch-action:manipulation}
.rr-simple-toggle input:after{content:'';position:absolute;left:3px;top:3px;width:16px;height:16px;background:#fff;border-radius:50%;box-shadow:0 1px 3px #0002;transition:left .16s}
.rr-simple-toggle input:checked{background:#25a8db}.rr-simple-toggle input:checked:after{left:18px}
.temp-guide{font-size:var(--rain-secondary-size);font-weight:400;color:var(--rain-muted);margin:0 0 10px}
.temp-chart-shell{border:1px solid #edf0f4;border-radius:12px;padding:12px 11px 6px;background:#fdfefe;min-width:0}
.temp-chart-top{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:2px}
.rr-section-heading{font-size:var(--rain-section-size);font-weight:750;color:var(--rain-section)}
.temp-chart-meta{font-size:var(--rain-micro-size);font-weight:400;color:var(--rain-muted)}
#temp-graph{width:100%;height:255px;display:block;overflow:visible;touch-action:pan-y}
.temp-plot-hit{fill:transparent;cursor:pointer;pointer-events:all;touch-action:manipulation}
.temp-day-button{touch-action:manipulation;-webkit-tap-highlight-color:transparent}
.temp-plot-hit:focus-visible{stroke:#159bd6;stroke-width:2;stroke-dasharray:3 4;outline:none}
.temp-legend{display:flex;align-items:center;justify-content:center;gap:15px;flex-wrap:wrap;margin:0 0 3px;color:var(--rain-muted);font-size:var(--rain-secondary-size)}
.temp-legend-entry{display:inline-flex;gap:5px;align-items:center}
.temp-legend-line{width:17px;height:3px;border-radius:5px;background:var(--temp-accent)}
.temp-legend-range{width:16px;height:9px;border-radius:3px;background:#B5DCEB88;border:1px solid #B5DCEB}
.temp-chart-selected{font-size:var(--rain-secondary-size);color:var(--rain-muted);text-align:center;margin:2px 0 0}
.temp-graph-options{color:var(--rain-muted);font-size:var(--rain-secondary-size);margin:9px 4px 2px}
.temp-graph-options summary{cursor:pointer;color:#576b80;font-weight:700}
.temp-graph-options-inner{display:flex;flex-wrap:wrap;align-items:center;gap:9px 18px;padding-top:8px}
.temp-graph-options label{display:inline-flex;align-items:center;gap:5px;cursor:pointer}
.temp-graph-options input{accent-color:#238FC2}

.temp-section{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin:17px 3px 7px}
.temp-section small{font-size:var(--rain-secondary-size);color:var(--rain-muted);font-weight:400}
.temp-days{display:flex;flex-direction:column;gap:2px}
.temp-day-wrap{border-radius:10px;min-width:0}
.temp-day-button{border:0;width:100%;display:grid;grid-template-columns:minmax(0,1fr) auto 15px;gap:12px;align-items:center;padding:9px 10px;background:transparent;color:var(--rain-ink);text-align:left;border-radius:10px;min-height:66px;transition:background-color .16s ease}
.temp-day-button:hover{background:#f6f8fb}
.temp-day-button[aria-expanded="true"]{background:#eef6fa;border-radius:10px 10px 0 0;box-shadow:inset 3px 0 0 #31ace0}
.temp-day-date{display:flex;flex-direction:column;gap:4px;min-width:0}
.temp-day-heading{display:flex;align-items:center;gap:6px;flex-wrap:wrap;line-height:1.1}
.temp-day-date .rr-today-tag{margin-bottom:2px}
.temp-expanded-date{display:flex;flex-direction:column;align-items:flex-start;gap:3px}
.rr-day-weekday{font-size:var(--rain-day-size);font-weight:700;color:var(--rain-ink)}
.rr-day-calendar{font-size:var(--rain-date-size);font-weight:400;color:var(--rain-date)}
.rr-today-tag{font-size:var(--rain-today-size);font-weight:400;color:var(--rain-today);letter-spacing:.025em}
.temp-day-numbers{display:flex;gap:14px;align-items:center;justify-content:flex-end;min-width:0;text-align:right}
.temp-day-metric{min-width:0;display:flex;align-items:flex-end;flex-direction:column;gap:2px}
.temp-day-metric b{font-size:var(--rain-value-size);font-weight:750;color:var(--rain-ink);line-height:1.05;white-space:nowrap}
.temp-day-metric small{font-size:var(--rain-secondary-size);font-weight:400;color:var(--rain-muted);line-height:1.2;white-space:nowrap}
.temp-day-metric.secondary b{font-size:var(--rain-secondary-size);font-weight:700;color:#516678}
.temp-day-metric.secondary small{font-size:var(--rain-micro-size)}
.temp-risk-dot{width:6px;height:6px;display:inline-block;border-radius:50%;background:var(--severity,#8894a1);vertical-align:middle;margin-right:5px}
.temp-day-metric b.temp-descriptor{font-size:var(--rain-descriptor-size);font-weight:680;line-height:1.13;white-space:normal;max-width:143px;overflow-wrap:normal;text-align:right}
.temp-chevron{width:14px;height:14px;stroke:#8593A2;stroke-width:2;fill:none;transition:transform .26s ease}
.temp-day-button[aria-expanded="true"] .temp-chevron{transform:rotate(180deg)}
.temp-detail{display:none;min-height:0;pointer-events:none}
.temp-detail.is-open{display:block;pointer-events:auto;animation:tempDetailAppear .20s ease-out both}
@keyframes tempDetailAppear{from{opacity:.68;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
.temp-detail-collapse{min-height:0;overflow:hidden}
.temp-detail-body{background:#f9fbfd;border:1px solid #dfe9f0;border-top:0;border-radius:0 0 10px 10px;padding:13px 12px}

.temp-detail-top{display:flex;justify-content:space-between;align-items:center;gap:10px}
.temp-detail-top strong{font-size:var(--rain-detail-title-size);font-weight:700;color:var(--rain-ink)}
.temp-detail-top .temp-detail-value{font-size:var(--rain-total-detail-size);font-weight:750;line-height:1.05;color:var(--rain-ink)}
.temp-detail-top .temp-detail-caption{font-size:var(--rain-secondary-size);font-weight:400;color:var(--rain-muted)}
.temp-detail-descriptor{display:inline-flex;align-items:center;justify-content:flex-end;gap:5px;margin-top:5px;font-size:var(--rain-secondary-size);font-weight:700;color:var(--rain-section);line-height:1.2}
.temp-detail-time{font-size:var(--rain-secondary-size);font-weight:400;color:var(--rain-muted);margin-top:3px}
.temp-details-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:13px}
.temp-summary-tile{padding:9px 8px;border:1px solid #e7eaf0;border-radius:9px;background:#fff;min-width:0}
.temp-summary-tile small{display:block;color:var(--rain-muted);font-size:var(--rain-secondary-size);line-height:1.2}
.temp-summary-tile b{display:block;margin-top:4px;font-weight:750;color:var(--rain-ink);font-size:var(--rain-value-size)}
.temp-summary-tile.is-hi-classified{background:var(--severity-bg,#fff);border-color:var(--severity-border,#e7eaf0)}
.temp-risk-inline{display:inline-flex;align-items:center;gap:5px;font-size:var(--rain-secondary-size);font-weight:650;color:var(--rain-section);margin:4px 0 0}
.temp-hour-title{margin:13px 0 6px}
.temp-hour-scroll{overflow-x:auto;overscroll-behavior-inline:contain;padding-bottom:4px}
.temp-hours{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(57px,1fr);gap:5px;min-width:max-content}
.temp-hour{min-width:57px;border-radius:8px;background:var(--severity-bg,#fff);border:1px solid var(--severity-border,#e7eaf0);padding:7px 5px;text-align:center;color:var(--rain-ink)}
.temp-hour.is-current-period{outline:2px solid var(--sw-accent,#1786B6);outline-offset:-2px}
.temp-hour-now{font-weight:700;color:var(--sw-accent,#1786B6);margin-left:3px}
.temp-hour-peak{font-weight:700;color:var(--rain-section)}
.temp-hour-time{font-size:var(--rain-micro-size);color:var(--rain-muted);display:block}
.temp-hour-main{font-size:var(--rain-value-size);font-weight:750;display:block;line-height:1.1;margin:5px 0 3px}
.temp-hour-secondary{font-size:var(--rain-micro-size);color:var(--rain-muted);display:block}
.temp-hour-labels{color:var(--rain-muted);font-size:var(--rain-micro-size);margin-top:4px}
.temp-note{color:var(--rain-muted);font-size:var(--rain-secondary-size);line-height:1.5;margin-top:12px}
.temp-note summary{cursor:pointer;font-weight:700;color:#576b80}
.temp-note p{margin:7px 0 0}
@media(max-width:430px){
  .temp-detail-top{flex-direction:column;align-items:stretch;gap:7px}
  .temp-detail-top>div:last-child{text-align:left!important}
  .temp-detail-descriptor{justify-content:flex-start}
}
@media(max-width:720px){.temp-root{border-radius:10px;padding:12px 9px}.temp-chart-shell{padding:10px 6px 5px}.temp-day-button{padding:9px 7px;gap:6px;min-height:62px}.temp-day-numbers{gap:10px}}
@media(max-width:355px){.rr-card-tabs button{padding:7px 8px;font-size:10.5px}.rr-simple-toggle{font-size:11px;gap:4px}.rr-simple-toggle input{width:33px;height:20px}.rr-simple-toggle input:after{width:14px;height:14px}.rr-simple-toggle input:checked:after{left:16px}.temp-day-numbers{gap:7px}.temp-day-metric b.temp-descriptor{max-width:101px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
__SHARED_RAIN_TYPOGRAPHY__
</style>
</head>
<body>
<main class="temp-root" id="temp-root">
  <h2 class="rr-card-title">Heat Index forecast for __LOCATION__</h2>
  <div class="temp-top-subtitle">ECMWF ENS · 3-hourly probabilistic estimates</div>
  <div class="temp-card-controls">
    <div class="rr-card-tabs" role="tablist" aria-label="Forecast variable">
      <button type="button" role="tab" data-variable="hi" aria-selected="true" aria-controls="temp-graph">Heat Index</button>
      <button type="button" role="tab" data-variable="temp" aria-selected="false" aria-controls="temp-graph">Air temperature</button>
    </div>
    <label class="rr-simple-toggle"><input type="checkbox" id="temp-simple">Simple mode</label>
  </div>
  <div class="temp-guide">Tap a peak or a day below to view details.</div>
  <section class="temp-chart-shell" aria-label="Heat index and air temperature graph">
    <div class="temp-chart-top"><span class="rr-section-heading" id="temp-chart-title">Heat Index over time</span><span class="temp-chart-meta">3-hourly most likely estimate</span></div>
    <svg id="temp-graph" role="img" aria-label="Interactive forecast graph with a marker for each daily peak. Tap a day marker to expand the corresponding daily forecast."></svg>
    <div class="temp-legend"><span class="temp-legend-entry"><span class="temp-legend-line" id="temp-legend-line"></span><span>Most likely</span></span><span class="temp-legend-entry"><span class="temp-legend-range" id="temp-legend-range"></span><span>Likely range</span></span></div>
    <div class="temp-chart-selected" id="temp-graph-caption" hidden></div>
    <details class="temp-graph-options"><summary>Graph options</summary>
      <div class="temp-graph-options-inner">
        <label><input type="checkbox" id="temp-show-range" checked> Likely range</label>
        <label id="temp-reference-label"><input type="checkbox" id="temp-show-reference" checked> Heat Index category lines</label>
        <label><input type="checkbox" id="temp-show-compare"> Compare both variables</label>
      </div>
    </details>
  </section>
  <div class="temp-section"><span class="rr-section-heading">Day-by-day outlook</span><small>Peak of the most likely estimate</small></div>
  <div class="temp-days" id="temp-days"></div>
  <details class="temp-note"><summary>How to read this forecast</summary>
    <p>The highlighted Heat Index is the default. Daily peaks are the highest available 3-hourly median estimates, not exact daily maxima. The shaded band shows P10–P90 estimates at individual times; it is not the probability range for daily maxima.</p>
    <p>Simple mode uses Heat Index caution categories and informal air-temperature descriptions in the daily overview. Detailed figures and graph values remain available. Daily periods use Philippine local time (UTC+8); partial days use only available forecast steps.</p>
  </details>
</main>
<script>
(() => {
'use strict';
const days=__TEMPERATURE_DATA__;
const root=document.getElementById('temp-root');
const svg=document.getElementById('temp-graph');
const list=document.getElementById('temp-days');
const tabs=Array.from(root.querySelectorAll('[data-variable]'));
const simple=document.getElementById('temp-simple');
const showRange=document.getElementById('temp-show-range');
const showReference=document.getElementById('temp-show-reference');
const showCompare=document.getElementById('temp-show-compare');
const categories=[
 {value:27,name:'Caution',color:'#F4C430'},
 {value:33,name:'Extreme caution',color:'#FF9F1C'},
 {value:42,name:'Danger',color:'#E8450C'},
 {value:52,name:'Extreme danger',color:'#C0122F'}
];
const NS='http://www.w3.org/2000/svg';
let variable='hi',selected=-1;
const ink='#283746', muted='#687585';
const palette={hi:{stroke:'#DF8744',shade:'#F6D6B4'},temp:{stroke:'#238FC2',shade:'#C8E6F1'}};
function fmt(value,unit='°C'){return Number.isFinite(value)?`${Math.round(value)}${unit}`:'—'}
function dec(value){return Number.isFinite(value)?`${value.toFixed(1)}°C`:'—'}
function severityFor(value){
  if(!Number.isFinite(value))return {label:'Unavailable',color:'#8994A1'};
  for(let i=categories.length-1;i>=0;i--)if(value>=categories[i].value)return {label:categories[i].name,color:categories[i].color};
  return {label:'Below caution',color:'#7B9BA4'};
}
function classColors(value){
  const tone=severityFor(value),hex=tone.color.replace('#','');
  const r=parseInt(hex.slice(0,2),16),g=parseInt(hex.slice(2,4),16),b=parseInt(hex.slice(4,6),16);
  return `--severity:${tone.color};--severity-bg:rgba(${r},${g},${b},.20);--severity-border:rgba(${r},${g},${b},.49)`;
}
function safe(text){return String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
let lastFrameHeight=0, frameHeightPending=false;
function height(){
  if(frameHeightPending)return;
  frameHeightPending=true;
  requestAnimationFrame(()=>{
    frameHeightPending=false;
    // Measure the card itself: document scrollHeight can retain a previously
    // expanded iframe's viewport height and leave a blank area on collapse.
    const h=Math.max(1,Math.ceil(root.getBoundingClientRect().height+2));
    if(Math.abs(h-lastFrameHeight)<2)return;
    lastFrameHeight=h;
    // Modern inline iframes can be adjusted directly when same-origin.
    // Never measure their current height to calculate h (feedback loop).
    try{
      const frame=window.frameElement;
      if(frame&&Math.abs(frame.getBoundingClientRect().height-h)>3){
        frame.style.setProperty('height',h+'px','important');
        frame.style.setProperty('min-height','0px','important');
      }
    }catch(e){} // Cross-origin embedding: the Streamlit message below handles it.
    try{window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:h},'*')}catch(e){}
  });
}
function animateHeight(){
  height();
  // Follow up after the DOM and any entrance animation have settled so the
  // embedding frame always receives the collapsed size as well as expanded.
  window.setTimeout(height,80);
  window.setTimeout(height,300);
}
function dayHTML(day,i){
  const selectedNow=selected===i;
  const todayTag=day.today?'<span class="rr-today-tag">TODAY</span>':'';
  const hiVal=day.hiPeak?.value??null,airVal=day.airPeak?.value??null;
  const p=day.hiPeak;
  const headline=p?`<div class="temp-detail-value">${fmt(p.value)}</div>`:'<div class="temp-detail-value">—</div>';
  const stats=[
    {name:'P10 · lower estimate',value:p?.p10??null,classify:true},
    {name:'P90 · higher estimate',value:p?.p90??null,classify:true},
    {name:'Peak air temperature',value:airVal,classify:false}
  ];
  const hours=day.steps.map((step,j)=>`<div class="temp-hour" data-valid-time="${safe(step.time)}" style="${classColors(step.hi.median)}">
    <span class="temp-hour-time">${safe(step.label.toLowerCase())}${p&&p.index===j?' <span class="temp-hour-peak">(peak)</span>':''}<span class="temp-hour-now" hidden>NOW</span></span><span class="temp-hour-main">${fmt(step.hi.median,'°')}</span>
    <span class="temp-hour-secondary">Air ${fmt(step.temp.median,'°')}</span></div>`).join('');
  const peakTime=p?.time?` (${safe(p.time.toLowerCase())})`:'';
  return `<div class="temp-day-wrap" data-day-wrap="${i}">
    <button type="button" class="temp-day-button" data-day="${i}" aria-expanded="${selectedNow}" aria-controls="temp-detail-${i}">
      <span class="temp-day-date">${todayTag}<span class="rr-day-weekday">${safe(day.shortWeekday)}</span><span class="rr-day-calendar">${safe(day.day)}</span></span>
      <span class="temp-day-numbers">
        <span class="temp-day-metric"><b class="temp-number" ${simple.checked?'hidden':''}>${fmt(hiVal,'°')}</b><b class="temp-descriptor" style="--severity:${day.severity.color}" ${simple.checked?'':'hidden'}><i class="temp-risk-dot" aria-hidden="true"></i>${safe(day.severity.label)}</b><small>Heat Index</small></span>
        <span class="temp-day-metric secondary"><b class="temp-number" ${simple.checked?'hidden':''}>${fmt(airVal,'°')}</b><b class="temp-descriptor" ${simple.checked?'':'hidden'}>${safe(day.airDescriptor)}</b><small>Air temp</small></span>
      </span><svg class="temp-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 6 5 5 5-5"/></svg>
    </button>
    <section class="temp-detail ${selectedNow?'is-open':''}" id="temp-detail-${i}" aria-label="Detailed Heat Index forecast for ${safe(day.full)}">
      <div class="temp-detail-collapse"><div class="temp-detail-body">
      <div class="temp-detail-top"><div class="temp-expanded-date">${todayTag}<strong>${safe(day.full)}</strong></div>
      <div style="text-align:right"><div class="temp-detail-caption">Peak Heat Index${peakTime}</div>${headline}
        <div class="temp-detail-descriptor"><span class="temp-risk-dot" style="--severity:${day.severity.color}"></span>${safe(day.severity.label)}</div>
      </div></div>
      <div class="temp-details-grid">${stats.map(stat=>`<div class="temp-summary-tile ${stat.classify?'is-hi-classified':''}" ${stat.classify?`style="${classColors(stat.value)}"`:''}>
        <small>${safe(stat.name)}</small><b>${fmt(stat.value)}</b></div>`).join('')}</div>
      <div class="rr-section-heading temp-hour-title">3-hourly forecast</div>
      <div class="temp-hour-scroll"><div class="temp-hours">${hours}</div></div>
      <div class="temp-hour-labels">Heat Index is emphasized; air temperature appears below each value.</div>
    </div></div></section>
  </div>`;
}
function highlightCurrentPeriod(){
  const now=Date.now(),period=3*60*60*1000;
  list.querySelectorAll('.temp-hour[data-valid-time]').forEach(tile=>{
    const start=Date.parse(tile.dataset.validTime);
    const current=Number.isFinite(start)&&start<=now&&now<start+period;
    tile.classList.toggle('is-current-period',current);
    const marker=tile.querySelector('.temp-hour-now');
    if(marker)marker.hidden=!current;
  });
}
function updateRows(){list.innerHTML=days.map(dayHTML).join('');highlightCurrentPeriod();height()}
function selectDay(i){if(i<0||i>=days.length)return;selected=selected===i?-1:i;updateRows();drawGraph();animateHeight()}
// One stable listener survives every re-render and works when tapping child spans/SVGs.
list.addEventListener('click',event=>{
  const button=event.target.closest?.('.temp-day-button');
  if(button)selectDay(Number(button.dataset.day));
});
function s(tag,attrs={},text=null){const el=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!==null)el.textContent=text;return el}
function add(tag,attrs={},text=null,parent=svg){const el=s(tag,attrs,text);parent.appendChild(el);return el}
function curve(points){
 if(!points.length)return '';
 if(points.length===1)return `M${points[0].x} ${points[0].y}`;
 let d=`M${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`;
 const slopes=[];
 for(let i=0;i<points.length;i++){
   const before=i>0?(points[i].y-points[i-1].y)/(points[i].x-points[i-1].x):null;
   const after=i<points.length-1?(points[i+1].y-points[i].y)/(points[i+1].x-points[i].x):null;
   slopes.push(before===null?after:after===null?before:before*after<=0?0:2*before*after/(before+after));
 }
 for(let i=0;i<points.length-1;i++){
   const a=points[i],b=points[i+1],dx=(b.x-a.x)/3;
   d+=` C${(a.x+dx).toFixed(2)} ${(a.y+slopes[i]*dx).toFixed(2)} ${(b.x-dx).toFixed(2)} ${(b.y-slopes[i+1]*dx).toFixed(2)} ${b.x.toFixed(2)} ${b.y.toFixed(2)}`;
 }
 return d;
}
function drawGraph(){
 svg.replaceChildren();
 const w=Math.max(260,Math.floor(svg.getBoundingClientRect().width)||320),h=255,l=43,r=w-12,t=32,b=211;
 svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
 const flat=[];days.forEach((day,i)=>day.steps.forEach(step=>flat.push({day:i,step,ms:Date.parse(step.time)})));
 if(!flat.length)return;
 const secondaryVariable=variable==='hi'?'temp':'hi';
 // Both charts always share the same Y-axis, independently of the selected tab.
 // Category lines are overlays only and do not expand the forecast-based range.
 const values=flat.flatMap(point=>[...Object.values(point.step.hi).filter(Number.isFinite),...Object.values(point.step.temp).filter(Number.isFinite)]);
 if(!values.length){add('text',{x:w/2,y:100,'text-anchor':'middle',fill:muted,'font-size':12},'No data for this forecast');return}
 const low=Math.floor((Math.min(...values)-2)/5)*5;
 let high=Math.ceil((Math.max(...values)+2)/5)*5;
 if(high-low<10)high=low+10;
 const x0=flat[0].ms,x1=flat[flat.length-1].ms;
 const x=ms=>l+(x1===x0 ? .5 :(ms-x0)/(x1-x0))*(r-l);
 const y=val=>b-(val-low)/(high-low)*(b-t);
 for(let i=0;i<=4;i++){
   const v=low+(high-low)*i/4,yy=y(v);
   add('line',{x1:l,y1:yy,x2:r,y2:yy,stroke:'#E7EDF2','stroke-width':1,'stroke-dasharray':'3 4'});
   add('text',{x:l-6,y:yy+3,'text-anchor':'end',fill:muted,'font-size':10},`${Math.round(v)}°`);
 }
 const cfg=palette[variable];
 if(showReference.checked&&variable==='hi')categories.forEach(cat=>{
    const yy=y(cat.value);
    if(yy<t||yy>b)return;
    add('line',{x1:l,y1:yy,x2:r,y2:yy,stroke:cat.color,'stroke-width':1.15,'stroke-dasharray':'4 4','stroke-opacity':'.85'});
    // Classification labels are just above the line, close to the Y-axis.
    const labelWidth=Math.min(w-60,cat.name.length*5.25+9);
    add('rect',{x:l+2,y:yy-16,width:labelWidth,height:13,rx:3,fill:'#fff','fill-opacity':'.9'});
    add('text',{x:l+6,y:yy-6,'text-anchor':'start',fill:cat.color,'font-size':9,'font-weight':750},cat.name);
 });
 if(showCompare.checked){
    const secondaryGroups=[];let secondary=[];
    flat.forEach(point=>{
      const value=point.step[secondaryVariable].median;
      if(Number.isFinite(value))secondary.push({x:x(point.ms),y:y(value)});
      else{if(secondary.length)secondaryGroups.push(secondary);secondary=[]}
    });if(secondary.length)secondaryGroups.push(secondary);
    secondaryGroups.forEach(g=>add('path',{d:curve(g),fill:'none',stroke:palette[secondaryVariable].stroke,'stroke-width':1.65,'stroke-dasharray':'4 4','stroke-opacity':'.85'}));
 }
 const groups=[];let segment=[];
 flat.forEach(point=>{
   const {median,p10,p90}=point.step[variable];
   if([median,p10,p90].every(Number.isFinite))segment.push({x:x(point.ms),lo:y(p10),hi:y(p90)});
   else{if(segment.length)groups.push(segment);segment=[]}
 });if(segment.length)groups.push(segment);
 if(showRange.checked)groups.forEach(group=>{
   if(group.length<2)return;
   const upper=group.map(p=>({x:p.x,y:p.hi})),lower=group.map(p=>({x:p.x,y:p.lo}));
   const fill=curve(upper)+' '+curve(lower.reverse()).replace(/^M/,'L')+' Z';
   add('path',{d:fill,fill:cfg.shade,'fill-opacity':'.5',stroke:'none'});
 });
 let midSeg=[];const midGroups=[];
 flat.forEach(point=>{
   const value=point.step[variable].median;
   if(Number.isFinite(value))midSeg.push({x:x(point.ms),y:y(value)});
   else{if(midSeg.length)midGroups.push(midSeg);midSeg=[]}
 });if(midSeg.length)midGroups.push(midSeg);
 midGroups.forEach(g=>add('path',{d:curve(g),fill:'none',stroke:cfg.stroke,'stroke-width':2.65,'stroke-linecap':'round','stroke-linejoin':'round'}));
 days.forEach((day,i)=>{
   const these=flat.filter(p=>p.day===i);if(!these.length)return;
   const first=x(these[0].ms),last=x(these[these.length-1].ms),middle=(first+last)/2;
   if(selected===i)add('rect',{x:Math.max(l,first-3),y:t,width:Math.max(4,last-first+6),height:b-t,rx:4,fill:cfg.stroke,'fill-opacity':'.045'});
   add('text',{x:middle,y:240,'text-anchor':'middle',fill:selected===i?ink:muted,'font-size':10,'font-weight':selected===i?700:400},`${day.weekday.slice(0,3)} ${day.day.split(' ')[0]}`);
   const peak=variable==='hi'?day.hiPeak:day.airPeak;
   if(peak&&these[peak.index]){
     const xx=x(these[peak.index].ms),yy=y(peak.value),active=selected===i;
     if(active)add('circle',{cx:xx,cy:yy,r:8,fill:cfg.stroke,'fill-opacity':'.15'});
     add('circle',{cx:xx,cy:yy,r:active?4.6:3.3,fill:cfg.stroke,stroke:'#fff','stroke-width':1.5});
     // Every daily peak stays labeled, whether or not its day is expanded.
     let ly=yy<t+19?yy+17:yy-11;
     if(variable==='hi'&&showReference.checked&&i===0){ly=yy<t+28?yy+19:Math.max(t+12,yy-24);}
     // Give the first peak label some breathing room from threshold labels at the Y-axis.
     const edgeLabel=variable==='hi'&&showReference.checked&&i===0;
     const labelX=edgeLabel?Math.min(r-32,xx+17):Math.max(l+11,Math.min(r-11,xx));
     add('text',{x:labelX,y:ly,'text-anchor':edgeLabel?'start':'middle',fill:cfg.stroke,'font-size':10.5,'font-weight':750},fmt(peak.value,'°'));
   }
 });
 // NOW marker is interpolated between the two surrounding three-hourly medians.
 // It is intentionally omitted if the current time is outside published steps.
 const nowMs=Date.now();
 if(nowMs>=x0&&nowMs<=x1){
   let before=null,after=null;
   for(const point of flat){
     if(point.ms<=nowMs&&Number.isFinite(point.step[variable].median))before=point;
     if(point.ms>=nowMs&&Number.isFinite(point.step[variable].median)){after=point;break;}
   }
   if(before&&after&&after.ms-before.ms<=6*3600*1000){
     const fraction=after.ms===before.ms?0:(nowMs-before.ms)/(after.ms-before.ms);
     const nowValue=before.step[variable].median+(after.step[variable].median-before.step[variable].median)*fraction;
     const xx=x(nowMs), yy=y(nowValue);
     const currentDot=add('circle',{cx:xx,cy:yy,r:6.6,fill:'#fff',stroke:cfg.stroke,'stroke-width':1.6,'pointer-events':'none','data-current-time':'true'});
     currentDot.appendChild(s('title',{},`Current time: ${fmt(nowValue)}`));
     add('circle',{cx:xx,cy:yy,r:3.2,fill:cfg.stroke,stroke:'#fff','stroke-width':.7,'pointer-events':'none'});
   }
 }
 // SVG hit zones cover all daily time intervals (not just narrow peak dots).
 days.forEach((day,i)=>{
   const these=flat.filter(p=>p.day===i);if(!these.length)return;
   let start=x(these[0].ms),end=x(these[these.length-1].ms);
   // Avoid Array.findLast() for older mobile Safari compatibility.
   let previous=null,next=null;
   for(let j=flat.length-1;j>=0;j--)if(flat[j].day===i-1){previous=flat[j];break;}
   for(let j=0;j<flat.length;j++)if(flat[j].day===i+1){next=flat[j];break;}
   if(previous)start=(start+x(previous.ms))/2;else start=l;
   if(next)end=(end+x(next.ms))/2;else end=r;
   const rect=add('rect',{x:start,y:t,width:Math.max(2,end-start),height:b-t,role:'button',tabindex:'0','data-chart-day':i,'aria-label':`Select ${day.full} forecast`,class:'temp-plot-hit'});
   rect.appendChild(s('title',{},`${day.full}: ${fmt((variable==='hi'?day.hiPeak:day.airPeak)?.value??null)} peak`));
 });
 root.querySelector('#temp-chart-title').textContent=(variable==='hi'?'Heat Index':'Air temperature')+' over time';
 root.querySelector('#temp-legend-line').style.background=cfg.stroke;
 root.querySelector('#temp-legend-range').style.background=cfg.shade+'88';
 root.querySelector('#temp-legend-range').parentElement.hidden=!showRange.checked;
 root.querySelector('#temp-reference-label').hidden=variable!=='hi';
 const caption=root.querySelector('#temp-graph-caption');
 caption.hidden=selected<0;
 caption.textContent=selected<0?'':`${days[selected].full} · ${fmt((variable==='hi'?days[selected].hiPeak:days[selected].airPeak)?.value??null)} peak`;
}
let graphPointerDown=null, lastGraphPointerUp=0;
const chartHit=target=>target&&target.closest?target.closest('[data-chart-day]'):null;
svg.addEventListener('pointerdown',event=>{
  const hit=chartHit(event.target);
  graphPointerDown=hit?{id:Number(hit.dataset.chartDay),x:event.clientX,y:event.clientY,pointerId:event.pointerId}:null;
});
svg.addEventListener('pointerup',event=>{
  const hit=chartHit(event.target);
  if(!hit||!graphPointerDown||graphPointerDown.pointerId!==event.pointerId||graphPointerDown.id!==Number(hit.dataset.chartDay))return;
  const moved=Math.hypot(event.clientX-graphPointerDown.x,event.clientY-graphPointerDown.y);
  graphPointerDown=null;
  if(moved>12)return;
  lastGraphPointerUp=Date.now();
  event.preventDefault();
  selectDay(Number(hit.dataset.chartDay));
});
svg.addEventListener('pointercancel',()=>{graphPointerDown=null});
svg.addEventListener('click',event=>{
  const hit=chartHit(event.target);
  if(hit&&Date.now()-lastGraphPointerUp>700)selectDay(Number(hit.dataset.chartDay));
});
svg.addEventListener('keydown',event=>{
  const hit=chartHit(event.target);
  if(hit&&(event.key==='Enter'||event.key===' ')){event.preventDefault();selectDay(Number(hit.dataset.chartDay));}
});
function refresh(){tabs.forEach(b=>b.setAttribute('aria-selected',String(b.dataset.variable===variable)));updateRows();drawGraph();animateHeight()}
tabs.forEach(b=>b.addEventListener('click',()=>{variable=b.dataset.variable;refresh()}));
simple.addEventListener('change',()=>{updateRows();animateHeight()});
[showRange,showReference,showCompare].forEach(input=>input.addEventListener('change',()=>{drawGraph();animateHeight()}));
// Only redraw when the chart width changes; root height changes on day expansion
// and should not trigger a ResizeObserver feedback loop on iOS.
let lastChartWidth=0;
if(typeof ResizeObserver!=='undefined'){
  new ResizeObserver(entries=>{
    const width=Math.round(entries[0]?.contentRect?.width||0);
    if(Math.abs(width-lastChartWidth)>2){lastChartWidth=width;drawGraph();}
    height();
  }).observe(root.querySelector('.temp-chart-shell'));
  new ResizeObserver(height).observe(root);
}
window.addEventListener('resize',()=>{drawGraph();height()});
if(document.fonts?.ready)document.fonts.ready.then(()=>{drawGraph();height()});
// Update the highlighted period as the local clock crosses a 3-hour boundary.
window.setInterval(highlightCurrentPeriod,60*1000);
refresh();
})();
</script>
</body></html>'''

# ---------------------------------------------------------------------------
# Observed weather: PAGASA WIS2 SYNOP first, OGIMET SYNOP second.
# All data are requested only when Compare to actual is enabled. Never
# infer a missing accumulation, report time, or Heat Index observation.
# ---------------------------------------------------------------------------
import base64
import hashlib
from io import BytesIO, StringIO
import re
import tempfile
from datetime import timedelta

import requests

_OBS_TIMEOUT = (5, 12)
_OBS_SESSION = requests.Session()
_OBS_SESSION.headers.update({"User-Agent": "Simpleweather/1.0 (synoptic observations; cached 30 minutes)", "Accept": "application/json,text/plain,*/*"})
_OBS_WIS_BASE = "https://wis.pagasa.dost.gov.ph/oapi/collections"
_OBS_OGIMET = "https://www.ogimet.com/cgi-bin/getsynop"


def _obs_fetch(url, params, *, max_bytes=3_000_000):
    """Bounded network read. An outage never interrupts the forecast app."""
    response = _OBS_SESSION.get(url, params=params, timeout=_OBS_TIMEOUT)
    response.raise_for_status()
    size = response.headers.get("Content-Length")
    if size and int(size) > max_bytes:
        raise ValueError("Observation response is too large")
    if len(response.content) > max_bytes:
        raise ValueError("Observation response is too large")
    return response


def _obs_distance_km(lat1, lon1, lat2, lon2):
    a, b = map(math.radians, (float(lat1), float(lat2)))
    dlat, dlon = b-a, math.radians(float(lon2)-float(lon1))
    h = math.sin(dlat/2)**2 + math.cos(a)*math.cos(b)*math.sin(dlon/2)**2
    return 6371.0088 * 2 * math.atan2(math.sqrt(h), math.sqrt(max(0, 1-h)))


@st.cache_data(ttl=24*3600, show_spinner=False)
def _obs_pagasa_stations():
    """PAGASA's official WIS2 published synoptic station inventory; no AWS/ARG."""
    result = []
    url = _OBS_WIS_BASE + "/stations/items"
    payload = _obs_fetch(url, {"f": "json", "limit": 200}).json()
    for item in payload.get("features", []):
        props = item.get("properties") or {}
        coord = (item.get("geometry") or {}).get("coordinates") or []
        wmo = str(props.get("traditional_station_identifier") or "")
        topic = " ".join(props.get("topics") or [])
        if (len(coord) < 2 or not re.fullmatch(r"98\d{3}", wmo)
                or "/synop" not in topic or props.get("facility_type") != "landFixed"
                or props.get("status", "operational").lower() != "operational"):
            continue
        result.append({"id": wmo, "wigos": props.get("wigos_station_identifier") or item.get("id"),
                       "name": props.get("name") or wmo, "lat": float(coord[1]),
                       "lon": float(coord[0]), "country": "PH"})
    return result


@st.cache_data(ttl=24*3600, show_spinner=False)
def _obs_noaa_stations():
    """Fallback metadata; actual reports must still be AAXX synoptic messages."""
    # The official Taiwanese station fallback below must remain available even
    # when the optional NOAA station inventory cannot be downloaded.
    try:
        raw = _obs_fetch("https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv", {}, max_bytes=5_000_000).content
        frame = pd.read_csv(BytesIO(raw), dtype=str)
    except (requests.RequestException, ValueError, pd.errors.ParserError):
        frame = pd.DataFrame()
    found = []
    for row in frame.to_dict("records"):
        code = str(row.get("USAF") or "")[:5]
        country = str(row.get("CTRY") or "")
        if ((country not in ("RP", "TW", "PH")) or
            not re.fullmatch(r"(?:98|46)\d{3}", code)):
            continue
        try:
            lat, lon = float(row["LAT"]), float(row["LON"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
        except (ValueError, TypeError):
            continue
        found.append({"id": code, "wigos": f"0-20000-0-{code}",
                      "name": str(row.get("STATION NAME") or code),
                      "lat": lat, "lon": lon,
                      "country": "TW" if country == "TW" else "PH"})
    # NOAA's ISD inventory is not a complete WMO/SYNOP catalogue. Taiwan's
    # official CWA synoptic-station list supplies trustworthy coordinates and
    # WMO identifiers even when the corresponding ISD record is absent.
    # These are traditional CWA SYNOP locations (not AWS/ARG). Their inclusion
    # in the catalogue does NOT imply that an observation is available.
    taiwan_synop = (
        ("46690", "Tamsui", 25.164889, 121.448906),
        ("46691", "Anbu", 25.182586, 121.529731),
        ("46692", "Taipei", 25.037658, 121.514853),
        ("46693", "Zhuzihu", 25.162078, 121.544547),
        ("46694", "Keelung", 25.133314, 121.740475),
        ("46695", "Pengjia Islet", 25.627975, 122.079744),
        ("46699", "Hualien", 23.9751, 121.6133),
        ("46706", "Su'ao", 24.5967, 121.8574),
        ("46708", "Yilan", 24.7640, 121.7565),
    )
    known = {s["id"] for s in found}
    for wmo, name, latitude, longitude in taiwan_synop:
        if wmo not in known:
            found.append({"id": wmo, "wigos": f"0-20000-0-{wmo}",
                          "name": name, "lat": latitude, "lon": longitude,
                          "country": "TW"})
    return found


def _obs_first_number(handle, keys):
    """Only read known ecCodes BUFR descriptors; missing data stay missing."""
    import eccodes
    for key in keys:
        try:
            vals = eccodes.codes_get_array(handle, key)
            for candidate in vals:
                val = float(candidate)
                if math.isfinite(val) and abs(val) < 1e8:
                    return val
        except Exception:
            continue
    return None


def _obs_heat_index(t_c, dew_c):
    """NOAA/NWS heat-index formula, computed from observed T and dewpoint."""
    if t_c is None or dew_c is None or dew_c > t_c + 0.25:
        return None
    rh = 100 * math.exp((17.625 * dew_c)/(243.04+dew_c)
                        - (17.625 * t_c)/(243.04+t_c))
    if not (0 <= rh <= 100.5):
        return None
    rh = min(100.0, rh)
    t_f = t_c * 9/5 + 32
    simple_f = 0.5*(t_f+61.0+((t_f-68.0)*1.2)+(rh*.094))
    if (simple_f + t_f) / 2 < 80:
        return (simple_f-32)*5/9
    hi = (-42.379+2.04901523*t_f+10.14333127*rh-.22475541*t_f*rh
          -.00683783*t_f**2-.05481717*rh**2+.00122874*t_f**2*rh
          +.00085282*t_f*rh**2-.00000199*t_f**2*rh**2)
    if rh < 13 and 80 <= t_f <= 112:
        hi -= (13-rh)/4*math.sqrt((17-abs(t_f-95))/17)
    elif rh > 85 and 80 <= t_f <= 87:
        hi += (rh-85)/10*((87-t_f)/5)
    return (hi-32)*5/9


def _obs_convert_kelvin(value):
    if value is None:
        return None
    return value - 273.15 if value > 150 else value


def _obs_decode_pagasa_bufr(content, when_utc):
    """PAGASA publishes BUFR; ecCodes comes with cfgrib on the app runtime.

    Only explicitly labelled 3/6-hour totals are used. We do NOT guess the
    accumulation period from generic precipitation descriptors.
    """
    try:
        import eccodes
        payload = base64.b64decode(content, validate=True)
        if not payload.startswith(b"BUFR"):
            return []
        stream = BytesIO(payload)
        output = []
        while True:
            msg = eccodes.codes_bufr_new_from_file(stream)
            if msg is None:
                break
            try:
                eccodes.codes_set(msg, "unpack", 1)
                t_c = _obs_convert_kelvin(_obs_first_number(msg, ("airTemperature", "airTemperatureAt2M")))
                d_c = _obs_convert_kelvin(_obs_first_number(msg, ("dewpointTemperature", "dewpointTemperatureAt2M")))
                rain3 = _obs_first_number(msg, ("totalPrecipitationPast3Hours", "totalPrecipitationInPast3Hours"))
                rain6 = _obs_first_number(msg, ("totalPrecipitationPast6Hours", "totalPrecipitationInPast6Hours"))
                base = {"time": when_utc.isoformat(), "temperature": t_c,
                        "heat_index": _obs_heat_index(t_c, d_c)}
                if t_c is not None:
                    output.append(base)
                for hours, rain in ((3, rain3), (6, rain6)):
                    if rain is not None and 0 <= rain <= 3000:
                        output.append({"time": when_utc.isoformat(), "rain_mm": rain,
                                       "rain_hours": hours})
            finally:
                eccodes.codes_release(msg)
        return output
    except Exception:
        return []


def _obs_read_pagasa(start_utc, end_utc):
    """One bounded WIS2 date-window request; optionally follows one next page."""
    result = {}
    params = {"f": "json", "limit": 1000, "datetime":
              f"{start_utc:%Y-%m-%dT%H:%M:%SZ}/{end_utc:%Y-%m-%dT%H:%M:%SZ}"}
    url = _OBS_WIS_BASE + "/messages/items"
    for _ in range(8):
        page = _obs_fetch(url, params).json()
        for feature in page.get("features", []):
            props = feature.get("properties") or {}
            wigos = str(props.get("wigos_station_identifier") or "")
            if not wigos.startswith("0-20000-0-98"):
                continue
            try:
                when = datetime.fromisoformat(str(props["datetime"]).replace("Z", "+00:00"))
                if not (start_utc <= when <= end_utc):
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            content = (props.get("content") or {}).get("value")
            if content:
                result.setdefault(wigos.split("-")[-1], []).extend(_obs_decode_pagasa_bufr(content, when))
        next_url = next((link.get("href") for link in page.get("links", []) if link.get("rel") == "next"), None)
        if not next_url or not next_url.startswith("https://wis.pagasa.dost.gov.ph/"):
            break
        url, params = next_url, {}
    return result


def _obs_rain_record(when, hours, amount=None, trace=False):
    """SYNOP RRRtR: the observation time is the END of the reported total."""
    return {"time": when.isoformat(),
            "rain_start": (when - timedelta(hours=hours)).isoformat(),
            "rain_end": when.isoformat(),
            "rain_hours": hours,
            **({"rain_trace": True} if trace else {"rain_mm": amount})}


def _obs_decode_synop(report, when):
    """Decode AAXX temperature/dew point and explicitly timed SYNOP rain.

    Section 1 and section 3 group 6RRRtR includes ONE reporting timestamp,
    and tR identifies the preceding accumulation interval (WMO code 4019).
    Other parts of SYNOP (including section 2) are never treated as rain.
    """
    terms = report.replace("=", " ").split()
    if len(terms) < 6 or terms[0] != "AAXX":
        return []
    try:
        # AAXX YYGGiw IIiii iRiXhVV Nddff [00fff] ...
        start_idx = 6 if len(terms) > 5 and re.fullmatch(r"00\d{3}", terms[5]) else 5
        section, supplemental = [], []
        current = section
        for word in terms[start_idx:]:
            if word.startswith("222"):
                current = None
                continue
            if word == "333":
                current = supplemental
                continue
            if word.startswith(("444", "555")):
                break
            if current is not None:
                current.append(word)
        t_c = d_c = None
        for word in section:
            if re.fullmatch(r"1[01]\d{3}", word):
                t_c = float(word[2:]) / 10 * (-1 if word[1] == "1" else 1)
            elif re.fullmatch(r"2[01]\d{3}", word):
                d_c = float(word[2:]) / 10 * (-1 if word[1] == "1" else 1)
        output = []
        if t_c is not None and -70 <= t_c <= 60:
            output.append({"time": when.isoformat(), "temperature": t_c,
                           "heat_index": _obs_heat_index(t_c, d_c)})
        durations = {1: 6, 2: 12, 3: 18, 4: 24, 5: 1,
                     6: 2, 7: 3, 8: 9, 9: 15}
        for word in section + supplemental:
            if not re.fullmatch(r"6\d{4}", word):
                continue
            hours = durations.get(int(word[4]))
            if hours is None:
                continue
            number = int(word[1:4])
            if number == 990:
                item = _obs_rain_record(when, hours, trace=True)
            elif number == 989:
                # "989 mm or more" is not a precise amount to compare.
                continue
            else:
                mm = (number - 990) / 10 if number >= 991 else float(number)
                if not 0 <= mm <= 3000:
                    continue
                item = _obs_rain_record(when, hours, amount=mm)
            if item not in output:
                output.append(item)
        return output
    except (TypeError, ValueError, OverflowError):
        return []


def _obs_read_ogimet(start_utc, end_utc, region, *, diagnostics=None, nearest_ids=()):
    """Retrieve raw OGIMET CSV (SYNOP, not METAR/AWS) with source diagnostics.

    For Taiwan, ask for the usual 466/467 prefixes first. If OGIMET returns
    no valid reports, try its documented state query and up to two exact WMO
    station IDs. This changes retrieval, never an observation value.
    """
    result, errors, request_notes = {}, [], []
    parsed_lines_total = 0
    decoded_reports_total = 0

    def get_one(params, label):
        nonlocal parsed_lines_total, decoded_reports_total
        try:
            response = _obs_fetch(_OBS_OGIMET, params, max_bytes=5_000_000)
            body = response.text
            rows, accepted = 0, 0
            for line in body.splitlines():
                items = line.strip().split(",", 6)
                if len(items) != 7:
                    continue
                station = items[0].strip()
                if not re.fullmatch(r"\d{5}", station):
                    continue
                # Strictly match the requested country, even for a state query.
                if region == "TW" and not station.startswith(("466", "467")):
                    continue
                if region == "PH" and not station.startswith("98"):
                    continue
                if params.get("block") and not station.startswith(params["block"]):
                    continue
                try:
                    when = datetime(*[int(p) for p in items[1:6]], tzinfo=timezone.utc)
                    if not start_utc <= when <= end_utc:
                        continue
                except (TypeError, ValueError):
                    continue
                rows += 1
                parsed = _obs_decode_synop(items[6].strip(), when)
                if parsed:
                    accepted += 1
                    result.setdefault(station, []).extend(parsed)
            parsed_lines_total += rows
            decoded_reports_total += accepted
            request_notes.append(f"{label}: {rows} CSV reports, {accepted} decoded")
            if not rows and "<html" in body[:2000].lower():
                errors.append(f"{label}: unexpected HTML instead of CSV")
            elif not rows and any(t in body.lower() for t in ("sorry", "too many", "temporarily", "error")):
                errors.append(f"{label}: provider returned an error message")
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{label}: {type(exc).__name__}")

    params = {"begin": start_utc.strftime("%Y%m%d%H%M"),
              "end": end_utc.strftime("%Y%m%d%H%M"),
              "lang": "eng", "header": "yes"}
    blocks = ("98",) if region == "PH" else ("466", "467")
    for block in blocks:
        get_one(dict(params, block=block), f"block {block}")

    if region == "TW" and decoded_reports_total == 0:
        get_one(dict(params, state="Taiwan"), "state Taiwan")
        if decoded_reports_total == 0:
            for station_id in list(dict.fromkeys(str(x) for x in nearest_ids))[:2]:
                if re.fullmatch(r"(?:466|467)\d{2}", station_id):
                    get_one(dict(params, block=station_id), f"station {station_id}")

    if diagnostics is not None:
        diagnostics.extend(request_notes)
        diagnostics.extend(errors)
    if errors and not result and parsed_lines_total == 0:
        raise RuntimeError("; ".join(errors))
    return result


def _obs_has_data(records, variable, hours):
    if variable == "rain":
        return any((r.get("rain_hours") in (1, 2, 3, 6) and
                    r.get("rain_hours") <= hours and
                    (r.get("rain_mm") is not None or r.get("rain_trace")))
                   for r in records)
    return any(r.get("temperature") is not None for r in records)


def _obs_match_rain(records, start, end, hours):
    """Match reporting-period intervals exactly; never subdivide longer totals.

    Example: SYNOP at 12Z with 6RRR1 represents the whole 06-12Z period.
    Permitted shorter reports may be summed ONLY if they cover the requested
    period completely, with no overlaps, gaps or inferred zeros.
    """
    if end - start != timedelta(hours=hours):
        return None
    by_start = {}
    for r in records:
        if r.get("rain_mm") is None and not r.get("rain_trace"):
            continue
        try:
            length = int(r["rain_hours"])
            if length not in (1, 2, 3, 6) or length > hours:
                continue
            # Explicit interval endpoints take precedence over derived ones.
            report_end = datetime.fromisoformat(r.get("rain_end") or r["time"])
            report_start = datetime.fromisoformat(r["rain_start"]) if r.get("rain_start") else report_end - timedelta(hours=length)
            if report_end - report_start != timedelta(hours=length):
                continue
            if not (start <= report_start < report_end <= end):
                continue
            by_start.setdefault(report_start, []).append((report_end, r))
        except (ValueError, TypeError, KeyError):
            continue

    # Prefer direct totals and then the fewest valid contiguous reports.
    from functools import lru_cache
    @lru_cache(None)
    def segments(cursor):
        if cursor == end:
            return ()
        options = []
        for next_end, r in by_start.get(cursor, ()):
            remaining = segments(next_end)
            if remaining is not None:
                options.append((r,) + remaining)
        return min(options, key=len) if options else None

    chosen = segments(start)
    if not chosen:
        return None
    if len(chosen) == 1 and chosen[0].get("rain_trace"):
        return {"rain_trace": True}
    if any(r.get("rain_trace") or r.get("rain_mm") is None for r in chosen):
        # A trace is not zero and cannot be summed as a precise quantity.
        return None
    return {"rain_mm": round(sum(r["rain_mm"] for r in chosen), 3)}


def _obs_dataset_summary(name, by_station):
    """Counts by rainfall DURATION, not misleading 'all decoded readings'."""
    rows = [record for station_rows in by_station.values() for record in station_rows]
    rain_counts = {h: sum(r.get("rain_hours") == h for r in rows) for h in (1, 2, 3, 6, 12, 24)}
    t_count = sum(r.get("temperature") is not None for r in rows)
    amounts = ", ".join(f"{h}h={rain_counts[h]}" for h in (1, 2, 3, 6, 12, 24) if rain_counts[h]) or "none"
    return f"{name}: {len(by_station)} stations, {t_count} temperature reports, rainfall intervals ({amounts})"


def _obs_match_temp(records, when):
    candidates = []
    for reading in records:
        if reading.get("temperature") is None:
            continue
        try:
            offset = abs((datetime.fromisoformat(reading["time"]) - when).total_seconds())
        except (ValueError, TypeError):
            continue
        if offset <= 30*60:
            candidates.append((offset, reading))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1]


@st.cache_data(ttl=30*60, show_spinner=False)
def _obs_for_forecast(lat, lon, variable, hours, periods_json, region):
    """Fetch lazily, then select the nearest station with ACTUAL aligned data.

    Diagnostic messages distinguish missing reports from request/decoder
    failures. No observation is guessed or shifted to another time window.
    """
    periods = json.loads(periods_json)
    now = datetime.now(timezone.utc)
    start = min((datetime.fromisoformat(p[0]) for p in periods), default=now-timedelta(hours=24))
    start = max(start-timedelta(hours=6), now-timedelta(days=3))
    end = min(now, max((datetime.fromisoformat(p[-1]) for p in periods), default=now))
    if start > end:
        start, end = end-timedelta(hours=6), end
    notes = []
    try:
        stations = _obs_pagasa_stations() if region == "PH" else _obs_noaa_stations()
    except Exception as exc:
        notes.append(f"Station inventory: {type(exc).__name__}")
        stations = []
    if not stations:
        try:
            stations = _obs_noaa_stations()
        except Exception as exc:
            notes.append(f"Fallback inventory: {type(exc).__name__}")
            stations = []
    # Only stations physically inside the chosen region. Taiwan's NOAA
    # inventory also includes non-Taiwanese East Asian 46xxx records.
    stations = [s for s in stations if s.get("country") == region]
    candidates = sorted((dict(s, distance_km=_obs_distance_km(lat, lon, s["lat"], s["lon"]))
                         for s in stations), key=lambda s: s["distance_km"])
    if not candidates:
        return {"status": "unavailable", "message":
                "No synoptic station metadata found for this region. " + "; ".join(notes)}
    datasets = []
    if region == "PH":
        try:
            data = _obs_read_pagasa(start, end)
            datasets.append(("PAGASA WIS2", data))
            notes.append(_obs_dataset_summary("PAGASA WIS2", data))
        except Exception as exc:
            notes.append(f"PAGASA WIS2: {type(exc).__name__}")
    try:
        ogimet_checks = []
        data = _obs_read_ogimet(start, end, region, diagnostics=ogimet_checks,
                                nearest_ids=[s["id"] for s in candidates[:3]])
        datasets.append(("OGIMET (SYNOP)", data))
        notes.append(_obs_dataset_summary("OGIMET", data))
        if region == "TW" and not data:
            notes.append("OGIMET attempts: " + "; ".join(ogimet_checks))
    except Exception as exc:
        notes.append(f"OGIMET: {type(exc).__name__}: {str(exc)[:100]}")
    any_relevant_reading = False
    for station in candidates:
        for source, by_station in datasets:
            readings = by_station.get(station["id"], [])
            if not _obs_has_data(readings, variable, hours):
                continue
            any_relevant_reading = True
            values = {}
            for start_text, end_text in periods:
                period_start, period_end = datetime.fromisoformat(start_text), datetime.fromisoformat(end_text)
                if period_end > now:
                    continue
                if variable == "rain":
                    amount = _obs_match_rain(readings, period_start, period_end, hours)
                    if amount is not None:
                        values[f"{int(period_start.timestamp()*1000)}-{int(period_end.timestamp()*1000)}"] = (
                            {"rain_trace": True} if amount.get("rain_trace")
                            else {"rain_mm": round(amount["rain_mm"], 2)})
                else:
                    report = _obs_match_temp(readings, period_end)
                    if report:
                        values[str(int(period_end.timestamp()*1000))] = {
                            "temperature": report.get("temperature"),
                            "heat_index": report.get("heat_index")}
            if values:
                return {"status": "ok", "source": source, "station": station["name"],
                        "distance_km": round(station["distance_km"], 1),
                        "distant": station["distance_km"] > 25,
                        "values": values, "fetched_at": now.isoformat()}
    if not datasets:
        diagnosis = "Observation providers could not be reached."
    elif any_relevant_reading:
        diagnosis = "Station observations exist, but none match the forecast periods exactly."
    else:
        diagnosis = f"No usable {hours}-hour {variable} observations in the requested period."
    return {"status": "unavailable", "message": diagnosis + " " + "; ".join(notes)}


def _obs_periods_rain(period_results, hours):
    bins = (period_results.get(hours) or {}).get("days") or []
    return json.dumps([(p["start_utc"].astimezone(timezone.utc).isoformat(),
                        p["end_utc"].astimezone(timezone.utc).isoformat()) for p in bins])


def _obs_periods_heat(result):
    return json.dumps([(step["time_utc"].astimezone(timezone.utc).isoformat(),
                        step["time_utc"].astimezone(timezone.utc).isoformat())
                       for step in result.get("steps", []) if isinstance(step.get("time_utc"), datetime)])


_OBS_COMPONENT_DIR = Path(tempfile.gettempdir()) / "simpleweather_compare_components"


def _render_observation_component(html, kind, lat, lon, periods_by_hours, region):
    """Bidirectional Streamlit component, built from app.py without extra repo files.

    A component value message triggers a Python rerun *only* when the user
    enables Compare to actual. Streamlit renders returned observations back
    to the same iframe without altering the existing forecast DOM/state.
    """
    # A bridge-code change must create a new component identity so Streamlit
    # cannot keep an iframe/cached component with obsolete JavaScript.
    fingerprint = hashlib.sha256((html + "\nobs-bridge-v3").encode("utf-8")).hexdigest()[:20]
    code = (_OBS_COMPARE_BRIDGE.replace("__OBS_KIND__", json.dumps(kind))
            .replace("__OBS_FINGERPRINT__", fingerprint))
    folder = _OBS_COMPONENT_DIR / (kind + "_" + fingerprint)
    folder.mkdir(parents=True, exist_ok=True)
    page = folder / "index.html"
    # The component name/URL is content-based. Refresh its HTML when the
    # observation bridge changes even if the forecast HTML is identical.
    if html.lstrip().lower().startswith("<!doctype"):
        rendered = html.replace("</body>", code + "</body>")
    else:
        rendered = "<!doctype html><html><head><meta charset='utf-8'></head><body>" + html + code + "</body></html>"
    if not page.exists() or page.read_text(encoding="utf-8") != rendered:
        page.write_text(rendered, encoding="utf-8")
    component = components.declare_component("sw_compare_" + kind + "_" + fingerprint, path=str(folder))
    session_key = "sw_obs_payload_" + kind + "_" + fingerprint
    event_key = session_key + "_last_event"
    result = component(obs_payload=st.session_state.get(session_key), key="sw_component_" + kind + "_" + fingerprint,
                       default=None)
    if isinstance(result, dict) and result.get("type") == "load":
        nonce = str(result.get("nonce", ""))
        if nonce and nonce != st.session_state.get(event_key):
            st.session_state[event_key] = nonce
            hours = 3 if kind == "heat" else int(result.get("hours", 3))
            if hours not in (3, 6):
                hours = 3
            periods = periods_by_hours.get(hours) or "[]"
            # The component expects responses keyed by accumulation length.
            # Keep both the 3-hour and 6-hour results when users switch tabs.
            all_payloads = dict(st.session_state.get(session_key) or {})
            all_payloads[str(hours)] = _obs_for_forecast(
                round(lat, 6), round(lon, 6), kind, hours, periods, region)
            st.session_state[session_key] = all_payloads
            st.rerun()

_OBS_COMPARE_BRIDGE = r'''
<style>
/* Compare-to-actual additions are scoped; all existing forecast styles remain. */
.sw-obs-action{display:flex;align-items:center;justify-content:flex-end;margin:9px 0 5px}
.sw-obs-toggle{display:inline-flex;gap:7px;align-items:center;cursor:pointer;user-select:none;
  font-family:inherit;font-size:12px;font-weight:650;color:#506070}
.sw-obs-toggle input{appearance:none;width:34px;height:20px;border-radius:99px;background:#d5dbe3;
  position:relative;margin:0;flex:none;cursor:pointer}
.sw-obs-toggle input:after{content:"";position:absolute;top:3px;left:3px;width:14px;height:14px;
  border-radius:50%;background:white;box-shadow:0 1px 3px #0002;transition:left .16s}
.sw-obs-toggle input:checked{background:#25a8db}.sw-obs-toggle input:checked:after{left:17px}
.sw-obs-toggle input:focus-visible{outline:2px solid #159bd6;outline-offset:3px}
.sw-obs-status{font-size:11px;color:#697789;line-height:1.4;margin:4px 0 7px}
.sw-obs-status .sw-obs-warning{display:block;color:#986228;font-weight:650}
.sw-obs-row{margin:9px 0 0;padding:8px 0 0;border-top:1px solid #e0e8ee}
.sw-obs-row .sw-obs-title{font-size:11px;font-weight:750;color:#516678;margin-bottom:6px}
.sw-obs-values{display:grid;min-width:0;align-items:center}
.sw-obs-value{font-size:10px;color:#283746;background:#edf6fb;border:1px solid #dcecf4;
 border-radius:5px;padding:4px 1px;text-align:center;font-weight:700;min-width:0;overflow:hidden}
.sw-obs-value.is-na{color:#8994a1;background:#f8fafc;border-color:#edf0f4;font-weight:400}
.temp-hour .sw-obs-temp{display:block;border-top:1px solid rgba(40,55,70,.14);margin-top:5px;
 padding-top:5px;font-size:10px;font-weight:650;line-height:1.4;white-space:normal;color:#283746}
.temp-hour .sw-obs-temp small{display:block;font-size:9px;font-weight:400;color:#506070}
.temp-hour .sw-obs-temp[hidden]{display:none!important}
.temp-detail .sw-obs-action{margin:9px 0 6px}
</style>
<script>
(()=>{
 'use strict';
 const kind=__OBS_KIND__;
 const storageKey='sw-compare-enabled:'+kind+':__OBS_FINGERPRINT__';
 let response={};
 let restored=[];
 try{restored=JSON.parse(window.sessionStorage.getItem(storageKey)||'[]');}catch(_err){}
 const enabled=new Set(Array.isArray(restored)?restored:[]);
 function preserveSelection(){try{window.sessionStorage.setItem(storageKey,JSON.stringify(Array.from(enabled)));}catch(_err){}}
 const root=document.querySelector(kind==='heat'?'.temp-root':'.rr-root');
 if(!root)return;
 function text(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
 function period(panel){return kind==='heat'?3:Number(panel.closest('.rr-variant')?.dataset.rrVariant?.split('-')[0]||3);}
 function dayKey(panel){
   if(kind==='heat')return String(panel.closest('[data-day-wrap]')?.dataset.dayWrap||'0');
   return String(period(panel))+':'+String(panel.dataset.rrDate||'');
 }
 function panels(){return Array.from(root.querySelectorAll(kind==='heat'?'.temp-detail':'.rr-detail'));}
 function componentValue(payload){
   window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setComponentValue',value:payload},'*');
 }
 function displayStatus(panel,data){
   let line=panel.querySelector('.sw-obs-status');
   if(!line)return;
   const p=period(panel),on=enabled.has(dayKey(panel));
   const row=panel.querySelector('.sw-obs-row');
   if(row)row.hidden=!on;
   line.hidden=!on;
   if(!on)return;
   if(!data){line.textContent='Loading synoptic observations…';return;}
   if(data.status!=='ok'){
     line.textContent=data.message||'Station observations are unavailable for these periods.';
     return;
   }
   line.innerHTML=`${text(data.source)} · ${text(data.station)} · ${Number(data.distance_km).toFixed(1)} km from selected location`
     +(data.distant?'<span class="sw-obs-warning">Station is over 25 km away; conditions at your location may differ.</span>':'');
 }
 function renderRain(panel,data){
   displayStatus(panel,data);
   const row=panel.querySelector('.sw-obs-row');
   if(!row)return;
   const slots=Array.from(panel.querySelectorAll('.rr-detail-bar-values .rr-slot'));
   const cells=row.querySelector('.sw-obs-values');
   if(!cells)return;
   cells.style.gridTemplateColumns=`repeat(${slots.length},minmax(0,1fr))`;
   cells.style.columnGap='2px';
   cells.replaceChildren();
   slots.forEach(slot=>{
     const span=document.createElement('span');
     const start=slot.dataset.startMs,end=slot.dataset.endMs;
     const record=data?.values?.[start+'-'+end];
     const value=record?.rain_mm,trace=Boolean(record?.rain_trace);
     span.className='sw-obs-value'+((Number.isFinite(value)||trace)?'':' is-na');
     span.textContent=trace?'trace':Number.isFinite(value)?((value<10?Number(value).toFixed(1):Math.round(value))+''):'—';
     span.title=trace?'Trace rainfall reported':Number.isFinite(value)?`Observed ${value} mm in this ${period(panel)}-hour period`:'No matching observation';
     cells.appendChild(span);
   });
 }
 function renderHeat(panel,data){
   displayStatus(panel,data);
   panel.querySelectorAll('.temp-hour[data-valid-time]').forEach(tile=>{
     let area=tile.querySelector('.sw-obs-temp');
     if(!area){area=document.createElement('span');area.className='sw-obs-temp';tile.appendChild(area);}
     const show=enabled.has(dayKey(panel));
     area.hidden=!show;
     if(!show)return;
     const record=data?.values?.[String(Date.parse(tile.dataset.validTime))];
     if(!record){area.innerHTML='<small>Observed</small>—';return;}
     const value=v=>Number.isFinite(v)?Math.round(v)+'°':'—';
     area.innerHTML=`<small>Observed</small>HI ${value(record.heat_index)}<br>Air ${value(record.temperature)}`;
   });
 }
 function refresh(){
   panels().forEach(panel=>{
     const hours=period(panel);
     const active=enabled.has(dayKey(panel));
     const toggle=panel.querySelector('.sw-obs-toggle input');
     if(toggle)toggle.checked=active;
     const payload=response[String(hours)];
     if(kind==='heat')renderHeat(panel,payload);
     else renderRain(panel,payload);
   });
   // Preserve the existing card's self-sizing logic on content changes.
   window.requestAnimationFrame(()=>{
     const h=Math.ceil(root.getBoundingClientRect().height+2);
     window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:h},'*');
   });
 }
 function install(){
   for(const panel of panels()){
     if(panel.querySelector('.sw-obs-action'))continue;
     const anchor=kind==='heat'?panel.querySelector('.temp-hour-title'):panel.querySelector('.rr-detail-bars');
     if(!anchor)continue;
     const action=document.createElement('div');action.className='sw-obs-action';
     action.innerHTML='<label class="sw-obs-toggle"><input type="checkbox" aria-label="Compare forecasts with synoptic station observations">Compare to actual</label>';
     const status=document.createElement('div');status.className='sw-obs-status';status.hidden=true;
     anchor.parentElement.insertBefore(action,anchor);
     anchor.parentElement.insertBefore(status,action.nextSibling);
     if(kind==='rain'){
       const obs=document.createElement('div');obs.className='sw-obs-row';obs.hidden=true;
       obs.innerHTML='<div class="sw-obs-title">Observed rainfall (mm)</div><div class="sw-obs-values"></div>';
       anchor.appendChild(obs);
     }
     action.querySelector('input').addEventListener('change',event=>{
       const key=dayKey(panel),hours=period(panel);
       if(event.target.checked)enabled.add(key);else enabled.delete(key);
       preserveSelection();
       refresh();
       if(event.target.checked&&!response[String(hours)]){
         componentValue({type:'load',hours,nonce:Date.now()+'-'+Math.random()});
       }
     });
   }
   refresh();
 }
 window.addEventListener('message',event=>{
   if(event.data?.type!=='streamlit:render')return;
   const incoming=event.data?.args?.obs_payload;
   if(incoming && typeof incoming==='object')response=incoming;
   install();
 });
 if(kind==='heat'){
   const dayList=root.querySelector('#temp-days');
   if(dayList&&typeof MutationObserver!=='undefined')new MutationObserver(install).observe(dayList,{childList:true});
 }
 install();
 window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:componentReady',apiVersion:1},'*');
})();
</script>
'''

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
        f"<th style='padding:8px 12px;text-align:left;color:{BASE_TEXT};border-bottom:2px solid {BORDER};'>Rainfall amount</th>"
        f"{header_cells}"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
        f"</div>"
    )

def _pick_dynamic_threshold(data: dict, window_label: str):
    for threshold in (100, 50, 20, 5):
        val = data.get(threshold, {}).get(window_label)
        if val is not None and val >= 65:
            return threshold, val
    return None

def _cell_style(threshold_mm: int, val) -> tuple[str, str]:
    if val is None:
        return "background-color:transparent;", BASE_TEXT
    br, bg_g, bb = _cell_rgb_for_value(threshold_mm, val)
    bg = f"background-color:rgb({br},{bg_g},{bb});"
    text_color = _cell_text_color(_relative_luminance(br, bg_g, bb))
    return bg, text_color

def _coords_caption(
    location_name: str, true_lat: float, true_lon: float, result: dict,
    *, include_live_metadata: bool = False,
) -> str:
    resolution = result.get("resolution") or "0p25"
    res_label = RESOLUTION_LABELS.get(resolution, resolution)
    parts = [
        f"**{location_name}, exact:** {true_lat:.6f}°N, {true_lon:.6f}°E",
        f"**{res_label} grid point used:** {result['grid_lat']:.3f}°N, {result['grid_lon']:.3f}°E",
    ]
    query = resolve_query_point(true_lat, true_lon, resolution)
    if query != (true_lat, true_lon):
        parts.append(f"(query point overridden to {query[0]}°N, {query[1]}°E for {res_label} data)")
    if include_live_metadata and result.get("run_time") is not None:
        run_time = result["run_time"]
        if run_time.tzinfo is None:
            run_time = run_time.replace(tzinfo=timezone.utc)
        run_time = run_time.astimezone(timezone.utc)
        parts.append(f"**Live ECMWF run:** {run_time:%HZ %d %b %Y} · **Resolution:** {res_label}")
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

if not taiwan_enabled():
    if st.session_state.get("last_rain", {}).get("region") == "taiwan":
        st.session_state.pop("last_rain", None)
    if st.session_state.get("last_temp_region") == "taiwan":
        for _key in ("last_temp_result", "last_temp_location_name", "last_temp_true", "last_temp_region"):
            st.session_state.pop(_key, None)

location_name, lat, lon, true_lat, true_lon = location_picker("shared")

# One request obtains both forecast types for the chosen location. Changing
# between the rainfall and temperature views only changes the presentation.
forecast_clicked = st.button("Get forecast", type="primary", key="forecast_button")
if forecast_clicked:
    request_started = time.perf_counter()
    st.session_state.pop("last_rain", None)
    for key in (
        "last_temp_result", "last_temp_location_name", "last_temp_true",
        "last_temp_was_cached", "last_temp_elapsed",
    ):
        st.session_state.pop(key, None)
    st.session_state.pop("last_forecast_elapsed", None)

    thr_out = pct_out = temp_out = None
    try:
        thr_out = get_forecast_with_progress(lat, lon, THRESHOLD_LEAD_DAYS)
    except Exception as exc:
        st.error(f"Rainfall chances are unavailable: {type(exc).__name__}: {exc}")
        with st.expander("Rainfall chances: error details"):
            st.exception(exc)
    try:
        pct_out = get_percentile_raw_with_progress(lat, lon, PERCENTILE_MAX_LEAD_HOURS)
    except Exception as exc:
        st.error(f"Rainfall amounts are unavailable: {type(exc).__name__}: {exc}")
        with st.expander("Rainfall amounts: error details"):
            st.exception(exc)
    try:
        temp_out = get_temperature_percentiles_with_progress(lat, lon, PERCENTILE_MAX_LEAD_HOURS)
    except Exception as exc:
        st.error(f"Heat Index forecast is unavailable: {type(exc).__name__}: {exc}")
        with st.expander("Heat Index forecast: error details"):
            st.exception(exc)

    pct_raw_new, pct_cached_new = pct_out if pct_out else (None, False)
    if thr_out is not None or pct_raw_new is not None:
        st.session_state["last_rain"] = {
            "location_name": location_name,
            "region": region_for_point(lat, lon),
            "true": (true_lat, true_lon),
            "thr": thr_out[0] if thr_out else None,
            "thr_cached": thr_out[1] if thr_out else True,
            "pct_raw": pct_raw_new,
            "pct_cached": pct_cached_new,
        }
    else:
        st.warning("Rainfall forecasts aren't available for this location right now.")

    if temp_out is not None and temp_out[0] is not None:
        temp_result, temp_was_cached = temp_out
        st.session_state["last_temp_result"] = temp_result
        st.session_state["last_temp_location_name"] = location_name
        st.session_state["last_temp_region"] = region_for_point(lat, lon)
        st.session_state["last_temp_true"] = (true_lat, true_lon)
        st.session_state["last_temp_was_cached"] = temp_was_cached
    else:
        st.warning("Heat Index forecast data isn't available for this location right now.")

    st.session_state["last_forecast_elapsed"] = time.perf_counter() - request_started
    # Only label the complete unified request "from cache" when all three
    # sources were available and served from cache.
    st.session_state["last_forecast_cached"] = all(
        entry is not None and entry[0] is not None and entry[1]
        for entry in (thr_out, pct_out, temp_out)
    )
    st.session_state["last_forecast_location"] = location_name

if st.session_state.get("last_forecast_location"):
    st.divider()
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "Rainfall Forecast"
# Preserve existing selections across the label change in already-running sessions.
if st.session_state.get("view_mode") == "Temperature Forecast":
    st.session_state["view_mode"] = "Heat Index Forecast"
if st.session_state.get("forecast_view_selector") == "Temperature Forecast":
    st.session_state["forecast_view_selector"] = "Heat Index Forecast"
# There is nothing to switch between before the first Get forecast request.
# Once forecasts have been requested, show the selector as usual.
if st.session_state.get("last_forecast_location"):
    _selected_mode = st.segmented_control(
        "View",
        ["Rainfall Forecast", "Heat Index Forecast"],
        default=st.session_state["view_mode"],
        key="forecast_view_selector",
    )
    if _selected_mode is not None:
        st.session_state["view_mode"] = _selected_mode
view_mode = st.session_state["view_mode"]
main = st.container()
with main:
    if view_mode == "Rainfall Forecast":
        if st.session_state.get("last_forecast_location"):
            st.caption("See how much rain to expect over the next six days, and your chances of rain over the next two weeks.")
        st.markdown(
            "<style>"
            "@media (orientation: portrait){"
            ".st-key-rain_split [data-testid='stHorizontalBlock']{flex-direction:column !important;}"
            ".st-key-rain_split [data-testid='stColumn'],.st-key-rain_split [data-testid='column']"
            "{width:100% !important;flex:1 1 auto !important;min-width:0 !important;}"
            "}"
            "</style>",
            unsafe_allow_html=True,
        )
        lead_days = THRESHOLD_LEAD_DAYS
        pct_lead_hours = PERCENTILE_MAX_LEAD_HOURS
        if "last_rain" in st.session_state:
            rain = st.session_state["last_rain"]
            rain_name = rain["location_name"]
            rain_true = rain["true"]
            thr_result = rain["thr"]
            pct_raw_result = rain["pct_raw"]
            card_thr = thr_result if (thr_result is not None and thr_result["windows"]) else None
            with st.container(key="rain_split"):
                left_col, right_col = st.columns([1, 1], gap="large")
            with left_col:
                pct_result = None
                if pct_raw_result is None:
                    st.warning(
                        "Rainfall amounts (percentile data) aren't available from the latest ingestion "
                        "right now (store unreachable or not yet published) -- please try again shortly."
                    )
                else:
                    period_results = {
                        h: aggregate_percentile_bins_anchored(pct_raw_result, period_hours=h)
                        for h in (3, 6) if h in AVAILABLE_PERIOD_HOURS
                    }
                    if not any(r.get("days") for r in period_results.values()):
                        st.warning("There isn't enough rainfall data for the available timing options.")
                    else:
                        rain_card_html = render_six_day_forecast_html(period_results, card_thr, rain_name)
                        _obs_region = "TW" if region_for_point(*rain_true) == "taiwan" else "PH"
                        _render_observation_component(
                            rain_card_html, "rain", rain_true[0], rain_true[1],
                            {h: _obs_periods_rain(period_results, h) for h in period_results},
                            _obs_region,
                        )
            with right_col:
                if thr_result is not None:
                    resolution = thr_result.get("resolution") or "0p25"
                    model_label = f"ECMWF ENS {RESOLUTION_LABELS.get(resolution, resolution)}"
                    if not thr_result["windows"]:
                        st.warning("The extended rain chances are temporarily unavailable.")
                    else:
                        today_ph = datetime.now(PH_TZ).date()
                        upcoming_windows = [
                            w for w in thr_result["windows"]
                            if w["start_utc"].astimezone(PH_TZ).date() >= today_ph
                        ]
                        num_days_shown = min(14, len(upcoming_windows or thr_result["windows"]))
                        outlook_html = render_14day_outlook_html(thr_result, rain_name)
                        if hasattr(st, "iframe"):
                            st.iframe(outlook_html, height="content")
                        else:
                            # The embedded outlook reports its height after each
                            # tab switch, day selection and details expansion.
                            components.html(outlook_html, height=980, scrolling=False)


    elif view_mode == "Heat Index Forecast":
        if "last_temp_result" in st.session_state:
            temp_result = st.session_state["last_temp_result"]
            temp_location_name = st.session_state["last_temp_location_name"]
            if not temp_result.get("steps"):
                st.warning("No Heat Index forecast is available for the selected period.")
            else:
                heat_card_html = render_temperature_forecast_html(temp_result, temp_location_name)
                _obs_lat, _obs_lon = st.session_state.get("last_temp_true", (true_lat, true_lon))
                _obs_region = "TW" if region_for_point(_obs_lat, _obs_lon) == "taiwan" else "PH"
                _render_observation_component(
                    heat_card_html, "heat", _obs_lat, _obs_lon,
                    {3: _obs_periods_heat(temp_result)}, _obs_region,
                )


# Keep load timing beneath the displayed forecasts, not near the fetch button.
if st.session_state.get("last_forecast_elapsed") is not None:
    st.caption(
        f"⏱️ Loaded in {st.session_state['last_forecast_elapsed']:.1f}s"
        + (" (from cache)" if st.session_state.get("last_forecast_cached") else "")
    )

# Keep technical methodology and exact locations in the rainfall page footnote.
if view_mode == "Rainfall Forecast":
    with st.expander("Forecast details & sources", expanded=False):
        st.markdown(
            "<style>"
            ".st-key-rain_footnote_text [data-testid='stMarkdown'] p{font-size:.82rem;line-height:1.42;}"
            ".st-key-rain_footnote_text [data-testid='stCaptionContainer'] p{font-size:.72rem;line-height:1.35;}"
            "details:has(.st-key-rain_footnote_text) summary{font-size:.88rem;}"
            "</style>",
            unsafe_allow_html=True,
        )
        with st.container(key="rain_footnote_text"):
            st.markdown(
                "**Forecast estimates.** ECMWF ENS uses 50 ensemble members. The blue bars and "
                "daily totals show the median (P50); the lighter bars and ‘High’ show P90 "
                "(a higher estimate, not a true maximum). Expanded details include P10, P25 and P75. "
                "The median is not necessarily the single most likely outcome."
            )
            st.markdown(
                "**Periods and probabilities.** Six-day totals cover 2 AM–2 AM PHT; "
                "14-day rainfall chances cover 8 AM–8 AM PHT. Each threshold percentage is "
                "the chance of at least that amount, and threshold chances cannot be added. "
                "Possible totals shows separate ranges that sum to 100% when all data is available. "
                "Missing values are not zeros; rainfall probability alone does not predict flooding."
            )
            st.markdown(
                "**Source and updates.** ECMWF Open Data (CC BY 4.0). Rainfall amounts use "
                "00/06/12/18Z runs; rainfall thresholds use 00/12Z. Data is normally fetched "
                "from stored forecast products (0.25° or 0.1°); stale threshold data may "
                "trigger a live 0.25° fetch. The exact coordinates, model grid point, "
                "resolution and available run information appear below."
            )
            if "last_rain" in st.session_state:
                _forecast = st.session_state["last_rain"]
                _site = _forecast["location_name"]
                _lat, _lon = _forecast["true"]
                if _forecast.get("pct_raw") is not None:
                    st.caption("Rainfall amounts: " + _coords_caption(_site, _lat, _lon, _forecast["pct_raw"]))
                if _forecast.get("thr") is not None:
                    st.caption(
                        "Rainfall chances: " + _coords_caption(
                            _site, _lat, _lon, _forecast["thr"],
                            include_live_metadata=_forecast["thr"].get("fetch_mode") in ("combined", "separate"),
                        )
                    )
                    if _forecast["thr"].get("fetch_mode") == "separate" and not _forecast["thr_cached"]:
                        st.caption("The threshold products were retrieved individually because the combined request wasn't available.")
            render_status_footer()
else:
    with st.expander("Forecast details & sources", expanded=False):
        st.markdown(
            "<style>"
            ".st-key-heat_footnote_text [data-testid='stMarkdown'] p{font-size:.82rem;line-height:1.42;}"
            ".st-key-heat_footnote_text [data-testid='stCaptionContainer'] p{font-size:.72rem;line-height:1.35;}"
            "details:has(.st-key-heat_footnote_text) summary{font-size:.88rem;}"
            "</style>",
            unsafe_allow_html=True,
        )
        with st.container(key="heat_footnote_text"):
            st.markdown(
                "**Forecast estimates.** Heat Index and air temperature are ECMWF ENS "
                "3-hourly probabilistic estimates. Daily peaks are the highest available "
                "3-hourly median values, not exact daily maxima. P10 and P90 describe "
                "the ensemble range at the indicated time."
            )
            st.markdown(
                "**Source and updates.** ECMWF Open Data (CC BY 4.0), "
                "retrieved from stored forecast products at the stated resolution. "
                "Forecasts represent model grid-point estimates, not measurements "
                "at the exact selected coordinates."
            )
            st.markdown(
                "**Compare to actual.** Synoptic-station measurements are retrieved "
                "on demand, with PAGASA WIS2 checked before OGIMET (SYNOP). "
                "Observed Heat Index is calculated from station air temperature and "
                "dew point where available. Comparisons are shown only when report "
                "times match the forecast periods; station distance and source "
                "appear alongside the observations."
            )
            if "last_temp_result" in st.session_state:
                _heat = st.session_state["last_temp_result"]
                _lat, _lon = st.session_state.get("last_temp_true", (true_lat, true_lon))
                st.caption(_coords_caption(
                    st.session_state.get("last_temp_location_name", "Selected location"),
                    _lat, _lon, _heat,
                ))
        render_status_footer()
