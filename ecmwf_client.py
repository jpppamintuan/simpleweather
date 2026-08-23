"""
ECMWF Open Data fetcher for 24h rainfall exceedance probabilities.

Uses the official `ecmwf-opendata` package to pull the ENS "probability"
stream (enfo / type=ep), which contains precomputed probability-of-exceedance
fields for 24h accumulated total precipitation: tpg1, tpg5, tpg10, tpg20,
tpg25, tpg50, tpg100 (mm). These are fixed thresholds baked into the model
output, computed from ECMWF's 50-member ensemble.

Only the 5 thresholds requested by the app (1/5/20/50/100 mm) are fetched,
and only the 24h windows that start at 00 UTC are kept (the raw product
also includes windows starting at 12 UTC, which we discard here).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import timedelta, timezone

import pandas as pd
import xarray as xr
from ecmwf.opendata import Client

AVAILABLE_THRESHOLDS_MM = [1, 5, 20, 50, 100]

THRESHOLD_COLORS = {
    1: "#00BFFF",
    5: "#FFFF00",
    20: "#FF8C00",
    50: "#800000",
    100: "#800080",
}

PH_TZ = timezone(timedelta(hours=8))
UTC = timezone.utc


def _param_for(threshold_mm: int) -> str:
    return f"tpg{threshold_mm}"


def _request_step_labels(max_lead_days: int) -> list[str]:
    """All 24h windows in 12h increments, e.g. '0-24', '12-36', ... '336-360'."""
    n = (max_lead_days * 24 - 24) // 12 + 1
    return [f"{12 * i}-{12 * i + 24}" for i in range(n)]


def _fetch_single_threshold(
    lat: float,
    lon: float,
    threshold_mm: int,
    client: Client,
    tmpdir: str,
    step_labels: list[str],
):
    """Retrieve one threshold's field and extract the nearest-grid-point
    values, keyed by the window's *end* step in hours (e.g. 24, 36, 48...).
    """
    param = _param_for(threshold_mm)
    target = str(Path(tmpdir) / f"{param}.grib2")

    client.retrieve(
        stream="enfo",
        type="ep",
        step=step_labels,
        param=param,
        target=target,
    )

    ds = xr.open_dataset(target, engine="cfgrib", backend_kwargs={"indexpath": ""})

    grid_lon_query = lon % 360 if float(ds.longitude.max()) > 180 else lon
    point = ds.sel(latitude=lat, longitude=grid_lon_query, method="nearest")

    run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)

    grid_lat = float(point.latitude.values)
    grid_lon = float(point.longitude.values)
    if grid_lon > 180:
        grid_lon -= 360

    var_name = list(ds.data_vars)[0]
    step_hours_end = (pd.to_timedelta(ds.step.values) / pd.Timedelta(hours=1)).astype(int)

    values_by_end_step = {}
    for idx, end_h in enumerate(step_hours_end):
        val = float(point[var_name].isel(step=idx).values)
        values_by_end_step[int(end_h)] = round(val, 1)

    return run_time, grid_lat, grid_lon, values_by_end_step


def _aligned_end_steps(run_hour: int, step_labels: list[str]) -> list[int]:
    """Keep only windows whose *start* aligns with 00 UTC (i.e. skip the
    12 UTC-to-12 UTC windows), returned as end-step hours (S1+24)."""
    aligned = []
    for label in step_labels:
        s1 = int(label.split("-")[0])
        if (run_hour + s1) % 24 == 0:
            aligned.append(s1 + 24)
    return aligned


def _dissemination_available_time(run_time):
    """Approximate when the full 15-day ENS probability product (Set III,
    derived products step 246-360) becomes available, per ECMWF's published
    dissemination schedule. 00Z -> ~08:01 UTC same day. 12Z -> ~20:01 UTC
    same day. (06Z/18Z runs don't carry the full 15-day probability set;
    same lag is used as a fallback estimate.)
    """
    run_hour = run_time.hour
    if run_hour == 0:
        return run_time.replace(hour=8, minute=1, second=0, microsecond=0)
    elif run_hour == 12:
        return run_time.replace(hour=20, minute=1, second=0, microsecond=0)
    else:
        return run_time + timedelta(hours=8, minutes=1)


def fetch_forecast_table(lat: float, lon: float, max_lead_days: int = 15) -> dict:
    """
    Fetch all thresholds for a location and return a structured result:

    {
        "run_time": datetime (UTC),
        "grid_lat": float, "grid_lon": float,
        "windows": [{"label": "0-24", "start_utc": dt, "end_utc": dt}, ...],
        "data": {1: {"0-24": 82.0, ...}, 5: {...}, ...},
        "available_since": datetime (UTC),
        "next_expected": datetime (UTC),
    }
    """
    step_labels = _request_step_labels(max_lead_days)
    client = Client(source="ecmwf")

    run_time = None
    grid_lat = grid_lon = None
    raw_by_threshold: dict[int, dict[int, float]] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for threshold_mm in AVAILABLE_THRESHOLDS_MM:
            rt, glat, glon, values = _fetch_single_threshold(
                lat, lon, threshold_mm, client, tmpdir, step_labels
            )
            if run_time is None:
                run_time, grid_lat, grid_lon = rt, glat, glon
            raw_by_threshold[threshold_mm] = values

    run_hour = run_time.hour
    aligned_end_steps = _aligned_end_steps(run_hour, step_labels)

    windows = []
    for end_h in aligned_end_steps:
        start_h = end_h - 24
        start_utc = run_time + timedelta(hours=start_h)
        end_utc = run_time + timedelta(hours=end_h)
        windows.append({
            "label": f"{start_h}-{end_h}",
            "end_step": end_h,
            "start_utc": start_utc,
            "end_utc": end_utc,
        })

    data = {
        threshold_mm: {
            w["label"]: raw_by_threshold[threshold_mm].get(w["end_step"])
            for w in windows
        }
        for threshold_mm in AVAILABLE_THRESHOLDS_MM
    }

    available_since = _dissemination_available_time(run_time)
    next_run_time = run_time + timedelta(hours=12)
    next_expected = _dissemination_available_time(next_run_time)

    return {
        "run_time": run_time,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "windows": windows,
        "data": data,
        "available_since": available_since,
        "next_expected": next_expected,
    }
