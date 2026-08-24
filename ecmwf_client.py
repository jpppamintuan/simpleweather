"""
ECMWF Open Data fetcher for 24h rainfall exceedance probabilities.

Uses the official `ecmwf-opendata` package to pull the ENS "probability"
stream (enfo / type=ep), which contains precomputed probability-of-exceedance
fields for 24h accumulated total precipitation: tpg1, tpg5, tpg10, tpg20,
tpg25, tpg50, tpg100 (mm). These are fixed thresholds baked into the model
output, computed from ECMWF's 50-member ensemble.

All 5 thresholds used by the app (1/5/20/50/100 mm) are fetched in a single
combined request (one file, one network round trip) rather than one request
per threshold -- this is the main optimization over the earlier version.
If the combined request fails for any reason, the client falls back to
fetching thresholds one at a time so the app still works, just slower.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import timedelta, timezone
from typing import Callable, Optional

import pandas as pd
import xarray as xr
from ecmwf.opendata import Client

AVAILABLE_THRESHOLDS_MM = [1, 5, 20, 50, 100]

THRESHOLD_COLORS = {
    1: "#00BFFF",
    5: "#FFFF00",
    20: "#FF8C00",
    50: "#FF0000",
    100: "#800080",
}

PH_TZ = timezone(timedelta(hours=8))
UTC = timezone.utc

ProgressFn = Optional[Callable[[float, str], None]]


def _param_for(threshold_mm: int) -> str:
    return f"tpg{threshold_mm}"


def _request_step_labels(max_lead_days: int) -> list[str]:
    """All 24h windows in 12h increments, e.g. '0-24', '12-36', ... up to
    just past max_lead_days*24 hours.

    Requests a 12h buffer beyond the requested range. Reason: whether the
    latest run is 00Z or 12Z changes which steps align to 00 UTC starts --
    a 12Z run "loses" one aligned window near the end of the requested
    range compared to a 00Z run, for the same max_lead_days. The buffer
    ensures both parities have enough steps to produce the full requested
    day count; fetch_forecast_table() then truncates to exactly
    max_lead_days after aligning. Capped at 360h (ECMWF's max ENS step) --
    requesting the full 15 days on a 12Z run may still fall one day short,
    since there's no more data to buffer with at that point."""
    target_hours = max_lead_days * 24
    buffered_hours = min(target_hours + 12, 360)
    n = (buffered_hours - 24) // 12 + 1
    return [f"{12 * i}-{12 * i + 24}" for i in range(n)]


def _notify(progress_callback: ProgressFn, frac: float, msg: str) -> None:
    if progress_callback is not None:
        progress_callback(frac, msg)


def _extract_point_series(ds: xr.Dataset, var_name: str, point) -> dict[int, float]:
    step_hours_end = (pd.to_timedelta(ds.step.values) / pd.Timedelta(hours=1)).astype(int)
    values = {}
    for idx, end_h in enumerate(step_hours_end):
        val = float(point[var_name].isel(step=idx).values)
        values[int(end_h)] = round(val, 1)
    return values


def _fetch_combined(lat, lon, step_labels, tmpdir, progress_callback: ProgressFn):
    """Single request for all thresholds at once. Raises on any failure so the
    caller can fall back to per-threshold requests."""
    client = Client(source="ecmwf")
    params = [_param_for(t) for t in AVAILABLE_THRESHOLDS_MM]
    target = str(Path(tmpdir) / "combined.grib2")

    _notify(progress_callback, 0.15, "Requesting latest ENS forecast (all 5 thresholds)...")
    result = client.retrieve(
        stream="enfo",
        type="ep",
        step=step_labels,
        param=params,
        target=target,
    )
    size_bytes = getattr(result, "size", None)

    _notify(progress_callback, 0.65, "Decoding GRIB2 data...")
    ds = xr.open_dataset(target, engine="cfgrib", backend_kwargs={"indexpath": ""})

    grid_lon_query = lon % 360 if float(ds.longitude.max()) > 180 else lon
    point = ds.sel(latitude=lat, longitude=grid_lon_query, method="nearest")

    run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)
    grid_lat = float(point.latitude.values)
    grid_lon = float(point.longitude.values)
    if grid_lon > 180:
        grid_lon -= 360

    _notify(progress_callback, 0.8, "Extracting nearest grid point for each threshold...")
    raw_by_threshold: dict[int, dict[int, float]] = {}
    for threshold_mm in AVAILABLE_THRESHOLDS_MM:
        var_name = _param_for(threshold_mm)
        if var_name not in ds.data_vars:
            raise KeyError(
                f"{var_name} not found in combined dataset (has: {list(ds.data_vars)})"
            )
        raw_by_threshold[threshold_mm] = _extract_point_series(ds, var_name, point)

    return run_time, grid_lat, grid_lon, raw_by_threshold, size_bytes


def _fetch_separate(lat, lon, step_labels, tmpdir, progress_callback: ProgressFn):
    """Fallback: one request per threshold (slower, but more forgiving)."""
    client = Client(source="ecmwf")
    run_time = grid_lat = grid_lon = None
    raw_by_threshold: dict[int, dict[int, float]] = {}
    size_bytes = 0

    for i, threshold_mm in enumerate(AVAILABLE_THRESHOLDS_MM):
        _notify(
            progress_callback,
            0.2 + 0.65 * (i / len(AVAILABLE_THRESHOLDS_MM)),
            f"Fetching {threshold_mm} mm threshold ({i + 1}/{len(AVAILABLE_THRESHOLDS_MM)})...",
        )
        param = _param_for(threshold_mm)
        target = str(Path(tmpdir) / f"{param}.grib2")
        result = client.retrieve(
            stream="enfo", type="ep", step=step_labels, param=param, target=target
        )
        size_bytes += getattr(result, "size", 0) or 0

        ds = xr.open_dataset(target, engine="cfgrib", backend_kwargs={"indexpath": ""})
        grid_lon_query = lon % 360 if float(ds.longitude.max()) > 180 else lon
        point = ds.sel(latitude=lat, longitude=grid_lon_query, method="nearest")

        if run_time is None:
            run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)
            grid_lat = float(point.latitude.values)
            grid_lon = float(point.longitude.values)
            if grid_lon > 180:
                grid_lon -= 360

        raw_by_threshold[threshold_mm] = _extract_point_series(ds, param, point)

    return run_time, grid_lat, grid_lon, raw_by_threshold, size_bytes


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
    same day. ECMWF's own docs describe this as "available between 7 and 9
    hours after the forecast starting date and time", which this matches.
    """
    run_hour = run_time.hour
    if run_hour == 0:
        return run_time.replace(hour=8, minute=1, second=0, microsecond=0)
    elif run_hour == 12:
        return run_time.replace(hour=20, minute=1, second=0, microsecond=0)
    else:
        return run_time + timedelta(hours=8, minutes=1)


def fetch_forecast_table(
    lat: float,
    lon: float,
    max_lead_days: int = 15,
    progress_callback: ProgressFn = None,
) -> dict:
    """
    Fetch all thresholds for a location and return a structured result:

    {
        "run_time": datetime (UTC),
        "grid_lat": float, "grid_lon": float,
        "windows": [{"label": "0-24", "start_utc": dt, "end_utc": dt}, ...],
        "data": {1: {"0-24": 82.0, ...}, 5: {...}, ...},
        "available_since": datetime (UTC),
        "next_expected": datetime (UTC),
        "downloaded_bytes": int | None,
        "fetch_mode": "combined" | "separate",
    }

    progress_callback, if given, is called as progress_callback(fraction, message)
    at various points during the fetch (fraction in [0, 1]).
    """
    step_labels = _request_step_labels(max_lead_days)

    _notify(progress_callback, 0.05, "Connecting to ECMWF Open Data...")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            run_time, grid_lat, grid_lon, raw_by_threshold, size_bytes = _fetch_combined(
                lat, lon, step_labels, tmpdir, progress_callback
            )
            fetch_mode = "combined"
        except Exception:
            _notify(
                progress_callback,
                0.2,
                "Combined request failed, retrying per threshold...",
            )
            run_time, grid_lat, grid_lon, raw_by_threshold, size_bytes = _fetch_separate(
                lat, lon, step_labels, tmpdir, progress_callback
            )
            fetch_mode = "separate"

    _notify(progress_callback, 0.9, "Computing aligned forecast windows...")

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
    windows = windows[:max_lead_days]

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

    _notify(progress_callback, 1.0, "Done")

    return {
        "run_time": run_time,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "windows": windows,
        "data": data,
        "available_since": available_since,
        "next_expected": next_expected,
        "downloaded_bytes": size_bytes,
        "fetch_mode": fetch_mode,
    }
