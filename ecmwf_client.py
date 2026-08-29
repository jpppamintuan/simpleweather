"""
ECMWF Open Data fetcher for 24h rainfall exceedance probabilities.

Uses the official `ecmwf-opendata` package to pull the ENS "probability"
stream (enfo / type=ep), which contains precomputed probability-of-exceedance
fields for 24h accumulated total precipitation: tpg1, tpg5, tpg10, tpg20,
tpg25, tpg50, tpg100 (mm). These are fixed thresholds baked into the model
output, computed from a 50-member ensemble.

Supports two ECMWF ensemble models via the `model` parameter: "ifs" (the
default physics-based ENS) and "aifs-ens" (ECMWF's newer AI-based
ensemble). Both publish the same product structure and parameter names --
only the Client's `model=` kwarg differs between them.

All 5 thresholds used by the app (1/5/20/50/100 mm) are fetched in a single
combined request per model (one file, one network round trip) rather than
one request per threshold -- this is the main optimization over the earlier
version. If the combined request fails for any reason, the client falls
back to fetching thresholds one at a time so the app still works, just
slower.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import timedelta, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr
from ecmwf.opendata import Client

AVAILABLE_THRESHOLDS_MM = [1, 5, 20, 50, 100]

THRESHOLD_COLORS = {
    1: "#00BFFF",
    5: "#FFFF00",
    20: "#FFA500",
    50: "#FF0000",
    100: "#800080",
}

# ECMWF's physics-based ensemble (the original/default) vs. their newer
# AI-based ensemble model. Both publish the same "ep" (probability) product
# type with the same tpg1/tpg5/tpg20/tpg50/tpg100 parameter names -- the
# only difference in the request is the `model=` kwarg on the Client.
MODEL_LABELS = {"ifs": "ECMWF ENS", "aifs-ens": "AIFS ENS"}
DEFAULT_MODEL = "ifs"

PH_TZ = timezone(timedelta(hours=8))
UTC = timezone.utc

ProgressFn = Optional[Callable[[float, str], None]]


def _param_for(threshold_mm: int) -> str:
    return f"tpg{threshold_mm}"


def _request_step_labels(max_lead_days: int, increment_hours: int = 12, max_step_hours: int = 360) -> list[str]:
    """All 24h windows in `increment_hours` increments, e.g. '0-24', '12-36',
    ... up to just past max_lead_days*24 hours, capped at max_step_hours.

    Requests a buffer beyond the requested range, sized to guarantee enough
    steps regardless of which hour the model's latest run falls on -- 12h
    spaced steps can only ever align to 00 UTC for a run at an even
    12-hour offset (00Z/12Z), so a run at 06Z or 18Z needs the fallback
    alignment in _aligned_end_steps() below rather than more buffer here.
    General formula: buffer = 24 - increment_hours.

    max_step_hours matters specifically for AIFS: per ECMWF's own
    documentation, AIFS ENS publishes the ep product at 12h-spaced steps
    for all four run times (00/06/12/18Z) -- same increment as IFS, not
    the 6h I'd assumed in an earlier attempt at this fix. The actual
    limitation is that 06Z and 18Z runs are capped at step 144 (day 6),
    while 00Z/12Z runs go to the full step 360 (day 15). Requesting beyond
    a run's actual cap is what caused "no aligned windows" -- the request
    for out-of-range steps was failing silently rather than erroring."""
    target_hours = max_lead_days * 24
    buffer_hours = 24 - increment_hours
    buffered_hours = min(target_hours + buffer_hours, max_step_hours)
    n = (buffered_hours - 24) // increment_hours + 1
    return [f"{increment_hours * i}-{increment_hours * i + 24}" for i in range(n)]


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


def _fetch_combined(lat, lon, step_labels, tmpdir, progress_callback: ProgressFn, model: str):
    """Single request for all thresholds at once. Raises on any failure so the
    caller can fall back to per-threshold requests."""
    client = Client(source="ecmwf", model=model)
    params = [_param_for(t) for t in AVAILABLE_THRESHOLDS_MM]
    target = str(Path(tmpdir) / "combined.grib2")

    _notify(progress_callback, 0.15, "Requesting latest forecast (5 thresholds)...")
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


def _fetch_separate(lat, lon, step_labels, tmpdir, progress_callback: ProgressFn, model: str):
    """Fallback: one request per threshold (slower, but more forgiving)."""
    client = Client(source="ecmwf", model=model)
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


def _aligned_end_steps(run_hour: int, step_labels: list[str], target_hour: int = 0) -> list[int]:
    """Keep only windows whose *start* aligns with `target_hour` UTC
    (default 00 UTC -- i.e. skip the 12 UTC-to-12 UTC windows for a 12h-
    spaced IFS request), returned as end-step hours (S1+24)."""
    aligned = []
    for label in step_labels:
        s1 = int(label.split("-")[0])
        if (run_hour + s1 - target_hour) % 24 == 0:
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
    model: str = DEFAULT_MODEL,
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
        "model": str,
    }

    progress_callback, if given, is called as progress_callback(fraction, message)
    at various points during the fetch (fraction in [0, 1]).

    model selects which ECMWF ensemble to query: "ifs" (the default
    physics-based ENS) or "aifs-ens" (ECMWF's newer AI-based ensemble).
    Note: available_since/next_expected are estimated from IFS's published
    dissemination schedule regardless of which model is requested -- AIFS
    runs on a different production pipeline with its own (undocumented,
    as far as this app knows) timing, so those two fields are a rougher
    estimate for AIFS than for IFS. Everything else (the actual forecast
    data, run time, grid point) comes directly from the fetched file and
    is exact either way.
    """
    if model == "aifs-ens":
        # 06Z/18Z AIFS runs only publish up to step 144 (day 6); 00Z/12Z
        # runs go to the full day 15 (step 360). We don't know which of
        # the 4 the "latest" run will resolve to until after fetching, so
        # request only the range that's safe regardless -- this is what
        # was missing before: requesting the full 15-day range against a
        # 06Z/18Z run (capped at day 6) asked for steps that don't exist
        # for that run, which produced zero usable data rather than an
        # error. Trade-off: AIFS is capped at 6 days here even on a
        # 00Z/12Z run where more would technically be available.
        effective_max_lead_days = min(max_lead_days, 6)
        step_labels = _request_step_labels(effective_max_lead_days, increment_hours=12, max_step_hours=144)
        capped_to_day6 = max_lead_days > 6
    else:
        step_labels = _request_step_labels(max_lead_days, increment_hours=12)
        capped_to_day6 = False

    _notify(progress_callback, 0.05, "Connecting to ECMWF Open Data...")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            run_time, grid_lat, grid_lon, raw_by_threshold, size_bytes = _fetch_combined(
                lat, lon, step_labels, tmpdir, progress_callback, model
            )
            fetch_mode = "combined"
        except Exception:
            _notify(
                progress_callback,
                0.2,
                "Combined request failed, retrying per threshold...",
            )
            run_time, grid_lat, grid_lon, raw_by_threshold, size_bytes = _fetch_separate(
                lat, lon, step_labels, tmpdir, progress_callback, model
            )
            fetch_mode = "separate"

    _notify(progress_callback, 0.9, "Computing aligned forecast windows...")

    run_hour = run_time.hour
    aligned_end_steps = _aligned_end_steps(run_hour, step_labels)
    aligned_to_utc_midnight = True

    if not aligned_end_steps:
        # 12h-spaced steps from a 06Z or 18Z run can never land on a 00 UTC
        # boundary (only 06/18 UTC ones) -- this happens for AIFS, which
        # runs 4x/day, unlike IFS which only ever produces this product at
        # 00Z/12Z. Rather than show nothing, align to the run's own hour
        # instead of insisting on exactly midnight UTC.
        aligned_end_steps = _aligned_end_steps(run_hour, step_labels, target_hour=run_hour)
        aligned_to_utc_midnight = False

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
        "model": model,
        "aligned_to_utc_midnight": aligned_to_utc_midnight,
        "capped_to_day6": capped_to_day6,
    }


DEFAULT_PERCENTILES = [10, 25, 50, 75, 90]


def fetch_percentile_rainfall(
    lat: float,
    lon: float,
    max_lead_days: int = 5,
    percentiles: list[int] = None,
    progress_callback: ProgressFn = None,
) -> dict:
    """
    Fetches raw total precipitation (tp) from all 50 ENS perturbed members
    (stream=enfo, type=pf) and computes daily-accumulation percentiles
    locally, since there's no server-side product for arbitrary
    percentiles the way there is for fixed-threshold exceedance (`ep`).

    This is a fundamentally heavier fetch than the rest of this module:
    50 individual member fields instead of one small precomputed product,
    for the same date range. Expect substantially longer load times --
    this is why it's a separate, opt-in feature rather than part of the
    main tables. tp is a cumulative value (total since forecast start),
    so each day's rainfall = tp at day-end minus tp at day-start, computed
    per member, then percentiles/mean/median are taken across the 50
    resulting daily values.

    Simplifications versus fetch_forecast_table() above, deliberate for
    this first version: no forcing of 00-UTC-aligned daily boundaries --
    "Day 1" here is simply the first 24h after whatever run gets fetched,
    labeled with its actual timestamp rather than assumed to start at
    midnight UTC. Also uses cfgrib/xarray like the rest of this module
    (decoding the whole downloaded file before extracting one point)
    rather than streaming/discarding per GRIB message the way a maximally
    memory-efficient implementation would -- simpler and consistent with
    the rest of this codebase, at the cost of holding more in memory
    briefly during decode. Worth revisiting if this becomes a bottleneck
    in practice.
    """
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    _notify(progress_callback, 0.05, "Connecting to ECMWF Open Data...")

    steps = list(range(0, max_lead_days * 24 + 1, 24))  # 0, 24, 48, ..., need N+1 points for N days

    client = Client(source="ecmwf")
    with tempfile.TemporaryDirectory() as tmpdir:
        target = str(Path(tmpdir) / "tp_members.grib2")
        _notify(progress_callback, 0.15, f"Requesting {len(steps)} steps x 50 ensemble members (this is the slow part)...")
        result = client.retrieve(
            stream="enfo",
            type="pf",
            param="tp",
            step=steps,
            number=list(range(1, 51)),
            target=target,
        )
        size_bytes = getattr(result, "size", None)

        _notify(progress_callback, 0.6, "Decoding GRIB2 data...")
        ds = xr.open_dataset(target, engine="cfgrib", backend_kwargs={"indexpath": ""})

        grid_lon_query = lon % 360 if float(ds.longitude.max()) > 180 else lon
        point = ds.sel(latitude=lat, longitude=grid_lon_query, method="nearest")

        run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)
        grid_lat = float(point.latitude.values)
        grid_lon = float(point.longitude.values)
        if grid_lon > 180:
            grid_lon -= 360

        _notify(progress_callback, 0.85, "Computing daily totals and percentiles...")
        var_name = list(ds.data_vars)[0]  # 'tp'
        step_hours = (pd.to_timedelta(ds.step.values) / pd.Timedelta(hours=1)).astype(int)
        order = np.argsort(step_hours)
        sorted_step_hours = step_hours[order]

        days = []
        for d in range(len(sorted_step_hours) - 1):
            start_idx = int(order[d])
            end_idx = int(order[d + 1])
            start_h = int(sorted_step_hours[d])
            end_h = int(sorted_step_hours[d + 1])

            # .isel(step=idx) keeps this correct regardless of how cfgrib
            # orders the "number"/"step" dimensions internally, rather than
            # assuming a fixed axis order via raw positional indexing.
            start_vals_mm = point[var_name].isel(step=start_idx).values * 1000.0
            end_vals_mm = point[var_name].isel(step=end_idx).values * 1000.0
            daily_mm = np.clip(end_vals_mm - start_vals_mm, 0, None)  # guard tiny negative float noise

            stats = {"mean": float(np.mean(daily_mm)), "median": float(np.median(daily_mm))}
            for p in percentiles:
                stats[f"p{p}"] = float(np.percentile(daily_mm, p))

            days.append({
                "start_utc": run_time + timedelta(hours=start_h),
                "end_utc": run_time + timedelta(hours=end_h),
                "stats": stats,
                "member_values_mm": daily_mm.tolist(),
            })

    _notify(progress_callback, 1.0, "Done")

    return {
        "run_time": run_time,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "days": days,
        "percentiles": percentiles,
        "downloaded_bytes": size_bytes,
    }
