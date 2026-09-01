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
PERCENTILE_BIN_HOURS = 3  # ECMWF's native accumulation granularity for this product, through step 144
AVAILABLE_PERIOD_HOURS = [3, 6, 12, 24]


def fetch_percentile_rainfall(
    lat: float,
    lon: float,
    max_lead_hours: int = 72,
    progress_callback: ProgressFn = None,
) -> dict:
    """
    Fetches raw total precipitation (tp) from all 50 ENS perturbed members
    (stream=enfo, type=pf) at EVERY native 3-hourly step, and returns each
    step's value as its own self-contained 3-hour precipitation bin.

    ACCUMULATION CONVENTION -- this is the important part. ECMWF's
    accumulated fields (tp included) are aggregated "up to step Y,
    starting at the previous available step X" -- NOT cumulative from the
    start of the forecast. Since tp is natively produced at 3-hourly
    resolution through step 144, every returned field already represents
    exactly the 3 hours since the prior native step, regardless of which
    steps are actually requested. Concretely, this means:

      - Skipping steps (e.g. requesting only step=6, 12, ... to get
        "6-hourly data" directly) does NOT give a 6-hour total. It
        silently returns whatever 3-hour slice ECMWF encoded for that
        step, mislabeled as if it covered the full 6 hours.
      - The only correct approach is to request every 3-hourly step and
        treat each one as an independent, self-contained bin -- no
        diffing against a neighboring step needed or wanted.

    Longer periods (6h/12h/24h) are built afterwards by SUMMING consecutive
    3-hour bins, member-by-member -- see aggregate_percentile_bins() below,
    which does this as a pure in-memory computation with no re-fetch. This
    is what lets the UI offer an adjustable time-step control that responds
    instantly instead of hitting the network again.

    This is a fundamentally heavier fetch than the threshold-forecast
    tables above: 50 individual member fields per step instead of one
    small precomputed product, and now at 3-hourly resolution instead of
    6-hourly, so roughly double the steps of an earlier version of this
    function. Expect noticeably longer load times -- this is why it's a
    separate, opt-in feature.

    max_lead_hours is hard-capped at 72 for now -- 3-hourly steps that far
    out are safely within ECMWF's documented step availability for type=pf
    (0-144h by 3h, for all four run times), so this isn't pushing into any
    step-availability edge case; it's a deliberate scope limit for this
    first version, not a server-side constraint.

    Simplifications, deliberate for this first version: no forcing of
    00-UTC-aligned bin boundaries -- the first bin simply covers the first
    3 hours after whatever run gets fetched, labeled with its actual
    timestamp rather than assumed to start at midnight UTC. Also uses
    cfgrib/xarray like the rest of this module (decoding the whole
    downloaded file before extracting one point) rather than
    streaming/discarding per GRIB message the way a maximally
    memory-efficient implementation would -- simpler and consistent with
    the rest of this codebase, at the cost of holding more in memory
    briefly during decode. Worth revisiting if this becomes a bottleneck
    in practice.

    Returns:
    {
        "run_time": datetime (UTC),
        "grid_lat": float, "grid_lon": float,
        "bins": [
            {"start_utc": dt, "end_utc": dt, "member_values_mm": [50 floats]},
            ...  # one entry per 3-hour bin, in chronological order
        ],
        "bin_hours": 3,
        "downloaded_bytes": int | None,
    }
    """
    max_lead_hours = min(max_lead_hours, 72)  # hard cap, see docstring
    bin_hours = PERCENTILE_BIN_HOURS

    # Every native 3-hourly step, each one its own self-contained bin --
    # deliberately NOT skipping any steps (see accumulation-convention
    # note in the docstring above for why that would silently corrupt
    # the data).
    steps = list(range(bin_hours, max_lead_hours + 1, bin_hours))

    _notify(progress_callback, 0.05, "Connecting to ECMWF Open Data...")

    client = Client(source="ecmwf")
    with tempfile.TemporaryDirectory() as tmpdir:
        target = str(Path(tmpdir) / "tp_members.grib2")
        _notify(
            progress_callback,
            0.15,
            f"Requesting {len(steps)} steps x 50 ensemble members (this is the slow part)...",
        )
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

        _notify(progress_callback, 0.85, "Reading per-member 3-hour totals...")
        var_name = list(ds.data_vars)[0]  # 'tp'
        step_hours = (pd.to_timedelta(ds.step.values) / pd.Timedelta(hours=1)).astype(int)
        # .isel(step=idx) keeps this correct regardless of how cfgrib
        # orders the "number"/"step" dimensions internally, rather than
        # assuming a fixed axis order via raw positional indexing.
        order = np.argsort(step_hours)

        bins = []
        for idx in order:
            idx = int(idx)
            end_h = int(step_hours[idx])
            start_h = end_h - bin_hours

            vals_mm = point[var_name].isel(step=idx).values * 1000.0
            vals_mm = np.clip(vals_mm, 0, None)  # guard tiny negative float noise

            bins.append({
                "start_utc": run_time + timedelta(hours=start_h),
                "end_utc": run_time + timedelta(hours=end_h),
                "member_values_mm": vals_mm.tolist(),
            })

    _notify(progress_callback, 1.0, "Done")

    return {
        "run_time": run_time,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "bins": bins,
        "bin_hours": bin_hours,
        "downloaded_bytes": size_bytes,
    }


def aggregate_percentile_bins(
    raw_result: dict,
    period_hours: int = 6,
    percentiles: list[int] = None,
) -> dict:
    """
    Groups the raw 3-hour bins from fetch_percentile_rainfall() into
    period_hours-long periods (must be a whole multiple of the raw bin
    size -- 3/6/12/24h are all valid) by SUMMING consecutive bins
    member-by-member, then computes mean/median/percentile stats on the
    summed per-member totals.

    Pure in-memory computation, no network access -- safe and cheap to
    call on every rerun, e.g. every time the person moves a time-step
    slider, against an already-fetched/cached raw_result.

    Only complete periods are returned: if the raw bin count isn't evenly
    divisible by (period_hours / bin_hours), the leftover trailing bins
    are dropped rather than shown as a partial/misleading period.

    Returns the same shape the app's rendering/table code has always
    expected:
    {
        "run_time": datetime (UTC),
        "grid_lat": float, "grid_lon": float,
        "days": [
            {"start_utc": dt, "end_utc": dt, "stats": {...}, "member_values_mm": [...]},
            ...
        ],
        "period_hours": int,
        "percentiles": list[int],
        "downloaded_bytes": int | None,
    }
    """
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    bin_hours = raw_result["bin_hours"]
    if period_hours % bin_hours != 0:
        raise ValueError(f"period_hours ({period_hours}) must be a multiple of {bin_hours}")

    bins_per_period = period_hours // bin_hours
    raw_bins = raw_result["bins"]
    num_periods = len(raw_bins) // bins_per_period

    periods = []
    for p in range(num_periods):
        group = raw_bins[p * bins_per_period: (p + 1) * bins_per_period]
        summed_mm = np.sum([b["member_values_mm"] for b in group], axis=0)

        stats = {"mean": float(np.mean(summed_mm)), "median": float(np.median(summed_mm))}
        for pct in percentiles:
            stats[f"p{pct}"] = float(np.percentile(summed_mm, pct))

        periods.append({
            "start_utc": group[0]["start_utc"],
            "end_utc": group[-1]["end_utc"],
            "stats": stats,
            "member_values_mm": summed_mm.tolist(),
        })

    return {
        "run_time": raw_result["run_time"],
        "grid_lat": raw_result["grid_lat"],
        "grid_lon": raw_result["grid_lon"],
        "days": periods,
        "period_hours": period_hours,
        "percentiles": percentiles,
        "downloaded_bytes": raw_result.get("downloaded_bytes"),
    }
