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


def _aligned_end_steps(run_hour: int, end_steps: list[int], target_hour: int = 0) -> list[int]:
    """Keep only windows whose *start* aligns with `target_hour` UTC
    (default 00 UTC -- i.e. skip the 12 UTC-to-12 UTC windows for a 12h-
    spaced IFS request). Takes plain end-step hours (not "S1-S2" range
    strings) so this can filter steps from either a live GRIB fetch or
    an already-cropped stored grid -- both just need to know which
    end-hours line up with a 00 UTC start."""
    aligned = []
    for end_h in end_steps:
        s1 = end_h - 24
        if (run_hour + s1 - target_hour) % 24 == 0:
            aligned.append(end_h)
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

    end_steps = [int(label.split("-")[1]) for label in step_labels]
    result = _build_threshold_result(
        run_time=run_time,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        raw_by_threshold=raw_by_threshold,
        end_steps=end_steps,
        max_lead_days=max_lead_days,
        model=model,
        downloaded_bytes=size_bytes,
        fetch_mode=fetch_mode,
        capped_to_day6=capped_to_day6,
    )
    _notify(progress_callback, 1.0, "Done")
    return result


def _build_threshold_result(
    run_time,
    grid_lat: float,
    grid_lon: float,
    raw_by_threshold: dict,
    end_steps: list[int],
    max_lead_days: int,
    model: str,
    downloaded_bytes,
    fetch_mode: str,
    capped_to_day6: bool,
) -> dict:
    """Shared tail logic for building the fetch_forecast_table() result
    shape (windows/data/metadata) from already-decoded per-threshold point
    data. Used both by the live-fetch path above and by
    read_threshold_result_from_store() below, which reads the same
    per-threshold point values out of a pre-fetched stored grid instead of
    a fresh GRIB download -- keeping this logic in one place means the
    24h-window alignment can't drift between the two paths."""
    run_hour = run_time.hour
    aligned_end_steps = _aligned_end_steps(run_hour, end_steps)
    aligned_to_utc_midnight = True

    if not aligned_end_steps:
        # 12h-spaced steps from a 06Z or 18Z run can never land on a 00 UTC
        # boundary (only 06/18 UTC ones) -- this happens for AIFS, which
        # runs 4x/day, unlike IFS which only ever produces this product at
        # 00Z/12Z. Rather than show nothing, align to the run's own hour
        # instead of insisting on exactly midnight UTC.
        aligned_end_steps = _aligned_end_steps(run_hour, end_steps, target_hour=run_hour)
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

    return {
        "run_time": run_time,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "windows": windows,
        "data": data,
        "available_since": available_since,
        "next_expected": next_expected,
        "downloaded_bytes": downloaded_bytes,
        "fetch_mode": fetch_mode,
        "model": model,
        "aligned_to_utc_midnight": aligned_to_utc_midnight,
        "capped_to_day6": capped_to_day6,
    }


DEFAULT_PERCENTILES = [10, 25, 50, 75, 90]
PERCENTILE_BIN_HOURS = 3  # ECMWF's native accumulation granularity for this product, through step 144
AVAILABLE_PERIOD_HOURS = [3, 6, 12, 24]


def check_latest_threshold_run(model: str = DEFAULT_MODEL):
    """Cheap metadata-only check for the most recent available threshold
    (ep) run -- does NOT download any data, per ecmwf-opendata's
    documented Client.latest() behavior. Used by ingestion to decide
    whether a full fetch is actually needed before paying for one."""
    client = Client(source="ecmwf", model=model)
    return client.latest(stream="enfo", type="ep", param=_param_for(AVAILABLE_THRESHOLDS_MM[0]))


def check_latest_percentile_run():
    """Cheap metadata-only check for the most recent available raw-member
    (pf) run -- does NOT download any data."""
    client = Client(source="ecmwf")
    return client.latest(stream="enfo", type="pf", param="tp")


def fetch_percentile_rainfall(
    lat: float,
    lon: float,
    max_lead_hours: int = 120,
    progress_callback: ProgressFn = None,
) -> dict:
    """
    Fetches raw total precipitation (tp) from all 50 ENS perturbed members
    (stream=enfo, type=pf) at EVERY native 3-hourly step, and returns the
    per-member precipitation that fell during each individual 3-hour bin.

    ACCUMULATION CONVENTION -- this is the important part, and confirmed
    empirically (an earlier version of this function got it backwards --
    see below). Each requested step's tp field is CUMULATIVE from the
    start of the forecast (step 0), not a self-contained "since the
    previous step" value. So step=6 holds total precip for 0-6h, step=9
    holds total precip for 0-9h, and so on -- each one strictly >= the
    last. That means:

      - Skipping steps (e.g. requesting only step=6, 12, ... to get
        "6-hourly data" directly) DOES still give a technically-valid
        cumulative value at each of those steps, but if you only request
        the coarser steps, the ecmwf-opendata client can end up
        subsetting/aligning to whatever's available at that spacing in a
        way that doesn't line up cleanly across all 5 threshold-style
        windows this app needs -- requesting every native 3-hourly step
        sidesteps that entirely and is what's used here.
      - To get the precip for an individual 3-hour bin, DIFF each step's
        cumulative value against the previous step's cumulative value
        (step=0's implicit value is 0, so the first bin is just step=3's
        value as-is). An earlier version of this function used the raw
        per-step values directly with no diffing, which produced
        constantly-increasing "totals so far" instead of per-period
        rainfall -- that was the bug; this version diffs consecutive
        cumulative values to recover the true per-bin amount.

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

    max_lead_hours is hard-capped at 120 for now -- 3-hourly steps that far
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
    max_lead_hours = min(max_lead_hours, 120)  # hard cap, see docstring
    bin_hours = PERCENTILE_BIN_HOURS

    # Every native 3-hourly step INCLUDING step=0 -- step=0 is needed as
    # the diffing baseline for the first bin (see accumulation-convention
    # note in the docstring above). Deliberately not skipping any steps
    # in between either, since tp is cumulative-from-start and diffing
    # only works correctly against the immediately preceding step.
    steps = list(range(0, max_lead_hours + 1, bin_hours))

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

        _notify(progress_callback, 0.85, "Computing per-member 3-hour totals...")
        var_name = list(ds.data_vars)[0]  # 'tp'
        step_hours = (pd.to_timedelta(ds.step.values) / pd.Timedelta(hours=1)).astype(int)
        # .isel(step=idx) keeps this correct regardless of how cfgrib
        # orders the "number"/"step" dimensions internally, rather than
        # assuming a fixed axis order via raw positional indexing.
        order = np.argsort(step_hours)

        bins = []
        prev_cumulative_mm = None  # step=0's implicit cumulative value is 0
        for idx in order:
            idx = int(idx)
            end_h = int(step_hours[idx])
            start_h = end_h - bin_hours

            cumulative_mm = point[var_name].isel(step=idx).values * 1000.0

            if end_h == 0:
                # This is the step=0 baseline itself -- not a bin, just
                # establishes the starting point for the first diff.
                prev_cumulative_mm = cumulative_mm
                continue

            period_vals_mm = cumulative_mm - prev_cumulative_mm
            period_vals_mm = np.clip(period_vals_mm, 0, None)  # guard tiny negative float noise
            prev_cumulative_mm = cumulative_mm

            bins.append({
                "start_utc": run_time + timedelta(hours=start_h),
                "end_utc": run_time + timedelta(hours=end_h),
                "member_values_mm": period_vals_mm.tolist(),
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


# ---------------------------------------------------------------------------
# Grid-fetch functions -- used by the scheduled ingestion job (ingest.py),
# NOT by the live Streamlit app.
#
# Everything above this point fetches then immediately narrows to a single
# nearest-neighbor point via .sel(..., method="nearest") -- fine for "show
# me this one location live," wrong for ingestion, which needs to store a
# whole region so a future query layer (a Cloudflare Worker, see project
# plan) can look up ANY point within it, not just today's 5 hardcoded
# locations. These functions return cropped xarray Datasets instead of
# point dicts, ready to write to Zarr.
# ---------------------------------------------------------------------------


def _crop_to_bbox(ds: xr.Dataset, bbox: dict) -> xr.Dataset:
    """Crops a decoded ECMWF dataset to a lat/lon box, handling the 0-360
    vs -180-180 longitude convention ECMWF's grids use (same normalization
    the point-extraction functions above already do)."""
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
    if float(ds.longitude.max()) > 180:
        lon_min, lon_max = lon_min % 360, lon_max % 360
    lat_slice = slice(bbox["lat_max"], bbox["lat_min"])  # ECMWF grids run north-to-south
    return ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))


def fetch_threshold_grid(
    bbox: dict,
    max_lead_days: int = 15,
    model: str = DEFAULT_MODEL,
    progress_callback: ProgressFn = None,
) -> xr.Dataset:
    """
    Ingestion counterpart to fetch_forecast_table(): fetches the same
    precomputed threshold-exceedance product (the 5 combined thresholds),
    but returns the CROPPED GRID over `bbox` instead of a single
    nearest-neighbor point.

    bbox: {"lat_min":, "lat_max":, "lon_min":, "lon_max":} in degrees,
    longitude in -180..180 (converted internally if the source grid uses
    0..360).

    Deliberately does NOT do the 24h-window alignment fetch_forecast_table()
    does (aligning 12h-spaced steps to 00 UTC boundaries etc.) -- that's a
    display/query concern, kept out of ingestion the same way percentile
    aggregation was kept out of fetch_percentile_rainfall() above. The
    returned dataset's "step" dimension is the raw request steps in hours;
    whatever reads this later (the Worker, in Phase 2) does that alignment
    at query time.
    """
    step_labels = _request_step_labels(max_lead_days, increment_hours=12)
    _notify(progress_callback, 0.05, "Connecting to ECMWF Open Data...")

    with tempfile.TemporaryDirectory() as tmpdir:
        client = Client(source="ecmwf", model=model)
        params = [_param_for(t) for t in AVAILABLE_THRESHOLDS_MM]
        target = str(Path(tmpdir) / "combined.grib2")

        _notify(progress_callback, 0.15, "Requesting latest forecast (5 thresholds, full grid)...")
        client.retrieve(stream="enfo", type="ep", step=step_labels, param=params, target=target)

        _notify(progress_callback, 0.6, "Decoding GRIB2 data...")
        ds = xr.open_dataset(target, engine="cfgrib", backend_kwargs={"indexpath": ""})

        run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)

        _notify(progress_callback, 0.8, "Cropping to bounding box...")
        ds = _crop_to_bbox(ds, bbox).load()  # .load() -- resolve into memory before the tmpdir is deleted

    ds.attrs["run_time"] = run_time.isoformat()
    ds.attrs["model"] = model
    _notify(progress_callback, 1.0, "Done")
    return ds


def fetch_percentile_grid(
    bbox: dict,
    max_lead_hours: int = 120,
    progress_callback: ProgressFn = None,
) -> xr.Dataset:
    """
    Ingestion counterpart to fetch_percentile_rainfall(): fetches raw tp
    from all 50 members at every native 3-hourly step, diffs consecutive
    cumulative steps into true per-3-hour-bin amounts (same accumulation
    logic as fetch_percentile_rainfall() -- see that docstring for why
    diffing against the previous step is required), and returns the
    CROPPED GRID over `bbox` with a "step" dimension of 3-hour bins,
    instead of collapsing straight to one point's percentile statistics.

    Storing per-member values (not already-computed percentiles) is
    deliberate: percentiles for an arbitrary future query point get
    computed on demand by whatever reads this (the Worker, in Phase 2),
    the same way aggregate_percentile_bins() does it today for the live
    app -- ingestion's job is just getting the raw numbers stored.
    """
    max_lead_hours = min(max_lead_hours, 120)
    bin_hours = PERCENTILE_BIN_HOURS
    steps = list(range(0, max_lead_hours + 1, bin_hours))

    _notify(progress_callback, 0.05, "Connecting to ECMWF Open Data...")

    with tempfile.TemporaryDirectory() as tmpdir:
        client = Client(source="ecmwf")
        target = str(Path(tmpdir) / "tp_members.grib2")
        _notify(progress_callback, 0.15, f"Requesting {len(steps)} steps x 50 members (full grid)...")
        client.retrieve(
            stream="enfo", type="pf", param="tp", step=steps, number=list(range(1, 51)), target=target
        )

        _notify(progress_callback, 0.6, "Decoding GRIB2 data...")
        ds = xr.open_dataset(target, engine="cfgrib", backend_kwargs={"indexpath": ""})

        run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)

        _notify(progress_callback, 0.75, "Cropping to bounding box...")
        ds = _crop_to_bbox(ds, bbox)

        _notify(progress_callback, 0.85, "Computing per-member 3-hour totals...")
        var_name = list(ds.data_vars)[0]  # 'tp', cumulative-since-forecast-start
        ds = ds.sortby("step")
        tp_mm = ds[var_name] * 1000.0
        period_mm = tp_mm.diff(dim="step")  # each entry = precip during that specific 3h bin
        period_mm = period_mm.clip(min=0)  # guard tiny negative float noise

        ds_out = period_mm.to_dataset(name="precip_3h_mm").load()

    ds_out.attrs["run_time"] = run_time.isoformat()
    ds_out.attrs["bin_hours"] = bin_hours
    _notify(progress_callback, 1.0, "Done")
    return ds_out


# ---------------------------------------------------------------------------
# Store-read functions -- used by the Streamlit app to build the exact same
# result shapes as the live-fetch functions above, but sourced from an
# already-fetched grid (opened from the GitHub-hosted Zarr store) instead of
# a fresh ECMWF download. This is what makes reading the stored data a
# few-second operation instead of ~500s: the expensive GRIB fetch/decode
# already happened once, in the scheduled ingestion job.
# ---------------------------------------------------------------------------


def extract_threshold_point_from_grid(ds: xr.Dataset, lat: float, lon: float) -> dict:
    """Nearest-neighbor point extraction from a threshold grid (as
    produced by fetch_threshold_grid() and stored via ingestion), in the
    same raw_by_threshold shape (mm -> {end_step_hour: value}) that
    _fetch_combined() produces for a live fetch -- so both feed the same
    _build_threshold_result() helper."""
    lon_query = lon % 360 if float(ds.longitude.max()) > 180 else lon
    point = ds.sel(latitude=lat, longitude=lon_query, method="nearest")

    raw_by_threshold: dict[int, dict[int, float]] = {}
    for threshold_mm in AVAILABLE_THRESHOLDS_MM:
        var_name = _param_for(threshold_mm)
        if var_name not in ds.data_vars:
            raise KeyError(f"{var_name} not found in stored grid (has: {list(ds.data_vars)})")
        raw_by_threshold[threshold_mm] = _extract_point_series(ds, var_name, point)

    return raw_by_threshold


def read_threshold_result_from_store(ds: xr.Dataset, lat: float, lon: float, max_lead_days: int) -> dict:
    """Builds the same result shape as fetch_forecast_table(), but reads
    from an already-open stored grid Dataset (e.g. opened from the
    GitHub-hosted Zarr store) instead of fetching from ECMWF live."""
    run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)
    lon_query = lon % 360 if float(ds.longitude.max()) > 180 else lon
    point = ds.sel(latitude=lat, longitude=lon_query, method="nearest")
    grid_lat = float(point.latitude.values)
    grid_lon = float(point.longitude.values)
    if grid_lon > 180:
        grid_lon -= 360

    raw_by_threshold = extract_threshold_point_from_grid(ds, lat, lon)
    end_steps = [int(h) for h in (pd.to_timedelta(ds.step.values) / pd.Timedelta(hours=1)).astype(int)]

    return _build_threshold_result(
        run_time=run_time,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        raw_by_threshold=raw_by_threshold,
        end_steps=end_steps,
        max_lead_days=max_lead_days,
        model=ds.attrs.get("model", DEFAULT_MODEL),
        downloaded_bytes=None,  # not meaningful here -- the heavy download already happened during ingestion
        fetch_mode="store",
        capped_to_day6=False,  # ingestion always requests the full 15-day range for IFS
    )


def read_percentile_raw_from_store(ds: xr.Dataset, lat: float, lon: float, max_lead_hours: int) -> dict:
    """Builds the same raw-bins shape as fetch_percentile_rainfall()
    (run_time/grid_lat/grid_lon/bins/bin_hours/downloaded_bytes), but reads
    from an already-open stored grid Dataset instead of fetching from
    ECMWF live. Feed the result straight into aggregate_percentile_bins()
    exactly as the live-fetch path does -- that function doesn't care
    where the raw bins came from."""
    run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)
    bin_hours = ds.attrs.get("bin_hours", PERCENTILE_BIN_HOURS)

    lon_query = lon % 360 if float(ds.longitude.max()) > 180 else lon
    point = ds.sel(latitude=lat, longitude=lon_query, method="nearest")

    var_name = list(ds.data_vars)[0]  # 'precip_3h_mm'
    step_hours = (pd.to_timedelta(ds.step.values) / pd.Timedelta(hours=1)).astype(int)
    order = np.argsort(step_hours)

    max_bins = max_lead_hours // bin_hours
    bins = []
    for idx in order[:max_bins]:
        idx = int(idx)
        end_h = int(step_hours[idx])
        start_h = end_h - bin_hours
        # mean(dim="number") would collapse members -- .values on the raw
        # per-member slice keeps every member, matching what
        # aggregate_percentile_bins() expects to compute percentiles over.
        member_values_mm = point[var_name].isel(step=idx).values.tolist()
        bins.append({
            "start_utc": run_time + timedelta(hours=start_h),
            "end_utc": run_time + timedelta(hours=end_h),
            "member_values_mm": member_values_mm,
        })

    return {
        "run_time": run_time,
        "grid_lat": float(point.latitude.values),
        "grid_lon": float(point.longitude.values) if float(point.longitude.values) <= 180 else float(point.longitude.values) - 360,
        "bins": bins,
        "bin_hours": bin_hours,
        "downloaded_bytes": None,  # heavy download already happened during ingestion
    }
