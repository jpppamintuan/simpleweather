"""
ECMWF Open Data fetcher for 24h rainfall exceedance probabilities.

Uses the official `ecmwf-opendata` package to pull the ENS "probability"
stream (enfo / type=ep), which contains precomputed probability-of-exceedance
fields for 24h accumulated total precipitation: tpg1, tpg5, tpg10, tpg20,
tpg25, tpg50, tpg100 (mm). These are fixed thresholds baked into the model
output, computed from a 50-member ensemble.

IFS ENS is published at 0.25 deg and (from 20 Oct 2026) 0.1 deg, the
latter ~2 hours after the former, as separate -sfc/-pl files. The
ingestion-side grid fetchers below take a `resol` ("0p25" / "0p1") and a
pinned run time; the live-fetch path (the app's last-resort fallback)
stays at 0.25 deg. The AIFS ensemble is no longer used.

All 5 thresholds used by the app (1/5/20/50/100 mm) are fetched in a single
combined request (one file, one network round trip) rather than
one request per threshold -- this is the main optimization over the earlier
version. If the combined request fails for any reason, the client falls
back to fetching thresholds one at a time so the app still works, just
slower.
"""

from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr
from ecmwf.opendata import Client

try:  # run diagnostics (see diag.py); the app doesn't need it
    from diag import log as _diag
except ImportError:
    def _diag(tag: str) -> None:
        pass

AVAILABLE_THRESHOLDS_MM = [1, 5, 20, 50, 100]

THRESHOLD_COLORS = {
    1: "#00BFFF",
    5: "#FFFF00",
    20: "#FFA500",
    50: "#FF0000",
    100: "#800080",
}

# Grid spacings IFS ENS open data is published at. 0.25 deg arrives first;
# 0.1 deg follows ~2h later as separate -sfc/-pl files (everything this app
# uses is surface-level, so only the -sfc file is ever requested).
RESOLUTIONS = ["0p25", "0p1"]  # ascending order of preference
RESOLUTION_RANK = {r: i for i, r in enumerate(RESOLUTIONS)}
RESOLUTION_LABELS = {"0p25": "0.25\u00b0", "0p1": "0.1\u00b0"}
DEFAULT_RESOLUTION = "0p25"

# 0.1 deg fields are ~6x bigger, so the grid fetchers download them in small
# batches of steps -- cropping each batch to the bbox and deleting its GRIB
# file before the next -- instead of one multi-GB file. None = one request.
STEPS_PER_CHUNK = {"0p25": None, "0p1": 6}

# ECMWF's portal serves roughly 3 MB/s per connection (observed on GitHub
# runners), so a ~600 MB batch takes ~3-4 min. Batches are independent, so
# at 0.1 deg several are downloaded at once. Only the DOWNLOAD runs in
# threads -- cfgrib/eccodes decoding stays on the main thread. Keep this
# modest: the portal caps simultaneous connections for everyone.
DOWNLOAD_WORKERS = {"0p25": 1, "0p1": 4}

# Diagnose mode (see ingest.py): only the first N batches of each fetch.
# Deliberately inert unless DIAGNOSE_ONLY is also set, so a stray value in
# the environment can never truncate a real ingest.
DEBUG_MAX_BATCHES = int(os.environ.get("DIAGNOSE_BATCHES") or 0) if os.environ.get("DIAGNOSE_ONLY") else 0


def _make_client(resol: str = DEFAULT_RESOLUTION) -> Client:
    return Client(source="ecmwf", resol=resol)


def _request_extras(resol: str) -> dict:
    """0.1 deg files are split into -sfc/-pl, so the level type must be
    named; 0.25 deg files are combined and need no such keyword."""
    return {"levtype": "sfc"} if resol == "0p1" else {}


def _run_kwargs(run_dt) -> dict:
    """Pins a request to one specific run (instead of 'latest'), so the run
    that was checked is the run that gets downloaded and labeled."""
    return {"date": run_dt.strftime("%Y%m%d"), "time": run_dt.hour}

# Generous bounding box around the Philippines -- covers all preset
# locations with plenty of room to spare, sized for arbitrary-point
# selection across the country. Shared single source of truth: ingest.py
# crops every stored grid to exactly this box, and app.py uses the same
# bounds to validate/limit any custom coordinate input, since a point
# outside this box has no stored data to look up at all.
PH_BBOX = {"lat_min": 4.0, "lat_max": 21.5, "lon_min": 115.0, "lon_max": 127.5}

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

    max_step_hours: 06Z and 18Z runs are capped at step 144 (day 6), while
    00Z/12Z runs go to the full step 360 (day 15). Requesting beyond a
    run's actual cap fails silently, which shows up later as "no aligned
    windows"."""
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


def _fetch_combined(lat, lon, step_labels, tmpdir, progress_callback: ProgressFn):
    """Single request for all thresholds at once. Raises on any failure so the
    caller can fall back to per-threshold requests."""
    client = _make_client(DEFAULT_RESOLUTION)
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


def _fetch_separate(lat, lon, step_labels, tmpdir, progress_callback: ProgressFn):
    """Fallback: one request per threshold (slower, but more forgiving)."""
    client = _make_client(DEFAULT_RESOLUTION)
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
    """Approximate when this run's threshold-exceedance product becomes
    available, per ECMWF's published dissemination schedule.

    NOTE on accuracy: these times are estimates, refined from
    user-sourced research against ECMWF's Confluence documentation, but
    NOT independently confirmed against an authoritative source for this
    specific product (ep/tpg). ECMWF distinguishes "forecast" dissemination
    times from "derived products" times (~20min later) for the day-15
    product; which bucket ep/tpg falls into wasn't confirmed, so the
    earlier ("forecast") time is used here as the working estimate. This
    only affects the "Last updated" / "Next update" display text -- actual
    ingestion timing doesn't depend on this being exactly right, since the
    scheduled checks (see ingest.yml) run every 15 minutes across windows
    generous enough to cover either possibility.
    """
    run_hour = run_time.hour
    if run_hour == 0:
        return run_time.replace(hour=7, minute=40, second=0, microsecond=0)
    elif run_hour == 12:
        return run_time.replace(hour=19, minute=40, second=0, microsecond=0)
    elif run_hour == 6:
        # 06Z is capped at day 6 -- no separate day-15 product applies.
        return run_time.replace(hour=13, minute=20, second=0, microsecond=0)
    elif run_hour == 18:
        # 18Z's day-6 product disseminates after midnight, the following day.
        next_day = run_time + timedelta(days=1)
        return next_day.replace(hour=1, minute=20, second=0, microsecond=0)
    else:
        return run_time + timedelta(hours=8, minutes=1)


# The 0.1 deg threshold (ep) product lands a measured ~2h21 after the 0.25 deg
# estimate above (portal listing, 00Z run: ep-sfc at 10:01 UTC vs. 07:40).
# Only a fallback -- ingestion records the real publish time (the file's
# Last-Modified header) whenever it can, and that is shown in preference.
RELEASE_LAG_0P1 = timedelta(hours=2, minutes=21)


def _estimated_release_time(run_time, resolution: str):
    base = _dissemination_available_time(run_time)
    return base + RELEASE_LAG_0P1 if resolution == "0p1" else base


def _next_update(run_time, resolution: str):
    """(when, description) of the next change to expect in the threshold
    data. Showing 0.25 deg: the 0.1 deg grid of the SAME run is next.
    Showing 0.1 deg: the next run's 0.25 deg release (6h later -- every
    run, 00/06/12/18Z, is chased) is next, with its 0.1 deg upgrade
    following ~2h21 after that."""
    if resolution != "0p1":
        return _estimated_release_time(run_time, "0p1"), f"0.1\u00b0 grid of the {run_time:%H}Z run"
    next_run = run_time + timedelta(hours=6)
    return _dissemination_available_time(next_run), f"{next_run:%H}Z run (0.25\u00b0 first)"


def _cap_for_run_hour(run_hour, requested_max_lead_days: int) -> tuple[int, bool]:
    """00Z/12Z runs carry the full 15-day threshold product; 06Z/18Z runs
    are capped at day 6 (step 144). Returns (lead days actually safe to
    request, whether a cap was applied). An unknown run hour gets the
    conservative 6-day cap."""
    if run_hour is None or run_hour in (6, 18):
        return min(requested_max_lead_days, 6), requested_max_lead_days > 6
    return requested_max_lead_days, False


def _effective_threshold_request(requested_max_lead_days: int) -> tuple[int, bool]:
    """Live-fetch path only (ingestion knows its run hour up front): one
    cheap metadata-only check for the latest 0.25 deg run, then the cap
    above."""
    try:
        latest = check_latest_threshold_run(DEFAULT_RESOLUTION)
        run_hour = getattr(latest, "datetime", latest).hour
    except Exception:
        run_hour = None
    return _cap_for_run_hour(run_hour, requested_max_lead_days)


def fetch_forecast_table(
    lat: float,
    lon: float,
    max_lead_days: int = 15,
    progress_callback: ProgressFn = None,
) -> dict:
    """
    Live fetch (0.25 deg only) of all thresholds for ONE location -- the
    app's last-resort fallback when the ingested store isn't fresh. Returns:

    {
        "run_time": datetime (UTC),
        "grid_lat": float, "grid_lon": float,
        "windows": [{"label": "0-24", "start_utc": dt, "end_utc": dt}, ...],
        "data": {1: {"0-24": 82.0, ...}, 5: {...}, ...},
        "available_since": datetime (UTC),
        "next_expected": datetime (UTC),
        "downloaded_bytes": int | None,
        "fetch_mode": "combined" | "separate",
        "resolution": "0p25",
    }

    progress_callback, if given, is called as progress_callback(fraction, message)
    (fraction in [0, 1]).
    """
    effective_max_lead_days, capped_to_day6 = _effective_threshold_request(max_lead_days)
    max_step_hours = 144 if capped_to_day6 else 360
    step_labels = _request_step_labels(effective_max_lead_days, increment_hours=12, max_step_hours=max_step_hours)

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

    end_steps = [int(label.split("-")[1]) for label in step_labels]
    result = _build_threshold_result(
        run_time=run_time,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        raw_by_threshold=raw_by_threshold,
        end_steps=end_steps,
        max_lead_days=max_lead_days,
        resolution=DEFAULT_RESOLUTION,
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
    resolution: str,
    downloaded_bytes,
    fetch_mode: str,
    capped_to_day6: bool,
    published_at=None,
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
        # boundary (only 06/18 UTC ones). Rather than show nothing, align
        # to the run's own hour instead of insisting on exactly midnight UTC.
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

    # Prefer the measured publish time (file Last-Modified, recorded at
    # ingestion); fall back to the schedule-based estimate for this resolution.
    available_since = published_at or _estimated_release_time(run_time, resolution)
    next_expected, next_description = _next_update(run_time, resolution)

    return {
        "run_time": run_time,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "windows": windows,
        "data": data,
        "available_since": available_since,
        "available_is_measured": published_at is not None,
        "next_expected": next_expected,
        "next_description": next_description,
        "downloaded_bytes": downloaded_bytes,
        "fetch_mode": fetch_mode,
        "resolution": resolution,
        "aligned_to_utc_midnight": aligned_to_utc_midnight,
        "capped_to_day6": capped_to_day6,
    }


DEFAULT_PERCENTILES = [10, 25, 50, 75, 90]
PERCENTILE_BIN_HOURS = 3  # ECMWF's native accumulation granularity for this product, through step 144
AVAILABLE_PERIOD_HOURS = [3, 6, 12, 24]


def check_latest_threshold_run(resol: str = DEFAULT_RESOLUTION):
    """Cheap metadata-only check for the most recent available threshold
    (ep) run at the given resolution -- does NOT download any data (per
    ecmwf-opendata's documented Client.latest() behavior). Raises or
    returns None if that product isn't published at that resolution."""
    return _make_client(resol).latest(
        stream="enfo", type="ep", param=_param_for(AVAILABLE_THRESHOLDS_MM[0]),
        **_request_extras(resol),
    )


def check_latest_pf_run(resol: str = DEFAULT_RESOLUTION):
    """Same, for the raw-member (pf) product that precipitation and
    temperature percentiles are computed from."""
    return _make_client(resol).latest(
        stream="enfo", type="pf", param="tp", **_request_extras(resol),
    )


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
        "resolution": raw_result.get("resolution"),
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
# point dicts, ready for ingestion to write to disk (currently NetCDF;
# see ingest.py -- this module stays storage-format-agnostic on purpose,
# it just returns plain xarray Datasets either way).
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


_CORE_COORDS = {"number", "step", "latitude", "longitude", "time"}


def _keep_core_coords(ds: xr.Dataset) -> xr.Dataset:
    """Drops the extra scalar coordinates cfgrib attaches (heightAboveGround,
    surface, valid_time, ...). None are used downstream (times are derived
    from `time` + `step`), and they only add merge/serialization noise."""
    extra = [c for c in ds.coords if c not in _CORE_COORDS]
    return ds.drop_vars(extra) if extra else ds


def _verify_run_time(ds: xr.Dataset, expected_run_dt):
    """Ingestion pins each request to a specific run; confirms what came
    back really is that run, rather than trusting the pin silently."""
    run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)
    if run_time.replace(tzinfo=None) != expected_run_dt.replace(tzinfo=None):
        raise RuntimeError(
            f"Downloaded data is for run {run_time.isoformat()}, "
            f"expected {expected_run_dt.isoformat()}"
        )
    return run_time


_RESULT_LOGGED = False


def _download_batch(resol: str, request: dict, extras: dict, chunk: list, target: Path):
    """One batch download; returns (bytes transferred, source URLs). Builds
    its own Client so nothing is shared between threads."""
    result = _make_client(resol).retrieve(step=chunk, target=str(target), **request, **extras)
    global _RESULT_LOGGED
    if not _RESULT_LOGGED:  # once: what does this client version actually expose?
        _RESULT_LOGGED = True
        public = {k: type(getattr(result, k, None)).__name__ for k in dir(result) if not k.startswith("_")}
        print(f"[diag] retrieve() returned {type(result).__name__} with attributes {public}; "
              f"size={getattr(result, 'size', None)!r}", flush=True)
    return (getattr(result, "size", 0) or 0), [str(u) for u in (getattr(result, "urls", None) or [])]


def _published_at(urls) -> str | None:
    """When ECMWF actually published these files: the latest Last-Modified
    header among them (a plain HEAD request each). The GRIB messages only
    carry the forecast BASE time, not a release time, so the portal's HTTP
    metadata is the only place the real publish time exists. Best effort --
    returns None if the client exposed no URLs or the server sent no
    Last-Modified, and the app then falls back to schedule estimates."""
    latest = None
    failures = 0
    for url in sorted(set(urls)):
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            header = resp.headers.get("Last-Modified")
            if not header:
                continue
            modified = parsedate_to_datetime(header)
            if latest is None or modified > latest:
                latest = modified
        except Exception:
            failures += 1
            if failures >= 3:
                break
    return latest.astimezone(UTC).isoformat() if latest else None


def _retrieve_cropped(
    *, resol: str, run_dt, request: dict, steps: list, bbox: dict, tmpdir: str,
    progress_callback: ProgressFn = None, progress_span=(0.1, 0.8), label: str = "data",
) -> xr.Dataset:
    """Downloads `steps` for `request` at `resol` in batches (see
    STEPS_PER_CHUNK), cropping each batch to `bbox` and deleting its GRIB
    file once decoded, then stitches the batches together along `step`.
    Up to DOWNLOAD_WORKERS batches download concurrently; decoding is
    serial. At 0.25 deg that's one batch, i.e. the same single request as
    before."""
    extras = {**_request_extras(resol), **_run_kwargs(run_dt)}
    per_chunk = STEPS_PER_CHUNK.get(resol) or len(steps)
    chunks = [steps[i:i + per_chunk] for i in range(0, len(steps), per_chunk)]
    if DEBUG_MAX_BATCHES:
        chunks = chunks[:DEBUG_MAX_BATCHES]
    workers = max(1, DOWNLOAD_WORKERS.get(resol, 1))
    lo, hi = progress_span

    pieces = []
    total_bytes = 0
    all_urls: list[str] = []
    for g in range(0, len(chunks), workers):
        group = list(enumerate(chunks))[g:g + workers]
        _notify(progress_callback, lo + (hi - lo) * g / len(chunks),
                f"Fetching {label} at {resol}, batches {group[0][0] + 1}-{group[-1][0] + 1} of {len(chunks)}...")
        targets = {i: Path(tmpdir) / f"{label}_{i}.grib2" for i, _ in group}

        if len(group) == 1:
            i, chunk = group[0]
            downloads = [_download_batch(resol, request, extras, chunk, targets[i])]
        else:
            with ThreadPoolExecutor(max_workers=len(group)) as pool:
                downloads = list(pool.map(
                    lambda ic: _download_batch(resol, request, extras, ic[1], targets[ic[0]]), group))
        total_bytes += sum(size for size, _ in downloads)
        for _, urls in downloads:
            all_urls.extend(urls)
        on_disk_mb = sum(targets[i].stat().st_size for i, _ in group) / 1e6
        _diag(f"{label} batches {group[0][0] + 1}-{group[-1][0] + 1}/{len(chunks)} downloaded "
              f"({on_disk_mb:.0f} MB on disk)")

        for i, _ in group:  # decode serially, in order
            _diag(f"{label} batch {i + 1}/{len(chunks)}: decoding {targets[i].stat().st_size / 1e6:.0f} MB")
            t0 = time.time()
            ds = xr.open_dataset(str(targets[i]), engine="cfgrib", backend_kwargs={"indexpath": ""})
            _diag(f"{label} batch {i + 1}/{len(chunks)}: index built in {time.time() - t0:.0f}s, cropping/loading")
            try:
                piece = _crop_to_bbox(ds, bbox).load()  # resolve into memory before the file goes
            finally:
                ds.close()
            targets[i].unlink(missing_ok=True)
            _diag(f"{label} batch {i + 1}/{len(chunks)}: decoded in {time.time() - t0:.0f}s -> {dict(piece.sizes)}")

            piece = piece.drop_vars("valid_time", errors="ignore")
            if "step" not in piece.dims:  # cfgrib squeezes a 1-step batch to a scalar coordinate
                piece = piece.expand_dims("step")
            pieces.append(piece)

    published = _published_at(all_urls)
    print(f"[{label}] downloaded {total_bytes / 1e9:.2f} GB at {resol} in {len(chunks)} batch(es); "
          f"ECMWF published: {published or 'unknown (no URLs / Last-Modified)'}", flush=True)
    out = pieces[0].sortby("step") if len(pieces) == 1 else \
        xr.concat(pieces, dim="step", coords="minimal", compat="override").sortby("step")
    if published:
        out.attrs["published_at"] = published
    return out


def fetch_threshold_grid(
    bbox: dict,
    run_dt,
    resol: str = DEFAULT_RESOLUTION,
    max_lead_days: int = 15,
    progress_callback: ProgressFn = None,
) -> xr.Dataset:
    """
    Ingestion counterpart to fetch_forecast_table(): fetches the same
    precomputed threshold-exceedance product (the 5 combined thresholds)
    for one specific run (`run_dt`, naive UTC) at `resol`, and returns the
    CROPPED GRID over `bbox` instead of a single nearest-neighbor point.

    bbox: {"lat_min":, "lat_max":, "lon_min":, "lon_max":} in degrees,
    longitude in -180..180 (converted internally if the source grid uses
    0..360).

    Deliberately does NOT do the 24h-window alignment fetch_forecast_table()
    does -- that's a display/query concern. The returned dataset's "step"
    dimension is the raw request steps in hours.
    """
    effective_max_lead_days, capped_to_day6 = _cap_for_run_hour(run_dt.hour, max_lead_days)
    max_step_hours = 144 if capped_to_day6 else 360
    step_labels = _request_step_labels(effective_max_lead_days, increment_hours=12, max_step_hours=max_step_hours)
    params = [_param_for(t) for t in AVAILABLE_THRESHOLDS_MM]

    _notify(progress_callback, 0.05, "Connecting to ECMWF Open Data...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _retrieve_cropped(
            resol=resol, run_dt=run_dt, request=dict(stream="enfo", type="ep", param=params),
            steps=step_labels, bbox=bbox, tmpdir=tmpdir,
            progress_callback=progress_callback, progress_span=(0.1, 0.9), label="threshold",
        )

    # Sanity-check the decode BEFORE trusting it. cfgrib silently skips
    # corrupted GRIB messages rather than raising -- a real, observed
    # failure mode that otherwise looks exactly like success. Checks every
    # expected threshold variable is present and the step count is close
    # to what was requested.
    missing_vars = [p for p in params if p not in ds.data_vars]
    if missing_vars:
        raise RuntimeError(
            f"Threshold grid ({resol}) is missing expected variable(s) {missing_vars} "
            f"(decoded dataset has: {list(ds.data_vars)}) -- likely cfgrib silently "
            f"skipped corrupted GRIB messages. Treating this as a failed fetch."
        )
    actual_step_count = ds.sizes.get("step", 0)
    expected_step_count = len(step_labels)
    if actual_step_count < expected_step_count * 0.9:
        raise RuntimeError(
            f"Threshold grid ({resol}) decoded with only {actual_step_count} of "
            f"{expected_step_count} expected steps -- likely the same corrupted-"
            f"message issue (or an incomplete cycle). Treating this as a failed fetch."
        )

    run_time = _verify_run_time(ds, run_dt)
    ds.attrs["run_time"] = run_time.isoformat()
    ds.attrs["resolution"] = resol
    # int, not bool -- NetCDF/h5netcdf has no boolean attribute type.
    ds.attrs["capped_to_day6"] = int(capped_to_day6)
    _notify(progress_callback, 1.0, "Done")
    return ds


def _relative_humidity_from_temp_dewpoint(temp_c, dewpoint_c):
    """Magnus-Tetens approximation (Alduchov & Eskridge 1996 constants).
    ECMWF publishes 2m dewpoint temperature, not 2m relative humidity,
    directly -- confirmed via ecmwf-opendata's own parameter list and
    ECMWF's own documentation -- so RH has to be derived from temperature
    + dewpoint rather than read as its own field."""
    a, b = 17.625, 243.04
    numerator = np.exp((a * dewpoint_c) / (b + dewpoint_c))
    denominator = np.exp((a * temp_c) / (b + temp_c))
    rh = 100.0 * (numerator / denominator)
    # .clip() as a DataArray METHOD, not np.clip(rh, ...) -- calling it on
    # the object itself is unambiguous about preserving DataArray type,
    # unlike passing a DataArray into a bare numpy function (see the
    # np.where() note in _heat_index_celsius for why that distinction
    # matters here).
    return rh.clip(0, 100)


def _heat_index_celsius(temp_c, rh_pct):
    """NWS Rothfusz regression -- the standard, near-universal Heat Index
    formula. PAGASA's own Heat Index Advisory almost certainly uses the
    same underlying formula (their own heat index product, iHeatMAP, is
    itself built from ECMWF data per DOST-PAGASA's own documentation) --
    PAGASA's specific category labels/colors are a UI-layer concern, not
    part of this calculation.

    Operates in Fahrenheit internally (the formula's native units),
    converting at the boundaries. Includes the NWS's standard simple-
    formula fallback for mild conditions and the two correction terms for
    very low/very high humidity, not just the bare regression --
    PAGASA's lowest category ("Caution") starts at 27C (~80F), right at
    the edge of the regression's documented valid range, so getting the
    full algorithm right (not a simplified version) matters here."""
    temp_f = temp_c * 9.0 / 5.0 + 32.0

    # NWS "simple formula" -- used as the final answer when conditions
    # are mild. Whether they count as "mild" is checked further down, by
    # averaging THIS value with temp_f (see the use_simple line below).
    hi_simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh_pct * 0.094)

    hi_full = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh_pct
        - 0.22475541 * temp_f * rh_pct
        - 0.00683783 * temp_f ** 2
        - 0.05481717 * rh_pct ** 2
        + 0.00122874 * temp_f ** 2 * rh_pct
        + 0.00085282 * temp_f * rh_pct ** 2
        - 0.00000199 * temp_f ** 2 * rh_pct ** 2
    )

    low_rh_mask = (rh_pct < 13) & (temp_f >= 80) & (temp_f <= 112)
    low_rh_adj = ((13 - rh_pct) / 4.0) * np.sqrt((17 - np.abs(temp_f - 95)) / 17.0)

    high_rh_mask = (rh_pct > 85) & (temp_f >= 80) & (temp_f <= 87)
    high_rh_adj = ((rh_pct - 85) / 10.0) * ((87 - temp_f) / 5.0)

    # xr.where(), NOT np.where() -- np.where() on xarray DataArrays is a
    # known trap: it can silently return a bare numpy array, stripping
    # all dimension/coordinate metadata, where xr.where() is specifically
    # built to preserve it. temp_c/rh_pct (and everything derived from
    # them here) are DataArrays, not plain arrays, so this matters.
    hi_full_adjusted = (
        hi_full
        - xr.where(low_rh_mask, low_rh_adj, 0.0)
        + xr.where(high_rh_mask, high_rh_adj, 0.0)
    )

    # The documented NWS switching criterion is the AVERAGE of the simple
    # estimate and the actual temperature, not the simple estimate alone
    # (confirmed against NOAA/NWS WPC's own description and an academic
    # paper's precise restatement of the same procedure) -- an earlier
    # version of this compared hi_simple directly against 80, which is
    # not what NWS's own algorithm does.
    use_simple = (hi_simple + temp_f) / 2.0 < 80
    hi_final_f = xr.where(use_simple, hi_simple, hi_full_adjusted)

    return (hi_final_f - 32.0) * 5.0 / 9.0


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str:
    """cfgrib's variable naming for a given GRIB shortName isn't something
    this code can verify against the live API ahead of time -- checks a
    few plausible names (both the raw GRIB shortName and cfgrib's more
    common CF-style rename) rather than hardcoding one and risking a
    silent KeyError in ingestion."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of {candidates} found in dataset (has: {list(ds.data_vars)})")


def fetch_precip_grid(
    bbox: dict,
    run_dt,
    resol: str = DEFAULT_RESOLUTION,
    max_lead_hours: int = 120,
    progress_callback: ProgressFn = None,
) -> xr.Dataset:
    """
    Ingestion counterpart to fetch_percentile_rainfall(): raw tp from all
    50 ENS perturbed members at every native 3-hourly step, for one
    specific run at `resol`, cropped to `bbox`, diffed from cumulative-
    since-start into true per-3-hour-bin amounts (see
    fetch_percentile_rainfall()'s docstring for why diffing consecutive
    steps is required). Variable: precip_3h_mm(number, step, lat, lon).

    Per-member values (not precomputed percentiles) are stored on purpose:
    percentiles for an arbitrary point, and sums into 6/12/24h periods,
    are computed on demand by whatever reads this.
    """
    max_lead_hours = min(max_lead_hours, 120)
    steps = list(range(0, max_lead_hours + 1, PERCENTILE_BIN_HOURS))
    numbers = list(range(1, 51))

    _notify(progress_callback, 0.05, "Connecting to ECMWF Open Data...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_tp = _retrieve_cropped(
            resol=resol, run_dt=run_dt,
            request=dict(stream="enfo", type="pf", param="tp", number=numbers),
            steps=steps, bbox=bbox, tmpdir=tmpdir,
            progress_callback=progress_callback, progress_span=(0.05, 0.9), label="precipitation",
        )

    run_time = _verify_run_time(ds_tp, run_dt)
    tp_mm = ds_tp[_find_var(ds_tp, ["tp"])] * 1000.0
    period_mm = tp_mm.diff(dim="step").clip(min=0)  # clip: guard tiny negative float noise
    ds_out = _keep_core_coords(period_mm.to_dataset(name="precip_3h_mm"))

    if ds_tp.attrs.get("published_at"):
        ds_out.attrs["published_at"] = ds_tp.attrs["published_at"]
    ds_out.attrs["run_time"] = run_time.isoformat()
    ds_out.attrs["resolution"] = resol
    ds_out.attrs["bin_hours"] = PERCENTILE_BIN_HOURS
    print(f"[fetch_precip_grid] dims={dict(ds_out.sizes)}, coords={list(ds_out.coords)}")
    _notify(progress_callback, 1.0, "Done")
    return ds_out


def fetch_temperature_grid(
    bbox: dict,
    run_dt,
    resol: str = DEFAULT_RESOLUTION,
    max_lead_hours: int = 120,
    progress_callback: ProgressFn = None,
) -> xr.Dataset:
    """
    2m temperature and 2m dewpoint from all 50 ENS perturbed members at
    every native 3-hourly step (including step=0, the analysis), for one
    specific run at `resol`, cropped to `bbox`. Both are instantaneous
    values, so no diffing. Relative humidity is derived from temperature +
    dewpoint (ECMWF doesn't publish 2m RH directly), then combined with
    temperature into Heat Index. Variables: temperature_c and
    heat_index_c, each (number, step, lat, lon).

    Stored as its own dataset (not merged with precipitation, as an
    earlier version did): the two views are independent, each file stays
    small enough to publish and to read, and there's no longer a
    dimension-size conflict to work around.
    """
    max_lead_hours = min(max_lead_hours, 120)
    steps = list(range(0, max_lead_hours + 1, PERCENTILE_BIN_HOURS))
    numbers = list(range(1, 51))

    _notify(progress_callback, 0.05, "Connecting to ECMWF Open Data...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_t = _retrieve_cropped(
            resol=resol, run_dt=run_dt,
            request=dict(stream="enfo", type="pf", param=["2t", "2d"], number=numbers),
            steps=steps, bbox=bbox, tmpdir=tmpdir,
            progress_callback=progress_callback, progress_span=(0.05, 0.85), label="temperature",
        )

    run_time = _verify_run_time(ds_t, run_dt)
    temp_c = ds_t[_find_var(ds_t, ["t2m", "2t"])] - 273.15
    dewpoint_c = ds_t[_find_var(ds_t, ["d2m", "2d"])] - 273.15

    print(
        f"[fetch_temperature_grid] raw temp_c range: "
        f"[{float(temp_c.min()):.1f}, {float(temp_c.max()):.1f}], "
        f"dewpoint_c range: [{float(dewpoint_c.min()):.1f}, {float(dewpoint_c.max()):.1f}]"
    )

    # Defensive clip to a physically plausible range BEFORE computing
    # RH/Heat Index: both the Magnus-Tetens approximation and the Rothfusz
    # regression are empirical fits that produce absurd outputs (this was
    # the confirmed cause of ~-20,000C heat index values) if extrapolated
    # on a wildly out-of-range input.
    temp_c = temp_c.clip(-10, 50)
    dewpoint_c = dewpoint_c.clip(-10, 40)

    _notify(progress_callback, 0.9, "Computing relative humidity and heat index...")
    rh_pct = _relative_humidity_from_temp_dewpoint(temp_c, dewpoint_c)
    heat_index_c = _heat_index_celsius(temp_c, rh_pct)

    print(
        f"[fetch_temperature_grid] clipped temp_c range: "
        f"[{float(temp_c.min()):.1f}, {float(temp_c.max()):.1f}], "
        f"rh_pct range: [{float(rh_pct.min()):.1f}, {float(rh_pct.max()):.1f}], "
        f"heat_index_c range: [{float(heat_index_c.min()):.1f}, {float(heat_index_c.max()):.1f}]"
    )

    ds_out = _keep_core_coords(xr.Dataset({"temperature_c": temp_c, "heat_index_c": heat_index_c}))
    if ds_t.attrs.get("published_at"):
        ds_out.attrs["published_at"] = ds_t.attrs["published_at"]
    ds_out.attrs["run_time"] = run_time.isoformat()
    ds_out.attrs["resolution"] = resol
    print(f"[fetch_temperature_grid] dims={dict(ds_out.sizes)}, coords={list(ds_out.coords)}")
    _notify(progress_callback, 1.0, "Done")
    return ds_out


# ---------------------------------------------------------------------------
# Store-read functions -- used by the Streamlit app to build the exact same
# result shapes as the live-fetch functions above, but sourced from an
# already-fetched grid (opened from the GitHub-hosted data file) instead of
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
    GitHub-hosted data file) instead of fetching from ECMWF live."""
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
        resolution=str(ds.attrs.get("resolution", DEFAULT_RESOLUTION)),
        published_at=(datetime.fromisoformat(ds.attrs["published_at"]) if ds.attrs.get("published_at") else None),
        downloaded_bytes=None,  # not meaningful here -- the heavy download already happened during ingestion
        fetch_mode="store",
        capped_to_day6=bool(ds.attrs.get("capped_to_day6", False)),  # stored as int 0/1, see write side
    )


def read_percentile_raw_from_store(ds: xr.Dataset, lat: float, lon: float, max_lead_hours: int) -> dict:
    """Builds the same raw-bins shape as fetch_percentile_rainfall()
    (run_time/grid_lat/grid_lon/bins/bin_hours/downloaded_bytes), but reads
    from an already-open stored grid Dataset instead of fetching from
    ECMWF live. Feed the result straight into aggregate_percentile_bins()
    exactly as the live-fetch path does -- that function doesn't care
    where the raw bins came from."""
    run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)
    bin_hours = int(ds.attrs.get("bin_hours", PERCENTILE_BIN_HOURS))  # NetCDF attrs round-trip as
    # numpy scalar types (e.g. numpy.int64), not plain Python int -- timedelta(hours=...) below
    # rejects that, so cast explicitly here rather than relying on it staying a plain int.

    lon_query = lon % 360 if float(ds.longitude.max()) > 180 else lon
    # .load() on just this point: the store dataset is opened lazily (it's
    # large at 0.1 deg), so this reads only the chunks covering one location.
    point = ds.sel(latitude=lat, longitude=lon_query, method="nearest").load()

    var_name = "precip_3h_mm"
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
        "resolution": str(ds.attrs.get("resolution", DEFAULT_RESOLUTION)),
    }


def read_temperature_percentiles_from_store(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    max_lead_hours: int,
    percentiles: list[int] = None,
) -> dict:
    """Reads per-member temperature and heat index values from the stored
    percentile grid (the temperature_c/heat_index_c variables written by
    fetch_temperature_grid(), computed from 2m temperature + derived
    relative humidity) and computes percentile stats across the 50
    members at each native 3-hourly step directly.

    Deliberately simpler than the rainfall percentile path: no raw-bins-
    plus-separate-aggregate-step split, because there's no adjustable
    time-step slider for temperature the way rainfall has one.
    Temperature doesn't accumulate -- each step's value is already
    meaningful on its own, so there's nothing to sum bins INTO. Includes
    step=0 (the analysis/"right now" reading), unlike the rainfall bins,
    which start at the first full 3-hour period.
    """
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    run_time = pd.Timestamp(ds.time.values).to_pydatetime().replace(tzinfo=UTC)

    lon_query = lon % 360 if float(ds.longitude.max()) > 180 else lon
    point = ds.sel(latitude=lat, longitude=lon_query, method="nearest").load()  # lazy store, see above

    step_hours = (pd.to_timedelta(point.step.values) / pd.Timedelta(hours=1)).astype(int)
    order = np.argsort(step_hours)
    max_steps = max_lead_hours // PERCENTILE_BIN_HOURS + 1  # +1 -- step=0 is included here, unlike precip's bins

    def _stats(vals) -> dict:
        s = {"mean": float(np.mean(vals)), "median": float(np.median(vals))}
        for p in percentiles:
            s[f"p{p}"] = float(np.percentile(vals, p))
        return s

    steps_out = []
    for idx in order[:max_steps]:
        idx = int(idx)
        h = int(step_hours[idx])
        temp_vals = point["temperature_c"].isel(step=idx).values
        hi_vals = point["heat_index_c"].isel(step=idx).values
        steps_out.append({
            "time_utc": run_time + timedelta(hours=h),
            "temperature_stats": _stats(temp_vals),
            "heat_index_stats": _stats(hi_vals),
        })

    return {
        "run_time": run_time,
        "grid_lat": float(point.latitude.values),
        "grid_lon": float(point.longitude.values) if float(point.longitude.values) <= 180 else float(point.longitude.values) - 360,
        "steps": steps_out,
        "percentiles": percentiles,
        "resolution": str(ds.attrs.get("resolution", DEFAULT_RESOLUTION)),
    }
