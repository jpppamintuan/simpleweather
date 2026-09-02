"""
Reads pre-fetched ECMWF grids from this repo's `data` branch (published by
the scheduled ingestion job -- see ingest.py and
.github/workflows/ingest.yml) instead of the app doing its own live ECMWF
fetch. This is what turns a ~500s page load into a few-second one: the
expensive GRIB download/decode already happened once, on a schedule,
independent of anyone actually visiting the page.

Every function here is designed to fail soft: any problem (network error,
missing file, data too old) returns None rather than raising, so the
caller in app.py can fall back to the existing live-fetch path exactly as
if this module didn't exist. Reading pre-fetched data is a fast path, not
the only path.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import fsspec
import requests
import xarray as xr

# EDIT THIS before deploying -- "owner/repo", e.g. "yourusername/simpleweather".
# Used to build the raw.githubusercontent.com URLs below.
GITHUB_REPO = "YOUR_GITHUB_USERNAME/simpleweather"

_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/data"
_MANIFEST_URL = f"{_BASE_URL}/manifest.json"

# How old a dataset's run_time can be before treating it as stale and
# falling back to a live fetch. ECMWF publishes ~every 12h; this allows a
# generous buffer for one missed/delayed ingestion cycle before giving up
# on the stored data rather than a tight window that flips to live-fetch
# on every minor scheduling hiccup.
MAX_AGE_HOURS = 15


def _fetch_manifest(timeout_seconds: float = 5.0) -> dict | None:
    """Small JSON fetch -- checked before opening any Zarr store, so a
    missing/failed ingestion run is detected in one cheap request instead
    of discovering it partway through opening a (possibly nonexistent)
    store."""
    try:
        resp = requests.get(_MANIFEST_URL, timeout=timeout_seconds)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def check_dataset_freshness(dataset_name: str) -> tuple[bool, dict | None]:
    """Returns (is_fresh, manifest). dataset_name is 'threshold' or
    'percentile', matching the keys ingest.py writes into run_times.
    is_fresh is False for any of: manifest unreachable, that dataset
    failed its last ingestion run, or its run_time is older than
    MAX_AGE_HOURS."""
    manifest = _fetch_manifest()
    if manifest is None:
        return False, None

    if dataset_name in manifest.get("failures", []):
        return False, manifest

    run_time_str = manifest.get("run_times", {}).get(dataset_name)
    if not run_time_str:
        return False, manifest

    try:
        run_time = datetime.fromisoformat(run_time_str)
    except ValueError:
        return False, manifest

    age = datetime.now(timezone.utc) - run_time
    is_fresh = age < timedelta(hours=MAX_AGE_HOURS)
    return is_fresh, manifest


def _open_remote_zarr(relative_path: str) -> xr.Dataset:
    """Opens a Zarr store from the data branch over HTTP. chunks=None
    deliberately disables xarray's default dask-backed lazy loading --
    the cropped Philippines-bbox grids here are small (a few MB), so
    eagerly loading the whole small grid in one go is simpler than lazy
    per-chunk fetching and doesn't require the 'dask' package. Genuine
    partial/lazy remote reads only start to matter at the much larger
    grid sizes Phase 2's point-query service is meant for."""
    url = f"{_BASE_URL}/{relative_path}"
    mapper = fsspec.get_mapper(url)
    ds = xr.open_zarr(mapper, consolidated=True, chunks=None)
    return ds.load()


def load_threshold_grid(model: str = "ifs") -> xr.Dataset | None:
    """Returns the stored threshold grid for the given model ('ifs' or
    'aifs-ens') if fresh, else None."""
    is_fresh, _ = check_dataset_freshness(f"threshold_{model}")
    if not is_fresh:
        return None
    try:
        return _open_remote_zarr(f"{model}/threshold_latest.zarr")
    except Exception:
        return None


def load_percentile_grid() -> xr.Dataset | None:
    """Returns the stored percentile grid if fresh, else None."""
    is_fresh, _ = check_dataset_freshness("percentile")
    if not is_fresh:
        return None
    try:
        return _open_remote_zarr("ifs/percentile_latest.zarr")
    except Exception:
        return None
