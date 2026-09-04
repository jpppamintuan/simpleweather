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
import time
from datetime import datetime, timedelta, timezone

import fsspec
import requests
import xarray as xr

# EDIT THIS before deploying -- "owner/repo", e.g. "yourusername/simpleweather".
# Used to build the raw.githubusercontent.com URLs below.
GITHUB_REPO = "YOUR_GITHUB_USERNAME/simpleweather"

_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/data"
_MANIFEST_URL = f"{_BASE_URL}/manifest.json"

# How old the manifest's generated_at can be before treating the store as
# stale and falling back to a live fetch. This is checking "was the
# check-then-fetch job here recently" (see ingest.yml's schedule -- every
# 15 min in the dense windows, hourly outside them), NOT how old the
# underlying model run is -- see the long comment on check_dataset_freshness()
# below for why those are different. 2h is a generous buffer over even the
# hourly sparse-check cadence, covering a couple of missed/delayed cycles
# before genuinely falling back.
MAX_MANIFEST_AGE_HOURS = 2

# Module-level cache for the manifest fetch, shared across every dataset
# freshness check within the same process. Previously each check
# (threshold-IFS, threshold-AIFS, percentile) called _fetch_manifest()
# independently -- up to 3 separate HTTP requests for the SAME file per
# page load, meaning one transient network hiccup could make everything
# look stale at once. One shared, short-lived cache fixes that.
_MANIFEST_CACHE_TTL_SECONDS = 30
_manifest_cache: dict = {"ts": 0.0, "data": None}


def _fetch_manifest(timeout_seconds: float = 8.0) -> dict | None:
    """Small JSON fetch -- checked before opening any Zarr store, so a
    missing/failed ingestion run is detected in one cheap request instead
    of discovering it partway through opening a (possibly nonexistent)
    store. Cached briefly (see _manifest_cache above) and retried once on
    failure before giving up."""
    now = time.time()
    if _manifest_cache["data"] is not None and (now - _manifest_cache["ts"] < _MANIFEST_CACHE_TTL_SECONDS):
        return _manifest_cache["data"]

    for attempt in range(2):
        try:
            resp = requests.get(_MANIFEST_URL, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            _manifest_cache["ts"] = now
            _manifest_cache["data"] = data
            return data
        except Exception:
            if attempt == 0:
                continue
            return None


def check_dataset_freshness(dataset_name: str) -> tuple[bool, dict | None]:
    """Returns (is_fresh, manifest). dataset_name is 'threshold_ifs',
    'threshold_aifs-ens', or 'percentile', matching the keys ingest.py
    writes into run_times.

    is_fresh checks how recently ingestion itself last ran/checked
    (manifest["generated_at"]), NOT how old the underlying model run is
    (manifest["run_times"][dataset_name]). Those are very different
    things: IFS's own run_time can legitimately be ~20 hours old right
    before the next run disseminates (00Z -> ~20:01 UTC for the 12Z run,
    accounting for ECMWF's own ~8h publish lag) without that being stale
    data -- it's just the correct current answer. What actually indicates
    staleness is whether the check-then-fetch job has recently confirmed
    that answer is still current, which is what MAX_MANIFEST_AGE_HOURS
    checks below."""
    manifest = _fetch_manifest()
    if manifest is None:
        return False, None

    if dataset_name in manifest.get("failures", []):
        return False, manifest

    if dataset_name not in manifest.get("run_times", {}):
        return False, manifest

    generated_at_str = manifest.get("generated_at")
    if not generated_at_str:
        return False, manifest

    try:
        generated_at = datetime.fromisoformat(generated_at_str)
    except ValueError:
        return False, manifest

    age = datetime.now(timezone.utc) - generated_at
    is_fresh = age < timedelta(hours=MAX_MANIFEST_AGE_HOURS)
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
