"""Read pre-fetched ECMWF grids from the Simpleweather GitHub data branch.

Philippine data retains its current freshness and live-fetch fallback behavior.
Optional Taiwan data is stored at separate paths and is never confused with
Philippine grid points. Only ingestion fetches the heavy ECMWF GRIB products.
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timedelta, timezone

import requests
import xarray as xr

from taiwan_support import taiwan_enabled

GITHUB_REPO = "jpppamintuan/simpleweather"
_DATA_BRANCH = "data"
_COMMIT_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{_DATA_BRANCH}"
MAX_MANIFEST_AGE_HOURS = 2
_MANIFEST_CACHE_TTL_SECONDS = 30
_manifest_cache: dict = {"ts": 0.0, "data": None}
_SHA_CACHE_TTL_SECONDS = 60
_sha_cache: dict = {"ts": 0.0, "sha": None}


def _get_latest_data_branch_sha(timeout_seconds: float = 8.0) -> str | None:
    """Use an immutable SHA to avoid stale branch-name CDN files and 404s."""
    now = time.time()
    if _sha_cache["sha"] is not None and (now - _sha_cache["ts"] < _SHA_CACHE_TTL_SECONDS):
        return _sha_cache["sha"]
    try:
        resp = requests.get(
            _COMMIT_API_URL,
            timeout=timeout_seconds,
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        sha = resp.json()["sha"]
        _sha_cache["ts"] = now
        _sha_cache["sha"] = sha
        return sha
    except Exception:
        return None


def _base_url() -> str:
    sha = _get_latest_data_branch_sha()
    ref = sha if sha else _DATA_BRANCH
    return f"https://raw.githubusercontent.com/{GITHUB_REPO}/{ref}"


def _fetch_manifest(timeout_seconds: float = 8.0) -> dict | None:
    """Fetch one small manifest with short caching and one retry."""
    now = time.time()
    if _manifest_cache["data"] is not None and (now - _manifest_cache["ts"] < _MANIFEST_CACHE_TTL_SECONDS):
        return _manifest_cache["data"]
    manifest_url = f"{_base_url()}/manifest.json"
    for attempt in range(2):
        try:
            resp = requests.get(manifest_url, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            _manifest_cache["ts"] = now
            _manifest_cache["data"] = data
            return data
        except Exception:
            if attempt == 0:
                continue
            return None


def get_manifest() -> dict | None:
    return _fetch_manifest()


def check_dataset_freshness(dataset_name: str) -> tuple[bool, dict | None]:
    """Confirm a recent ingestion check, not merely a recent model run."""
    manifest = _fetch_manifest()
    if manifest is None:
        return False, None
    if (dataset_name in manifest.get("failures", [])
            or dataset_name in manifest.get("taiwan_failures", [])):
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
    return age < timedelta(hours=MAX_MANIFEST_AGE_HOURS), manifest


def _open_remote_netcdf(relative_path: str, timeout_seconds: float = 60.0,
                        lazy: bool = False) -> xr.Dataset:
    """HTTP GET for one compressed NetCDF; lazily open large ensemble files."""
    url = f"{_base_url()}/{relative_path}"
    resp = requests.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    ds = xr.open_dataset(io.BytesIO(resp.content), engine="h5netcdf")
    return ds if lazy else ds.load()


def _taiwan_file_is_published(name: str) -> bool:
    manifest = _fetch_manifest()
    return bool(manifest and f"taiwan_{name}" in manifest.get("run_times", {}))


def load_threshold_grid(region: str = "ph") -> xr.Dataset | None:
    """Stored threshold grid, or None when a live threshold fallback is needed."""
    if region == "taiwan":
        if not taiwan_enabled() or not _taiwan_file_is_published("threshold"):
            return None
        # Consistent with PH behavior: do not silently show stale Taiwan
        # thresholds if the ingest checks have stopped running.
        fresh, _ = check_dataset_freshness("taiwan_threshold")
        if not fresh:
            return None
        return _open_remote_netcdf("taiwan/threshold_latest.nc")
    if region != "ph":
        raise ValueError(f"Unknown region {region!r}")
    fresh, _ = check_dataset_freshness("threshold")
    if not fresh:
        return None
    return _open_remote_netcdf("ifs/threshold_latest.nc")


def load_percentile_grid(region: str = "ph") -> xr.Dataset:
    """Stored per-member rainfall. Like PH, stale data beats no data."""
    if region == "taiwan":
        if not taiwan_enabled():
            raise ValueError("Taiwan coverage is disabled")
        if not _taiwan_file_is_published("percentile"):
            raise FileNotFoundError("Taiwan rainfall amounts have not been ingested yet")
        return _open_remote_netcdf("taiwan/percentile_latest.nc", lazy=True)
    if region != "ph":
        raise ValueError(f"Unknown region {region!r}")
    return _open_remote_netcdf("ifs/percentile_latest.nc", lazy=True)


def load_temperature_grid(region: str = "ph") -> xr.Dataset:
    """Stored per-member temperature/heat index."""
    if region == "taiwan":
        if not taiwan_enabled():
            raise ValueError("Taiwan coverage is disabled")
        if not _taiwan_file_is_published("temperature"):
            raise FileNotFoundError("Taiwan temperature has not been ingested yet")
        return _open_remote_netcdf("taiwan/temperature_latest.nc", lazy=True)
    if region != "ph":
        raise ValueError(f"Unknown region {region!r}")
    return _open_remote_netcdf("ifs/temperature_latest.nc", lazy=True)
