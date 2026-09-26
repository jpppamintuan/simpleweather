"""
Reads pre-fetched ECMWF grids from this repo's `data` branch (published by
the scheduled ingestion job -- see ingest.py and
.github/workflows/ingest.yml) instead of the app doing its own live ECMWF
fetch. This is what turns a ~500s page load into a few-second one: the
expensive GRIB download/decode already happened once, on a schedule,
independent of anyone actually visiting the page.

Every function here reads pre-computed data, never fetches live from
ECMWF. Failure handling differs deliberately by case: legitimate
unavailability (nothing fresh yet) returns None so callers can fall back
appropriately, but a genuine read/parse error is left to raise rather
than being silently swallowed -- see the docstrings on
load_threshold_grid() and load_percentile_grid() below for why that
distinction matters.
"""

from __future__ import annotations

import io
import time
from datetime import datetime, timedelta, timezone

import requests
import xarray as xr

# EDIT THIS before deploying -- "owner/repo", e.g. "yourusername/simpleweather".
# Used to build the raw.githubusercontent.com and api.github.com URLs below.
GITHUB_REPO = "jpppamintuan/simpleweather"

_DATA_BRANCH = "data"
_COMMIT_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{_DATA_BRANCH}"

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

# Cache for the resolved commit SHA (see _base_url() below for why this
# exists at all). Short TTL -- long enough to avoid hammering GitHub's API
# on every rerun, short enough that a fresh ingestion commit becomes
# visible quickly rather than being pinned to a minutes-old SHA.
_SHA_CACHE_TTL_SECONDS = 60
_sha_cache: dict = {"ts": 0.0, "sha": None}


def _get_latest_data_branch_sha(timeout_seconds: float = 8.0) -> str | None:
    """Resolves the `data` branch's current commit SHA via GitHub's API.
    This exists to work around a real problem: raw.githubusercontent.com
    aggressively caches responses for branch-name-referenced URLs
    (.../data/path), and that cache does NOT reliably invalidate just
    because the branch got force-pushed to a new commit -- ingest.yml's
    force_orphan publish replaces the branch's history entirely on every
    run, but the CDN can keep serving a response (including a 404) cached
    from BEFORE that push, for an unpredictable amount of time. This was
    silently causing exactly the two symptoms this whole debugging thread
    was chasing: a stale cached manifest.json (making generated_at look
    older than it really was) and a stale cached 404 for .zmetadata (a
    file confirmed to genuinely exist in the branch via GitHub's own web
    UI, which does not go through this CDN).

    A commit-SHA-pinned URL (.../<full-sha>/path) is a distinct URL for
    every new commit, so it can never be served a stale response left
    over from a previous one -- that's the actual fix, not just a
    mitigation. Returns None on failure (network error, rate limit) so
    callers can fall back to the branch name -- degraded back into the
    caching issue, but not broken outright."""
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
    """Raw-content base URL for the current data, pinned to an exact
    commit SHA when resolvable (see _get_latest_data_branch_sha() for
    why), falling back to the mutable branch name only if GitHub's API
    call itself fails -- keeps the app working (just re-exposed to the
    caching issue) rather than breaking outright on a transient API
    hiccup or rate limit."""
    sha = _get_latest_data_branch_sha()
    ref = sha if sha else _DATA_BRANCH
    return f"https://raw.githubusercontent.com/{GITHUB_REPO}/{ref}"


def _fetch_manifest(timeout_seconds: float = 8.0) -> dict | None:
    """Small JSON fetch -- checked before opening any data file, so a
    missing/failed ingestion run is detected in one cheap request instead
    of discovering it partway through opening a (possibly nonexistent)
    store. Cached briefly (see _manifest_cache above) and retried once on
    failure before giving up."""
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


def _open_remote_netcdf(relative_path: str, timeout_seconds: float = 30.0) -> xr.Dataset:
    """Fetches a NetCDF file via a single, plain HTTP GET -- the same
    simple, proven pattern _fetch_manifest() has always used successfully
    -- and opens it from the downloaded bytes. No remote-filesystem
    abstraction, no multi-file layout, no consolidated metadata: just one
    file, one request. Replaces an earlier Zarr-based approach that ran
    into a persistent, never-fully-diagnosed KeyError('.zmetadata') --
    see ingest.py's module docstring for the full story."""
    url = f"{_base_url()}/{relative_path}"
    resp = requests.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    return xr.open_dataset(io.BytesIO(resp.content), engine="h5netcdf").load()


def load_threshold_grid(model: str = "ifs") -> xr.Dataset | None:
    """Returns the stored threshold grid for the given model ('ifs' or
    'aifs-ens') if fresh, else None (the normal, expected case when
    there's nothing fresh yet -- app.py's caller falls back to a live
    fetch for this, no error involved).

    Does NOT catch exceptions from _open_remote_netcdf() itself -- unlike
    the "not fresh yet" case above, a real read/parse failure is a bug,
    and should surface loudly via app.py's existing error display rather
    than silently degrading into another "just live fetch" case. That
    silent swallowing here (an earlier version of this function) is
    exactly what hid a persistent read-side bug for weeks: every store
    read was failing, but it just looked like "still slow, guess the
    store isn't fresh yet" instead of a visible, diagnosable error."""
    is_fresh, _ = check_dataset_freshness(f"threshold_{model}")
    if not is_fresh:
        return None
    return _open_remote_netcdf(f"{model}/threshold_latest.nc")


def load_percentile_grid() -> xr.Dataset:
    """Returns the stored percentile grid. Deliberately does NOT gate on
    freshness the way load_threshold_grid() does -- there's no live
    fetch fallback for percentile anymore (the raw-member fetch is ~1.7GB
    and takes 10+ minutes, completely inappropriate as an on-demand
    fallback for a user request), so showing whatever ingestion most
    recently produced is always better than showing nothing. Freshness is
    entirely the scheduled ingestion job's responsibility now (see
    ingest.yml's schedule -- it already chases every run, 00/06/12/18Z).

    Raises on failure rather than swallowing to None -- kept deliberately
    loud (unlike load_threshold_grid()) so a real read-side problem shows
    up as a visible traceback via the app's existing error-details
    expander, instead of silently degrading into an opaque "not
    available" message with no diagnostic value."""
    return _open_remote_netcdf("ifs/percentile_latest.nc")
