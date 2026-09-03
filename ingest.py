"""
Scheduled ingestion script for the ECMWF rainfall forecast platform.

Run on a schedule via GitHub Actions (see .github/workflows/ingest.yml).
Fetches the latest ECMWF IFS ENS forecast -- both the precomputed
threshold-exceedance product and the raw ensemble-member product used for
percentile calculations -- crops each to a bounding box covering the
Philippines (generous enough to support arbitrary-point queries later, not
just the app's current 5 fixed locations), and writes the result to a
local ./output directory as Zarr.

This script itself has NO knowledge of where ./output ends up -- that's
the GitHub Actions workflow's job (it checks out the current `data`
branch into ./output before this script runs, then publishes ./output
back as a single fresh commit after, via peaceiris/actions-gh-pages with
force_orphan: true, so the branch never accumulates history -- matches
the "rolling latest only" retention decision). Keeping storage/publishing
out of this script means no cloud credentials are needed here at all.

Check-then-fetch: before doing any expensive download, each dataset's
availability is checked cheaply via Client.latest() (metadata only, no
data transferred) and compared against what's already published. If
nothing's newer, that dataset's existing files are left untouched rather
than re-downloaded -- this is what makes it safe to run this workflow
often (see the schedule in ingest.yml) without wasting bandwidth on
repeatedly re-fetching the same unchanged forecast run.

Zarr specifically (not GRIB2/NetCDF): it's chunked, and both
raw.githubusercontent.com and (later, if adopted) a CDN in front of it
support HTTP range requests -- so a future query layer (Phase 2 -- a
Cloudflare Worker) can fetch just the chunk containing a queried point
instead of downloading the whole file. Chunk sizes below are a first
guess (small lat/lon tiles, full step/member dimensions) sized for
"a query wants every step for one point" -- worth re-tuning once Phase 2
makes the real access pattern visible.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

# Reuses the exact fetch/decode/crop logic in ecmwf_client.py -- this
# script is intentionally a thin wrapper, not a reimplementation, so
# ingestion and the live app's own fetch code can't quietly drift apart.
from ecmwf_client import (
    fetch_threshold_grid,
    fetch_percentile_grid,
    check_latest_threshold_run,
    check_latest_percentile_run,
    MODEL_LABELS,
)

# Generous bounding box around the Philippines -- covers all 5 of the
# app's current fixed locations with plenty of room to spare, sized for
# future arbitrary-point selection across the country rather than just
# today's handful of points. Revisit if the app ever needs to cover areas
# outside this box.
PH_BBOX = {"lat_min": 4.0, "lat_max": 21.5, "lon_min": 115.0, "lon_max": 127.5}

# Both ECMWF's physics-based ENS and their AI-based ensemble are ingested
# for the threshold dataset. Percentile stays IFS-only for now -- that
# view has no model selector in the UI yet, so ingesting AIFS data for it
# would have nowhere to be shown. Revisit both together if/when that UI
# need comes up.
THRESHOLD_MODELS = ["ifs", "aifs-ens"]

OUTPUT_DIR = Path("output")
PERCENTILE_ZARR_PATH = OUTPUT_DIR / "ifs" / "percentile_latest.zarr"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# First-guess chunk sizes -- see module docstring.
# Chunk size for lat/lon, in grid points. Deliberately large enough to
# exceed any realistic bbox grid size, so _write_zarr_locally's min(request,
# actual_size) logic collapses this to ONE chunk per dimension -- i.e. one
# chunk per data variable overall, not many small tiles.
#
# This was originally set to 10 (small tiles), reasoning ahead to Phase 2's
# point-query Worker, which only needs to fetch the one tile containing a
# queried point. That reasoning was premature: right now, in Phase 1, the
# only consumer is the Streamlit app, which reads the ENTIRE small grid
# eagerly every time (github_data_source.py) -- and fine-grained tiling
# only hurts that pattern, since reading "everything" then means fetching
# every tile as its own separate HTTP request. With a 71x51 grid split into
# 10x10 tiles, that was ~48 chunks x 5 threshold variables = ~240 individual
# HTTP requests, which is exactly what made loads take ~25s instead of the
# few seconds a single request per variable should take. One chunk per
# variable now; revisit tiling if/when Phase 2's Worker actually needs it.
LATLON_CHUNK = 10_000


def _write_zarr_locally(ds: xr.Dataset, path: Path, extra_chunks: dict | None = None) -> None:
    # NOTE: deliberately NOT using ds.chunk(...) here -- that's xarray's
    # dask-chunking API (it tries to wrap the data in dask arrays), which
    # requires the 'dask' package and isn't otherwise needed anywhere in
    # this pipeline (everything is already materialized via .load() by
    # the time it gets here). Passing chunk sizes through to_zarr's
    # `encoding` argument instead sets zarr's on-disk chunk layout
    # directly, with no dask dependency at all.
    chunk_sizes = {"latitude": LATLON_CHUNK, "longitude": LATLON_CHUNK}
    if extra_chunks:
        chunk_sizes.update(extra_chunks)

    encoding = {}
    for var_name, var in ds.data_vars.items():
        chunk_shape = []
        for dim in var.dims:
            requested = chunk_sizes.get(dim, -1)  # -1 (or unlisted) = one chunk covering the whole dimension
            dim_size = var.sizes[dim]
            chunk_shape.append(dim_size if requested in (-1, None) else min(requested, dim_size))
        encoding[var_name] = {"chunks": tuple(chunk_shape)}

    # mode="w" -- the "rolling latest" overwrite happens at the git-publish
    # level (force_orphan), but writing fresh here too avoids ever mixing
    # stale chunk files with new ones within a single local run.
    ds.to_zarr(path, mode="w", encoding=encoding)


def _load_existing_manifest() -> dict:
    """Reads whatever manifest.json is already sitting in ./output -- the
    workflow checks out the current `data` branch into ./output BEFORE
    running this script (see ingest.yml), so this reflects the currently
    published state. Used both to know each dataset's last-ingested
    run_time (for the check-then-fetch comparison below) and, implicitly,
    to leave already-correct data files untouched when nothing's new."""
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except Exception:
            return {}
    return {}


def _normalize_latest_result(latest):
    """Client.latest() is documented to return the date of the most
    recent matching forecast. Normalizes to a plain datetime regardless
    of whether it comes back bare or wrapped in a small result-like
    object -- worth spot-checking against the printed log on the first
    real run, since this couldn't be verified against the live API
    ahead of time."""
    if latest is None:
        return None
    return getattr(latest, "datetime", latest)


def _needs_fetch(dataset_name: str, latest_check, existing_run_times: dict) -> bool:
    """True if a full fetch is warranted for this dataset: either the
    cheap availability check itself failed (better to retry the full
    fetch than silently stay stale), there's no prior recorded run_time,
    or the latest available run doesn't match what's already stored."""
    latest_dt = _normalize_latest_result(latest_check)
    if latest_dt is None:
        return True

    current_str = existing_run_times.get(dataset_name)
    if not current_str:
        return True

    try:
        current_dt = datetime.fromisoformat(current_str)
    except ValueError:
        return True

    # Compare on the naive value -- Client.latest() returns a naive UTC
    # datetime, while our stored run_time is timezone-aware (both UTC),
    # so a direct aware-vs-naive comparison would raise.
    return latest_dt.replace(tzinfo=None) != current_dt.replace(tzinfo=None)
    """Returns the run_time string on success, None on failure (caller logs)."""
    label = MODEL_LABELS.get(model, model)
    print(f"[threshold:{model}] Fetching {label} threshold-exceedance grid...")
    ds = fetch_threshold_grid(
        bbox=PH_BBOX,
        max_lead_days=15,
        model=model,
        progress_callback=lambda frac, msg: print(f"[threshold:{model}] {frac:.0%} {msg}"),
    )
    print(f"[threshold:{model}] Fetched. run_time={ds.attrs.get('run_time')}, shape={dict(ds.sizes)}")
    path = OUTPUT_DIR / model / "threshold_latest.zarr"
    print(f"[threshold:{model}] Writing locally to {path}")
    _write_zarr_locally(ds, path)
    print(f"[threshold:{model}] Done.")
    return ds.attrs.get("run_time")


def ingest_percentile_data() -> str | None:
    """Returns the run_time string on success, None on failure (caller logs)."""
    print("[percentile] Fetching IFS ENS raw-member percentile grid...")
    ds = fetch_percentile_grid(
        bbox=PH_BBOX,
        max_lead_hours=120,
        progress_callback=lambda frac, msg: print(f"[percentile] {frac:.0%} {msg}"),
    )
    print(f"[percentile] Fetched. run_time={ds.attrs.get('run_time')}, shape={dict(ds.sizes)}")
    print(f"[percentile] Writing locally to {PERCENTILE_ZARR_PATH}")
    # step (25 bins) and number (50 members) are kept as single chunks --
    # a point query wants the whole forecast + all members for that point,
    # so splitting those dimensions would only mean more chunk files to
    # fetch for the same query, not less data transferred.
    _write_zarr_locally(ds, PERCENTILE_ZARR_PATH, extra_chunks={"step": -1, "number": -1})
    print("[percentile] Done.")
    return ds.attrs.get("run_time")


def main() -> int:
    # NOTE: no longer wipes ./output -- the workflow checks out the
    # current `data` branch into ./output before this script runs, so it
    # already contains the last-published state (including datasets we're
    # about to decide NOT to refetch). Only mkdir if that checkout didn't
    # happen (e.g. very first run ever, before a `data` branch exists).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_manifest = _load_existing_manifest()
    existing_run_times = existing_manifest.get("run_times", {})
    # Seed this run's manifest from the existing one -- datasets we skip
    # below (because nothing's new) keep their carried-forward run_time
    # and their on-disk files untouched, rather than disappearing from
    # the published output.
    run_times = dict(existing_run_times)
    failures = []

    jobs = [(f"threshold_{m}", m, "threshold") for m in THRESHOLD_MODELS]
    jobs.append(("percentile", None, "percentile"))

    for name, model, product in jobs:
        try:
            latest_check = (
                check_latest_threshold_run(model) if product == "threshold"
                else check_latest_percentile_run()
            )
        except Exception:
            print(f"[{name}] Availability check failed -- will attempt a full fetch anyway.")
            latest_check = None

        if not _needs_fetch(name, latest_check, existing_run_times):
            print(f"[{name}] Already up to date (run {existing_run_times.get(name)}) -- skipping.")
            continue

        try:
            if product == "threshold":
                run_times[name] = ingest_threshold_forecast(model)
            else:
                run_times[name] = ingest_percentile_data()
        except Exception:
            print(f"[{name}] FAILED:", file=sys.stderr)
            traceback.print_exc()
            failures.append(name)

    # Small manifest alongside the data -- lets the Streamlit app (and
    # later, the Worker) check "how fresh is this?" with one small fetch
    # instead of opening a Zarr store just to read an attribute.
    manifest = {
        "generated_at": _now_iso(),
        "run_times": run_times,
        "failures": failures,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    if failures:
        print(f"Ingestion finished with failures: {failures}", file=sys.stderr)
        return 1

    print("Ingestion finished successfully.")
    return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    sys.exit(main())
