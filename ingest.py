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
the GitHub Actions workflow's job (it publishes ./output to a dedicated
`data` branch as a single fresh commit each run, via
peaceiris/actions-gh-pages with force_orphan: true, so the branch never
accumulates history -- matches the "rolling latest only" retention
decision). Keeping storage/publishing out of this script means no cloud
credentials are needed here at all.

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
import shutil
import sys
import traceback
from pathlib import Path

import xarray as xr

# Reuses the exact fetch/decode/crop logic in ecmwf_client.py -- this
# script is intentionally a thin wrapper, not a reimplementation, so
# ingestion and the live app's own fetch code can't quietly drift apart.
from ecmwf_client import fetch_threshold_grid, fetch_percentile_grid

# Generous bounding box around the Philippines -- covers all 5 of the
# app's current fixed locations with plenty of room to spare, sized for
# future arbitrary-point selection across the country rather than just
# today's handful of points. Revisit if the app ever needs to cover areas
# outside this box.
PH_BBOX = {"lat_min": 4.0, "lat_max": 21.5, "lon_min": 115.0, "lon_max": 127.5}

OUTPUT_DIR = Path("output")
THRESHOLD_ZARR_PATH = OUTPUT_DIR / "ifs" / "threshold_latest.zarr"
PERCENTILE_ZARR_PATH = OUTPUT_DIR / "ifs" / "percentile_latest.zarr"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# First-guess chunk sizes -- see module docstring.
LATLON_CHUNK = 10  # grid points per chunk, each dimension


def _write_zarr_locally(ds: xr.Dataset, path: Path, extra_chunks: dict | None = None) -> None:
    chunks = {"latitude": LATLON_CHUNK, "longitude": LATLON_CHUNK}
    if extra_chunks:
        chunks.update(extra_chunks)
    ds = ds.chunk(chunks)
    # mode="w" -- the "rolling latest" overwrite happens at the git-publish
    # level (force_orphan), but writing fresh here too avoids ever mixing
    # stale chunk files with new ones within a single local run.
    ds.to_zarr(path, mode="w")


def ingest_threshold_forecast() -> str | None:
    """Returns the run_time string on success, None on failure (caller logs)."""
    print("[threshold] Fetching IFS ENS threshold-exceedance grid...")
    ds = fetch_threshold_grid(
        bbox=PH_BBOX,
        max_lead_days=15,
        model="ifs",
        progress_callback=lambda frac, msg: print(f"[threshold] {frac:.0%} {msg}"),
    )
    print(f"[threshold] Fetched. run_time={ds.attrs.get('run_time')}, shape={dict(ds.sizes)}")
    print(f"[threshold] Writing locally to {THRESHOLD_ZARR_PATH}")
    _write_zarr_locally(ds, THRESHOLD_ZARR_PATH)
    print("[threshold] Done.")
    return ds.attrs.get("run_time")


def ingest_percentile_data() -> str | None:
    """Returns the run_time string on success, None on failure (caller logs)."""
    print("[percentile] Fetching IFS ENS raw-member percentile grid...")
    ds = fetch_percentile_grid(
        bbox=PH_BBOX,
        max_lead_hours=72,
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
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)  # clean slate -- no leftover files from a previous local run
    OUTPUT_DIR.mkdir(parents=True)

    # Both run to completion even if one fails -- a broken percentile
    # fetch shouldn't prevent the (cheaper, more central to the app)
    # threshold data from updating, and vice versa. Streamlit's fallback
    # to a live fetch (kept in place, see the app-side plan) covers
    # whichever one didn't make it. If one fails, its ./output subfolder
    # simply won't exist -- the workflow still publishes whatever DID
    # succeed, rather than an all-or-nothing failure wiping out a good
    # threshold fetch just because percentile had a bad day.
    run_times = {}
    failures = []

    for name, fn in [("threshold", ingest_threshold_forecast), ("percentile", ingest_percentile_data)]:
        try:
            run_times[name] = fn()
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
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    sys.exit(main())
