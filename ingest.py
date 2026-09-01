"""
Scheduled ingestion script for the ECMWF rainfall forecast platform.

Run on a schedule via GitHub Actions (see .github/workflows/ingest.yml).
Fetches the latest ECMWF IFS ENS forecast -- both the precomputed
threshold-exceedance product and the raw ensemble-member product used for
percentile calculations -- crops each to a bounding box covering the
Philippines (generous enough to support arbitrary-point queries later, not
just the app's current 5 fixed locations), and writes the result to
Cloudflare R2 as Zarr.

Zarr specifically (not GRIB2/NetCDF): R2 supports HTTP range requests and
Zarr is chunked, so a future query layer (Phase 2 -- a Cloudflare Worker)
can read just the chunk containing a queried point instead of downloading
the whole file per request.

"Latest run" storage model: every run overwrites the same object keys
(ifs/threshold_latest.zarr, ifs/percentile_latest.zarr) rather than
accumulating timestamped runs -- matches the "rolling latest only"
retention decision. Wanting historical/verification data later is a
deliberate, separate change to this retention model, not something to
grow into by accident.

Required environment variables (set as GitHub Actions secrets -- see the
setup notes shared alongside this script, NEVER commit these):
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME
"""

from __future__ import annotations

import os
import sys
import traceback

import s3fs
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

R2_ENDPOINT_TEMPLATE = "https://{account_id}.r2.cloudflarestorage.com"

THRESHOLD_OBJECT_KEY = "ifs/threshold_latest.zarr"
PERCENTILE_OBJECT_KEY = "ifs/percentile_latest.zarr"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"This must be set as a GitHub Actions secret (see setup notes)."
        )
    return value


def _r2_filesystem() -> s3fs.S3FileSystem:
    account_id = _require_env("R2_ACCOUNT_ID")
    return s3fs.S3FileSystem(
        key=_require_env("R2_ACCESS_KEY_ID"),
        secret=_require_env("R2_SECRET_ACCESS_KEY"),
        endpoint_url=R2_ENDPOINT_TEMPLATE.format(account_id=account_id),
    )


def _write_zarr_to_r2(ds: xr.Dataset, object_key: str) -> None:
    bucket = _require_env("R2_BUCKET_NAME")
    fs = _r2_filesystem()
    store = s3fs.S3Map(root=f"{bucket}/{object_key}", s3=fs, check=False)
    # mode="w" IS the "rolling latest" overwrite -- each run replaces the
    # previous object wholesale rather than appending to it.
    ds.to_zarr(store, mode="w")


def ingest_threshold_forecast() -> None:
    print("[threshold] Fetching IFS ENS threshold-exceedance grid...")
    ds = fetch_threshold_grid(
        bbox=PH_BBOX,
        max_lead_days=15,
        model="ifs",
        progress_callback=lambda frac, msg: print(f"[threshold] {frac:.0%} {msg}"),
    )
    print(f"[threshold] Fetched. run_time={ds.attrs.get('run_time')}, shape={dict(ds.sizes)}")
    print(f"[threshold] Writing to R2: {THRESHOLD_OBJECT_KEY}")
    _write_zarr_to_r2(ds, THRESHOLD_OBJECT_KEY)
    print("[threshold] Done.")


def ingest_percentile_data() -> None:
    print("[percentile] Fetching IFS ENS raw-member percentile grid...")
    ds = fetch_percentile_grid(
        bbox=PH_BBOX,
        max_lead_hours=72,
        progress_callback=lambda frac, msg: print(f"[percentile] {frac:.0%} {msg}"),
    )
    print(f"[percentile] Fetched. run_time={ds.attrs.get('run_time')}, shape={dict(ds.sizes)}")
    print(f"[percentile] Writing to R2: {PERCENTILE_OBJECT_KEY}")
    _write_zarr_to_r2(ds, PERCENTILE_OBJECT_KEY)
    print("[percentile] Done.")


def main() -> int:
    # Both run to completion even if one fails -- a broken percentile
    # fetch shouldn't prevent the (cheaper, more central to the app)
    # threshold data from updating, and vice versa. Streamlit's fallback
    # to a live fetch (kept in place, see the app-side plan) covers
    # whichever one didn't make it.
    failures = []

    for name, fn in [("threshold", ingest_threshold_forecast), ("percentile", ingest_percentile_data)]:
        try:
            fn()
        except Exception:
            print(f"[{name}] FAILED:", file=sys.stderr)
            traceback.print_exc()
            failures.append(name)

    if failures:
        print(f"Ingestion finished with failures: {failures}", file=sys.stderr)
        return 1

    print("Ingestion finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
