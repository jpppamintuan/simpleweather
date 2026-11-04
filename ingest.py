"""
Scheduled ingestion script for the ECMWF forecast platform.

Run on a schedule via GitHub Actions (see .github/workflows/ingest.yml).
Fetches the latest ECMWF IFS ENS data -- the precomputed threshold-
exceedance product, plus raw ensemble members for rainfall and for
temperature/heat-index percentiles -- crops each to a bounding box
covering the Philippines, and writes one NetCDF file per dataset to a
local ./output directory. Publishing ./output to the `data` branch is the
workflow's job (it checks the branch out into ./output before this runs
and force-pushes it back as a single orphan commit after), so no cloud
credentials are needed here.

TWO RESOLUTIONS, ONE STORE. ECMWF publishes each IFS run at 0.25 deg
first and at 0.1 deg about two hours later. Each dataset is therefore
ingested in two stages per run: the 0.25 deg version as soon as it
exists, then the 0.1 deg version of the SAME run once that appears,
replacing it. Selection rule (see _candidates / _is_newer):

  * 0.1 deg is preferred whenever its latest run is at least as new as
    the latest 0.25 deg run;
  * a newer 0.25 deg run beats an older 0.1 deg run (so the store never
    sits on a stale high-res run while a fresher run is available);
  * stored data is never replaced by the same run at lower resolution,
    nor by an older run -- a transient failed availability check can't
    downgrade anything.

If the 0.1 deg product isn't published (or its fetch fails), ingestion
falls back to 0.25 deg for that run. Set try_0p1 to False in DATASETS to
keep a dataset on 0.25 deg.

Check-then-fetch: availability is checked cheaply via Client.latest()
(metadata only) and compared with what's already published, so running
this often costs almost nothing when nothing is new.

NetCDF, not Zarr: an earlier Zarr version hit a never-diagnosed
KeyError('.zmetadata') when read over raw.githubusercontent.com. A single
NetCDF file fetched with one plain HTTP GET avoids that whole class of
problem.

FILE SIZE. GitHub rejects files over 100 MB, and 0.1 deg grids have ~6x
the points of 0.25 deg. So large per-member variables are packed to int16
(scale_factor), zlib-compressed with the shuffle filter, and written
atomically: a file is only moved into place if it's under MAX_FILE_BYTES,
otherwise the previous good file stays published and the run is reported
as failed with the actual size.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

import diag

# Reuses the exact fetch/decode/crop logic in ecmwf_client.py -- this
# script is intentionally a thin wrapper, not a reimplementation.
from ecmwf_client import (
    DEBUG_MAX_BATCHES,
    DEFAULT_RESOLUTION,
    PERCENTILE_MAX_LEAD_HOURS,
    PH_BBOX,
    THRESHOLD_LEAD_DAYS,
    RESOLUTION_RANK,
    check_latest_pf_run,
    check_latest_threshold_run,
    fetch_precip_grid,
    fetch_temperature_grid,
    fetch_threshold_grid,
)

OUTPUT_DIR = Path("output")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# GitHub's hard per-file limit is 100 MB; stay clear of it.
MAX_FILE_BYTES = 95 * 1024 * 1024

# Diagnose mode: DIAGNOSE_ONLY="percentile" (comma-separated names) with
# DIAGNOSE_BATCHES=1 fetches and decodes just the first batch(es) of those
# datasets, exercises the writer against a temp dir, and touches neither
# ./output nor the manifest -- a 2-4 minute reproduction instead of an
# 80-minute full run. Set from the workflow's manual-run inputs.
DIAGNOSE_ONLY = [n.strip() for n in os.environ.get("DIAGNOSE_ONLY", "").split(",") if n.strip()]


def _progress(tag: str):
    return lambda frac, msg: print(f"[{tag}] {frac:.0%} {msg}", flush=True)


# name -> config.
#   path     where the file lands inside ./output (the app reads these)
#   try_0p1  attempt the 0.1 deg product too (False = stay on 0.25 deg)
#   check    cheap metadata-only availability check, called as check(resol)
#   fetch    full fetch, called as fetch(run_dt, resol) -> xr.Dataset
#   packing  {variable: scale_factor} -- stored as int16, see _write_netcdf_locally
DATASETS = {
    "threshold": {
        "path": OUTPUT_DIR / "ifs" / "threshold_latest.nc",
        "try_0p1": True,
        "check": check_latest_threshold_run,
        "fetch": lambda run_dt, resol: fetch_threshold_grid(
            PH_BBOX, run_dt, resol, max_lead_days=THRESHOLD_LEAD_DAYS, progress_callback=_progress("threshold")),
        "packing": {},
    },
    "percentile": {
        "path": OUTPUT_DIR / "ifs" / "percentile_latest.nc",
        "try_0p1": True,
        "check": check_latest_pf_run,
        "fetch": lambda run_dt, resol: fetch_precip_grid(
            PH_BBOX, run_dt, resol, max_lead_hours=PERCENTILE_MAX_LEAD_HOURS, progress_callback=_progress("percentile")),
        # 0.02 mm steps, range +/-655 mm per 3h bin (values are clipped to it)
        "packing": {"precip_3h_mm": 0.02},
    },
    "temperature": {
        "path": OUTPUT_DIR / "ifs" / "temperature_latest.nc",
        "try_0p1": True,
        "check": check_latest_pf_run,
        "fetch": lambda run_dt, resol: fetch_temperature_grid(
            PH_BBOX, run_dt, resol, max_lead_hours=PERCENTILE_MAX_LEAD_HOURS, progress_callback=_progress("temperature")),
        "packing": {"temperature_c": 0.1, "heat_index_c": 0.1},
    },
}


def _write_netcdf_locally(ds: xr.Dataset, path: Path, packing: dict | None = None) -> int:
    """Writes a single NetCDF file and returns its size in bytes.

    Variables named in `packing` are stored as int16 with the given
    scale_factor (clipped to the representable range first -- an
    out-of-range float would otherwise wrap around silently). The scale
    is a float32 so xarray decodes back to float32 rather than float64,
    which matters for memory when the app opens the file. All variables
    get zlib + shuffle (shuffle is what makes int16 compress well).

    Written to a temp file and moved into place only if it's under
    MAX_FILE_BYTES; mode is effectively "w" -- an existing file is
    replaced, matching the "rolling latest only" retention decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    packing = packing or {}

    encoding = {}
    for var in ds.data_vars:
        enc = {"zlib": True, "complevel": 4, "shuffle": True}
        scale = packing.get(var)
        if scale is not None:
            limit = 32766 * scale
            ds[var] = ds[var].clip(-limit, limit)
            enc.update({"dtype": "int16", "scale_factor": np.float32(scale), "_FillValue": -32768})
        encoding[var] = enc

    tmp_path = path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp_path, engine="h5netcdf", encoding=encoding)
    size = tmp_path.stat().st_size
    if size > MAX_FILE_BYTES:
        tmp_path.unlink()
        raise RuntimeError(
            f"{path.name} came out at {size / 1e6:.1f} MB, over the "
            f"{MAX_FILE_BYTES / 1e6:.0f} MB limit for a file on the data branch. "
            f"Previous file left in place. Options: coarser packing, a smaller "
            f"bbox or fewer steps, or set try_0p1 to False for this dataset."
        )
    os.replace(tmp_path, path)
    print(f"[write] {path} -- {size / 1e6:.1f} MB")
    return size


def _load_existing_manifest() -> dict:
    """Reads whatever manifest.json is already sitting in ./output -- the
    workflow checks out the current `data` branch into ./output BEFORE
    running this script, so this reflects the currently published state."""
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except Exception:
            return {}
    return {}


def _naive(dt: datetime) -> datetime:
    # Client.latest() returns naive UTC datetimes while stored run_times
    # are timezone-aware (also UTC) -- compare on the naive values.
    return dt.replace(tzinfo=None)


def _parse_run_time(value) -> datetime | None:
    if not value:
        return None
    try:
        return _naive(datetime.fromisoformat(value))
    except ValueError:
        return None


def _normalize_latest_result(latest):
    """Client.latest() is documented to return the date of the most recent
    matching forecast; normalize to a plain naive datetime whether it comes
    back bare or wrapped in a small result-like object."""
    if latest is None:
        return None
    dt = getattr(latest, "datetime", latest)
    return _naive(dt)


def _safe_latest(check, resol: str, name: str) -> datetime | None:
    try:
        return _normalize_latest_result(check(resol))
    except Exception as e:
        print(f"[{name}] {resol} availability check failed or product not published "
              f"({type(e).__name__}: {e})")
        return None


def _candidates(name: str, cfg: dict):
    """Returns ([(run_dt, resol), ...] in order of preference, latest 0.25
    run or None). The second entry, when present, is the fallback if the
    first fails to download."""
    latest_025 = _safe_latest(cfg["check"], "0p25", name)
    latest_01 = _safe_latest(cfg["check"], "0p1", name) if cfg["try_0p1"] else None

    candidates = []
    # 0.1 deg only wins if it's at least as new as the newest 0.25 deg run.
    if latest_01 is not None and (latest_025 is None or latest_01 >= latest_025):
        candidates.append((latest_01, "0p1"))
    if latest_025 is not None:
        candidates.append((latest_025, "0p25"))
    return candidates, latest_025


def _is_newer(run, resol, stored_run, stored_resol) -> bool:
    """True if (run, resol) should replace what's stored: a newer run
    always does; the same run does only at a higher resolution. Never an
    older run, never the same run at equal or lower resolution."""
    if stored_run is None:
        return True
    if run != stored_run:
        return run > stored_run
    return RESOLUTION_RANK.get(resol, 0) > RESOLUTION_RANK.get(stored_resol, 0)


def _ingest(name: str, cfg: dict, run_dt: datetime, resol: str) -> tuple[str, str | None]:
    """Returns (run_time, published_at) on success -- published_at being
    when ECMWF actually released the files (None if it couldn't be
    determined); raises on failure."""
    diag.log(f"[{name}] Fetching run {run_dt:%Y-%m-%d %H}Z at {resol}")
    ds = cfg["fetch"](run_dt, resol)
    print(f"[{name}] Fetched. run_time={ds.attrs.get('run_time')}, resolution={ds.attrs.get('resolution')}, "
          f"dims={dict(ds.sizes)}", flush=True)
    diag.log(f"[{name}] writing")
    _write_netcdf_locally(ds, cfg["path"], cfg["packing"])
    diag.log(f"[{name}] done")
    return ds.attrs["run_time"], ds.attrs.get("published_at")


def _diagnose() -> int:
    import tempfile
    rc = 0
    for name in DIAGNOSE_ONLY:
        cfg = DATASETS.get(name)
        if cfg is None:
            print(f"[diagnose] unknown dataset {name!r}; choose from {list(DATASETS)}")
            rc = 1
            continue
        candidates, _ = _candidates(name, cfg)
        if not candidates:
            print(f"[diagnose:{name}] no available run found")
            rc = 1
            continue
        run_dt, resol = candidates[0]
        diag.log(f"[diagnose:{name}] fetching run {run_dt:%Y-%m-%d %H}Z at {resol}, "
                 f"first {DEBUG_MAX_BATCHES or 'all'} batch(es); nothing will be published")
        try:
            ds = cfg["fetch"](run_dt, resol)
            diag.log(f"[diagnose:{name}] fetched, dims={dict(ds.sizes)}")
            with tempfile.TemporaryDirectory() as tmp:
                size = _write_netcdf_locally(ds, Path(tmp) / f"{name}.nc", cfg["packing"])
            diag.log(f"[diagnose:{name}] test file written, {size / 1e6:.1f} MB (partial data if batches were limited)")
        except Exception:
            traceback.print_exc()
            diag.log(f"[diagnose:{name}] FAILED (the threshold step-count check failing on partial data is expected)")
            rc = 1
    return rc


def main() -> int:
    diag.install()
    diag.start_monitor()
    diag.log(f"start | python {sys.version.split()[0]} | event={os.environ.get('GITHUB_EVENT_NAME', '?')} "
             f"run={os.environ.get('GITHUB_RUN_ID', '?')} attempt={os.environ.get('GITHUB_RUN_ATTEMPT', '?')}")
    try:
        return _diagnose() if DIAGNOSE_ONLY else _run()
    finally:
        diag.log("end")


def _run() -> int:
    # NOT wiped: the workflow already checked the current `data` branch out
    # into ./output, so it holds the last-published state (including
    # datasets we're about to decide NOT to refetch).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Retired model -- remove its leftover files from the published branch.
    shutil.rmtree(OUTPUT_DIR / "aifs-ens", ignore_errors=True)

    force_refresh = os.environ.get("FORCE_REFRESH", "false").strip().lower() == "true"
    if force_refresh:
        print("FORCE_REFRESH is set -- ignoring the check-then-fetch skip logic, "
              "refetching every dataset regardless of what's stored.")

    existing = _load_existing_manifest()
    # Seed from the existing manifest so datasets we skip keep their
    # carried-forward run_time/resolution. Entries for datasets that no
    # longer exist (threshold_ifs, threshold_aifs-ens) are dropped, so the
    # first run after this change refetches everything once.
    run_times = {k: v for k, v in existing.get("run_times", {}).items() if k in DATASETS}
    resolutions = {k: v for k, v in existing.get("resolutions", {}).items() if k in DATASETS}
    published_ats = {k: v for k, v in existing.get("published_at", {}).items() if k in DATASETS}
    failures: list[str] = []          # stored data is behind what's available
    upgrade_failures: list[str] = []  # stored data is current; only the 0.1 deg upgrade failed

    def write_manifest():
        # Written after EVERY dataset, not just at the end: a run that gets
        # cut off (timeout, cancel) still publishes a manifest matching the
        # files it did finish, instead of new files under a stale manifest.
        MANIFEST_PATH.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_times": run_times,
            "resolutions": resolutions,
            "published_at": published_ats,
            "failures": failures,
            "upgrade_failures": upgrade_failures,
        }, indent=2))

    for name, cfg in DATASETS.items():
        existing_run = _parse_run_time(run_times.get(name))
        stored_run = None if force_refresh else existing_run
        stored_resol = resolutions.get(name, DEFAULT_RESOLUTION)  # a pre-0.1 manifest means 0.25

        candidates, latest_025 = _candidates(name, cfg)
        pending = [(run, res) for run, res in candidates if _is_newer(run, res, stored_run, stored_resol)]
        if not pending:
            print(f"[{name}] Nothing newer than stored run {existing_run} at {stored_resol} -- skipping.")
            write_manifest()  # keeps generated_at fresh for the app's staleness check
            continue

        succeeded = False
        for run_dt, resol in pending:
            try:
                run_times[name], published = _ingest(name, cfg, run_dt, resol)
                resolutions[name] = resol
                published_ats.pop(name, None)
                if published:
                    published_ats[name] = published
                succeeded = True
                break
            except Exception:
                print(f"[{name}] FAILED at {resol}:", file=sys.stderr)
                traceback.print_exc()

        if not succeeded:
            still_current = existing_run is not None and latest_025 is not None and existing_run >= latest_025
            (upgrade_failures if still_current else failures).append(name)
        write_manifest()

    write_manifest()

    if failures or upgrade_failures:
        print(f"Ingestion finished with failures: {failures}, upgrade failures: {upgrade_failures}",
              file=sys.stderr)
        return 1

    print("Ingestion finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
