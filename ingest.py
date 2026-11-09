"""ECMWF ENS scheduled ingestion, with optional and independently stored Taiwan crops.

PH storage and the 95 MiB per-file guard are unchanged. With Taiwan enabled,
ECMWF is fetched once per product/run/resolution over a transient combined box,
then each region is written separately. No expiry or scheduling changes.
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
from regions import TAIWAN_BBOX
from taiwan_support import combined_fetch_bbox, crop_region, taiwan_enabled

OUTPUT_DIR = Path("output")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
MAX_FILE_BYTES = 95 * 1024 * 1024
DIAGNOSE_ONLY = [n.strip() for n in os.environ.get("DIAGNOSE_ONLY", "").split(",") if n.strip()]


def _progress(tag: str):
    return lambda frac, msg: print(f"[{tag}] {frac:.0%} {msg}", flush=True)


# These are the same three products and packing settings as the PH-only app.
# They temporarily include Taiwan in the fetch, if ENABLE_TAIWAN is on.
DATASETS = {
    "threshold": {
        "path": OUTPUT_DIR / "ifs" / "threshold_latest.nc",
        "try_0p1": True,
        "check": check_latest_threshold_run,
        "fetch": lambda run_dt, resol: fetch_threshold_grid(
            combined_fetch_bbox(), run_dt, resol,
            max_lead_days=THRESHOLD_LEAD_DAYS,
            progress_callback=_progress("threshold")
        ),
        "packing": {},
    },
    "percentile": {
        "path": OUTPUT_DIR / "ifs" / "percentile_latest.nc",
        "try_0p1": True,
        "check": check_latest_pf_run,
        "fetch": lambda run_dt, resol: fetch_precip_grid(
            combined_fetch_bbox(), run_dt, resol,
            max_lead_hours=PERCENTILE_MAX_LEAD_HOURS,
            progress_callback=_progress("percentile")
        ),
        "packing": {"precip_3h_mm": 0.02},
    },
    "temperature": {
        "path": OUTPUT_DIR / "ifs" / "temperature_latest.nc",
        "try_0p1": True,
        "check": check_latest_pf_run,
        "fetch": lambda run_dt, resol: fetch_temperature_grid(
            combined_fetch_bbox(), run_dt, resol,
            max_lead_hours=PERCENTILE_MAX_LEAD_HOURS,
            progress_callback=_progress("temperature")
        ),
        "packing": {"temperature_c": 0.1, "heat_index_c": 0.1},
    },
}


def _write_netcdf_locally(ds: xr.Dataset, path: Path, packing: dict | None = None) -> int:
    """Atomically publish one compressed dataset if it fits GitHub's file cap."""
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
    try:
        ds.to_netcdf(tmp_path, engine="h5netcdf", encoding=encoding)
        size = tmp_path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise RuntimeError(
                f"{path.name} came out at {size / 1e6:.1f} MB, over the "
                f"{MAX_FILE_BYTES / 1e6:.0f} MB data-branch limit. "
                "The previous file was left in place. Try a smaller bbox, "
                "coarser packing, fewer steps, or 0.25-degree data."
            )
        os.replace(tmp_path, path)
    finally:
        # Also remove the temporary file on a serialization/size failure.
        if tmp_path.exists():
            tmp_path.unlink()
    print(f"[write] {path} -- {size / 1e6:.1f} MB", flush=True)
    return size


def _load_existing_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except Exception:
            return {}
    return {}


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None)


def _parse_run_time(value) -> datetime | None:
    if not value:
        return None
    try:
        return _naive(datetime.fromisoformat(value))
    except (TypeError, ValueError):
        return None


def _normalize_latest_result(latest):
    if latest is None:
        return None
    dt = getattr(latest, "datetime", latest)
    return _naive(dt)


def _safe_latest(check, resol: str, name: str) -> datetime | None:
    try:
        return _normalize_latest_result(check(resol))
    except Exception as exc:
        print(f"[{name}] {resol} availability check failed or product not published "
              f"({type(exc).__name__}: {exc})")
        return None


def _candidates(name: str, cfg: dict):
    """Newest 0.1-degree candidate (if current), then 0.25-degree fallback."""
    latest_025 = _safe_latest(cfg["check"], "0p25", name)
    latest_01 = _safe_latest(cfg["check"], "0p1", name) if cfg["try_0p1"] else None
    candidates = []
    if latest_01 is not None and (latest_025 is None or latest_01 >= latest_025):
        candidates.append((latest_01, "0p1"))
    if latest_025 is not None:
        candidates.append((latest_025, "0p25"))
    return candidates, latest_025


def _is_newer(run, resol, stored_run, stored_resol) -> bool:
    if stored_run is None:
        return True
    if run != stored_run:
        return run > stored_run
    return RESOLUTION_RANK.get(resol, 0) > RESOLUTION_RANK.get(stored_resol, 0)


def _ingest(name: str, cfg: dict, run_dt: datetime, resol: str,
            *, write_ph: bool, write_tw: bool) -> tuple[str, str | None, bool]:
    """One fetch, separate atomic writes; a TW *write* failure never blocks PH."""
    diag.log(f"[{name}] Fetching run {run_dt:%Y-%m-%d %H}Z at {resol}")
    ds = cfg["fetch"](run_dt, resol)
    run_time = ds.attrs["run_time"]
    published = ds.attrs.get("published_at")
    print(f"[{name}] Fetched run={run_time}, resolution={resol}, dims={dict(ds.sizes)}", flush=True)

    try:
        if write_ph:
            ph = crop_region(ds, PH_BBOX) if taiwan_enabled() else ds
            diag.log(f"[{name}] writing Philippines")
            _write_netcdf_locally(ph, cfg["path"], cfg["packing"])
            if ph is not ds:
                del ph

        tw_written = False
        if write_tw and taiwan_enabled():
            try:
                tw = crop_region(ds, TAIWAN_BBOX)
                tw_path = OUTPUT_DIR / "taiwan" / f"{name}_latest.nc"
                diag.log(f"[taiwan_{name}] writing {tw_path}")
                _write_netcdf_locally(tw, tw_path, cfg["packing"])
                del tw
                tw_written = True
            except Exception:
                print(f"[taiwan_{name}] WARNING: Taiwan update failed; PH files unaffected.",
                      file=sys.stderr)
                traceback.print_exc()
    finally:
        del ds
    diag.log(f"[{name}] done")
    return run_time, published, tw_written


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
                ph = crop_region(ds, PH_BBOX) if taiwan_enabled() else ds
                size = _write_netcdf_locally(ph, Path(tmp) / f"{name}.nc", cfg["packing"])
                if taiwan_enabled():
                    tw = crop_region(ds, TAIWAN_BBOX)
                    _write_netcdf_locally(tw, Path(tmp) / f"taiwan_{name}.nc", cfg["packing"])
            diag.log(f"[diagnose:{name}] test file written, {size / 1e6:.1f} MB")
        except Exception:
            traceback.print_exc()
            diag.log(f"[diagnose:{name}] FAILED")
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
    # The workflow checked out the data branch into ./output first.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUTPUT_DIR / "aifs-ens", ignore_errors=True)
    force_refresh = os.environ.get("FORCE_REFRESH", "false").strip().lower() == "true"
    if force_refresh:
        print("FORCE_REFRESH is set -- refetching all datasets regardless of stored state.")

    existing = _load_existing_manifest()
    active_names = set(DATASETS) | {f"taiwan_{name}" for name in DATASETS}
    # Preserve TW metadata after switching the feature off for easy reactivation.
    run_times = {k: v for k, v in existing.get("run_times", {}).items() if k in active_names}
    resolutions = {k: v for k, v in existing.get("resolutions", {}).items() if k in active_names}
    published_ats = {k: v for k, v in existing.get("published_at", {}).items() if k in active_names}
    failures: list[str] = []
    upgrade_failures: list[str] = []
    taiwan_failures: list[str] = []

    def write_manifest():
        MANIFEST_PATH.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_times": run_times,
            "resolutions": resolutions,
            "published_at": published_ats,
            "failures": failures,
            "upgrade_failures": upgrade_failures,
            "taiwan_failures": taiwan_failures,
        }, indent=2))

    for name, cfg in DATASETS.items():
        ph_old = _parse_run_time(run_times.get(name))
        ph_old_res = resolutions.get(name, DEFAULT_RESOLUTION)
        tw_name = f"taiwan_{name}"
        tw_old = _parse_run_time(run_times.get(tw_name))
        tw_old_res = resolutions.get(tw_name, DEFAULT_RESOLUTION)
        tw_path = OUTPUT_DIR / "taiwan" / f"{name}_latest.nc"
        ph_missing = not cfg["path"].is_file()
        tw_missing = not tw_path.is_file()

        candidates, latest_025 = _candidates(name, cfg)
        pending = []
        for run_dt, resol in candidates:
            need_ph = force_refresh or ph_missing or _is_newer(run_dt, resol, ph_old, ph_old_res)
            need_tw = taiwan_enabled() and (
                force_refresh or tw_missing or _is_newer(run_dt, resol, tw_old, tw_old_res)
            )
            if need_ph or need_tw:
                pending.append((run_dt, resol, need_ph, need_tw))

        if not pending:
            print(f"[{name}] Nothing new for Philippines/Taiwan; skipping.")
            write_manifest()
            continue

        ph_succeeded = False
        fetch_succeeded = False
        for run_dt, resol, need_ph, need_tw in pending:
            try:
                run_time, published, tw_written = _ingest(
                    name, cfg, run_dt, resol, write_ph=need_ph, write_tw=need_tw
                )
                if need_ph:
                    run_times[name] = run_time
                    resolutions[name] = resol
                    published_ats.pop(name, None)
                    if published:
                        published_ats[name] = published
                    ph_succeeded = True
                    ph_missing = False
                if tw_written:
                    run_times[tw_name] = run_time
                    resolutions[tw_name] = resol
                    published_ats.pop(tw_name, None)
                    if published:
                        published_ats[tw_name] = published
                    tw_missing = False
                elif need_tw:
                    taiwan_failures.append(tw_name)
                fetch_succeeded = True
                break
            except Exception:
                print(f"[{name}] FAILED at {resol}:", file=sys.stderr)
                traceback.print_exc()

        if any(ph for _, _, ph, _ in pending) and not ph_succeeded:
            current = ph_old is not None and latest_025 is not None and ph_old >= latest_025
            (upgrade_failures if current else failures).append(name)
        if not fetch_succeeded and taiwan_enabled() and any(tw for _, _, _, tw in pending):
            taiwan_failures.append(tw_name)
        write_manifest()

    write_manifest()
    if failures or upgrade_failures:
        print(f"Ingestion finished with failures: {failures}, upgrade failures: {upgrade_failures}",
              file=sys.stderr)
        return 1
    if taiwan_failures:
        print(f"Ingestion finished; optional Taiwan updates failed: {taiwan_failures}",
              file=sys.stderr)
    print("Ingestion finished successfully for all required Philippine datasets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
