"""
ECMWF Open Data fetcher for 24h rainfall exceedance probabilities.

Uses the official `ecmwf-opendata` package to pull the ENS "probability"
stream (enfo / type=ep), which contains precomputed probability-of-exceedance
fields for 24h accumulated total precipitation: tpg1, tpg5, tpg10, tpg20,
tpg25, tpg50, tpg100 (mm). These are fixed thresholds baked into the model
output, computed from ECMWF's 50-member ensemble (each member = 2% weight).

No raw ensemble-member processing needed for these standard thresholds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import xarray as xr
from ecmwf.opendata import Client

# Fixed thresholds ECMWF publishes for 24h accumulated precipitation, in mm
AVAILABLE_THRESHOLDS_MM = [1, 5, 10, 20, 25, 50, 100]


def _threshold_to_param(threshold_mm: int) -> str:
    if threshold_mm not in AVAILABLE_THRESHOLDS_MM:
        raise ValueError(
            f"Threshold {threshold_mm} mm not available. "
            f"Choose one of {AVAILABLE_THRESHOLDS_MM}."
        )
    return f"tpg{threshold_mm}"


def _default_steps(max_lead_days: int = 10) -> list[str]:
    """24h windows starting every 12h: '0-24', '12-36', '24-48', ..."""
    max_lead_hours = max_lead_days * 24
    n = (max_lead_hours - 24) // 12 + 1
    return [f"{12 * i}-{12 * i + 24}" for i in range(n)]


def fetch_exceedance_probabilities(
    lat: float,
    lon: float,
    threshold_mm: int,
    max_lead_days: int = 10,
) -> dict[str, float]:
    """
    Fetch the latest available ECMWF ENS forecast run and return the
    probability (%) that 24h accumulated precipitation exceeds
    `threshold_mm` at the given location, for each forecast window.

    Returns e.g. {"0-24": 82.0, "12-36": 71.0, "24-48": 58.0, ...}
    """
    param = _threshold_to_param(threshold_mm)
    steps = _default_steps(max_lead_days)

    # No date/time specified -> client automatically resolves to the
    # latest complete forecast run available on the open data servers.
    client = Client(source="ecmwf")

    with tempfile.TemporaryDirectory() as tmpdir:
        target = str(Path(tmpdir) / "data.grib2")
        client.retrieve(
            stream="enfo",
            type="ep",
            step=steps,
            param=param,
            target=target,
        )

        ds = xr.open_dataset(
            target,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )

        # ECMWF open data grids run 0-360 degrees longitude; normalize
        # the requested longitude to match before selecting a point.
        grid_lon = lon % 360 if float(ds.longitude.max()) > 180 else lon

        point = ds.sel(latitude=lat, longitude=grid_lon, method="nearest")

        var_name = list(ds.data_vars)[0]
        results: dict[str, float] = {}
        for i, step_label in enumerate(steps):
            try:
                val = float(point[var_name].isel(step=i).values)
                results[step_label] = round(val, 1)
            except IndexError:
                break

        return results
