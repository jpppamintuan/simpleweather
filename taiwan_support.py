"""Opt-in Taiwan coverage helpers; disabled when regions.ENABLE_TAIWAN is False.

Existing Philippine datasets and the existing regional PH_BBOX are unchanged.
"""
from __future__ import annotations

from ecmwf_client import PH_BBOX
from regions import ENABLE_TAIWAN, TAIWAN_BBOX


def taiwan_enabled() -> bool:
    return bool(ENABLE_TAIWAN)


def inside_box(lat: float, lon: float, bbox: dict) -> bool:
    return (bbox["lat_min"] <= float(lat) <= bbox["lat_max"]
            and bbox["lon_min"] <= float(lon) <= bbox["lon_max"])


def region_for_point(lat: float, lon: float) -> str:
    if inside_box(lat, lon, PH_BBOX):
        return "ph"
    if taiwan_enabled() and inside_box(lat, lon, TAIWAN_BBOX):
        return "taiwan"
    raise ValueError("Location is outside the currently supported forecast areas.")


def combined_fetch_bbox() -> dict:
    """Minimal rectangle covering both crops; enables only one ECMWF retrieval.

    The combined grid is transient: it is cropped again before publication,
    so Philippine NetCDF size remains unchanged. Data over the ocean between
    the two countries is temporarily retained in memory during processing.
    """
    if not taiwan_enabled():
        return dict(PH_BBOX)
    return {
        "lat_min": min(PH_BBOX["lat_min"], TAIWAN_BBOX["lat_min"]),
        "lat_max": max(PH_BBOX["lat_max"], TAIWAN_BBOX["lat_max"]),
        "lon_min": min(PH_BBOX["lon_min"], TAIWAN_BBOX["lon_min"]),
        "lon_max": max(PH_BBOX["lon_max"], TAIWAN_BBOX["lon_max"]),
    }


def crop_region(ds, bbox: dict):
    """Slice already-cropped xarray dataset; latitude is usually descending.

    Keep a deep copy so the small Taiwan dataset doesn't hold the larger
    decoded parent array in memory while writing.
    """
    lat_values = ds.latitude.values
    descending = bool(lat_values[0] > lat_values[-1])
    lat_slice = slice(bbox["lat_max"], bbox["lat_min"]) if descending else slice(bbox["lat_min"], bbox["lat_max"])
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
    if float(ds.longitude.max()) > 180:
        lon_min, lon_max = lon_min % 360, lon_max % 360
    result = ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    if not result.sizes.get("latitude") or not result.sizes.get("longitude"):
        raise ValueError(f"No ECMWF grid points were found inside {bbox}")
    return result.copy(deep=True)
