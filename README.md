# Rainfall Exceedance Forecast (Stage 1)

Probability that 24-hour accumulated rainfall exceeds a chosen threshold,
for Guiguinto, Bulacan, Philippines (14.842279, 120.859681), using ECMWF's
free ENS Open Data.

## How it works

- `ecmwf.py` uses the official `ecmwf-opendata` package to download the
  precomputed **probability-of-exceedance** fields (`tpg1` ... `tpg100`)
  from ECMWF's ensemble forecast — no need to process 50 raw ensemble
  members yourself.
- The GRIB2 response is opened with `xarray`/`cfgrib`, and the value at
  the nearest grid point (ECMWF ENS is ~0.25° resolution, so up to ~25 km
  away in the worst case) to the requested coordinates is extracted.
- `app.py` is the Streamlit front end: pick a threshold and forecast
  range, click **Get forecast**, see a probability timeline in 12-hour
  steps out to your chosen lead time (max 15 days).

## Available thresholds

1, 5, 10, 20, 25, 50, 100 mm (24h accumulated). These are the only
thresholds ECMWF publishes pre-computed — anything in between would
require pulling the raw 50-member ensemble and computing the exceedance
fraction yourself, which is a heavier fetch (future stage).

## Run locally

```bash
pip install -r requirements.txt
# cfgrib also needs the eccodes system library:
#   macOS:  brew install eccodes
#   Ubuntu: sudo apt install libeccodes0 libeccodes-tools
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On share.streamlit.io, point a new app at `app.py` in that repo.
3. `packages.txt` (apt deps) and `requirements.txt` (pip deps) are read
   automatically during the build — no extra config needed.

## Notes / known limitations

- ECMWF's open-data forecast run isn't published instantly — the 00Z
  run typically lands ~7-8 hours later. The client automatically falls
  back to the most recent *complete* run, so early in the cycle you may
  get the previous run's data.
- Results are cached for 3 hours in the app to avoid re-downloading on
  every UI interaction (ENS updates 4x/day: 00/06/12/18 UTC).
- This queries a single point per request. Precomputing many
  locations/thresholds on a schedule (e.g., via GitHub Actions) is the
  natural next stage rather than fetching on every user click.

## Next stages

- Add GEFS (NOAA) alongside ECMWF for model comparison.
- Add ICON-EPS as a third model.
- Add arbitrary thresholds (50, 75, 90 mm, etc.) via raw ensemble members.
- Move to a scheduled GitHub Action that precomputes results into a small
  JSON/SQLite file, with Streamlit just reading and displaying it.
