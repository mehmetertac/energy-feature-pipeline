# Data sources

Cross-links: [README](../README.md), [methodology](methodology.md), [decisions](decisions.md).

## Electricity generation — OPSD

- **Provider:** [Open Power System Data](https://open-power-system-data.org/) — `time_series` package (60 min single-index CSV).
- **Loader:** [`energy_features/data_loader.py`](../energy_features/data_loader.py) — `load_opsd_generation`.
- **Resolution order:** explicit path → `ENERGY_FP_OPSD_PATH` → `data/raw/opsd_time_series_60min.parquet` → HTTP download (large; cache strongly recommended).
- **Columns used (Germany example):** `DE_solar_generation_actual`, `DE_wind_onshore_generation_actual`, `DE_wind_offshore_generation_actual`.

## Electricity load — ENTSO-E (sibling project)

- **Artifact:** Parquet from [energy-ts-fundamentals](../energy-ts-fundamentals/README.md), e.g. `data/raw/de_lu_load_hourly.parquet`.
- **Loader:** `load_opsd_load` / `resolve_load_parquet_path` — respects `ENERGY_TS_LOAD_PARQUET`.

## Reanalysis — ERA5 (CDS)

- **API:** Copernicus CDS — [`cdsapi`](https://pypi.org/project/cdsapi/).
- **Credential file:** `~/.cdsapirc` or env `CDSAPI_URL` / `CDSAPI_KEY` (see [.env.example](../.env.example)); **never commit** keys.
- **Downloader:** [`fetch_era5_year`](../energy_features/weather.py) — twelve monthly hourly NetCDF files in `data/raw/era5_{year}_de_bbox5deg_{mm}.nc` (default: **2019**, **5°×5°** box around **52.5°N, 13.4°E**, six variables: 2 m temperature, 10 m u/v wind, surface solar radiation downwards, total cloud cover, surface pressure). Monthly chunks stay under CDS cost limits.
- **Licence:** accept the [ERA5 single-levels licence](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences) on the CDS website before the first download.
- **CLI:** `py scripts/download_era5.py` (skips months already on disk unless `force_download=True`).
- **xarray helpers:** [`open_era5`](../energy_features/weather.py), [`sel_nearest_time`](../energy_features/weather.py), [`interp_to_point`](../energy_features/weather.py), [`wind_speed_direction_from_uv`](../energy_features/weather.py), [`resample_era5`](../energy_features/weather.py), [`extract_era5_point`](../energy_features/weather.py) — see [methodology](methodology.md).
- **Merge with generation:** [`ERA5Loader`](../energy_features/weather.py) — ``loader.load(lat, lon, start, end)`` → tidy hourly frame (:data:`ERA5_MERGE_COLUMNS`, default tz ``Europe/Berlin``).

## Numerical weather prediction (NWP)

- **Herbie:** [`herbie-data`](https://pypi.org/project/herbie-data/) for GRIB subsets (e.g. HRRR, GFS — pick one consistent product per study).
- **Planned module:** same `weather.py` stub; document chosen model grid + variables in [decisions.md](decisions.md) when fixed.
