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
- **Credential file:** `~/.cdsapirc` (see [.env.example](../.env.example)); **never commit** keys.
- **Planned module:** [`energy_features/weather.py`](../energy_features/weather.py) (stub today).

## Numerical weather prediction (NWP)

- **Herbie:** [`herbie-data`](https://pypi.org/project/herbie-data/) for GRIB subsets (e.g. HRRR, GFS — pick one consistent product per study).
- **Planned module:** same `weather.py` stub; document chosen model grid + variables in [decisions.md](decisions.md) when fixed.
