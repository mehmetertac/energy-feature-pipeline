# Methodology

Cross-links: [README](../README.md), [data_sources](data_sources.md), [decisions](decisions.md).

## Task 1 — Calendar & lag baselines

### Feature blocks

1. **Calendar / holiday** — [`make_calendar_features`](../energy_features/calendar.py): `hour`, `day_of_week`, `month`, ISO `week_of_year`, `is_weekend`; optional sin/cos encodings for **hour** and **day-of-year**; `holidays`-based flags for `country` (default **DE**), plus days to / since nearest holiday.
2. **Lag / rolling** — [`make_lag_features`](../energy_features/lags.py), [`make_rolling_features`](../energy_features/lags.py): **feature-engine** `LagFeatures` / `WindowFeatures` with default windows **24 h** and **168 h** and stats **mean, std, min, max**. `WindowFeatures(..., periods=shift)` with default **`shift=1`** matches the library's causal forecasting semantics (no future values; contemporaneous target excluded from the rolling slice at \(t\)). [`make_lag_rolling_block`](../energy_features/lags.py) wires both via a sklearn `Pipeline`.
3. **Solar** — [`add_solar_features`](../energy_features/solar.py): pvlib [`get_solarposition`](../energy_features/solar.py) plus clear-sky **GHI / DNI / DHI** (`ineichen` or `simplified_solis`); optional **HDD / CDD** columns via [`degree_days_from_temperature`](../energy_features/thermal.py) when a dry-bulb temperature column is supplied.
4. **Wind** — [`extrapolate_wind_speed`](../energy_features/wind.py): power-law shear (default **α = 1/7**) from measurement height to hub; [`air_density_kg_m3`](../energy_features/wind.py) / [`density_corrected_wind_speed`](../energy_features/wind.py) from pressure (Pa) and temperature (K or °C); [`run_reference_turbine_power`](../energy_features/wind.py) via windpowerlib **ModelChain** on **Enercon E-126/4200** (135 m hub). High-level [`add_wind_features`](../energy_features/wind.py) merges hub wind, density, and reference power when ERA5/NWP columns are joined.
5. **Thermal (HDD/CDD)** — [`degree_days_from_temperature`](../energy_features/thermal.py): per timestep **HDD** = max(0, T_base − T), **CDD** = max(0, T − T_base) in °C with default **T_base = 18** — use after joining **ERA5 / NWP** (or other) temperature.

### Baselines & lift

- Fit **LightGBMRegressor** (or sklearn baseline for debugging) on **train**, evaluate on **held-out tail** (e.g. last 90 days — configurable).
- Compare **calendar-only** \(X_{\text{cal}}\) vs **calendar + lags/rolls** \(X_{\text{cal+lag}}\).
- Report **lift %** per technology:
  \[
    \text{lift}_\text{MAE} = \frac{\text{MAE}_{\text{cal}} - \text{MAE}_{\text{cal+lag}}}{\text{MAE}_{\text{cal}}}
  \]
  (same for RMSE / sMAPE if desired).

### Honesty checks

- Strict **time ordering** — no random split across time for operational forecasts.
- Inspect **residual ACF** after calendar+lags (notebook) — persistent structure motivates **weather** features next.

## Later milestones

### ERA5 gridded fundamentals — [`weather.py`](../energy_features/weather.py)

1. **Open / concat** — `open_era5` (handles CDS zip bundles), `open_era5_year`; `standardize_era5` renames `valid_time` → **`time`** (UTC) and CDS short names → long names.
2. **Select / interpolate** — `sel_nearest_time(ds, …)` wraps `.sel(time=…, method="nearest")`; `interp_to_point(ds, lat, lon)` wraps `.interp(latitude=…, longitude=…)`.
3. **Wind** — `wind_speed_direction_from_uv(u, v)` → speed (m/s) and meteorological direction (° from north, wind **from**).
4. **Time** — ERA5 is UTC; `ensure_utc_time`, `convert_time_zone(ds, "Europe/Berlin")`, and :class:`ERA5Loader` for a tz-aware pandas frame aligned with OPSD.
5. **Resample** — `resample_era5(ds, "1D")`; hourly `surface_solar_radiation_downwards` (J/m²) converted to W/m² before averaging.
6. **Merge-ready loader** — :class:`ERA5Loader` → tidy hourly ``DataFrame`` at ``(lat, lon)`` with ``t2m_k``, ``wind_speed_10m``, ``surface_pressure_pa``, etc. (see :data:`ERA5_MERGE_COLUMNS`).

### NWP forecast (GFS via Herbie) — [`weather.py`](../energy_features/weather.py)

1. **Fetch** — :func:`fetch_gfs_forecast` / :func:`fetch_nwp`: one **init time**, multiple **lead hours** (default 6 / 12 / 24 / 48); Herbie subsets GRIB2 from NOAA open data.
2. **Standardize** — :func:`standardize_nwp`: cfgrib short names → ERA5-like long names; longitude wrapped to [-180, 180].
3. **Point extract** — :func:`extract_nwp_forecast_point` / :func:`nwp_forecast_point_from_dataset`: bilinear ``interp`` at the ERA5 study point (52.5°N, 13.4°E); output includes ``init_time``, ``valid_time``, ``lead_hours`` plus weather columns in :data:`NWP_FORECAST_MERGE_COLUMNS`.
4. **Reanalysis vs forecast** — ERA5 is hindcast truth for feature joins; GFS rows are **forecasts** (what was available at ``init_time`` for each lead). Do not treat GFS as observed weather when scoring against generation actuals unless emulating an operational setting.
5. **Demo notebook** — [`03_era5_vs_nwp.ipynb`](../notebooks/03_era5_vs_nwp.ipynb): one-week ERA5 vs GFS plots and lead-time MAE weather skill baseline.

- **Weather vs model (next):** align ERA5 / NWP leads with generation targets; ablations: calendar-only → +weather → +physics (+ wind physics block above).
