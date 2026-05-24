"""Feature builders for renewable energy forecasting (calendar, lags, weather, physics)."""

from energy_features.calendar import make_calendar_features
from energy_features.data_loader import load_opsd_generation, load_opsd_load
from energy_features.lags import (
    make_lag_features,
    make_lag_rolling_block,
    make_rolling_features,
)
from energy_features.solar import (
    add_solar_features,
    get_clearsky_irradiance,
    get_solarposition,
    localize_to_tz,
)
from energy_features.thermal import degree_days_from_temperature
from energy_features.weather import (
    ERA5_MERGE_COLUMNS,
    ERA5Loader,
    convert_time_zone,
    ensure_utc_time,
    extract_era5_point,
    fetch_era5,
    fetch_era5_year,
    interp_to_point,
    open_era5,
    open_era5_range,
    open_era5_year,
    resample_era5,
    sel_nearest_time,
    standardize_era5,
    wind_speed_direction_from_uv,
)
from energy_features.wind import (
    add_wind_features,
    air_density_kg_m3,
    create_reference_model_chain,
    density_corrected_wind_speed,
    extrapolate_wind_speed,
    prepare_modelchain_weather,
    run_reference_turbine_power,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ERA5Loader",
    "ERA5_MERGE_COLUMNS",
    "add_solar_features",
    "add_wind_features",
    "air_density_kg_m3",
    "convert_time_zone",
    "create_reference_model_chain",
    "degree_days_from_temperature",
    "density_corrected_wind_speed",
    "ensure_utc_time",
    "extract_era5_point",
    "extrapolate_wind_speed",
    "fetch_era5",
    "fetch_era5_year",
    "get_clearsky_irradiance",
    "get_solarposition",
    "interp_to_point",
    "localize_to_tz",
    "load_opsd_generation",
    "load_opsd_load",
    "make_calendar_features",
    "make_lag_features",
    "make_lag_rolling_block",
    "make_rolling_features",
    "open_era5",
    "open_era5_range",
    "open_era5_year",
    "prepare_modelchain_weather",
    "resample_era5",
    "run_reference_turbine_power",
    "sel_nearest_time",
    "standardize_era5",
    "wind_speed_direction_from_uv",
]
