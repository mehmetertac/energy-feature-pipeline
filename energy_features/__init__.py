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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "add_solar_features",
    "degree_days_from_temperature",
    "get_clearsky_irradiance",
    "get_solarposition",
    "localize_to_tz",
    "load_opsd_generation",
    "load_opsd_load",
    "make_calendar_features",
    "make_lag_features",
    "make_lag_rolling_block",
    "make_rolling_features",
]
