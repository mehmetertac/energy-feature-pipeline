"""Wind shear extrapolation, air-density correction, and reference-turbine power (windpowerlib)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from windpowerlib import ModelChain, WindTurbine

# Dry air, J/(kg·K) — ideal-gas density at hub conditions.
DRY_AIR_GAS_CONSTANT_J_KG_K = 287.058
STANDARD_AIR_DENSITY_KG_M3 = 1.225
DEFAULT_SHEAR_EXPONENT = 1.0 / 7.0

REFERENCE_TURBINE_TYPE = "E-126/4200"
REFERENCE_HUB_HEIGHT_M = 135.0

TemperatureUnit = Literal["K", "C"]


def extrapolate_wind_speed(
    ws: float | pd.Series | np.ndarray,
    z_from: float,
    z_to: float,
    alpha: float = DEFAULT_SHEAR_EXPONENT,
) -> float | pd.Series | np.ndarray:
    """Extrapolate wind speed vertically with a power-law wind profile.

    .. math::

        v(z_to) = v(z_from) * (z_to / z_from)^alpha

    Default ``alpha = 1/7`` is a common open-terrain shear exponent.
    """
    if z_from <= 0.0 or z_to <= 0.0:
        msg = "z_from and z_to must be positive heights (m)"
        raise ValueError(msg)

    ratio = (float(z_to) / float(z_from)) ** float(alpha)
    if isinstance(ws, pd.Series):
        return ws.astype(float) * ratio
    if isinstance(ws, np.ndarray):
        return ws.astype(float) * ratio
    return float(ws) * ratio


def _as_kelvin(
    temperature: float | pd.Series | np.ndarray, unit: TemperatureUnit
) -> np.ndarray | pd.Series | float:
    if unit == "K":
        if isinstance(temperature, pd.Series):
            return temperature.astype(float)
        if isinstance(temperature, np.ndarray):
            return temperature.astype(float)
        return float(temperature)
    offset = 273.15
    if isinstance(temperature, pd.Series):
        return temperature.astype(float) + offset
    if isinstance(temperature, np.ndarray):
        return temperature.astype(float) + offset
    return float(temperature) + offset


def air_density_kg_m3(
    pressure_pa: float | pd.Series | np.ndarray,
    temperature: float | pd.Series | np.ndarray,
    *,
    temperature_unit: TemperatureUnit = "K",
) -> float | pd.Series | np.ndarray:
    """Air density (kg/m³) from ideal gas law using pressure and dry-bulb temperature.

    Parameters
    ----------
    pressure_pa
        Static pressure in pascals.
    temperature
        Dry-bulb temperature; kelvin by default (``temperature_unit="K"``) or Celsius
        when ``temperature_unit="C"``.
    """
    t_k = _as_kelvin(temperature, temperature_unit)
    if isinstance(pressure_pa, pd.Series):
        p = pressure_pa.astype(float)
        t = t_k if isinstance(t_k, pd.Series) else float(t_k)
        return p / (DRY_AIR_GAS_CONSTANT_J_KG_K * t)
    if isinstance(pressure_pa, np.ndarray):
        p = pressure_pa.astype(float)
        t = np.asarray(t_k, dtype=float)
        return p / (DRY_AIR_GAS_CONSTANT_J_KG_K * t)
    p = float(pressure_pa)
    t = float(t_k)
    return p / (DRY_AIR_GAS_CONSTANT_J_KG_K * t)


def density_corrected_wind_speed(
    wind_speed: float | pd.Series | np.ndarray,
    pressure_pa: float | pd.Series | np.ndarray,
    temperature: float | pd.Series | np.ndarray,
    *,
    temperature_unit: TemperatureUnit = "K",
    rho_ref: float = STANDARD_AIR_DENSITY_KG_M3,
) -> float | pd.Series | np.ndarray:
    """Adjust hub-height wind speed for non-standard air density (IEC-style).

    Returns ``wind_speed * (rho / rho_ref)^(1/3)`` so power-curve lookups at ``rho_ref``
    remain consistent when site density differs from standard conditions.
    """
    rho = air_density_kg_m3(pressure_pa, temperature, temperature_unit=temperature_unit)
    factor = (rho / float(rho_ref)) ** (1.0 / 3.0)
    if isinstance(wind_speed, pd.Series):
        return wind_speed.astype(float) * factor
    if isinstance(wind_speed, np.ndarray):
        return wind_speed.astype(float) * factor
    return float(wind_speed) * float(factor)


def prepare_modelchain_weather(
    df: pd.DataFrame,
    *,
    wind_speed_col: str,
    wind_speed_height_m: float,
    temperature_col: str,
    temperature_height_m: float = 2.0,
    temperature_unit: TemperatureUnit = "K",
    pressure_col: str,
    pressure_height_m: float = 0.0,
    roughness_length: float | str = 0.15,
) -> pd.DataFrame:
    """Build a windpowerlib weather frame (MultiIndex columns) from flat dataframe columns."""
    idx = df.index
    wind = df[wind_speed_col].astype(float)
    temp = _as_kelvin(df[temperature_col], temperature_unit)
    if isinstance(temp, pd.Series):
        temp = temp.astype(float)
    pressure = df[pressure_col].astype(float)

    if isinstance(roughness_length, str):
        roughness = df[roughness_length].astype(float)
    else:
        roughness = pd.Series(float(roughness_length), index=idx, dtype=float)

    return pd.DataFrame(
        {
            ("wind_speed", float(wind_speed_height_m)): wind,
            ("temperature", float(temperature_height_m)): temp,
            ("pressure", float(pressure_height_m)): pressure,
            ("roughness_length", 0.0): roughness,
        },
        index=idx,
    )


def create_reference_model_chain(
    *,
    turbine_type: str = REFERENCE_TURBINE_TYPE,
    hub_height: float | None = None,
    density_correction: bool = True,
    wind_speed_model: str = "logarithmic",
    **kwargs: Any,
) -> ModelChain:
    """Return a :class:`windpowerlib.ModelChain` for a reference onshore turbine."""
    turbine = WindTurbine(
        turbine_type=turbine_type,
        hub_height=hub_height if hub_height is not None else REFERENCE_HUB_HEIGHT_M,
    )
    return ModelChain(
        turbine,
        wind_speed_model=wind_speed_model,
        density_correction=density_correction,
        **kwargs,
    )


def run_reference_turbine_power(
    weather: pd.DataFrame,
    *,
    turbine_type: str = REFERENCE_TURBINE_TYPE,
    hub_height: float | None = None,
    density_correction: bool = True,
    model_chain: ModelChain | None = None,
    **model_chain_kwargs: Any,
) -> pd.Series:
    """Run windpowerlib ``ModelChain`` for the reference turbine and return power (W).

    ``weather`` must use windpowerlib's MultiIndex column layout — see
    :func:`prepare_modelchain_weather`.
    """
    mc = model_chain or create_reference_model_chain(
        turbine_type=turbine_type,
        hub_height=hub_height,
        density_correction=density_correction,
        **model_chain_kwargs,
    )
    mc.run_model(weather)
    return mc.power_output.rename("ref_turbine_power_w")


def add_wind_features(
    df: pd.DataFrame,
    *,
    wind_speed_col: str,
    wind_speed_height_m: float,
    hub_height_m: float = REFERENCE_HUB_HEIGHT_M,
    shear_exponent: float = DEFAULT_SHEAR_EXPONENT,
    pressure_col: str | None = None,
    temperature_col: str | None = None,
    temperature_unit: TemperatureUnit = "K",
    roughness_length: float | str = 0.15,
    turbine_type: str = REFERENCE_TURBINE_TYPE,
    density_correction: bool = True,
) -> pd.DataFrame:
    """Append hub-height wind, density, and reference-turbine power columns.

    Requires ``wind_speed_col``. When ``pressure_col`` and ``temperature_col`` are set,
    adds air density, density-corrected hub wind, and Enercon E-126 reference power.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = "index must be a pandas.DatetimeIndex"
        raise TypeError(msg)
    if wind_speed_col not in df.columns:
        msg = f"wind_speed_col {wind_speed_col!r} not in dataframe columns"
        raise KeyError(msg)

    ws_hub = extrapolate_wind_speed(
        df[wind_speed_col],
        wind_speed_height_m,
        hub_height_m,
        alpha=shear_exponent,
    )
    parts: list[pd.DataFrame] = [
        df,
        pd.DataFrame({"wind_speed_hub_mps": ws_hub}, index=df.index),
    ]

    if pressure_col is not None and temperature_col is not None:
        for col in (pressure_col, temperature_col):
            if col not in df.columns:
                msg = f"column {col!r} not in dataframe columns"
                raise KeyError(msg)
        rho = air_density_kg_m3(
            df[pressure_col],
            df[temperature_col],
            temperature_unit=temperature_unit,
        )
        ws_corr = density_corrected_wind_speed(
            ws_hub,
            df[pressure_col],
            df[temperature_col],
            temperature_unit=temperature_unit,
        )
        weather = prepare_modelchain_weather(
            df,
            wind_speed_col=wind_speed_col,
            wind_speed_height_m=wind_speed_height_m,
            temperature_col=temperature_col,
            temperature_unit=temperature_unit,
            pressure_col=pressure_col,
            roughness_length=roughness_length,
        )
        power = run_reference_turbine_power(
            weather,
            turbine_type=turbine_type,
            hub_height=hub_height_m,
            density_correction=density_correction,
        )
        parts.append(
            pd.DataFrame(
                {
                    "air_density_kg_m3": rho,
                    "wind_speed_hub_density_corr_mps": ws_corr,
                    "ref_turbine_power_w": power,
                },
                index=df.index,
            )
        )

    out = pd.concat(parts, axis=1)
    if out.columns.duplicated().any():
        dupes = out.columns[out.columns.duplicated()].tolist()
        msg = f"column collision after merging wind features: {dupes}"
        raise ValueError(msg)
    return out
