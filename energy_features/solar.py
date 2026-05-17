"""Solar geometry and clear-sky irradiance (pvlib).

Optional HDD/CDD in :func:`add_solar_features` delegates to :mod:`energy_features.thermal`.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import pandas as pd
from pvlib import atmosphere, clearsky, irradiance, solarposition

from energy_features.thermal import degree_days_from_temperature

ClearSkyModel = Literal["ineichen", "simplified_solis"]

_SUN_PREFIX = "sun_"
_CS_PREFIX = "cs_"


def _require_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        msg = "index must be a pandas.DatetimeIndex"
        raise TypeError(msg)
    return index


def localize_to_tz(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Interpret or convert timestamps to timezone ``tz`` for site-local solar geometry."""
    idx = _require_datetime_index(index)
    if idx.tz is None:
        return idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    return idx.tz_convert(tz)


def get_solarposition(
    time: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    *,
    altitude: float = 0.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Solar position via :func:`pvlib.solarposition.get_solarposition`.

    Parameters
    ----------
    time
        Timestamps (timezone-aware recommended; use :func:`localize_to_tz` for a target zone).
    latitude, longitude
        Site coordinates in decimal degrees.
    altitude
        Site elevation (meters above sea level).
    **kwargs
        Forwarded to pvlib (e.g. ``pressure``, ``temperature``, ``method``).
    """
    _require_datetime_index(time)
    return solarposition.get_solarposition(time, latitude, longitude, altitude=altitude, **kwargs)


def get_clearsky_irradiance(
    time: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    *,
    model: ClearSkyModel = "ineichen",
    altitude: float = 0.0,
    solar_position: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Clear-sky GHI, DNI, and DHI using Ineichen or simplified SOLIS.

    Parameters
    ----------
    time
        Same timestamps used for solar position (must align with ``solar_position`` if passed).
    latitude, longitude
        Used for Linke turbidity (Ineichen) when ``model == "ineichen"``.
    model
        ``"ineichen"`` or ``"simplified_solis"`` (pvlib implementations).
    altitude
        Site elevation (meters).
    solar_position
        Optional precomputed pvlib solar position frame. If omitted, it is computed here.
    **kwargs
        Ineichen: passed to :func:`pvlib.clearsky.ineichen` (e.g. ``perez_enhancement``).
        Simplified SOLIS: passed to :func:`pvlib.clearsky.simplified_solis`
        (e.g. ``aod700``, ``precipitable_water``, ``pressure``).
    """
    _require_datetime_index(time)
    sol = (
        solar_position
        if solar_position is not None
        else get_solarposition(time, latitude, longitude, altitude=altitude)
    )

    if model == "ineichen":
        linke = clearsky.lookup_linke_turbidity(time, latitude, longitude)
        pressure = atmosphere.alt2pres(altitude)
        rel_am = atmosphere.get_relative_airmass(sol["apparent_zenith"])
        abs_am = atmosphere.get_absolute_airmass(rel_am, pressure)
        dni_extra = irradiance.get_extra_radiation(time)
        return clearsky.ineichen(
            sol["apparent_zenith"],
            abs_am,
            linke,
            altitude,
            dni_extra,
            **kwargs,
        )

    if model == "simplified_solis":
        dni_extra = irradiance.get_extra_radiation(time)
        return clearsky.simplified_solis(
            sol["apparent_elevation"],
            dni_extra=dni_extra,
            **kwargs,
        )

    msg = f"unknown clearsky model: {model!r}"
    raise ValueError(msg)


def add_solar_features(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    tz: str,
    *,
    temp_col: str | None = None,
    degree_day_base_c: float = 18.0,
    clearsky_model: ClearSkyModel = "ineichen",
    altitude_m: float = 0.0,
    solar_columns: tuple[str, ...] = (
        "apparent_zenith",
        "apparent_elevation",
        "azimuth",
    ),
    solar_position_kwargs: dict[str, Any] | None = None,
    clearsky_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Append solar position, clear-sky irradiance, and optionally HDD/CDD columns.

    Uses site-local time ``tz`` for solar geometry (naive indices are localized to ``tz``).

    Column prefixes: ``sun_*`` for geometry, ``cs_*`` for clear-sky irradiance, ``hdd`` / ``cdd``
    when ``temp_col`` is set.
    """
    idx = _require_datetime_index(df.index)
    times_local = localize_to_tz(idx, tz)
    s_kwargs = solar_position_kwargs or {}
    c_kwargs = clearsky_kwargs or {}

    sol_full = get_solarposition(times_local, latitude, longitude, altitude=altitude_m, **s_kwargs)
    missing = [c for c in solar_columns if c not in sol_full.columns]
    if missing:
        msg = f"unknown solar columns requested: {missing}"
        raise KeyError(msg)
    sol = sol_full[list(solar_columns)].add_prefix(_SUN_PREFIX)
    sol.index = idx

    cs = get_clearsky_irradiance(
        times_local,
        latitude,
        longitude,
        model=clearsky_model,
        altitude=altitude_m,
        solar_position=sol_full,
        **c_kwargs,
    ).add_prefix(_CS_PREFIX)
    cs.index = idx

    parts: list[pd.DataFrame] = [df, sol, cs]
    if temp_col is not None:
        if temp_col not in df.columns:
            msg = f"temp_col {temp_col!r} not in dataframe columns"
            raise KeyError(msg)
        dd = degree_days_from_temperature(df[temp_col], base_temperature_c=degree_day_base_c)
        if not dd.index.equals(idx):
            msg = "internal degree-day index mismatch"
            raise AssertionError(msg)
        parts.append(dd)
    elif degree_day_base_c != 18.0:
        warnings.warn(
            "degree_day_base_c has no effect without temp_col",
            stacklevel=2,
        )

    out = pd.concat(parts, axis=1)
    if out.columns.duplicated().any():
        dupes = out.columns[out.columns.duplicated()].tolist()
        msg = f"column collision after merging solar features: {dupes}"
        raise ValueError(msg)
    return out
