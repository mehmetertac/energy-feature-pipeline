"""Heating / cooling degree metrics from dry-bulb temperature (e.g. after ERA5 joins)."""

from __future__ import annotations

import pandas as pd


def degree_days_from_temperature(
    temperature_c: pd.Series,
    *,
    base_temperature_c: float = 18.0,
) -> pd.DataFrame:
    """Per-timestep heating and cooling degree *amounts* relative to a balance temperature (°C).

    These are the standard clipped imbalances (not necessarily aggregated to calendar days):

    - **HDD** (heating): ``max(0, T_base - T)``
    - **CDD** (cooling): ``max(0, T - T_base)``

    ``T`` is dry-bulb temperature in °C; default ``T_base`` is **18** °C.
    """
    t = temperature_c.astype(float)
    base = float(base_temperature_c)
    hdd = (base - t).clip(lower=0.0)
    cdd = (t - base).clip(lower=0.0)
    return pd.DataFrame({"hdd": hdd, "cdd": cdd}, index=temperature_c.index)
