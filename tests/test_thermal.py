"""Tests for heating / cooling degree metrics."""

from __future__ import annotations

import pandas as pd
from energy_features.thermal import degree_days_from_temperature


def test_hdd_cdd_balance_18c() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    t = pd.Series([15.0, 22.0], index=idx)
    dd = degree_days_from_temperature(t, base_temperature_c=18.0)
    assert float(dd["hdd"].iloc[0]) == 3.0
    assert float(dd["cdd"].iloc[0]) == 0.0
    assert float(dd["hdd"].iloc[1]) == 0.0
    assert float(dd["cdd"].iloc[1]) == 4.0


def test_hdd_cdd_default_base_18c() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-06-01", tz="UTC")])
    dd = degree_days_from_temperature(pd.Series([18.0], index=idx))
    assert float(dd["hdd"].iloc[0]) == 0.0
    assert float(dd["cdd"].iloc[0]) == 0.0
