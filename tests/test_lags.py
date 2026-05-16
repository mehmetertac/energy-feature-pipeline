"""Tests for lag / rolling features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_features.lags import (
    make_lag_features,
    make_lag_rolling_block,
    make_rolling_features,
)


def test_lag_columns_match_shift(hourly_series: pd.Series) -> None:
    L = make_lag_features(hourly_series, [1, 24])
    name = hourly_series.name or "value"
    pd.testing.assert_series_equal(
        L[f"{name}_lag_1"],
        hourly_series.shift(1),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        L[f"{name}_lag_24"],
        hourly_series.shift(24),
        check_names=False,
    )


def test_lag_negative_raises(hourly_series: pd.Series) -> None:
    with pytest.raises(ValueError, match="positive"):
        make_lag_features(hourly_series, [-1])


def test_lag_two_steps_back_matches_series_shift() -> None:
    idx = pd.date_range("2025-06-01", periods=5, freq="h", tz="UTC")
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx, name="load")
    L = make_lag_features(s, [2])
    assert float(L["load_lag_2"].iloc[2]) == pytest.approx(10.0)
    assert float(L["load_lag_2"].iloc[4]) == pytest.approx(30.0)


def test_rolling_uses_shifted_past_only() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="h", tz="UTC")
    s = pd.Series(np.arange(6, dtype=float), index=idx, name="y")
    R = make_rolling_features(s, windows=[2], stats=("mean",), shift=1, min_periods=1)
    # shifted: nan,0,1,2,3,4 — rolling2 mean at positions:
    # i0 nan, i1 nan?, rolling min_periods 1 on window 2...
    # pandas rolling with min_periods=1: i1 uses only one non-nan?
    # Actually window 2 needs 2 rows for mean?
    # shifted [nan,0]: mean at idx1 = 0
    expected_i2 = (0.0 + 1.0) / 2.0
    assert np.isclose(R.iloc[2, 0], expected_i2)


def test_rolling_shift_negative_raises(hourly_series: pd.Series) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        make_rolling_features(hourly_series, windows=[3], shift=-1)


def test_rolling_shift_zero_raises(hourly_series: pd.Series) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        make_rolling_features(hourly_series, windows=[3], shift=0)


def test_rolling_unknown_stat(hourly_series: pd.Series) -> None:
    with pytest.raises(ValueError, match="Unknown stat"):
        make_rolling_features(hourly_series, windows=[3], stats=("not_a_function",))


def test_min_periods_enforced() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    s = pd.Series(np.arange(10.0), index=idx)
    R = make_rolling_features(s, windows=[3], stats=("mean",), shift=1, min_periods=3)
    col = R.columns[0]
    assert pd.isna(R[col].iloc[:3]).all()
    assert not pd.isna(R[col].iloc[3])


def test_lag_rolling_block_joins(hourly_series: pd.Series) -> None:
    B = make_lag_rolling_block(hourly_series, [1], [24], prefix="x")
    assert "x_lag_1" in B.columns
    assert any(c.startswith("x_roll24_") for c in B.columns)
