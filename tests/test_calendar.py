"""Tests for calendar / holiday features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_features.calendar import make_calendar_features


def test_calendar_aligns_and_no_nan(hourly_series: pd.Series) -> None:
    X = make_calendar_features(hourly_series.index, country="DE")
    assert len(X) == len(hourly_series)
    assert X.index.equals(hourly_series.index)
    assert X.notna().all().all()


def test_calendar_german_christmas_is_holiday() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2023-12-25 12:00:00", tz="Europe/Berlin")])
    X = make_calendar_features(idx, country="DE")
    assert int(X["is_holiday"].iloc[0]) == 1


def test_calendar_plain_weekday_not_holiday_de() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-10 15:00:00", tz="Europe/Berlin")])
    X = make_calendar_features(idx, country="DE")
    assert int(X["is_holiday"].iloc[0]) == 0


def test_calendar_rejects_non_monotonic_index() -> None:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")]
    )
    with pytest.raises(ValueError, match="monotonic"):
        make_calendar_features(idx)


def test_calendar_rejects_duplicates() -> None:
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    idx = pd.DatetimeIndex([ts, ts])
    with pytest.raises(ValueError, match="duplicates"):
        make_calendar_features(idx)


def test_expected_calendar_columns(hourly_series: pd.Series) -> None:
    X = make_calendar_features(hourly_series.index, country="DE", include_cyclical=True)
    for col in (
        "hour",
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_day_of_year",
        "cos_day_of_year",
        "is_holiday",
    ):
        assert col in X.columns


def test_cyclical_unit_circle(hourly_series: pd.Series) -> None:
    X = make_calendar_features(hourly_series.index, country="DE", include_cyclical=True)
    for sh, ch in (("sin_hour", "cos_hour"), ("sin_day_of_year", "cos_day_of_year")):
        mag = np.sqrt(X[sh].to_numpy(dtype=float) ** 2 + X[ch].to_numpy(dtype=float) ** 2)
        assert np.allclose(mag, 1.0, rtol=0, atol=1e-9)


def test_calendar_without_cyclical(hourly_series: pd.Series) -> None:
    X = make_calendar_features(hourly_series.index, country="DE", include_cyclical=False)
    assert not any(c.startswith("sin_") or c.startswith("cos_") for c in X.columns)
