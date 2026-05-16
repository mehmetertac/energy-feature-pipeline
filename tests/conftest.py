"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_series() -> pd.Series:
    """Synthetic hourly series with daily + weekly seasonality (UTC)."""
    idx = pd.date_range("2024-01-01", periods=30 * 24, freq="h", tz="UTC")
    t = np.arange(len(idx), dtype=float)
    y = 100.0 + 10.0 * np.sin(2.0 * np.pi * t / 24.0) + 5.0 * np.sin(2.0 * np.pi * t / (24.0 * 7.0))
    return pd.Series(y, index=idx, name="gen")
