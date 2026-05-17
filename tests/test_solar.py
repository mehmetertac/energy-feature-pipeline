"""Tests for solar geometry, clear-sky irradiance, and degree days."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_features.solar import (
    add_solar_features,
    get_clearsky_irradiance,
    get_solarposition,
    localize_to_tz,
)


def test_localize_naive_to_berlin() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2020-06-01 12:00:00")])
    loc = localize_to_tz(idx, "Europe/Berlin")
    assert loc.tz is not None
    assert str(loc.tz) in {"Europe/Berlin"}


def test_get_solarposition_noon_berlin_elevated() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2020-06-01 12:00:00", tz="Europe/Berlin")])
    sp = get_solarposition(idx, 52.52, 13.405)
    assert "apparent_zenith" in sp.columns
    # Near local solar noon in summer, sun is high (low zenith)
    assert float(sp["apparent_zenith"].iloc[0]) < 40.0


@pytest.mark.parametrize("model", ["ineichen", "simplified_solis"])
def test_clearsky_daylight_positive(model: str) -> None:
    idx = pd.date_range("2020-06-01 08:00", periods=6, freq="h", tz="Europe/Berlin")
    cs = get_clearsky_irradiance(idx, 52.5, 13.4, model=model)  # type: ignore[arg-type]
    assert set(cs.columns) == {"ghi", "dni", "dhi"}
    assert (cs["ghi"] > 0.0).all()


def test_add_solar_features_columns_and_align() -> None:
    idx = pd.date_range("2020-06-01", periods=12, freq="h", tz="UTC")
    df = pd.DataFrame({"load": np.linspace(1.0, 12.0, len(idx)), "t_c": 20.0}, index=idx)
    out = add_solar_features(df, 52.5, 13.4, "Europe/Berlin", temp_col="t_c")
    assert out.index.equals(idx)
    for col in (
        "sun_apparent_zenith",
        "sun_apparent_elevation",
        "sun_azimuth",
        "cs_ghi",
        "cs_dni",
        "cs_dhi",
        "hdd",
        "cdd",
    ):
        assert col in out.columns
    assert (out["cdd"] > 0.0).all()


def test_clearsky_unknown_model() -> None:
    idx = pd.date_range("2020-06-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="unknown clearsky"):
        get_clearsky_irradiance(idx, 50.0, 10.0, model="not_a_model")  # type: ignore[arg-type]


def test_add_solar_features_requires_datetime_index() -> None:
    df = pd.DataFrame({"a": [1]}, index=pd.Index([0]))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        add_solar_features(df, 50.0, 10.0, "UTC")


def test_add_solar_features_unknown_solar_column() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
    with pytest.raises(KeyError, match="unknown solar columns"):
        add_solar_features(df, 50.0, 10.0, "UTC", solar_columns=("not_a_column",))


def test_package_exports_solar_helpers() -> None:
    import energy_features as ef

    for name in (
        "add_solar_features",
        "get_solarposition",
        "get_clearsky_irradiance",
        "localize_to_tz",
    ):
        assert hasattr(ef, name), name


def test_package_exports_thermal() -> None:
    import energy_features as ef

    assert hasattr(ef, "degree_days_from_temperature")
