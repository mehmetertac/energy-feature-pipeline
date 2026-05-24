"""Tests for ERA5 xarray helpers (open, select, interp, wind, resample, time zones)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from energy_features.weather import (
    ERA5_MERGE_COLUMNS,
    ERA5Loader,
    convert_time_zone,
    extract_era5_point,
    interp_to_point,
    open_era5,
    resample_era5,
    sel_nearest_time,
    standardize_era5,
    wind_speed_direction_from_uv,
)


def _synthetic_era5(*, n_time: int = 24) -> xr.Dataset:
    times = pd.date_range("2019-06-01", periods=n_time, freq="h", tz="UTC")
    lat = np.linspace(51.0, 54.0, 4)
    lon = np.linspace(11.0, 16.0, 5)
    shape = (n_time, 4, 5)
    coords = {"valid_time": times.tz_localize(None), "latitude": lat, "longitude": lon}
    dims = ("valid_time", "latitude", "longitude")
    return xr.Dataset(
        {
            "u10": xr.DataArray(np.full(shape, 3.0), coords=coords, dims=dims),
            "v10": xr.DataArray(np.full(shape, 4.0), coords=coords, dims=dims),
            "t2m": xr.DataArray(np.full(shape, 290.0), coords=coords, dims=dims),
            "sp": xr.DataArray(np.full(shape, 101325.0), coords=coords, dims=dims),
            "tcc": xr.DataArray(np.full(shape, 0.5), coords=coords, dims=dims),
            "ssrd": xr.DataArray(np.full(shape, 3600.0 * 200.0), coords=coords, dims=dims),
        }
    )


def test_standardize_era5_renames_time_and_variables() -> None:
    ds = _synthetic_era5()
    out = standardize_era5(ds)
    assert "time" in out.coords
    assert "valid_time" not in out.coords
    assert "10m_u_component_of_wind" in out.data_vars
    assert out["time"].attrs.get("timezone") == "UTC"


def test_sel_nearest_time() -> None:
    ds = standardize_era5(_synthetic_era5())
    picked = sel_nearest_time(ds, "2019-06-01T12:20:00")
    assert pd.Timestamp(picked["time"].values).hour == 12


def test_interp_to_point() -> None:
    ds = standardize_era5(_synthetic_era5())
    point = interp_to_point(ds, latitude=52.5, longitude=13.4)
    assert "latitude" not in point.dims
    assert float(point["10m_u_component_of_wind"].isel(time=0).values) == pytest.approx(3.0)


def test_wind_speed_direction_from_uv() -> None:
    speed, direction = wind_speed_direction_from_uv(3.0, 4.0)
    assert speed == pytest.approx(5.0)
    assert direction == pytest.approx(216.869898, rel=1e-5)

    speed_east, direction_east = wind_speed_direction_from_uv(1.0, 0.0)
    assert speed_east == pytest.approx(1.0)
    assert direction_east == pytest.approx(270.0)


def test_convert_time_zone_berlin() -> None:
    ds = standardize_era5(_synthetic_era5(n_time=1))
    local = convert_time_zone(ds, "Europe/Berlin")
    assert local["time"].attrs.get("timezone") == "Europe/Berlin"
    assert pd.Timestamp(local["time"].values[0]).hour == 2  # 00 UTC → 02 CEST


def test_resample_era5_daily_mean() -> None:
    ds = standardize_era5(_synthetic_era5())
    daily = resample_era5(ds, "1D", method="mean")
    assert daily.sizes["time"] == 1
    assert float(daily["2m_temperature"].mean()) == pytest.approx(290.0)


def test_extract_era5_point_dataframe() -> None:
    ds = standardize_era5(_synthetic_era5())
    df = extract_era5_point(ds, 52.5, 13.4, tz="Europe/Berlin")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "Europe/Berlin"
    assert "wind_speed_10m" in df.columns
    assert "wind_direction_10m" in df.columns
    assert float(df["wind_speed_10m"].iloc[0]) == pytest.approx(5.0)


@pytest.mark.integration
def test_open_era5_real_month() -> None:
    path = Path("data/raw/era5_2019_de_bbox5deg_01.nc")
    if not path.is_file():
        pytest.skip("ERA5 monthly file not cached locally")

    ds = open_era5(path)
    assert ds.sizes["time"] == 744
    assert "2m_temperature" in ds.data_vars
    assert ds["time"].attrs.get("timezone") == "UTC"

    point = extract_era5_point(ds, 52.5, 13.4, tz="UTC")
    assert len(point) == 744
    assert point["2m_temperature"].between(250.0, 310.0).all()


def test_era5_loader_merge_columns_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = standardize_era5(_synthetic_era5())

    def fake_open_era5_range(*args, **kwargs):
        return ds

    monkeypatch.setattr("energy_features.weather.open_era5_range", fake_open_era5_range)
    loader = ERA5Loader(tz="UTC")
    df = loader.load(52.5, 13.4, "2019-06-01", "2019-06-01")

    assert list(df.columns) == list(ERA5_MERGE_COLUMNS)
    assert str(df.index.tz) == "UTC"
    assert len(df) == 24
    assert float(df["wind_speed_10m"].iloc[0]) == pytest.approx(5.0)
    assert float(df["t2m_c"].iloc[0]) == pytest.approx(float(df["t2m_k"].iloc[0]) - 273.15)


@pytest.mark.integration
def test_era5_loader_real_january_week() -> None:
    path = Path("data/raw/era5_2019_de_bbox5deg_01.nc")
    if not path.is_file():
        pytest.skip("ERA5 monthly file not cached locally")

    loader = ERA5Loader()
    df = loader.load(52.5, 13.4, "2019-01-01", "2019-01-07")
    # ERA5 is UTC-hourly; Jan 1 00:00 Europe/Berlin needs Dec-31 UTC (prior month file).
    assert len(df) == 24 * 7 - 1
    assert df.index[0].hour == 1
    assert set(ERA5_MERGE_COLUMNS).issubset(df.columns)
    assert str(df.index.tz) == "Europe/Berlin"
    assert df.index.is_monotonic_increasing
