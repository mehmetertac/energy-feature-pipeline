"""Tests for wind shear, air density, and reference-turbine power."""

from __future__ import annotations

import pandas as pd
import pytest
from energy_features.wind import (
    STANDARD_AIR_DENSITY_KG_M3,
    add_wind_features,
    air_density_kg_m3,
    create_reference_model_chain,
    density_corrected_wind_speed,
    extrapolate_wind_speed,
    prepare_modelchain_weather,
    run_reference_turbine_power,
)


def test_extrapolate_wind_speed_scalar_power_law() -> None:
    # 5 m/s at 10 m → hub 100 m with alpha=0.2 → 5 * (10)^0.2
    out = extrapolate_wind_speed(5.0, 10.0, 100.0, alpha=0.2)
    assert out == pytest.approx(5.0 * (10.0**0.2))


def test_extrapolate_wind_speed_series_preserves_index() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    ws = pd.Series([4.0, 5.0, 6.0], index=idx)
    out = extrapolate_wind_speed(ws, 10.0, 80.0, alpha=1.0 / 7.0)
    assert isinstance(out, pd.Series)
    assert out.index.equals(idx)
    assert float(out.iloc[0]) == pytest.approx(4.0 * (8.0 ** (1.0 / 7.0)))


def test_extrapolate_wind_speed_invalid_height() -> None:
    with pytest.raises(ValueError, match="positive heights"):
        extrapolate_wind_speed(5.0, 0.0, 100.0)


def test_air_density_standard_conditions() -> None:
    rho = air_density_kg_m3(101325.0, 288.15)
    assert rho == pytest.approx(STANDARD_AIR_DENSITY_KG_M3, rel=0.01)


def test_air_density_celsius() -> None:
    rho_k = air_density_kg_m3(101325.0, 15.0, temperature_unit="C")
    rho_direct = air_density_kg_m3(101325.0, 288.15)
    assert rho_k == pytest.approx(rho_direct)


def test_density_corrected_wind_speed_at_standard_density() -> None:
    corr = density_corrected_wind_speed(10.0, 101325.0, 288.15)
    assert corr == pytest.approx(10.0, rel=1e-4)


def test_density_corrected_wind_speed_lower_density() -> None:
    rho = air_density_kg_m3(90000.0, 288.15)
    corr = density_corrected_wind_speed(10.0, 90000.0, 288.15)
    expected = 10.0 * (rho / STANDARD_AIR_DENSITY_KG_M3) ** (1.0 / 3.0)
    assert corr == pytest.approx(expected)


def test_prepare_modelchain_weather_multiindex() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "ws10": [5.0, 8.0],
            "t2m": [280.0, 285.0],
            "sp": [101325.0, 101000.0],
        },
        index=idx,
    )
    weather = prepare_modelchain_weather(
        df,
        wind_speed_col="ws10",
        wind_speed_height_m=10.0,
        temperature_col="t2m",
        pressure_col="sp",
    )
    assert isinstance(weather.columns, pd.MultiIndex)
    assert ("wind_speed", 10.0) in weather.columns


def test_run_reference_turbine_power_positive() -> None:
    idx = pd.date_range("2020-06-01 12:00", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "ws10": [3.0, 6.0, 10.0, 14.0],
            "t2m": [285.0, 285.0, 285.0, 285.0],
            "sp": [101325.0, 101325.0, 101325.0, 101325.0],
        },
        index=idx,
    )
    weather = prepare_modelchain_weather(
        df,
        wind_speed_col="ws10",
        wind_speed_height_m=10.0,
        temperature_col="t2m",
        pressure_col="sp",
    )
    power = run_reference_turbine_power(weather)
    assert power.name == "ref_turbine_power_w"
    assert (power >= 0.0).all()
    assert float(power.iloc[-1]) > float(power.iloc[0])


def test_create_reference_model_chain_default_turbine() -> None:
    mc = create_reference_model_chain()
    assert mc.power_plant.turbine_type == "E-126/4200"


def test_add_wind_features_hub_wind_only() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"ws10": [5.0, 6.0, 7.0]}, index=idx)
    out = add_wind_features(df, wind_speed_col="ws10", wind_speed_height_m=10.0)
    assert "wind_speed_hub_mps" in out.columns
    assert "ref_turbine_power_w" not in out.columns
    assert out.index.equals(idx)


def test_add_wind_features_with_weather_and_power() -> None:
    idx = pd.date_range("2020-06-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "ws10": [4.0, 8.0, 12.0],
            "t2m": [285.0, 285.0, 285.0],
            "sp": [101325.0, 101325.0, 101325.0],
        },
        index=idx,
    )
    out = add_wind_features(
        df,
        wind_speed_col="ws10",
        wind_speed_height_m=10.0,
        pressure_col="sp",
        temperature_col="t2m",
    )
    for col in (
        "wind_speed_hub_mps",
        "air_density_kg_m3",
        "wind_speed_hub_density_corr_mps",
        "ref_turbine_power_w",
    ):
        assert col in out.columns
    assert (out["ref_turbine_power_w"] >= 0.0).all()


def test_add_wind_features_requires_datetime_index() -> None:
    df = pd.DataFrame({"ws10": [5.0]}, index=pd.Index([0]))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        add_wind_features(df, wind_speed_col="ws10", wind_speed_height_m=10.0)


def test_package_exports_wind_helpers() -> None:
    import energy_features as ef

    for name in (
        "extrapolate_wind_speed",
        "air_density_kg_m3",
        "density_corrected_wind_speed",
        "run_reference_turbine_power",
        "add_wind_features",
    ):
        assert hasattr(ef, name), name
