"""Tests for OPSD loader (local parquet + mocked HTTP)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import requests
from energy_features.data_loader import (
    DEFAULT_PARQUET_NAME,
    load_opsd_generation,
    resolve_opsd_parquet_path,
)


def test_resolve_opsd_prefers_explicit_local(tmp_path: Path) -> None:
    p = tmp_path / "custom.parquet"
    df = pd.DataFrame(
        {"DE_solar_generation_actual": [1.0]}, index=[pd.Timestamp("2020-01-01", tz="UTC")]
    )
    df.to_parquet(p)

    got = resolve_opsd_parquet_path(local_path=p, cache_dir=tmp_path)
    assert got == p.resolve()


def test_load_from_local_parquet(tmp_path: Path) -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC")])
    df = pd.DataFrame(
        {
            "DE_solar_generation_actual": [1.0],
            "DE_wind_onshore_generation_actual": [2.0],
            "DE_wind_offshore_generation_actual": [3.0],
        },
        index=idx,
    )
    path = tmp_path / DEFAULT_PARQUET_NAME
    df.to_parquet(path)

    out = load_opsd_generation(local_path=path, cache_dir=tmp_path)
    assert list(out.columns) == [
        "DE_solar_generation_actual",
        "DE_wind_onshore_generation_actual",
        "DE_wind_offshore_generation_actual",
    ]
    assert len(out) == 1


def test_force_download_mocked_writes_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = (
        "utc_timestamp,DE_solar_generation_actual,"
        "DE_wind_onshore_generation_actual,DE_wind_offshore_generation_actual\n"
        "2020-01-01T00:00:00Z,1.0,2.0,3.0\n"
        "2020-01-01T01:00:00Z,1.1,2.1,3.1\n"
    )

    class Resp:
        content = csv.encode("utf-8")

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(requests, "get", lambda url, timeout=None: Resp())

    out = load_opsd_generation(force_download=True, cache_dir=tmp_path)
    assert len(out) == 2
    assert (tmp_path / DEFAULT_PARQUET_NAME).is_file()
