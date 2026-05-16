"""Load OPSD time series (generation) with local-first resolution and optional download."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests

OPSD_TIMESERIES_URL = (
    "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
)
DEFAULT_PARQUET_NAME = "opsd_time_series_60min.parquet"
UTC_TS_COL = "utc_timestamp"


def _normalize_country(code: str) -> str:
    return code.strip().upper()


def resolve_opsd_parquet_path(
    *,
    local_path: str | Path | None = None,
    cache_dir: Path | None = None,
    env_var: str = "ENERGY_FP_OPSD_PATH",
) -> Path | None:
    """Return first existing parquet path among explicit value, env, and default cache."""
    candidates: list[Path] = []
    if local_path is not None:
        candidates.append(Path(local_path))
    env = os.environ.get(env_var, "").strip()
    if env:
        candidates.append(Path(env))
    base = cache_dir or Path("data/raw")
    candidates.append(base / DEFAULT_PARQUET_NAME)

    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.is_absolute() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    return None


def _generation_column_names(country: str, tech: tuple[str, ...]) -> list[str]:
    ctry = _normalize_country(country)
    return [f"{ctry}_{t}_generation_actual" for t in tech]


def _read_opsd_csv_bytes(data: bytes, country: str, tech: tuple[str, ...]) -> pd.DataFrame:
    cols = [UTC_TS_COL, *_generation_column_names(country, tech)]
    df = pd.read_csv(
        io.BytesIO(data),
        usecols=lambda c: c in set(cols),
        encoding="utf-8-sig",
    )
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {sorted(missing)}")
    df[UTC_TS_COL] = pd.to_datetime(df[UTC_TS_COL], utc=True)
    df = df.set_index(UTC_TS_COL).sort_index()
    df = df.astype("float64")
    return df


def _download_opsd_generation(country: str, tech: tuple[str, ...]) -> pd.DataFrame:
    resp = requests.get(OPSD_TIMESERIES_URL, timeout=600)
    resp.raise_for_status()
    return _read_opsd_csv_bytes(resp.content, country=country, tech=tech)


def load_opsd_generation(
    *,
    local_path: str | Path | None = None,
    cache_dir: Path = Path("data/raw"),
    country: str = "DE",
    tech: tuple[str, ...] = ("solar", "wind_onshore", "wind_offshore"),
    force_download: bool = False,
) -> pd.DataFrame:
    """Load hourly OPSD *actual generation* columns for ``country`` and ``tech``.

    Resolution order (unless ``force_download``):

    #. ``local_path``
    #. ``ENERGY_FP_OPSD_PATH``
    #. ``{cache_dir}/opsd_time_series_60min.parquet`` if it exists
    #. Download from OPSD and write that parquet

    Note
    ----
    The upstream CSV is large; prefer caching the parquet once or setting
    ``ENERGY_FP_OPSD_PATH`` to a local extract.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force_download:
        resolved = resolve_opsd_parquet_path(local_path=local_path, cache_dir=cache_dir)
        if resolved is not None:
            df = pd.read_parquet(resolved)
            expected = _generation_column_names(country, tech)
            missing = [c for c in expected if c not in df.columns]
            if missing:
                raise ValueError(f"Parquet at {resolved} missing columns: {missing}")
            out = df[expected].sort_index()
            if not isinstance(out.index, pd.DatetimeIndex):
                out.index = pd.to_datetime(out.index, utc=True)
            return out.astype("float64")

    df_dl = _download_opsd_generation(country=country, tech=tech)
    out_path = cache_dir / DEFAULT_PARQUET_NAME
    df_dl.to_parquet(out_path)
    return df_dl


def resolve_load_parquet_path(
    parquet_path: str | Path | None = None,
    *,
    env_var: str = "ENERGY_TS_LOAD_PARQUET",
    extra_candidates: tuple[str | Path, ...] = (),
) -> Path:
    """Resolve ENTSO-E / parquet load path (compatible with energy-ts-fundamentals env)."""
    candidates: list[Path] = []
    if parquet_path is not None:
        candidates.append(Path(parquet_path))
    env = os.environ.get(env_var, "").strip()
    if env:
        candidates.append(Path(env))
    candidates.extend(Path(p) for p in extra_candidates)

    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.is_absolute() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find hourly load parquet. Tried: " + ", ".join(str(p) for p in candidates)
    )


def load_opsd_load(
    parquet_path: str | Path | None = None,
    *,
    tz: str = "Europe/Berlin",
    value_col: str = "load_mw",
    extra_path_candidates: tuple[str | Path, ...] = (),
) -> pd.Series:
    """Load hourly load from parquet (e.g. sibling ``de_lu_load_hourly.parquet``)."""
    path = resolve_load_parquet_path(
        parquet_path,
        extra_candidates=extra_path_candidates,
    )
    load_hourly = pd.read_parquet(path)
    idx = load_hourly.index
    if not isinstance(idx, pd.DatetimeIndex):
        load_hourly.index = pd.to_datetime(idx, utc=True)
    if load_hourly.index.tz is None:
        load_hourly.index = load_hourly.index.tz_localize(
            tz, ambiguous="infer", nonexistent="shift_forward"
        )
    else:
        load_hourly.index = load_hourly.index.tz_convert(tz)

    return (
        load_hourly[value_col]
        .sort_index()
        .asfreq("1h")
        .interpolate(limit_direction="both")
        .astype(float)
    )
