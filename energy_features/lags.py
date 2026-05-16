"""Lag and rolling statistics for forecasting (causal / past-only via feature-engine)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
from sklearn.pipeline import Pipeline

DEFAULT_ROLL_WINDOWS: tuple[int, int] = (24, 168)
DEFAULT_ROLL_STATS: tuple[str, ...] = ("mean", "std", "min", "max")
_ALLOWED_ROLL_FUNCS = frozenset(
    {"mean", "std", "min", "max", "sum", "median", "var", "sem", "skew", "kurt"}
)


def _ensure_datetime_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series.index must be a pandas.DatetimeIndex")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series.index must be monotonic increasing")
    if series.index.has_duplicates:
        raise ValueError("series.index must not contain duplicates")
    return series


def _series_prefix(series: pd.Series, prefix: str | None) -> str:
    if prefix is not None:
        return prefix
    if series.name:
        return str(series.name)
    return "value"


def _base_frame(series: pd.Series, name: str) -> pd.DataFrame:
    """Single-column frame with float values (feature-engine expects numeric, no NaNs)."""
    if series.isna().any():
        raise ValueError("series must not contain NaNs for lag/window features.")
    return pd.DataFrame({name: series.astype(float).to_numpy()}, index=series.index)


def _rename_window_columns(frame: pd.DataFrame, base: str) -> pd.DataFrame:
    """Map ``{base}_window_{w}_{fn}`` → ``{base}_roll{w}_{fn}`` for backward compatibility."""
    prefix = f"{base}_window_"
    mapping: dict[str, str] = {}
    for col in frame.columns:
        if not col.startswith(prefix):
            continue
        rest = col[len(prefix) :]
        window_part, _, fn_part = rest.partition("_")
        if window_part.isdigit() and fn_part:
            mapping[col] = f"{base}_roll{window_part}_{fn_part}"
    return frame.rename(columns=mapping)


def make_lag_features(
    series: pd.Series,
    lags: Sequence[int],
    *,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add lag columns ``{prefix}_lag_{k}`` via :class:`LagFeatures` (strictly past values)."""
    s = _ensure_datetime_series(series)
    name = _series_prefix(s, prefix)
    if not lags:
        return pd.DataFrame(index=s.index)

    for k in lags:
        if k < 1:
            raise ValueError(f"lags must be positive integers, got {k}")

    df = _base_frame(s, name)
    tx = LagFeatures(
        variables=[name],
        periods=list(dict.fromkeys(lags)),
        missing_values="raise",
        drop_original=True,
        sort_index=True,
    )
    return tx.fit_transform(df)


def make_rolling_features(
    series: pd.Series,
    windows: Sequence[int] | None = None,
    *,
    stats: Sequence[str] = DEFAULT_ROLL_STATS,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.DataFrame:
    """Rolling mean/std/min/max via :class:`WindowFeatures` (causal).

    ``WindowFeatures`` aggregates past observations then shifts by ``periods``. We set
    ``periods`` equal to ``shift`` so that at time *t* the window **never** uses target
    values after *t* (default ``shift=1`` excludes the contemporaneous value from the
    rolling slice used at *t*).

    Column names follow the historical convention ``{name}_roll{window}_{stat}``.

    Notes
    -----
    Pandas rolling ``std`` uses ``ddof=1`` (sample std) unless you pass a custom
    aggregation via feature-engine supported callables.

    Parameters
    ----------
    windows :
        Window lengths in rows (e.g. hours for hourly data). Defaults to **24** and **168**.
    shift :
        Passed to ``WindowFeatures(periods=…)``. Must be ``>= 1`` for leakage-safe defaults.
    """
    s = _ensure_datetime_series(series)
    name = _series_prefix(s, None)

    if windows is None:
        windows = DEFAULT_ROLL_WINDOWS
    win_list = list(windows)
    if not win_list:
        return pd.DataFrame(index=s.index)

    if shift < 1:
        raise ValueError(
            "shift must be >= 1 for causal window features (maps to WindowFeatures.periods)."
        )

    unknown = [fn for fn in stats if fn not in _ALLOWED_ROLL_FUNCS]
    if unknown:
        raise ValueError(f"Unknown stat {unknown[0]!r}; allowed {sorted(_ALLOWED_ROLL_FUNCS)}")

    for w in win_list:
        if w < 1:
            raise ValueError(f"windows must be positive integers, got {w}")

    df = _base_frame(s, name)
    wx = WindowFeatures(
        variables=[name],
        window=win_list,
        functions=list(stats),
        periods=shift,
        min_periods=min_periods,
        missing_values="raise",
        drop_original=True,
        sort_index=True,
    )
    out = wx.fit_transform(df)
    return _rename_window_columns(out, name)


def make_lag_rolling_block(
    series: pd.Series,
    lags: Sequence[int],
    windows: Sequence[int] | None = None,
    *,
    rolling_stats: Sequence[str] = DEFAULT_ROLL_STATS,
    min_periods: int | None = None,
    shift: int = 1,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Lag columns plus rolling blocks in one pass (:class:`Pipeline`)."""
    s = series.copy()
    if prefix is not None:
        s = s.rename(prefix)

    s = _ensure_datetime_series(s)
    name = _series_prefix(s, None)

    if windows is None:
        windows = DEFAULT_ROLL_WINDOWS
    win_list = list(windows)

    if shift < 1:
        raise ValueError(
            "shift must be >= 1 for causal window features (maps to WindowFeatures.periods)."
        )

    unknown = [fn for fn in rolling_stats if fn not in _ALLOWED_ROLL_FUNCS]
    if unknown:
        raise ValueError(f"Unknown stat {unknown[0]!r}; allowed {sorted(_ALLOWED_ROLL_FUNCS)}")

    for k in lags:
        if k < 1:
            raise ValueError(f"lags must be positive integers, got {k}")
    for w in win_list:
        if w < 1:
            raise ValueError(f"windows must be positive integers, got {w}")

    df = _base_frame(s, name)

    steps: list[tuple[str, LagFeatures | WindowFeatures]] = []

    if lags:
        steps.append(
            (
                "lags",
                LagFeatures(
                    variables=[name],
                    periods=list(dict.fromkeys(lags)),
                    missing_values="raise",
                    drop_original=not win_list,
                    sort_index=True,
                ),
            )
        )

    if win_list:
        steps.append(
            (
                "windows",
                WindowFeatures(
                    variables=[name],
                    window=win_list,
                    functions=list(rolling_stats),
                    periods=shift,
                    min_periods=min_periods,
                    missing_values="raise",
                    drop_original=True,
                    sort_index=True,
                ),
            )
        )

    if not steps:
        return pd.DataFrame(index=s.index)

    pipe: Pipeline = Pipeline(steps)
    out = pipe.fit_transform(df)

    if name in out.columns:
        out = out.drop(columns=[name])

    return _rename_window_columns(out, name)
