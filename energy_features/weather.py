"""ERA5 / NWP ingestion (xarray, cdsapi, herbie-data)."""

from __future__ import annotations

import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

ERA5_DATASET = "reanalysis-era5-single-levels"

DEFAULT_ERA5_VARIABLES: tuple[str, ...] = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_cloud_cover",
    "surface_pressure",
)

ERA5_SHORT_TO_LONG: dict[str, str] = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "ssrd": "surface_solar_radiation_downwards",
    "tcc": "total_cloud_cover",
    "sp": "surface_pressure",
}

ERA5_LONG_TO_SHORT: dict[str, str] = {v: k for k, v in ERA5_SHORT_TO_LONG.items()}

# Germany study region — matches solar.py / notebook coords (~52.5°N, 13.4°E).
DEFAULT_LATITUDE = 52.5
DEFAULT_LONGITUDE = 13.4
DEFAULT_BBOX_HALF_DEG = 2.5  # 5° × 5°

TIME_DIM = "time"
ResampleMethod = Literal["mean", "sum"]

# Tidy columns returned by :class:`ERA5Loader` (ready to join with OPSD generation).
ERA5_MERGE_COLUMNS: tuple[str, ...] = (
    "t2m_k",
    "t2m_c",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_u_10m",
    "wind_v_10m",
    "surface_pressure_pa",
    "surface_solar_radiation_w_m2",
    "total_cloud_cover",
)


def _bbox_from_center(latitude: float, longitude: float, half_deg: float) -> list[float]:
    """Return CDS ``area`` as [North, West, South, East]."""
    return [latitude + half_deg, longitude - half_deg, latitude - half_deg, longitude + half_deg]


def _monthly_output_path(cache_dir: Path, year_str: str, month: int) -> Path:
    return cache_dir / f"era5_{year_str}_de_bbox5deg_{month:02d}.nc"


def resolve_era5_monthly_paths(
    year: str | int,
    *,
    cache_dir: Path = Path("data/raw"),
) -> list[Path]:
    """Return sorted monthly ERA5 paths for ``year`` (raises if any month is missing)."""
    year_str = str(year)
    paths = [_monthly_output_path(cache_dir, year_str, month) for month in range(1, 13)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        msg = f"Missing ERA5 monthly files for {year_str}: {missing[:3]}"
        if len(missing) > 3:
            msg += f" … ({len(missing)} total)"
        raise FileNotFoundError(msg)
    return paths


def _months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """Return ``(year, month)`` pairs covering the closed interval ``[start, end]``."""
    y, m = int(start.year), int(start.month)
    end_y, end_m = int(end.year), int(end.month)
    out: list[tuple[int, int]] = []
    while (y, m) <= (end_y, end_m):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def resolve_era5_paths_for_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_dir: Path = Path("data/raw"),
) -> list[Path]:
    """Return monthly ERA5 files overlapping ``[start, end]``."""
    paths: list[Path] = []
    missing: list[str] = []
    for year, month in _months_in_range(start, end):
        path = _monthly_output_path(cache_dir, str(year), month)
        if path.is_file():
            paths.append(path)
        else:
            missing.append(str(path))
    if missing:
        msg = f"Missing ERA5 monthly files for requested range: {missing[:3]}"
        if len(missing) > 3:
            msg += f" … ({len(missing)} total)"
        raise FileNotFoundError(msg)
    return paths


def _coerce_time_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tz: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse ``start`` / ``end`` in ``tz``; date-only ``end`` is inclusive through 23:00."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize(tz)
    else:
        start_ts = start_ts.tz_convert(tz)
    if end_ts.tz is None:
        end_ts = end_ts.tz_localize(tz)
    else:
        end_ts = end_ts.tz_convert(tz)

    if len(str(end).strip()) <= 10:
        end_ts = end_ts.normalize() + pd.Timedelta(hours=23)

    if start_ts > end_ts:
        msg = f"start {start_ts} is after end {end_ts}"
        raise ValueError(msg)
    return start_ts, end_ts


def _utc_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a tz-aware timestamp to timezone-naive UTC (for xarray ``.sel``)."""
    if ts.tz is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def fetch_era5(
    *,
    output_path: str | Path | None = None,
    year: str | int = 2019,
    month: int | None = None,
    variables: Sequence[str] = DEFAULT_ERA5_VARIABLES,
    area: Sequence[float] | None = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    bbox_half_deg: float = DEFAULT_BBOX_HALF_DEG,
    cache_dir: Path = Path("data/raw"),
    force_download: bool = False,
) -> Path:
    """Download hourly ERA5 single-level fields from Copernicus CDS.

    Credentials: ``~/.cdsapirc`` or env ``CDSAPI_URL`` / ``CDSAPI_KEY``
    (see `.env.example` and docs/data_sources.md).

    For a full calendar year, prefer :func:`fetch_era5_year` — CDS rejects
    one-shot requests that exceed account cost limits.
    """
    if month is None:
        raise ValueError("month is required; use fetch_era5_year() for a full calendar year.")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    year_str = str(year)
    if output_path is None:
        out = _monthly_output_path(cache_dir, year_str, month)
    else:
        out = Path(output_path)

    if out.is_file() and not force_download:
        return out

    if area is None:
        area = _bbox_from_center(latitude, longitude, bbox_half_deg)

    request = {
        "product_type": "reanalysis",
        "variable": list(variables),
        "year": year_str,
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": list(area),
        "format": "netcdf",
    }

    cdsapi.Client().retrieve(ERA5_DATASET, request, str(out))
    return out


def fetch_era5_year(
    *,
    year: str | int = 2019,
    variables: Sequence[str] = DEFAULT_ERA5_VARIABLES,
    area: Sequence[float] | None = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    bbox_half_deg: float = DEFAULT_BBOX_HALF_DEG,
    cache_dir: Path = Path("data/raw"),
    force_download: bool = False,
) -> list[Path]:
    """Download one calendar year as twelve monthly NetCDF files (CDS cost-limit friendly)."""
    paths: list[Path] = []
    for month in range(1, 13):
        paths.append(
            fetch_era5(
                year=year,
                month=month,
                variables=variables,
                area=area,
                latitude=latitude,
                longitude=longitude,
                bbox_half_deg=bbox_half_deg,
                cache_dir=cache_dir,
                force_download=force_download,
            )
        )
    return paths


def _load_era5_zip(path: Path) -> xr.Dataset:
    """Read a CDS zip bundle (instant + accumulated streams) into one in-memory dataset."""
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as zf:
        zf.extractall(tmp)
        tmp_path = Path(tmp)
        parts = sorted(tmp_path.glob("*.nc"))
        if not parts:
            msg = f"No NetCDF members found in ERA5 archive: {path}"
            raise ValueError(msg)
        opened = [xr.open_dataset(part, engine="netcdf4") for part in parts]
        try:
            merged = xr.merge(opened, compat="override")
            return merged.load()
        finally:
            for ds in opened:
                ds.close()


def open_era5(path: str | Path) -> xr.Dataset:
    """Open one monthly ERA5 file (CDS zip or plain NetCDF) and standardize coordinates."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        ds = _load_era5_zip(path)
    else:
        with xr.open_dataset(path, engine="netcdf4") as opened:
            ds = opened.load()
    return standardize_era5(ds)


def open_era5_year(
    year: str | int = 2019,
    *,
    cache_dir: Path = Path("data/raw"),
) -> xr.Dataset:
    """Open and concatenate twelve monthly ERA5 files for ``year``."""
    paths = resolve_era5_monthly_paths(year, cache_dir=cache_dir)
    parts = [open_era5(path) for path in paths]
    combined = xr.concat(parts, dim="valid_time" if "valid_time" in parts[0].dims else TIME_DIM)
    return standardize_era5(combined)


def open_era5_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    cache_dir: Path = Path("data/raw"),
    tz: str = "Europe/Berlin",
) -> xr.Dataset:
    """Open and concatenate monthly ERA5 files for ``[start, end]`` (inclusive)."""
    start_ts, end_ts = _coerce_time_range(start, end, tz)
    paths = resolve_era5_paths_for_range(start_ts, end_ts, cache_dir=cache_dir)
    parts = [open_era5(path) for path in paths]
    combined = xr.concat(parts, dim=TIME_DIM)
    return combined.sel(
        {
            TIME_DIM: slice(
                _utc_naive(start_ts.tz_convert("UTC")),
                _utc_naive(end_ts.tz_convert("UTC")),
            )
        }
    )


def standardize_era5(
    ds: xr.Dataset,
    *,
    rename_variables: bool = True,
    mark_utc: bool = True,
) -> xr.Dataset:
    """Normalize CDS ERA5: ``valid_time`` → ``time`` (UTC-naive), drop scalar dims, rename vars."""
    out = ds.copy()

    if "valid_time" in out.coords and TIME_DIM not in out.coords:
        out = out.rename({"valid_time": TIME_DIM})

    for coord in ("number", "expver"):
        if coord in out.dims and out.sizes.get(coord, 0) <= 1:
            out = out.squeeze(coord, drop=True)
        elif coord in out.coords and coord not in out.dims:
            out = out.drop_vars(coord, errors="ignore")

    if mark_utc:
        out = ensure_utc_time(out)

    if rename_variables:
        renames = {
            short: long for short, long in ERA5_SHORT_TO_LONG.items() if short in out.data_vars
        }
        if renames:
            out = out.rename(renames)

    return out


def ensure_utc_time(ds: xr.Dataset) -> xr.Dataset:
    """Mark ``time`` as UTC (stored timezone-naive — xarray datetime64 convention)."""
    if TIME_DIM not in ds.coords:
        msg = f"dataset has no {TIME_DIM!r} coordinate"
        raise ValueError(msg)

    out = ds.assign_coords({TIME_DIM: pd.DatetimeIndex(ds[TIME_DIM].values)})
    out[TIME_DIM].attrs["timezone"] = "UTC"
    return out


def convert_time_zone(ds: xr.Dataset, tz: str) -> xr.Dataset:
    """Relabel ``time`` in ``tz`` (stored timezone-naive for OPSD alignment)."""
    if TIME_DIM not in ds.coords:
        msg = f"dataset has no {TIME_DIM!r} coordinate"
        raise ValueError(msg)

    index = pd.DatetimeIndex(ds[TIME_DIM].values)
    if index.tz is None:
        source_tz = ds[TIME_DIM].attrs.get("timezone", "UTC")
        index = index.tz_localize(source_tz)
    local = index.tz_convert(tz)
    out = ds.assign_coords({TIME_DIM: local.tz_localize(None)})
    out[TIME_DIM].attrs["timezone"] = tz
    return out


def _ensure_time_dim(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.coords and TIME_DIM not in ds.coords:
        return standardize_era5(ds, rename_variables=False, mark_utc=False)
    return ds


def sel_nearest_time(
    ds: xr.Dataset,
    time: str | pd.Timestamp | pd.DatetimeIndex | Sequence[str | pd.Timestamp],
) -> xr.Dataset:
    """Select the nearest ERA5 timestep(s) along ``time``."""
    ds = _ensure_time_dim(ds)
    return ds.sel({TIME_DIM: time}, method="nearest")


def interp_to_point(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
) -> xr.Dataset:
    """Bilinear interpolation to a single ``latitude`` / ``longitude`` point."""
    return ds.interp(latitude=latitude, longitude=longitude)


def wind_speed_direction_from_uv(
    u: xr.DataArray | np.ndarray | float,
    v: xr.DataArray | np.ndarray | float,
) -> tuple[xr.DataArray | np.ndarray | float, xr.DataArray | np.ndarray | float]:
    """Convert 10 m u/v wind (m/s) to speed and meteorological direction (degrees from north).

    Direction follows the meteorological convention: clockwise degrees from north,
    indicating where the wind is **coming from**.
    """
    speed = np.hypot(u, v)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed, direction


def ssrd_hourly_to_flux(ssrd_j_m2: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
    """Convert hourly accumulated surface solar radiation (J/m²) to mean flux (W/m²)."""
    return ssrd_j_m2 / 3600.0


def resample_era5(
    ds: xr.Dataset,
    rule: str,
    *,
    method: ResampleMethod = "mean",
) -> xr.Dataset:
    """Resample along ``time`` (e.g. ``rule='1D'`` for daily aggregates).

    Uses ``mean`` for instantaneous fields. For ``surface_solar_radiation_downwards`` with
    ``method='mean'``, converts hourly J/m² accumulations to W/m² before averaging.
    """
    if TIME_DIM not in ds.dims:
        msg = f"dataset has no {TIME_DIM!r} dimension"
        raise ValueError(msg)

    work = ds
    ssrd_name = "surface_solar_radiation_downwards"
    if method == "mean" and ssrd_name in work.data_vars:
        work = work.assign({ssrd_name: ssrd_hourly_to_flux(work[ssrd_name])})

    resampled = work.resample({TIME_DIM: rule})
    if method == "mean":
        return resampled.mean()
    if method == "sum":
        return resampled.sum()
    msg = f"unsupported resample method: {method!r}"
    raise ValueError(msg)


def extract_era5_point(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
    *,
    tz: str | None = "Europe/Berlin",
    add_wind: bool = True,
    ssrd_as_flux: bool = True,
) -> pd.DataFrame:
    """Point time series at ``latitude`` / ``longitude`` as a pandas DataFrame.

    Steps: ``standardize_era5`` → ``interp`` → optional wind speed/direction → optional
    local timezone on the index.

    For merge-ready column names (``t2m_k``, ``wind_speed_10m``, …), use :class:`ERA5Loader`.
    """
    ds = standardize_era5(ds)
    point = _extract_point_dataset(
        ds,
        latitude,
        longitude,
        add_wind=add_wind,
        ssrd_as_flux=ssrd_as_flux,
    )

    df = point.to_dataframe()
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        source_tz = point[TIME_DIM].attrs.get("timezone", "UTC")
        df.index = df.index.tz_localize(source_tz)
    if tz is not None:
        df.index = df.index.tz_convert(tz)
    return df.sort_index()


def _extract_point_dataset(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
    *,
    add_wind: bool = True,
    ssrd_as_flux: bool = True,
) -> xr.Dataset:
    point = interp_to_point(ds, latitude, longitude)

    if ssrd_as_flux and "surface_solar_radiation_downwards" in point.data_vars:
        flux = ssrd_hourly_to_flux(point["surface_solar_radiation_downwards"])
        point = point.assign(surface_solar_radiation_downwards_w_m2=flux)

    if add_wind:
        u_name = "10m_u_component_of_wind"
        v_name = "10m_v_component_of_wind"
        if u_name in point.data_vars and v_name in point.data_vars:
            speed, direction = wind_speed_direction_from_uv(point[u_name], point[v_name])
            point = point.assign(
                wind_speed_10m=speed,
                wind_direction_10m=direction,
            )
    return point


def _to_merge_ready_dataframe(point: xr.Dataset, tz: str) -> pd.DataFrame:
    """Map interpolated ERA5 variables to tidy columns for generation joins."""
    idx = pd.DatetimeIndex(point[TIME_DIM].values)
    source_tz = point[TIME_DIM].attrs.get("timezone", "UTC")
    if idx.tz is None:
        idx = idx.tz_localize(source_tz)

    raw: dict[str, np.ndarray] = {}

    if "2m_temperature" in point.data_vars:
        t_k = np.asarray(point["2m_temperature"].values, dtype=float)
        raw["t2m_k"] = t_k
        raw["t2m_c"] = t_k - 273.15

    u_name = "10m_u_component_of_wind"
    v_name = "10m_v_component_of_wind"
    if u_name in point.data_vars:
        raw["wind_u_10m"] = np.asarray(point[u_name].values, dtype=float)
    if v_name in point.data_vars:
        raw["wind_v_10m"] = np.asarray(point[v_name].values, dtype=float)

    if "wind_speed_10m" in point.data_vars:
        raw["wind_speed_10m"] = np.asarray(point["wind_speed_10m"].values, dtype=float)
    elif u_name in point.data_vars and v_name in point.data_vars:
        speed, _ = wind_speed_direction_from_uv(point[u_name], point[v_name])
        raw["wind_speed_10m"] = np.asarray(speed, dtype=float)

    if "wind_direction_10m" in point.data_vars:
        raw["wind_direction_10m"] = np.asarray(point["wind_direction_10m"].values, dtype=float)
    elif u_name in point.data_vars and v_name in point.data_vars:
        _, direction = wind_speed_direction_from_uv(point[u_name], point[v_name])
        raw["wind_direction_10m"] = np.asarray(direction, dtype=float)

    if "surface_pressure" in point.data_vars:
        raw["surface_pressure_pa"] = np.asarray(point["surface_pressure"].values, dtype=float)

    if "surface_solar_radiation_downwards_w_m2" in point.data_vars:
        raw["surface_solar_radiation_w_m2"] = np.asarray(
            point["surface_solar_radiation_downwards_w_m2"].values,
            dtype=float,
        )
    elif "surface_solar_radiation_downwards" in point.data_vars:
        raw["surface_solar_radiation_w_m2"] = np.asarray(
            ssrd_hourly_to_flux(point["surface_solar_radiation_downwards"].values),
            dtype=float,
        )

    if "total_cloud_cover" in point.data_vars:
        raw["total_cloud_cover"] = np.asarray(point["total_cloud_cover"].values, dtype=float)

    ordered = {col: raw[col] for col in ERA5_MERGE_COLUMNS if col in raw}
    df = pd.DataFrame(ordered, index=idx)
    df.index = df.index.tz_convert(tz)
    return df.sort_index()


class ERA5Loader:
    """Load tidy hourly ERA5 point weather for merging with generation data.

    Parameters
    ----------
    cache_dir
        Directory with monthly ``era5_{year}_de_bbox5deg_{mm}.nc`` files from
        :func:`fetch_era5_year`.
    tz
        Timezone for the returned index (default ``Europe/Berlin`` — matches OPSD notebooks).

    Examples
    --------
    >>> loader = ERA5Loader()
    >>> wx = loader.load(52.5, 13.4, "2019-01-01", "2019-01-31")
    >>> gen.join(wx, how="inner")  # merge with OPSD generation on hourly index
    """

    def __init__(
        self,
        *,
        cache_dir: Path | str = Path("data/raw"),
        tz: str = "Europe/Berlin",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.tz = tz

    def load(
        self,
        latitude: float,
        longitude: float,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        tz: str | None = None,
    ) -> pd.DataFrame:
        """Return hourly weather at ``latitude`` / ``longitude`` for ``[start, end]``.

        Output columns (see :data:`ERA5_MERGE_COLUMNS`):

        - ``t2m_k`` / ``t2m_c`` — 2 m temperature (K / °C)
        - ``wind_speed_10m``, ``wind_direction_10m``, ``wind_u_10m``, ``wind_v_10m``
        - ``surface_pressure_pa`` — Pa (for :func:`~energy_features.wind.add_wind_features`)
        - ``surface_solar_radiation_w_m2`` — hourly mean flux (W/m²)
        - ``total_cloud_cover`` — fraction 0–1

        Naive ``start`` / ``end`` strings are interpreted in ``tz`` (default
        :attr:`~ERA5Loader.tz`). Date-only ``end`` includes all hours that calendar day.

        The index is ERA5 UTC timestamps converted to ``tz`` (not reindexed to every
        local wall-clock hour). Use ``DataFrame.join(..., how="inner")`` with generation
        data; at month boundaries the first local hour may be absent without the prior
        month's file.
        """
        out_tz = tz if tz is not None else self.tz
        start_ts, end_ts = _coerce_time_range(start, end, out_tz)

        ds = open_era5_range(start_ts, end_ts, cache_dir=self.cache_dir, tz=out_tz)
        point = _extract_point_dataset(ds, latitude, longitude)
        df = _to_merge_ready_dataframe(point, out_tz)
        return df.loc[start_ts:end_ts]


# --- NWP forecast (Herbie / GFS) -------------------------------------------------

DEFAULT_NWP_MODEL = "gfs"
# GFS 0.5° grid (Herbie ``pgrb2.0p50`` product id).
DEFAULT_GFS_PRODUCT = "pgrb2.0p50"
DEFAULT_GFS_LEAD_HOURS: tuple[int, ...] = (6, 12, 24, 48)

# Herbie search string — 2 m temperature, 10 m wind, downward shortwave at surface.
GFS_HERBIE_SEARCH = r":(?:TMP:2 m above|UGRD:10 m above|VGRD:10 m above|DSWRF:surface|DSWRF|SDSWRF)"

# cfgrib short names → ERA5-compatible long names (solar flux is already W/m² in GFS).
GFS_CFGrib_TO_ERA5: dict[str, str] = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "dswrf": "surface_solar_radiation_flux",
    "sdswrf": "surface_solar_radiation_flux",
}

NWP_FORECAST_MERGE_COLUMNS: tuple[str, ...] = (
    "init_time",
    "valid_time",
    "lead_hours",
    "t2m_k",
    "t2m_c",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_u_10m",
    "wind_v_10m",
    "surface_solar_radiation_w_m2",
)


def _normalize_grid_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename lat/lon and wrap longitude to [-180, 180] for ERA5-compatible interpolation."""
    out = ds.copy()
    renames: dict[str, str] = {}
    if "lat" in out.coords and "latitude" not in out.coords:
        renames["lat"] = "latitude"
    if "lon" in out.coords and "longitude" not in out.coords:
        renames["lon"] = "longitude"
    if renames:
        out = out.rename(renames)

    if "longitude" in out.coords and float(out["longitude"].max()) > 180.0:
        lon = ((out["longitude"] + 180.0) % 360.0) - 180.0
        out = out.assign_coords(longitude=lon).sortby("longitude")
    return out


def standardize_nwp(ds: xr.Dataset, *, model: str = DEFAULT_NWP_MODEL) -> xr.Dataset:
    """Normalize Herbie GRIB output: regular lat/lon grid and ERA5-like variable names."""
    out = _normalize_grid_coords(ds)

    if model == DEFAULT_NWP_MODEL:
        renames = {
            short: long for short, long in GFS_CFGrib_TO_ERA5.items() if short in out.data_vars
        }
        if renames:
            out = out.rename(renames)

    if "step" in out.coords and "lead_hours" not in out.coords:
        lead = (pd.to_timedelta(out["step"].values) / pd.Timedelta(hours=1)).astype(int)
        out = out.assign_coords(lead_hours=("step", lead))

    if TIME_DIM in out.coords:
        if TIME_DIM in out.dims:
            out = ensure_utc_time(out)
        else:
            out[TIME_DIM].attrs["timezone"] = "UTC"
    return out


def _herbie_xarray(herbie_obj: object, search: str) -> xr.Dataset:
    ds = herbie_obj.xarray(search, remove_grib=True)  # type: ignore[attr-defined]
    if isinstance(ds, list):
        ds = xr.merge(ds, compat="override")
    return ds


def fetch_gfs_forecast(
    init_time: str | pd.Timestamp,
    fxx: Sequence[int] = DEFAULT_GFS_LEAD_HOURS,
    *,
    product: str = DEFAULT_GFS_PRODUCT,
    search: str = GFS_HERBIE_SEARCH,
    verbose: bool = False,
) -> xr.Dataset:
    """Download GFS open-data forecast fields for one init cycle and multiple lead times.

    Unlike ERA5 reanalysis (hindcast), GFS is an **operational forecast**: each row is
    what the model predicted at ``init_time`` for ``init_time + lead_hours``.

    Uses Herbie to subset GRIB2 from NOAA open-data archives (AWS by default).
    """
    from herbie import Herbie

    init = pd.Timestamp(init_time)
    if init.tz is not None:
        init = init.tz_convert("UTC").tz_localize(None)

    parts: list[xr.Dataset] = []
    for lead in fxx:
        lead_int = int(lead)
        herbie = Herbie(init, model="gfs", product=product, fxx=lead_int, verbose=verbose)
        if herbie.grib is None:
            msg = f"GFS cycle not found: init={init} f{lead_int:03d} product={product}"
            raise FileNotFoundError(msg)
        part = _herbie_xarray(herbie, search)
        if "step" not in part.dims and "step" not in part.coords:
            part = part.expand_dims(step=[pd.Timedelta(hours=lead_int)])
        part = part.assign_coords(lead_hours=int(lead_int))
        parts.append(part)

    combined = (
        parts[0]
        if len(parts) == 1
        else xr.concat(parts, dim="step", compat="override", coords="minimal")
    )
    return standardize_nwp(combined, model=DEFAULT_NWP_MODEL)


def fetch_nwp(
    init_time: str | pd.Timestamp,
    fxx: Sequence[int] = DEFAULT_GFS_LEAD_HOURS,
    *,
    model: str = DEFAULT_NWP_MODEL,
    **kwargs: object,
) -> xr.Dataset:
    """Fetch NWP forecast grid for one init cycle (currently GFS via Herbie)."""
    if model != DEFAULT_NWP_MODEL:
        msg = f"unsupported NWP model {model!r}; only {DEFAULT_NWP_MODEL!r} is implemented"
        raise NotImplementedError(msg)
    return fetch_gfs_forecast(init_time, fxx=fxx, **kwargs)  # type: ignore[arg-type]


def _scalar_timestamp(ds: xr.Dataset, name: str) -> pd.Timestamp | None:
    if name not in ds.coords:
        return None
    value = ds[name].values
    if value.ndim == 0:
        return pd.Timestamp(value).tz_localize("UTC")
    return pd.Timestamp(value.flat[0]).tz_localize("UTC")


def _nwp_forecast_point_to_dataframe(point: xr.Dataset, tz: str | None) -> pd.DataFrame:
    """Map an interpolated multi-lead NWP point dataset to a tidy forecast DataFrame."""
    iter_dim = "step" if "step" in point.dims else None
    n_rows = point.sizes[iter_dim] if iter_dim else 1

    u_name = "10m_u_component_of_wind"
    v_name = "10m_v_component_of_wind"
    if (
        u_name in point.data_vars
        and v_name in point.data_vars
        and "wind_speed_10m" not in point.data_vars
    ):
        speed, direction = wind_speed_direction_from_uv(point[u_name], point[v_name])
        point = point.assign(wind_speed_10m=speed, wind_direction_10m=direction)

    rows: list[dict[str, object]] = []
    for idx in range(n_rows):
        sl = point.isel(step=idx) if iter_dim else point
        init_ts = _scalar_timestamp(sl, TIME_DIM)
        valid_ts = _scalar_timestamp(sl, "valid_time")
        if valid_ts is None and init_ts is not None and "step" in sl.coords:
            valid_ts = init_ts + pd.Timedelta(sl["step"].values)
        if valid_ts is None:
            valid_ts = init_ts

        lead_hours = int(sl["lead_hours"].values) if "lead_hours" in sl.coords else 0
        if lead_hours == 0 and "step" in sl.coords:
            lead_hours = int(pd.Timedelta(sl["step"].values) / pd.Timedelta(hours=1))

        row: dict[str, object] = {
            "init_time": init_ts,
            "valid_time": valid_ts,
            "lead_hours": lead_hours,
        }

        if "2m_temperature" in sl.data_vars:
            t_k = float(sl["2m_temperature"].values)
            row["t2m_k"] = t_k
            row["t2m_c"] = t_k - 273.15
        if u_name in sl.data_vars:
            row["wind_u_10m"] = float(sl[u_name].values)
        if v_name in sl.data_vars:
            row["wind_v_10m"] = float(sl[v_name].values)
        if "wind_speed_10m" in sl.data_vars:
            row["wind_speed_10m"] = float(sl["wind_speed_10m"].values)
        if "wind_direction_10m" in sl.data_vars:
            row["wind_direction_10m"] = float(sl["wind_direction_10m"].values)
        if "surface_solar_radiation_flux" in sl.data_vars:
            row["surface_solar_radiation_w_m2"] = float(sl["surface_solar_radiation_flux"].values)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["init_time"] = pd.to_datetime(df["init_time"], utc=True)
    if tz is not None:
        df["valid_time"] = df["valid_time"].dt.tz_convert(tz)
        df["init_time"] = df["init_time"].dt.tz_convert(tz)
    df = df.set_index("valid_time")
    ordered = [c for c in NWP_FORECAST_MERGE_COLUMNS if c in df.columns or c == "valid_time"]
    out = df[[c for c in ordered if c != "valid_time"]].sort_index()
    out.index.name = "valid_time"
    return out


def nwp_forecast_point_from_dataset(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
    *,
    tz: str | None = "Europe/Berlin",
) -> pd.DataFrame:
    """Extract forecast point values at ``latitude`` / ``longitude`` from a multi-lead NWP grid."""
    ds = standardize_nwp(ds)
    point = interp_to_point(ds, latitude, longitude)
    for dim in ("latitude", "longitude"):
        if dim in point.dims and point.sizes.get(dim, 0) == 1:
            point = point.squeeze(dim, drop=True)
    return _nwp_forecast_point_to_dataframe(point, tz)


def extract_nwp_forecast_point(
    init_time: str | pd.Timestamp,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    fxx: Sequence[int] = DEFAULT_GFS_LEAD_HOURS,
    *,
    model: str = DEFAULT_NWP_MODEL,
    product: str = DEFAULT_GFS_PRODUCT,
    tz: str | None = "Europe/Berlin",
    verbose: bool = False,
) -> pd.DataFrame:
    """Fetch one NWP init cycle and return point forecasts at ERA5 study coordinates.

    Default location matches :data:`DEFAULT_LATITUDE` / :data:`DEFAULT_LONGITUDE`
    (52.5°N, 13.4°E — same as ERA5 downloads).

    Returns one row per lead time with :data:`NWP_FORECAST_MERGE_COLUMNS`.
    """
    if model != DEFAULT_NWP_MODEL:
        msg = f"unsupported NWP model {model!r}; only {DEFAULT_NWP_MODEL!r} is implemented"
        raise NotImplementedError(msg)

    ds = fetch_gfs_forecast(
        init_time,
        fxx=fxx,
        product=product,
        verbose=verbose,
    )
    return nwp_forecast_point_from_dataset(ds, latitude, longitude, tz=tz)
