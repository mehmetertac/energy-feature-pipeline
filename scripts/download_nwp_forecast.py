"""CLI: pull a handful of GFS forecast lead times at the ERA5 study point."""

from __future__ import annotations

import argparse

from energy_features.weather import (
    DEFAULT_GFS_LEAD_HOURS,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    GFS_HERBIE_SEARCH,
    extract_nwp_forecast_point,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "init_time",
        nargs="?",
        default="2024-06-01 00:00",
        help="Model initialization time (UTC), e.g. '2024-06-01 00:00'",
    )
    parser.add_argument(
        "-f",
        "--fxx",
        type=int,
        nargs="+",
        default=list(DEFAULT_GFS_LEAD_HOURS),
        help=f"Forecast lead hours (default: {' '.join(str(h) for h in DEFAULT_GFS_LEAD_HOURS)})",
    )
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE)
    args = parser.parse_args()

    df = extract_nwp_forecast_point(
        args.init_time,
        latitude=args.lat,
        longitude=args.lon,
        fxx=args.fxx,
        tz="UTC",
    )
    print(f"GFS point forecast at ({args.lat}, {args.lon}) init={args.init_time} leads={args.fxx}")
    print(f"Herbie search: {GFS_HERBIE_SEARCH}")
    print(df.to_string())


if __name__ == "__main__":
    main()
