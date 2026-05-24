"""CLI: request a small ERA5 hourly slice for the DE study region."""

from __future__ import annotations

from energy_features.weather import DEFAULT_ERA5_VARIABLES, fetch_era5_year


def main() -> None:
    paths = fetch_era5_year()
    print(f"ERA5: {len(paths)} monthly files in {paths[0].parent}")
    for path in paths:
        print(f"  {path.name}")
    print(f"Variables: {', '.join(DEFAULT_ERA5_VARIABLES)}")


if __name__ == "__main__":
    main()
