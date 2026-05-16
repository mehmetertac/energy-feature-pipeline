"""ERA5 / NWP ingestion (xarray, cdsapi, herbie-data) — planned for a later milestone."""

from __future__ import annotations


def fetch_era5(*args, **kwargs) -> None:  # pragma: no cover
    """Placeholder for CDS ERA5 downloads."""
    raise NotImplementedError("ERA5 pipeline is not implemented yet; see docs/data_sources.md.")


def fetch_nwp(*args, **kwargs) -> None:  # pragma: no cover
    """Placeholder for NWP retrieval (e.g. Herbie)."""
    raise NotImplementedError("NWP pipeline is not implemented yet; see docs/data_sources.md.")
