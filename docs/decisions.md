# Architecture / decisions log

Cross-links: [README](../README.md), [data_sources](data_sources.md), [methodology](methodology.md).

| ID | Date | Decision | Rationale |
|----|------|----------|-----------|
| ADR-001 | 2026-05-16 | OPSD loader is **local-first** with optional HTTP download into `data/raw/opsd_time_series_60min.parquet`. | Full OPSD CSV is large; reproducible workflows should pin a cached parquet or env path. |
| ADR-002 | 2026-05-16 | Rolling features default to **`shift=1`** (exclude current timestep from rolling window). | Matches common forecasting hygiene — avoids peeking at \(y_t\) when predicting \(y_t\). |
| ADR-003 | 2026-05-16 | Holiday flags compare **calendar dates** (`date()`), not tz-aware timestamps vs naive holidays. | Fixes mismatches between `holidays` date keys and tz-aware `DatetimeIndex` values. |
| ADR-004 | 2026-05-16 | Pre-push hooks run **pytest + coverage** and **documentation freshness** vs upstream. | Enforces testing + docs-before-push policy from README. |
| ADR-005 | 2026-05-16 | Calendar frame uses **`day_of_week`** / **`week_of_year`** (ISO); cyclical layer is **only** `sin_*` / `cos_*` for **hour** and **day-of-year** (365/366-aware). | Matches OPSD-DE baselines; avoids redundant DOW Fourier when `day_of_week` is explicit. |
| ADR-006 | 2026-05-16 | Lag / rolling features use **feature-engine** `LagFeatures` + `WindowFeatures`; rolling defaults **24 / 168** steps; **`periods = shift`** (default 1) for causal alignment with the forecasting transformers. | Standardizes on tested sklearn-compatible transformers; maps `shift` to library ``periods`` explicitly. |
| ADR-007 | 2026-05-25 | NWP ingestion uses **GFS 0.5°** via **Herbie** (`pgrb2.0p50`); default leads **6 / 12 / 24 / 48 h**; point extraction at **52.5°N, 13.4°E** (same as ERA5). | Open NOAA data on AWS; no extra credentials; aligns forecast vs reanalysis ablations on one grid point. ECMWF/ICON deferred until a second product is needed. |

_Add new rows with increasing ADR numbers when changing data sources, split strategy, or weather alignment._
