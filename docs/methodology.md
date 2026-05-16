# Methodology

Cross-links: [README](../README.md), [data_sources](data_sources.md), [decisions](decisions.md).

## Task 1 — Calendar & lag baselines

### Feature blocks

1. **Calendar / holiday** — [`make_calendar_features`](../energy_features/calendar.py): `hour`, `day_of_week`, `month`, ISO `week_of_year`, `is_weekend`; optional sin/cos encodings for **hour** and **day-of-year**; `holidays`-based flags for `country` (default **DE**), plus days to / since nearest holiday.
2. **Lag / rolling** — [`make_lag_features`](../energy_features/lags.py), [`make_rolling_features`](../energy_features/lags.py): **feature-engine** `LagFeatures` / `WindowFeatures` with default windows **24 h** and **168 h** and stats **mean, std, min, max**. `WindowFeatures(..., periods=shift)` with default **`shift=1`** matches the library's causal forecasting semantics (no future values; contemporaneous target excluded from the rolling slice at \(t\)). [`make_lag_rolling_block`](../energy_features/lags.py) wires both via a sklearn `Pipeline`.

### Baselines & lift

- Fit **LightGBMRegressor** (or sklearn baseline for debugging) on **train**, evaluate on **held-out tail** (e.g. last 90 days — configurable).
- Compare **calendar-only** \(X_{\text{cal}}\) vs **calendar + lags/rolls** \(X_{\text{cal+lag}}\).
- Report **lift %** per technology:
  \[
    \text{lift}_\text{MAE} = \frac{\text{MAE}_{\text{cal}} - \text{MAE}_{\text{cal+lag}}}{\text{MAE}_{\text{cal}}}
  \]
  (same for RMSE / sMAPE if desired).

### Honesty checks

- Strict **time ordering** — no random split across time for operational forecasts.
- Inspect **residual ACF** after calendar+lags (notebook) — persistent structure motivates **weather** features next.

## Later milestones (stubs today)

- **Physics:** pvlib clear-sky / POA (`solar.py`), windpowerlib curves (`wind.py`).
- **Weather vs model:** align ERA5 / NWP leads with generation targets; ablations: calendar-only → +weather → +physics.
