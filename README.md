# energy-feature-pipeline

Master the **data layer** of renewable forecasting — calendar / physics features and **NWP / ERA5** weather pipelines — and quantify how much forecast skill is **“weather”** vs **“model”**.

This repo scaffolds:

1. **Join** solar / wind generation (Open Power System Data — OPSD) with ERA5 reanalysis + a forecast NWP product (later milestones).
2. **Baselines**: calendar-only vs richer feature blocks; measure **lift** over naive / seasonal baselines with honest time-based splits.
3. **Credentials**: register for a free Copernicus CDS account and store your API key in `~/.cdsapirc` (see [.env.example](.env.example)); **do not commit** secrets.

Related course repo (hourly load parquet built there): [energy-ts-fundamentals README](../energy-ts-fundamentals/README.md).

---

## Roadmap (implemented vs planned)

| Milestone | Scope |
|-----------|--------|
| **Task 1 (this scaffold)** | Package layout, `calendar.py`, `lags.py`, `thermal.py` (HDD/CDD), `solar.py` (pvlib clear-sky), OPSD loader with local-first + download fallback, notebook comparing calendar-only vs calendar+lags + LightGBM, tests + hooks. |
| Next | `wind.py` (windpowerlib), `weather.py` (ERA5 `cdsapi`, NWP e.g. Herbie), weather-vs-model ablations. |

---

## Quick start

```bash
# Python ≥ 3.11 recommended (3.13 works locally if wheels exist).
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type pre-push
pytest -q --cov=energy_features --cov-report=term-missing
```

On Windows, if `python` is not on `PATH`, use `py -m pip` / `py -m pytest` instead. The same applies to **`py -m pre_commit …`** if the `pre-commit` launcher is not on `PATH`.

**OPSD generation:** set `ENERGY_FP_OPSD_PATH` to a cached parquet/CSV extract, or let [`load_opsd_generation`](energy_features/data_loader.py) download the large OPSD CSV once into `data/raw/opsd_time_series_60min.parquet` (slow; cache preferred).

**Sibling ENTSO-E load parquet:** point `ENERGY_TS_LOAD_PARQUET` at `../energy-ts-fundamentals/data/raw/de_lu_load_hourly.parquet` (or copy the file).

---

## Layout

| Path | Role |
|------|------|
| [`energy_features/calendar.py`](energy_features/calendar.py) | Calendar + holiday features + sin/cos hour & day-of-year |
| [`energy_features/lags.py`](energy_features/lags.py) | Lag / rolling via feature-engine (`LagFeatures`, `WindowFeatures`; default 24h / 168h) |
| [`energy_features/data_loader.py`](energy_features/data_loader.py) | OPSD + optional load parquet |
| [`energy_features/thermal.py`](energy_features/thermal.py) | Heating/cooling degree amounts from temperature (HDD/CDD vs 18 °C base) |
| [`energy_features/solar.py`](energy_features/solar.py) | pvlib solar position + clear-sky GHI/DNI/DHI |
| [`energy_features/wind.py`](energy_features/wind.py) | Stub → windpowerlib later |
| [`energy_features/weather.py`](energy_features/weather.py) | Stub → ERA5 / NWP later |
| [`notebooks/01_calendar_lag_baseline.ipynb`](notebooks/01_calendar_lag_baseline.ipynb) | Task 1 demo |
| [`docs/`](docs/) | Data sources, methodology, decisions |
| [`tests/`](tests/) | Unit tests |

---

## Project rules (configuration)

These rules apply to humans and agents working in this repository.

### Coding standards

- **Python** ≥ 3.11; prefer modern typing (`from __future__ import annotations` where helpful).
- Format / lint with **Ruff** (`ruff format`, `ruff check`) via pre-commit.
- **No source file should exceed 1,000 lines.** If a file approaches that size, split by feature family (e.g. `_holiday_features`, loaders vs transforms) and update imports/tests.

### Documentation rules

- **`README.md`** and at least one relevant file under **`docs/`** must be updated **in the same commit** as any behavior change under `energy_features/`.
- **Before every push to the remote**, documentation must be current with the changes being pushed (see **Git hooks**).

### Testing requirements

- **Always add minimal unit tests** for behavior changes under `energy_features/` (`tests/`).
- Add **`tests/integration/`** (marker `integration`) for downloads / heavy IO when the project supports them; run with `pytest -m integration`.
- Default CI / pre-push runs `pytest -q --cov=energy_features` with **`fail_under = 70`** in `[tool.coverage.report]` ([`pyproject.toml`](pyproject.toml)).

### Git hooks

- Framework: [**pre-commit**](https://pre-commit.com/).
- **pre-commit** stage: trailing whitespace / YAML checks, **Ruff** format + check, **nbstripout** on notebooks.
- **pre-push** stage: **`pytest --cov`** with the coverage floor, plus **`scripts/check_docs_updated.py`** (fails if `energy_features/` changed vs upstream but neither `README.md` / `AGENTS.md` nor `docs/` did).

Install hooks:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

If hooks are missing in a fresh clone, **create or restore** `.pre-commit-config.yaml` and run the command above.

### References — where to keep reading

| Doc | Purpose |
|-----|---------|
| [docs/data_sources.md](docs/data_sources.md) | OPSD, CDS ERA5, NWP / Herbie notes |
| [docs/methodology.md](docs/methodology.md) | Baselines, splits, feature blocks, metrics |
| [docs/decisions.md](docs/decisions.md) | ADR-style decisions |
| [AGENTS.md](AGENTS.md) | Short agent-facing summary of these rules |
| [energy-ts-fundamentals README](../energy-ts-fundamentals/README.md) | Hourly load parquet + ENTSO-E workflow |

External links:

- [Open Power System Data — Time series](https://open-power-system-data.org/)
- [Copernicus CDS](https://cds.climate.copernicus.eu/)
- [Herbie (NWP downloads)](https://github.com/blaylockbk/Herbie)

---

## License

See [LICENSE](LICENSE).
