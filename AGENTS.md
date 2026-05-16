# Agent rules (energy-feature-pipeline)

Short mirror of [README.md](README.md) **Project rules**:

1. **Coding:** Python ≥ 3.11, Ruff format/lint, **no file > 1,000 lines** — refactor early.
2. **Docs:** Update **README.md** and relevant **docs/** with every `energy_features/` behavior change; **before every push**, docs must match the branch (pre-push guard).
3. **Tests:** Add **unit tests** for any package change; integration tests under `tests/integration/` with `@pytest.mark.integration` when appropriate; coverage gate **70%** on `energy_features` (see `pyproject.toml`).
4. **Hooks:** Use **pre-commit** (commit: Ruff + nbstripout; push: pytest + docs check). Run `pre-commit install --hook-type pre-commit --hook-type pre-push`.
5. **Read next:** [docs/data_sources.md](docs/data_sources.md), [docs/methodology.md](docs/methodology.md), [docs/decisions.md](docs/decisions.md), sibling [energy-ts-fundamentals README](../energy-ts-fundamentals/README.md).
