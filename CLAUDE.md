# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**lightcurve-strategies** — a Python library providing [Hypothesis](https://hypothesis.readthedocs.io/) strategies for generating physically valid [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet) objects and exoplanet transit light curves. Used for property-based testing of light-curve analysis pipelines.

GitHub org: `mit-kavli-institute`

## Environment Setup

- Python 3.14 via Homebrew (`/opt/homebrew/opt/python@3.14/bin/python3.14`)
- Virtual environment in `devpython/` (not the standard `venv/` name)
- Activate: `source devpython/bin/activate`
- Always set `JAX_PLATFORMS=cpu` when running jaxoplanet code

## Build & Install

- Build system: **hatchling** (no dynamic versioning — `pyproject.toml` `version` is the single source of truth)
- `pip install -e .` for runtime deps
- `pip install -e ".[dev]"` for linting/testing
- `pip install -e ".[docs]"` for Sphinx docs

## Project Structure

```
src/lightcurve_strategies/
├── __init__.py          # Public API re-exports
├── _units.py            # Astropy unit helpers
├── transit.py           # transit_orbits() strategy
├── keplerian.py         # centrals(), bodies(), systems() strategies
├── starry.py            # surfaces(), surface_systems() strategies
└── light_curves.py      # light_curves(), noise factories, kernels
tests/
docs/
├── conf.py              # Sphinx config (version from importlib.metadata)
├── _static/             # Committed PNGs for README/docs
└── *.rst                # Sphinx pages
```

## Common Commands

```bash
# Tests
pytest tests/ -v

# Linting (matches CI)
ruff check
ruff format --check
mypy src/
bandit -c pyproject.toml -r src/
codespell

# Pre-commit (runs all of the above)
pre-commit run --all-files

# Docs
sphinx-build docs docs/_build/html -W --keep-going
```

## CI/CD

- **`.github/workflows/ci.yml`** — triggers on push to `main`/`master` and PRs. Three jobs: lint (3.12), test matrix (3.11–3.14), docs build + GitHub Pages deploy.
- **`.github/workflows/release.yml`** — triggers on `v*` tags. Validates tag matches `pyproject.toml` version, builds sdist+wheel, creates GitHub Release with dist artifacts. No PyPI publish — dist files are consumed by a private PyPI index from the GitHub Release.
- Docs are deployed to GitHub Pages via the CI workflow (push only, not PRs).

## Release Process

1. Bump `version` in `pyproject.toml`
2. Commit: `git commit -m "Bump version to X.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push && git push origin vX.Y.Z`

The release workflow handles the rest.

## Key Conventions

- Target Python: `>=3.11`, CI tests 3.11–3.14
- Ruff: line length 88, rules `E,F,I,N,W,UP`
- mypy: `python_version = "3.11"`, third-party imports ignored
- Pre-commit hooks must pass before committing (ruff, ruff-format, mypy, bandit, codespell)
- `docs/conf.py` reads version dynamically via `importlib.metadata` — never hardcode it
- `docs/index.rst` and `README.rst` have parallel content with different image paths (`_static/` vs `docs/_static/`)
